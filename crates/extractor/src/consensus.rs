use web_search_common::models::Page;
use web_search_common::models::PageMetadata;
use web_search_common::config::ExtractorConfig;
use chrono::Utc;
use regex::Regex;
use serde_json::Value;

use crate::hydration;
use crate::metadata;
use crate::readability;
use crate::trafilatura;
use crate::ExtractionResult;

/// Strip script, style, svg, noscript, and other non-content tags from HTML
/// BEFORE passing to extraction algorithms. Prevents CSS/JS leaks in body text.
fn sanitize_html(html: &str) -> String {
    // Remove entire tag blocks that contain non-content
    let patterns = [
        r"(?si)<script[^>]*>.*?</script>",
        r"(?si)<style[^>]*>.*?</style>",
        r"(?si)<svg[^>]*>.*?</svg>",
        r"(?si)<noscript[^>]*>.*?</noscript>",
        r"(?si)<iframe[^>]*>.*?</iframe>",
        r"(?si)<!--.*?-->",
    ];

    let mut cleaned = html.to_string();
    for pattern in &patterns {
        let re = Regex::new(pattern).unwrap();
        cleaned = re.replace_all(&cleaned, "").to_string();
    }

    // Remove inline style attributes
    let style_attr = Regex::new(r#"\s*style="[^"]*""#).unwrap();
    cleaned = style_attr.replace_all(&cleaned, "").to_string();

    cleaned
}

/// Clean extracted body text: remove residual CSS, collapse whitespace.
fn clean_css_artifacts(text: &str) -> String {
    let mut cleaned = text.to_string();

    // Remove any CSS that leaked through (e.g., .mw-parser-output ul.cslist{...})
    let css_block = Regex::new(r"(?s)[.\w][\w.-]*(?:\s+[\w.#\[\]=:,>+~ -]+)*\s*\{[^}]*\}").unwrap();
    cleaned = css_block.replace_all(&cleaned, "").to_string();

    // Remove CSS-like property declarations
    let _css_prop = Regex::new(r"[a-z-]+\s*:\s*[^;]+;\s*").unwrap();
    // Only remove if line looks entirely like CSS (no regular sentence structure)
    let lines: Vec<&str> = cleaned.lines().collect();
    let cleaned_lines: Vec<&str> = lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            // Skip lines that look like pure CSS
            if trimmed.starts_with('.') && trimmed.contains('{') {
                return false;
            }
            if trimmed.starts_with("@media") || trimmed.starts_with("@import") {
                return false;
            }
            true
        })
        .copied()
        .collect();
    cleaned = cleaned_lines.join("\n");

    // Collapse multiple whitespace/newlines
    let multi_newline = Regex::new(r"\n{3,}").unwrap();
    cleaned = multi_newline.replace_all(&cleaned, "\n\n").to_string();

    let multi_space = Regex::new(r"[ \t]{2,}").unwrap();
    cleaned = multi_space.replace_all(&cleaned, " ").to_string();

    cleaned.trim().to_string()
}

/// Extract content from HTML using multi-pass consensus voting.
///
/// Pre-step: Sanitize HTML (strip script/style/svg/noscript)
/// Pass 1: Trafilatura-RS (density-based, F1~0.85)
/// Pass 2: Readability (DOM scoring, F1~0.80)
///
/// Post-step: Clean body text (remove residual CSS, collapse whitespace)
///
/// Default entry point — data-layer acquisition is OFF, so output is identical to
/// the pre-data-layer pipeline. To enable in-band hydration salvage / structured
/// promotion (ADR 0002), call [`extract_page_with_config`] with a config that has
/// `enable_data_layer = true`.
pub fn extract_page(html: &str, url: &str) -> ExtractionResult {
    extract_base(html, url)
}

/// Config-aware extraction. Runs the standard consensus pipeline, then — only when
/// `cfg.enable_data_layer` is set — applies the two in-band, near-free data-layer
/// techniques from ADR 0002:
///   1.5.2 structured-data promotion (JSON-LD `articleBody` / OG → primary content)
///   1.5.1 hydration-state salvage (`__NEXT_DATA__`/`__NUXT__`/state/RSC blobs)
///
/// Both are additive: metadata fields are only *filled when missing*, and the body
/// is only *replaced when the data-layer text is richer* than the consensus body.
/// With `enable_data_layer = false` this is byte-for-byte equal to [`extract_page`].
pub fn extract_page_with_config(html: &str, url: &str, cfg: &ExtractorConfig) -> ExtractionResult {
    let mut result = extract_base(html, url);

    if !cfg.enable_data_layer {
        return result;
    }

    if cfg.data_layer_structured_promotion {
        promote_structured(&mut result);
    }
    if cfg.data_layer_hydration {
        salvage_from_hydration(html, &mut result);
    }

    result
}

/// The original consensus extraction. Kept separate so [`extract_page`] stays
/// byte-for-byte identical regardless of data-layer changes.
fn extract_base(html: &str, url: &str) -> ExtractionResult {
    // Sanitize HTML before extraction
    let clean_html = sanitize_html(html);

    // Run both extraction passes in parallel (saves ~50-150ms per page)
    let clean_for_read = clean_html.clone();
    let (traf_result, read_result) = rayon::join(
        || trafilatura::extract(&clean_html),
        || readability::extract(&clean_for_read),
    );

    // Pick best body text by confidence, then clean
    let (body_text, confidence) = if traf_result.confidence >= read_result.confidence {
        (clean_css_artifacts(&traf_result.body_text), traf_result.confidence)
    } else {
        (clean_css_artifacts(&read_result.body_text), read_result.confidence)
    };

    // If both produced content, boost confidence
    let both_agree = !traf_result.body_text.is_empty() && !read_result.body_text.is_empty();
    let final_confidence = if both_agree {
        (confidence * 1.1).min(1.0)
    } else {
        confidence
    };

    // Merge headings (prefer trafilatura, deduplicate)
    let headings = if !traf_result.headings.is_empty() {
        traf_result.headings
    } else {
        read_result.headings
    };

    // Pick title: prefer trafilatura, fallback readability
    let title = traf_result.title.or(read_result.title);

    // Extract metadata
    let meta = metadata::extract_metadata(html, url);

    ExtractionResult {
        title,
        author: meta.author,
        published_date: meta.published_date,
        body_text,
        headings,
        links: meta.links,
        tables: meta.tables,
        language: meta.language,
        description: meta.description,
        json_ld: meta.json_ld,
        open_graph: meta.open_graph,
        extraction_confidence: final_confidence,
    }
}

// ── Data-Layer Acquisition (ADR 0002, Rung -1 IN-band) ───────────────────────

/// Body length (chars) at/below which the consensus extraction is considered
/// "thin" — i.e. likely soft-blocked or a JS shell, a candidate for salvage.
const THIN_BODY_CHARS: usize = 200;
/// Confidence floor below which we treat the extraction as low-quality.
const LOW_CONFIDENCE: f32 = 0.6;

/// True when `candidate` is substantive enough to replace `current` as body text.
/// Requires the candidate to be meaningful on its own AND either the current body
/// is near-empty or the candidate is meaningfully (>20%) longer.
fn is_richer(candidate: &str, current: &str) -> bool {
    let c = candidate.trim().chars().count();
    if c < THIN_BODY_CHARS {
        return false;
    }
    let cur = current.trim().chars().count();
    cur < 50 || (c as f32) > (cur as f32) * 1.2
}

/// 1.5.2 — Promote already-parsed structured data (JSON-LD Article / OpenGraph)
/// to primary content when richer than the consensus body. Fills missing metadata
/// unconditionally; replaces the body only when `is_richer`.
fn promote_structured(result: &mut ExtractionResult) {
    if let Some(article) = result.json_ld.as_ref().and_then(json_ld_article) {
        if result.title.is_none() {
            result.title = article.headline.clone();
        }
        if result.description.is_none() {
            result.description = article.description.clone();
        }
        if result.author.is_none() {
            result.author = article.author.clone();
        }
        if result.published_date.is_none() {
            if let Some(d) = article.date.as_deref().and_then(metadata::parse_date) {
                result.published_date = Some(d);
            }
        }
        if let Some(body) = article.article_body.as_deref() {
            if is_richer(body, &result.body_text) {
                result.body_text = clean_body_text(body);
                result.extraction_confidence = result.extraction_confidence.max(0.75);
            }
        }
    }

    // OG carries title/description only (never a full body) — fill if still missing.
    if let Some(og) = result.open_graph.as_ref() {
        if result.title.is_none() {
            result.title = og.og_title.clone();
        }
        if result.description.is_none() {
            result.description = og.og_description.clone();
        }
    }
}

/// 1.5.1 — Salvage content from an embedded hydration blob when the consensus
/// extraction is thin (soft-block / JS shell). Fills missing metadata and replaces
/// the body only when the hydration text is richer.
fn salvage_from_hydration(html: &str, result: &mut ExtractionResult) {
    let still_thin = result.extraction_confidence < LOW_CONFIDENCE
        || result.body_text.trim().chars().count() < THIN_BODY_CHARS;
    if !still_thin {
        return;
    }

    let Some(h) = hydration::extract(html) else {
        return;
    };

    if result.title.is_none() {
        result.title = h.title;
    }
    if result.description.is_none() {
        result.description = h.description;
    }
    if result.author.is_none() {
        result.author = h.author;
    }
    if result.published_date.is_none() {
        if let Some(d) = h.published.as_deref().and_then(metadata::parse_date) {
            result.published_date = Some(d);
        }
    }
    if is_richer(&h.body_text, &result.body_text) {
        tracing::debug!(source = h.source, "data-layer: salvaged body from hydration blob");
        result.body_text = clean_body_text(&h.body_text);
        result.extraction_confidence = result.extraction_confidence.max(0.6);
    }
}

/// Structured article fields harvested from a JSON-LD blob.
struct StructuredArticle {
    headline: Option<String>,
    description: Option<String>,
    author: Option<String>,
    date: Option<String>,
    article_body: Option<String>,
}

/// Locate an Article-like node in a JSON-LD value (handles a bare node, a top-level
/// array of nodes, and `@graph` arrays) and pull out its content fields.
fn json_ld_article(value: &Value) -> Option<StructuredArticle> {
    fn is_article_type(node: &Value) -> bool {
        let t = &node["@type"];
        let matches = |s: &str| {
            let s = s.to_ascii_lowercase();
            s.contains("article") || s == "blogposting" || s == "report"
        };
        match t {
            Value::String(s) => matches(s),
            Value::Array(arr) => arr.iter().any(|v| v.as_str().map(matches).unwrap_or(false)),
            _ => false,
        }
    }

    fn from_node(node: &Value) -> Option<StructuredArticle> {
        if !is_article_type(node) {
            return None;
        }
        let get = |k: &str| node[k].as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let author = match &node["author"] {
            Value::String(s) => Some(s.trim().to_string()),
            Value::Object(_) => node["author"]["name"].as_str().map(|s| s.trim().to_string()),
            Value::Array(arr) => arr
                .first()
                .and_then(|a| a.get("name").and_then(|n| n.as_str()).or_else(|| a.as_str()))
                .map(|s| s.trim().to_string()),
            _ => None,
        }
        .filter(|s| !s.is_empty());

        Some(StructuredArticle {
            headline: get("headline").or_else(|| get("name")),
            description: get("description"),
            author,
            date: get("datePublished").or_else(|| get("dateCreated")),
            article_body: get("articleBody"),
        })
    }

    // Candidate node lists: the value itself, its array elements, or its @graph.
    if let Some(a) = from_node(value) {
        return Some(a);
    }
    if let Value::Array(arr) = value {
        for node in arr {
            if let Some(a) = from_node(node) {
                return Some(a);
            }
        }
    }
    if let Value::Array(graph) = &value["@graph"] {
        for node in graph {
            if let Some(a) = from_node(node) {
                return Some(a);
            }
        }
    }
    None
}

/// Convert extraction result + crawl metadata into a Page model.
pub fn to_page(
    extraction: &ExtractionResult,
    url: &str,
    domain: &str,
    content_hash: &str,
    status_code: u16,
    response_time_ms: u64,
    content_length: usize,
) -> Page {
    Page {
        url: url.to_string(),
        domain: domain.to_string(),
        title: extraction.title.clone(),
        author: extraction.author.clone(),
        published_date: extraction.published_date,
        body_text: extraction.body_text.clone(),
        headings: extraction.headings.clone(),
        links: extraction.links.clone(),
        tables: extraction.tables.clone(),
        metadata: PageMetadata {
            language: extraction.language.clone(),
            description: extraction.description.clone(),
            content_type: "text/html".to_string(),
            status_code,
            response_time_ms,
            content_length,
            extraction_confidence: extraction.extraction_confidence,
            json_ld: extraction.json_ld.clone(),
            open_graph: extraction.open_graph.clone(),
        },
        content_hash: content_hash.to_string(),
        crawled_at: Utc::now(),
    }
}

// ── Data Quality Filters ─────────────────────────────────────────────

/// Public filter: remove code blocks, CTAs, navigation, citations, and fix unicode.
/// Call on body_text after extraction to clean up non-factual content.
pub fn clean_body_text(text: &str) -> String {
    let cleaned: String = text
        .lines()
        .filter(|line| !is_code_line(line))
        .filter(|line| !is_cta_line(line))
        .filter(|line| !is_navigation_line(line))
        .filter(|line| !is_citation_footnote_line(line))
        .filter(|line| !is_retrieved_or_archived_line(line))
        .map(|line| strip_reference_brackets(line))
        .collect::<Vec<_>>()
        .join("\n");
    normalize_unicode(&cleaned)
}

/// Detect if a line is source code.
fn is_code_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    let code_starts = [
        "fn ", "pub fn ", "async fn ", "impl ", "struct ", "enum ", "trait ", "mod ",
        "use ", "#[", "//", "/*", "*/",
        "def ", "class ", "import ", "from ", "__",
        "function ", "const ", "let ", "var ", "=>",
        "#include", "int main", "void ", "std::", "namespace ",
        "package ", "public class", "private ", "protected ",
    ];
    if code_starts.iter().any(|p| trimmed.starts_with(p)) {
        return true;
    }

    // High density of code-like characters
    let code_chars = trimmed
        .chars()
        .filter(|c| matches!(c, '{' | '}' | '(' | ')' | ';' | '#' | '=' | '<' | '>' | '[' | ']'))
        .count();
    if trimmed.len() > 20 && code_chars as f32 / trimmed.len() as f32 > 0.15 {
        return true;
    }

    // Stack trace lines
    if trimmed.starts_with("at ") && trimmed.contains('(') && trimmed.contains(':') {
        return true;
    }

    false
}

/// Detect CTA / marketing / boilerplate lines.
fn is_cta_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    if lower.is_empty() {
        return false;
    }
    let patterns = [
        "contact us",
        "sign up",
        "subscribe",
        "free trial",
        "get started",
        "book a demo",
        "schedule a call",
        "request a quote",
        "click here",
        "download now",
        "buy now",
        "order now",
        "follow us on",
        "share this",
        "tweet this",
        "cookie policy",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "copyright ©",
        "powered by",
        "free consultation",
        "free estimate",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

/// Detect navigation/menu lines.
fn is_navigation_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 {
        return true;
    }
    // Breadcrumbs
    if trimmed.contains(" > ") && trimmed.split(" > ").count() >= 3 {
        return true;
    }
    false
}

/// Detect Wikipedia citation/footnote lines like "^ a b c Brandom, Russell..."
fn is_citation_footnote_line(line: &str) -> bool {
    let trimmed = line.trim();
    // Lines starting with "^" followed by spaces and single letters (footnote back-references)
    // e.g. "^ a b c Brandom, Russell (March 5, 2026)..."
    if trimmed.starts_with('^') {
        let rest = trimmed[1..].trim_start();
        // Check if it starts with single-letter markers like "a b c" or is a footnote body
        if rest.is_empty() {
            return true;
        }
        // Pattern: "^ a b c ..." — single letters separated by spaces
        let mut chars = rest.chars();
        if let Some(first) = chars.next() {
            if first.is_ascii_alphabetic() {
                // Next char is space or end — it's a footnote marker line
                match chars.next() {
                    None => return true,
                    Some(' ') => return true,
                    _ => {}
                }
            }
        }
        // Also catch lines like "^ Brandom, Russell" or "^ \"Title of article\""
        // These are citation/reference lines starting with ^
        if rest.chars().next().map_or(false, |c| c.is_ascii_uppercase() || c == '"' || c == '\'') {
            return true;
        }
    }
    false
}

/// Detect "Retrieved [Month] [Day], [Year]" and "Archived from the original on..." lines.
fn is_retrieved_or_archived_line(line: &str) -> bool {
    let trimmed = line.trim();
    let lower = trimmed.to_lowercase();

    // "Retrieved January 5, 2026" or "Retrieved 2026-01-05" or ". Retrieved ..."
    if lower.contains("retrieved ") {
        let months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ];
        // Check for "retrieved <Month>" pattern
        if let Some(pos) = lower.find("retrieved ") {
            let after = &lower[pos + 10..];
            let after_trimmed = after.trim_start();
            // Matches "retrieved January ..." or "retrieved 2" (year start)
            if months.iter().any(|m| after_trimmed.starts_with(m)) {
                return true;
            }
            // Matches "retrieved 2026-..." or "retrieved 20..."
            if after_trimmed.starts_with("20") || after_trimmed.starts_with("19") {
                return true;
            }
        }
    }

    // "Archived from the original on ..."
    if lower.contains("archived from the original") {
        return true;
    }
    // Also match "Archived (PDF) from the original" etc.
    if lower.contains("archived") && lower.contains("from the original") {
        return true;
    }

    false
}

/// Strip reference bracket markers like [1], [2], [citation needed], [edit] from a line.
fn strip_reference_brackets(line: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' {
            // Collect the content inside brackets
            let mut bracket_content = String::new();
            let mut found_close = false;
            for inner in chars.by_ref() {
                if inner == ']' {
                    found_close = true;
                    break;
                }
                bracket_content.push(inner);
            }

            if !found_close {
                // No closing bracket — keep the original text
                result.push('[');
                result.push_str(&bracket_content);
            } else if should_strip_bracket(&bracket_content) {
                // Strip this bracket entirely (don't add anything)
            } else {
                // Keep brackets that look like meaningful content (e.g., [example])
                result.push('[');
                result.push_str(&bracket_content);
                result.push(']');
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Determine if a bracket's content should be stripped (reference markers, editorial tags).
fn should_strip_bracket(content: &str) -> bool {
    let trimmed = content.trim();

    // Empty brackets
    if trimmed.is_empty() {
        return true;
    }

    // Pure numeric references: [1], [23], [145]
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }

    // Numeric ranges or lists: [1][2], already split by caller, but handle [1,2] or [1-3]
    if trimmed.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '-' || c == ' ') {
        return true;
    }

    // Common editorial/Wikipedia markers
    let lower = trimmed.to_lowercase();
    let editorial_markers = [
        "citation needed",
        "edit",
        "note",
        "nb",
        "clarification needed",
        "when?",
        "who?",
        "where?",
        "which?",
        "what?",
        "why?",
        "how?",
        "dubious",
        "discuss",
        "disputed",
        "unreliable source",
        "unreliable source?",
        "better source needed",
        "failed verification",
        "not in citation given",
        "original research?",
        "primary source needed",
        "year needed",
        "page needed",
        "full citation needed",
        "dead link",
        "permanent dead link",
        "self-published source",
        "self-published source?",
        "update",
        "needs update",
    ];
    if editorial_markers.iter().any(|m| lower == *m) {
        return true;
    }

    false
}

/// Normalize unicode: fix smart quotes, remove control chars.
fn normalize_unicode(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2013}' | '\u{2014}' => '-',
            '\u{00A0}' | '\u{200B}' | '\u{FEFF}' => ' ',
            c if c.is_control() && c != '\n' && c != '\r' && c != '\t' => ' ',
            c => c,
        })
        .collect()
}

/// Detect if content is a search engine results page (SERP/listing).
pub fn is_serp_page(url: &str, body_text: &str) -> bool {
    let url_lower = url.to_lowercase();
    let serp_patterns = [
        "/search?",
        "/search/?",
        "?q=",
        "?query=",
        "/releases/search",
        "/packages?",
        "/repositories?",
        "special:search",
        "/w/index.php?search",
    ];

    if serp_patterns.iter().any(|p| url_lower.contains(p)) {
        let lines: Vec<&str> = body_text.lines().filter(|l| l.len() > 10).collect();
        if lines.len() > 5 {
            let short = lines.iter().filter(|l| l.len() < 100).count();
            if short as f32 / lines.len() as f32 > 0.7 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consensus_extracts_article() {
        let html = r#"<html>
        <head>
            <title>Test Article - MySite</title>
            <meta name="author" content="Author Name">
            <meta name="description" content="A test article">
            <meta property="og:title" content="Test Article OG">
        </head>
        <body>
            <nav><a href="/">Home</a></nav>
            <article class="post-content">
                <h1>Test Article</h1>
                <p>This is the first paragraph of a test article with enough content for the extraction algorithms to correctly identify it as the main content block of the page.</p>
                <p>This is the second paragraph providing additional context and details about the topic being discussed in this test article for extraction testing purposes.</p>
                <p>A third paragraph rounds out the article with concluding thoughts and summary points about the overall subject matter covered here.</p>
            </article>
            <footer><p>Copyright 2025</p></footer>
        </body></html>"#;

        let result = extract_page(html, "https://example.com/article");

        assert!(result.title.is_some());
        assert!(result.body_text.contains("first paragraph"));
        assert_eq!(result.author.as_deref(), Some("Author Name"));
        assert_eq!(result.description.as_deref(), Some("A test article"));
        assert!(result.extraction_confidence > 0.5);
    }

    #[test]
    fn consensus_extracts_metadata() {
        let html = r#"<html lang="en">
        <head>
            <script type="application/ld+json">{"@type":"Article","headline":"JSON-LD Title"}</script>
            <meta property="og:title" content="OG Title">
            <meta property="og:type" content="article">
        </head>
        <body>
            <p>Some content for the extraction algorithms to process and identify as main text.</p>
        </body></html>"#;

        let result = extract_page(html, "https://example.com/");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert!(result.json_ld.is_some());
        assert!(result.open_graph.is_some());
    }

    #[test]
    fn consensus_handles_empty() {
        let result = extract_page("<html><body></body></html>", "https://example.com/");
        assert!(result.body_text.len() < 50);
        assert!(result.extraction_confidence <= 0.4);
    }

    #[test]
    fn to_page_creates_valid_model() {
        let extraction = ExtractionResult {
            title: Some("Title".into()),
            author: None,
            published_date: None,
            body_text: "Body text content".into(),
            headings: vec![],
            links: vec![],
            tables: vec![],
            language: Some("en".into()),
            description: None,
            json_ld: None,
            open_graph: None,
            extraction_confidence: 0.9,
        };

        let page = to_page(&extraction, "https://example.com", "example.com", "hash123", 200, 150, 5000);
        assert_eq!(page.url, "https://example.com");
        assert_eq!(page.domain, "example.com");
        assert_eq!(page.title.as_deref(), Some("Title"));
        assert_eq!(page.metadata.status_code, 200);
        assert_eq!(page.metadata.extraction_confidence, 0.9);
    }

    #[test]
    fn sanitize_strips_script_and_style() {
        let html = r#"<html><head>
            <style>.mw-parser-output { color: red; }</style>
            <script>var x = 1;</script>
        </head><body>
            <article>
                <p>Clean content that should survive sanitization and be properly extracted by the readability algorithm.</p>
                <p>More clean content in a second paragraph for the extraction to identify as main text.</p>
            </article>
        </body></html>"#;

        let result = extract_page(html, "https://example.com/");
        assert!(!result.body_text.contains("mw-parser-output"));
        assert!(!result.body_text.contains("var x = 1"));
        assert!(result.body_text.contains("Clean content"));
    }

    #[test]
    fn clean_body_removes_css_leaks() {
        let dirty = ".mw-parser-output ul.cslist{margin:0;padding:0}\nActual content here.\nMore real text.";
        let cleaned = clean_css_artifacts(dirty);
        assert!(!cleaned.contains("mw-parser-output"));
        assert!(cleaned.contains("Actual content"));
    }

    #[test]
    fn sanitize_preserves_content_structure() {
        let html = r#"<html><body>
            <h1>Title</h1>
            <p>Paragraph one with meaningful content for extraction testing purposes.</p>
            <table><tr><th>A</th></tr><tr><td>1</td></tr></table>
            <svg><path d="M0 0"/></svg>
            <p>Paragraph two after the SVG that should still be extracted properly.</p>
        </body></html>"#;

        let result = extract_page(html, "https://example.com/");
        assert!(!result.body_text.contains("svg"));
        assert!(!result.body_text.contains("path"));
    }

    // ── Data-Layer Acquisition tests (ADR 0002) ──────────────────────────────

    fn data_layer_on() -> ExtractorConfig {
        let mut c = web_search_common::config::Config::default().extractor;
        c.enable_data_layer = true;
        c
    }

    /// A page whose rendered DOM is a thin JS shell (no real paragraphs).
    const THIN_SHELL: &str = r#"<html><head><title>Loading…</title></head>
        <body><div id="root">Please enable JavaScript to view this site.</div></body></html>"#;

    #[test]
    fn structured_promotion_lifts_jsonld_article_body() {
        // Thin DOM, but a JSON-LD Article carries the full body (1.5.2).
        let html = r#"<html><head><title>News</title>
        <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"NewsArticle",
         "headline":"Reactor sets fusion record",
         "description":"A tokamak sustained plasma for a record duration.",
         "author":{"name":"Pat Rivera"},
         "datePublished":"2026-04-02T12:00:00Z",
         "articleBody":"The experimental tokamak sustained a burning plasma for a record one hundred seconds during a run this week, more than doubling the previous mark. Engineers credited improved magnetic confinement and a redesigned divertor for handling the intense heat flux at the reactor wall."}
        </script></head>
        <body><div id="app"></div></body></html>"#;

        // Off → byte-for-byte base: body stays thin, no promotion.
        let off = extract_page(html, "https://news.example/r");
        assert!(off.body_text.chars().count() < THIN_BODY_CHARS);

        // On → body promoted from JSON-LD; missing metadata filled (title already
        // present from <title>, so it is preserved — promotion is additive).
        let on = extract_page_with_config(html, "https://news.example/r", &data_layer_on());
        assert!(on.body_text.contains("burning plasma"));
        assert_eq!(on.author.as_deref(), Some("Pat Rivera"));
        assert_eq!(on.description.as_deref(), Some("A tokamak sustained plasma for a record duration."));
        assert!(on.published_date.is_some());
        assert!(on.extraction_confidence >= 0.75);
    }

    #[test]
    fn hydration_salvage_recovers_next_data_body() {
        // Thin DOM, but __NEXT_DATA__ carries the article (1.5.1 salvage).
        let html = r#"<html><head><title>Soft block</title>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"article":{
            "headline":"Trade talks resume",
            "articleBody":"Negotiators returned to the table on Monday after a two-week pause, signaling renewed willingness to compromise on tariffs. Officials from both delegations described the opening session as constructive and said working groups would meet through the week to draft a framework agreement covering agriculture and technology.",
            "datePublished":"2026-03-10T09:30:00Z"
        }}},"buildId":"z9"}
        </script></head>
        <body><div id="__next">Loading…</div></body></html>"#;

        let off = extract_page(html, "https://soft.example/a");
        assert!(off.body_text.chars().count() < THIN_BODY_CHARS);

        let on = extract_page_with_config(html, "https://soft.example/a", &data_layer_on());
        assert!(on.body_text.contains("Negotiators returned to the table"));
        assert!(on.published_date.is_some());
    }

    #[test]
    fn rich_consensus_body_is_not_overridden() {
        // A normal article with a real DOM body must NOT be replaced by a shorter
        // JSON-LD/hydration field — promotion only fires when data-layer is richer.
        let html = r#"<html><head><title>Real Article - Site</title>
        <script type="application/ld+json">{"@type":"Article","headline":"Real Article","description":"short"}</script>
        </head><body><article class="post-content">
            <h1>Real Article</h1>
            <p>This is a fully server-rendered article with substantial paragraph content that the consensus extractors identify confidently as the main body of the page without any help.</p>
            <p>A second paragraph adds more depth and detail, ensuring the consensus body is long and high-confidence so the data-layer pass has no reason to override it at all.</p>
            <p>A third paragraph rounds things out with a concluding thought and a few more sentences of genuine article prose for good measure here.</p>
        </article></body></html>"#;

        let base = extract_page(html, "https://real.example/x");
        let on = extract_page_with_config(html, "https://real.example/x", &data_layer_on());
        assert_eq!(base.body_text, on.body_text, "rich body must be preserved");
    }

    #[test]
    fn no_blob_data_layer_is_noop() {
        // Data-layer ON but no structured data and no hydration blob → identical
        // to base extraction (negative case).
        let base = extract_page(THIN_SHELL, "https://plain.example/");
        let on = extract_page_with_config(THIN_SHELL, "https://plain.example/", &data_layer_on());
        assert_eq!(base.body_text, on.body_text);
        assert_eq!(base.title, on.title);
    }

    #[test]
    fn data_layer_off_equals_base() {
        // Even with a salvageable blob present, OFF must equal extract_page exactly.
        let html = r#"<html><head><script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"article":{"articleBody":"Some recoverable body text that is long enough to be considered a meaningful salvage candidate for the data layer path when it is enabled by config."}}}}
        </script></head><body><div id="__next"></div></body></html>"#;

        let cfg_off = web_search_common::config::Config::default().extractor; // enable_data_layer = false
        let base = extract_page(html, "https://x.example/");
        let off = extract_page_with_config(html, "https://x.example/", &cfg_off);
        assert_eq!(base.body_text, off.body_text);
    }
}
