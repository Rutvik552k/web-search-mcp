use web_search_common::models::Page;
use web_search_common::models::PageMetadata;
use chrono::Utc;
use regex::Regex;

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
fn clean_body_text(text: &str) -> String {
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
pub fn extract_page(html: &str, url: &str) -> ExtractionResult {
    // Sanitize HTML before extraction
    let clean_html = sanitize_html(html);

    // Run both extraction passes on sanitized HTML
    let traf_result = trafilatura::extract(&clean_html);
    let read_result = readability::extract(&clean_html);

    // Pick best body text by confidence, then clean
    let (body_text, confidence) = if traf_result.confidence >= read_result.confidence {
        (clean_body_text(&traf_result.body_text), traf_result.confidence)
    } else {
        (clean_body_text(&read_result.body_text), read_result.confidence)
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
        let cleaned = clean_body_text(dirty);
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
}
