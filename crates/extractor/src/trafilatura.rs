use scraper::{ElementRef, Html, Selector};
use web_search_common::models::Heading;

use crate::ExtractionPass;

/// Trafilatura-inspired content extraction.
///
/// Multi-stage approach:
/// 1. Remove known boilerplate elements (nav, footer, ads)
/// 2. Score remaining blocks by text density + class heuristics
/// 3. Extract blocks above threshold
/// 4. Preserve heading structure
///
/// Based on the Trafilatura algorithm (F1=0.958 on benchmarks).
pub fn extract(html: &str) -> ExtractionPass {
    let document = Html::parse_document(html);

    let title = extract_title(&document);
    let headings = extract_headings(&document);

    // Collect text blocks with scores
    let blocks = score_blocks(&document);

    // Filter blocks above threshold
    let threshold = compute_threshold(&blocks);
    let mut content_blocks: Vec<&ScoredBlock> = blocks
        .iter()
        .filter(|b| b.score >= threshold && b.text.len() > 30)
        .collect();

    // Fix B — thin-body listing recovery. The primary threshold filter rejects
    // listing pages (post indexes, doc TOCs): every cell is a short link-bearing
    // `<td>`/`<li>`, none clears the prose-tuned threshold, so the body collapses
    // to a stub. When the primary selection is thin yet the page is dominated by
    // short link-cells under one container, fall back to keeping every link-cell
    // whose anchor text is a real title (>= MIN_ANCHOR_TITLE_LEN chars). This is
    // the block-selection analogue of the c026dbb truncation guard: one extra
    // O(n) pass over already-collected blocks, only when the primary path is thin.
    let primary_chars: usize = content_blocks.iter().map(|b| b.text.chars().count()).sum();
    if primary_chars < LISTING_RECOVERY_FLOOR_CHARS {
        let link_cells = blocks
            .iter()
            .filter(|b| b.is_link_cell && b.anchor_chars >= MIN_ANCHOR_TITLE_LEN)
            .count();
        if link_cells >= MIN_LISTING_CELLS {
            content_blocks = blocks
                .iter()
                .filter(|b| {
                    (b.score >= threshold && b.text.len() > 30)
                        || (b.is_link_cell && b.anchor_chars >= MIN_ANCHOR_TITLE_LEN)
                })
                .collect();
        }
    }

    let body_text = content_blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<&str>>()
        .join("\n\n");

    let confidence = if content_blocks.len() > 2 && body_text.len() > 300 {
        0.85
    } else if body_text.len() > 100 {
        0.6
    } else {
        0.3
    };

    ExtractionPass {
        title,
        body_text,
        headings,
        confidence,
    }
}

/// Minimum anchor-text length (chars) for a link-bearing cell to count as a real
/// title (e.g. a post heading) rather than a one-word nav label ("Home"/"About").
const MIN_ANCHOR_TITLE_LEN: usize = 15;
/// Below this many chars in the primary (threshold-passed) selection, treat the
/// page as a thin listing and attempt Fix B recovery.
const LISTING_RECOVERY_FLOOR_CHARS: usize = 600;
/// Minimum number of title-anchor link-cells required before Fix B fires — a
/// genuine listing (index/TOC) has many; a normal article has ~0.
const MIN_LISTING_CELLS: usize = 5;

#[derive(Debug)]
struct ScoredBlock {
    text: String,
    score: f32,
    /// True for an `li`/`td`/`dd` block whose text is mostly anchor text
    /// (`link_ratio > LINK_HEAVY_RATIO`) — a candidate listing cell.
    is_link_cell: bool,
    /// Total chars of anchor text inside this block.
    anchor_chars: usize,
}

/// Link-ratio above which a block is considered "mostly links" (nav OR a content
/// listing cell). Used both for the penalty in `score_block` and to flag
/// candidate listing cells for Fix B recovery.
const LINK_HEAVY_RATIO: f32 = 0.6;

/// Class/id substrings on an ancestor that mark a region as content listing
/// (post index, doc TOC, archive) rather than chrome — the primary discriminator
/// that lets a link-heavy cell keep most of its score (Fix A).
const POSITIVE_LISTING_ANCESTORS: [&str; 6] =
    ["posts", "post-list", "content", "entry", "index", "archive"];

/// Score each text-bearing block in the document.
fn score_blocks(document: &Html) -> Vec<ScoredBlock> {
    let block_sel = Selector::parse("p, li, blockquote, td, pre, h1, h2, h3, h4, h5, h6, figcaption, dt, dd").unwrap();
    let mut blocks = Vec::new();

    for element in document.select(&block_sel) {
        // Skip if inside boilerplate ancestor
        if is_boilerplate_ancestor(element) {
            continue;
        }

        let text = element.text().collect::<String>();
        let text = text.trim().to_string();

        if text.is_empty() {
            continue;
        }

        let (score, is_link_cell, anchor_chars) = score_block(element, &text);
        blocks.push(ScoredBlock { text, score, is_link_cell, anchor_chars });
    }

    blocks
}

/// Score a single text block. Returns `(score, is_link_cell, anchor_chars)` where
/// `is_link_cell` flags an `li`/`td`/`dd` cell that is mostly anchor text (a
/// listing-cell candidate for Fix B), and `anchor_chars` is its total anchor length.
fn score_block(element: ElementRef, text: &str) -> (f32, bool, usize) {
    let mut score: f32 = 0.0;
    let text_len = text.len() as f32;

    // Text length: longer blocks more likely content
    score += (text_len / 50.0).min(10.0);

    // Sentence count heuristic: content has sentences
    let sentence_count = text.matches(|c: char| c == '.' || c == '!' || c == '?').count();
    score += (sentence_count as f32) * 2.0;

    // Word count
    let word_count = text.split_whitespace().count();
    if word_count > 10 {
        score += 5.0;
    }

    // Tag bonuses
    let tag = element.value().name();
    match tag {
        "p" => score += 3.0,
        "blockquote" => score += 2.0,
        "pre" => score += 2.0,
        "li" => score += 1.0,
        "h1" | "h2" | "h3" => score += 4.0,
        "td" => score -= 1.0, // tables often navigation
        _ => {}
    }

    // Class/ID heuristics on ancestors
    if let Some(parent) = element.parent().and_then(ElementRef::wrap) {
        let class_id = format!(
            "{} {}",
            parent.value().attr("class").unwrap_or(""),
            parent.value().attr("id").unwrap_or("")
        ).to_lowercase();

        let positive = ["article", "content", "post", "entry", "story", "text", "body"];
        let negative = ["comment", "sidebar", "nav", "menu", "footer", "widget", "ad",
                        "social", "share", "related", "meta", "tags", "breadcrumb"];

        for w in &positive {
            if class_id.contains(w) {
                score += 5.0;
            }
        }
        for w in &negative {
            if class_id.contains(w) {
                score -= 10.0;
            }
        }
    }

    // Link density: penalize blocks that are mostly links (nav bars are link-only).
    // Fix A — but a content listing (post index, doc TOC) is structurally identical
    // to a nav bar: a cell of nothing but a link. Blanket-penalizing both deletes
    // real listing content (root cause of the blog.rust-lang.org 552-char bug).
    // Discriminate: for `li`/`td`/`dd` cells with a *real-title* anchor (>= 15
    // chars) OR sitting under a positive listing ancestor, apply a soft penalty so
    // the cell can still clear threshold. Keep the full penalty for short link-only
    // cells (true nav like "Home"/"About").
    let a_sel = Selector::parse("a").unwrap();
    let link_chars: usize = element
        .select(&a_sel)
        .map(|a| a.text().collect::<String>().chars().count())
        .sum();
    let text_chars = text.chars().count();
    let link_ratio = link_chars as f32 / (text_chars as f32).max(1.0);

    let is_cell = matches!(tag, "li" | "td" | "dd");
    let link_heavy = link_ratio > LINK_HEAVY_RATIO;
    let is_link_cell = is_cell && link_heavy;

    if link_heavy {
        let long_anchor = link_chars >= MIN_ANCHOR_TITLE_LEN;
        let under_positive = has_positive_listing_ancestor(element);
        if is_cell && (long_anchor || under_positive) {
            // Real listing content — soft penalty, lets the cell survive threshold.
            score -= 4.0;
        } else {
            // True nav / link-only chrome — full penalty.
            score -= 15.0;
        }
    }

    (score, is_link_cell, link_chars)
}

/// Walk ancestors looking for an id/class that marks a content-listing region
/// (post index, doc TOC, archive). Bounded depth to stay O(1) per block.
fn has_positive_listing_ancestor(element: ElementRef) -> bool {
    let mut current = element.parent().and_then(ElementRef::wrap);
    let mut depth = 0;
    while let Some(el) = current {
        if depth > 10 {
            break;
        }
        let class_id = format!(
            "{} {}",
            el.value().attr("class").unwrap_or(""),
            el.value().attr("id").unwrap_or("")
        )
        .to_lowercase();
        if POSITIVE_LISTING_ANCESTORS.iter().any(|w| class_id.contains(w)) {
            return true;
        }
        current = el.parent().and_then(ElementRef::wrap);
        depth += 1;
    }
    false
}

/// Check if element is inside a boilerplate container.
fn is_boilerplate_ancestor(element: ElementRef) -> bool {
    let boilerplate_tags = ["nav", "footer", "aside", "noscript"];
    let boilerplate_classes = ["sidebar", "nav", "menu", "footer", "comment",
                               "widget", "ad-", "advertisement", "social-share",
                               "cookie", "popup", "modal", "banner"];

    let mut current = element.parent().and_then(ElementRef::wrap);
    let mut depth = 0;

    while let Some(el) = current {
        if depth > 10 {
            break;
        }

        let tag = el.value().name();
        if boilerplate_tags.contains(&tag) {
            return true;
        }

        let classes = el.value().attr("class").unwrap_or("").to_lowercase();
        let id = el.value().attr("id").unwrap_or("").to_lowercase();

        for bp in &boilerplate_classes {
            if classes.contains(bp) || id.contains(bp) {
                return true;
            }
        }

        current = el.parent().and_then(ElementRef::wrap);
        depth += 1;
    }

    false
}

/// Compute adaptive threshold: median score of all blocks.
fn compute_threshold(blocks: &[ScoredBlock]) -> f32 {
    if blocks.is_empty() {
        return 0.0;
    }

    let mut scores: Vec<f32> = blocks.iter().map(|b| b.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use 40th percentile as threshold (keep top 60%)
    let idx = (scores.len() as f32 * 0.4) as usize;
    scores.get(idx).copied().unwrap_or(0.0)
}

fn extract_title(document: &Html) -> Option<String> {
    let sel = Selector::parse("title").unwrap();
    document.select(&sel).next().map(|el| {
        let text = el.text().collect::<String>().trim().to_string();
        text.split(" - ").next()
            .or_else(|| text.split(" | ").next())
            .unwrap_or(&text)
            .trim()
            .to_string()
    }).filter(|t| !t.is_empty())
}

fn extract_headings(document: &Html) -> Vec<Heading> {
    let mut headings = Vec::new();
    for level in 1..=6 {
        let sel = Selector::parse(&format!("h{level}")).unwrap();
        for el in document.select(&sel) {
            if is_boilerplate_ancestor(el) {
                continue;
            }
            let text = el.text().collect::<String>().trim().to_string();
            if !text.is_empty() {
                headings.push(Heading { level: level as u8, text });
            }
        }
    }
    headings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_article_content() {
        let html = r#"<html><head><title>Article Title</title></head><body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <div class="article-content">
                <h1>Article Title</h1>
                <p>First paragraph of the article has substantial content that the trafilatura algorithm should identify as main content based on text density scoring.</p>
                <p>Second paragraph continues the story with additional details and context that further establishes this block as genuine article content.</p>
                <p>Third paragraph wraps up the article with conclusions and final thoughts about the topic being discussed in this test case.</p>
            </div>
            <div class="sidebar">
                <a href="/related1">Related 1</a>
                <a href="/related2">Related 2</a>
            </div>
        </body></html>"#;

        let result = extract(html);
        assert_eq!(result.title.as_deref(), Some("Article Title"));
        assert!(result.body_text.contains("First paragraph"));
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn skips_nav_and_footer() {
        let html = r#"<html><body>
            <nav><p>Navigation text that should be skipped by the extractor because it lives inside a nav element.</p></nav>
            <main>
                <p>This is the main content paragraph that should be extracted. It contains real article text that users want to read.</p>
                <p>Another paragraph of main content with enough words to score well in the text density analysis.</p>
            </main>
            <footer><p>Footer text about copyright and legal stuff that should also be skipped from the extraction results.</p></footer>
        </body></html>"#;

        let result = extract(html);
        assert!(result.body_text.contains("main content"));
        assert!(!result.body_text.contains("Navigation text"));
        assert!(!result.body_text.contains("Footer text"));
    }

    #[test]
    fn skips_comment_sections() {
        let html = r#"<html><body>
            <article>
                <p>The real article content about an important topic that readers care about and the extraction algorithm should identify.</p>
                <p>More article content with additional details about the subject matter being discussed in this piece.</p>
            </article>
            <div class="comments-section">
                <p>User comment that should not be extracted as part of the main article content because it is in the comments area.</p>
            </div>
        </body></html>"#;

        let result = extract(html);
        assert!(result.body_text.contains("real article"));
        assert!(!result.body_text.contains("User comment"));
    }

    #[test]
    fn handles_empty_page() {
        let result = extract("<html><body></body></html>");
        assert!(result.body_text.is_empty() || result.body_text.len() < 30);
        assert!(result.confidence <= 0.3);
    }

    #[test]
    fn listing_recovery_floor_recovers_title_cells() {
        // A pure listing: a table of N link-only `<td>` cells, each a real-title
        // anchor (>= MIN_ANCHOR_TITLE_LEN chars). Every cell is link_ratio == 1.0,
        // so without Fix A/B all are penalized below threshold and the body is a
        // stub. Assert the recovery path fires and the titles survive.
        let rows: String = (0..8)
            .map(|i| {
                format!(
                    "<tr><td><a href=\"/post/{i}\">Announcing Release Number {i} Of The Project</a></td></tr>"
                )
            })
            .collect();
        let html = format!(
            "<html><head><title>Archive</title></head><body>\
             <section class=\"posts\"><table class=\"post-list\">{rows}</table></section>\
             </body></html>"
        );

        let result = extract(&html);

        // Several distinct title cells must be recovered (not a single stub).
        for i in [0, 3, 7] {
            let title = format!("Announcing Release Number {i} Of The Project");
            assert!(
                result.body_text.contains(&title),
                "recovery floor failed to recover listing cell {i:?}; body was {:?}",
                result.body_text
            );
        }
    }

    #[test]
    fn short_link_only_cells_are_not_recovered() {
        // Negative pair for the recovery floor: a table of short one-word link
        // cells (true nav) must NOT be recovered — their anchors are below
        // MIN_ANCHOR_TITLE_LEN, so neither Fix A's soft penalty nor Fix B applies.
        let html = r#"<html><head><title>Nav</title></head><body>
            <section class="posts"><table class="post-list">
                <tr><td><a href="/a">Home</a></td></tr>
                <tr><td><a href="/b">About</a></td></tr>
                <tr><td><a href="/c">Docs</a></td></tr>
                <tr><td><a href="/d">Blog</a></td></tr>
                <tr><td><a href="/e">Login</a></td></tr>
                <tr><td><a href="/f">Help</a></td></tr>
            </table></section>
        </body></html>"#;

        let result = extract(html);
        // None of the short nav labels survive (no real-title anchor present).
        for label in ["Home", "About", "Login"] {
            assert!(
                !result.body_text.contains(label),
                "short link-only cell {label:?} was wrongly recovered as content"
            );
        }
    }
}
