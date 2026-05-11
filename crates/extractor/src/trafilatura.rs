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
    let content_blocks: Vec<&ScoredBlock> = blocks
        .iter()
        .filter(|b| b.score >= threshold && b.text.len() > 30)
        .collect();

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

#[derive(Debug)]
struct ScoredBlock {
    text: String,
    score: f32,
}

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

        let score = score_block(element, &text);
        blocks.push(ScoredBlock { text, score });
    }

    blocks
}

/// Score a single text block.
fn score_block(element: ElementRef, text: &str) -> f32 {
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

    // Link density in paragraph: penalize blocks that are mostly links
    let a_sel = Selector::parse("a").unwrap();
    let link_chars: usize = element
        .select(&a_sel)
        .map(|a| a.text().collect::<String>().len())
        .sum();
    let link_ratio = link_chars as f32 / text_len.max(1.0);
    if link_ratio > 0.6 {
        score -= 15.0;
    }

    score
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
}
