use scraper::{ElementRef, Html, Selector};
use web_search_common::models::Heading;

use crate::ExtractionPass;

/// Mozilla Readability-inspired content extraction.
///
/// Scores each DOM node by:
/// - Text density (chars / child tags ratio)
/// - Link density (link chars / total chars)
/// - Tag penalties (nav, sidebar, footer, ad → negative)
/// - Tag bonuses (article, main, section, p → positive)
///
/// Keeps highest-scoring subtree as main content.
pub fn extract(html: &str) -> ExtractionPass {
    let document = Html::parse_document(html);

    let title = extract_title(&document);
    let headings = extract_headings(&document);

    // Score candidate content blocks
    let candidates = score_candidates(&document);

    // Pick best candidate
    let body_text = if let Some(best) = candidates.first() {
        best.text.clone()
    } else {
        // Fallback: extract all <p> tags
        fallback_extract(&document)
    };

    let confidence = if body_text.len() > 200 { 0.8 } else { 0.4 };

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

fn score_candidates(document: &Html) -> Vec<ScoredBlock> {
    // Target: div, article, section, main blocks
    let block_sel = Selector::parse("article, main, [role='main'], .post-content, .article-body, .entry-content, #content, .content").unwrap();
    let p_sel = Selector::parse("p").unwrap();

    let mut candidates: Vec<ScoredBlock> = Vec::new();

    // Try semantic containers first
    for element in document.select(&block_sel) {
        let text = collect_text(element);
        if text.len() < 100 {
            continue;
        }
        let score = score_element(element, &text);
        candidates.push(ScoredBlock { text, score });
    }

    // If no semantic containers found, score all divs with substantial <p> content
    if candidates.is_empty() {
        let div_sel = Selector::parse("div").unwrap();
        for element in document.select(&div_sel) {
            let paragraphs: Vec<String> = element
                .select(&p_sel)
                .map(|p| p.text().collect::<String>().trim().to_string())
                .filter(|t| t.len() > 40)
                .collect();

            if paragraphs.len() < 2 {
                continue;
            }

            let text = paragraphs.join("\n\n");
            let score = score_element(element, &text);
            candidates.push(ScoredBlock { text, score });
        }
    }

    // Sort by score descending
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates
}

fn score_element(element: ElementRef, text: &str) -> f32 {
    let mut score: f32 = 0.0;

    let text_len = text.len() as f32;
    if text_len < 50.0 {
        return -100.0;
    }

    // Text length bonus (more text = more likely content)
    score += (text_len / 100.0).min(50.0);

    // Paragraph bonus
    let p_sel = Selector::parse("p").unwrap();
    let p_count = element.select(&p_sel).count() as f32;
    score += p_count * 3.0;

    // Tag name bonus/penalty
    let tag = element.value().name();
    match tag {
        "article" => score += 30.0,
        "main" => score += 25.0,
        "section" => score += 10.0,
        "div" => score += 0.0,
        "aside" | "nav" | "footer" | "header" => score -= 50.0,
        _ => {}
    }

    // Class/ID signals
    let classes = element.value().attr("class").unwrap_or("");
    let id = element.value().attr("id").unwrap_or("");
    let class_id = format!("{} {}", classes, id).to_lowercase();

    let positive = ["article", "content", "post", "entry", "text", "body", "main", "story"];
    let negative = ["sidebar", "nav", "menu", "footer", "comment", "widget", "ad", "banner",
                    "social", "share", "related", "recommend", "popup", "modal"];

    for word in &positive {
        if class_id.contains(word) {
            score += 15.0;
        }
    }
    for word in &negative {
        if class_id.contains(word) {
            score -= 20.0;
        }
    }

    // Link density penalty
    let a_sel = Selector::parse("a").unwrap();
    let link_text_len: usize = element
        .select(&a_sel)
        .map(|a| a.text().collect::<String>().len())
        .sum();
    let link_density = link_text_len as f32 / text_len.max(1.0);
    if link_density > 0.5 {
        score -= 30.0; // more than half text is links = probably nav
    }

    score
}

fn collect_text(element: ElementRef) -> String {
    // Collect text, skipping script/style/nav elements
    let skip_tags = ["script", "style", "nav", "footer", "header", "aside", "noscript"];
    let mut parts = Vec::new();

    collect_text_recursive(element, &skip_tags, &mut parts);

    let text = parts.join(" ");
    // Collapse whitespace
    let collapsed: String = text
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");
    collapsed
}

fn collect_text_recursive(element: ElementRef, skip_tags: &[&str], parts: &mut Vec<String>) {
    for child in element.children() {
        if let Some(el) = ElementRef::wrap(child) {
            let tag = el.value().name();
            if skip_tags.contains(&tag) {
                continue;
            }
            collect_text_recursive(el, skip_tags, parts);
        } else if let Some(text_node) = child.value().as_text() {
            let trimmed = text_node.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
    }
}

fn extract_title(document: &Html) -> Option<String> {
    // Try <title> tag
    let title_sel = Selector::parse("title").unwrap();
    if let Some(el) = document.select(&title_sel).next() {
        let text = el.text().collect::<String>().trim().to_string();
        if !text.is_empty() {
            // Clean common suffixes like " - Site Name" or " | Site Name"
            let cleaned = text
                .split(" - ")
                .next()
                .or_else(|| text.split(" | ").next())
                .unwrap_or(&text)
                .trim()
                .to_string();
            return Some(cleaned);
        }
    }

    // Try <h1>
    let h1_sel = Selector::parse("h1").unwrap();
    if let Some(el) = document.select(&h1_sel).next() {
        let text = el.text().collect::<String>().trim().to_string();
        if !text.is_empty() {
            return Some(text);
        }
    }

    // Try og:title
    let og_sel = Selector::parse(r#"meta[property="og:title"]"#).unwrap();
    if let Some(el) = document.select(&og_sel).next() {
        if let Some(content) = el.value().attr("content") {
            return Some(content.trim().to_string());
        }
    }

    None
}

fn extract_headings(document: &Html) -> Vec<Heading> {
    let mut headings = Vec::new();

    for level in 1..=6 {
        let sel = Selector::parse(&format!("h{level}")).unwrap();
        for el in document.select(&sel) {
            let text = el.text().collect::<String>().trim().to_string();
            if !text.is_empty() {
                headings.push(Heading {
                    level: level as u8,
                    text,
                });
            }
        }
    }

    headings
}

fn fallback_extract(document: &Html) -> String {
    let p_sel = Selector::parse("p").unwrap();
    let paragraphs: Vec<String> = document
        .select(&p_sel)
        .map(|p| p.text().collect::<String>().trim().to_string())
        .filter(|t| t.len() > 20)
        .collect();
    paragraphs.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_from_article() {
        let html = r#"<html><head><title>Test Article - MySite</title></head><body>
            <nav>Menu items here</nav>
            <article>
                <h1>Main Heading</h1>
                <p>This is the first paragraph of the article with enough content to be meaningful for extraction purposes and testing.</p>
                <p>This is the second paragraph with more substantial content that helps the readability algorithm identify this as the main content area.</p>
                <p>A third paragraph ensures we have enough text density to score well against other candidate blocks in the document.</p>
            </article>
            <footer>Footer stuff</footer>
        </body></html>"#;

        let result = extract(html);
        assert_eq!(result.title.as_deref(), Some("Test Article"));
        assert!(result.body_text.contains("first paragraph"));
        assert!(result.body_text.contains("second paragraph"));
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn extract_title_from_og() {
        let html = r#"<html><head><meta property="og:title" content="OG Title"></head><body><p>Content</p></body></html>"#;
        let result = extract(html);
        assert_eq!(result.title.as_deref(), Some("OG Title"));
    }

    #[test]
    fn extract_headings_all_levels() {
        let html = r#"<html><body>
            <h1>Title</h1>
            <h2>Section</h2>
            <h3>Subsection</h3>
            <p>Content here to make extraction work properly with enough text.</p>
        </body></html>"#;

        let result = extract(html);
        assert!(result.headings.len() >= 3);
        assert_eq!(result.headings[0].level, 1);
        assert_eq!(result.headings[0].text, "Title");
    }

    #[test]
    fn penalize_nav_heavy_blocks() {
        let html = r#"<html><body>
            <div class="sidebar">
                <a href="/a">Link 1</a><a href="/b">Link 2</a><a href="/c">Link 3</a>
                <a href="/d">Link 4</a><a href="/e">Link 5</a><a href="/f">Link 6</a>
            </div>
            <article>
                <p>This is the real article content that should be extracted by the readability algorithm because it has much higher text density and lower link density than the sidebar.</p>
                <p>Second paragraph with more content to ensure this block scores higher than the navigation-heavy sidebar block that contains mostly links.</p>
            </article>
        </body></html>"#;

        let result = extract(html);
        assert!(result.body_text.contains("real article content"));
        assert!(!result.body_text.contains("Link 1"));
    }

    #[test]
    fn fallback_to_paragraphs() {
        let html = r#"<html><body>
            <p>This is a plain page with no semantic HTML structure but enough paragraph content to extract something meaningful from the page.</p>
            <p>Second paragraph ensures the fallback extraction picks up multiple paragraphs from the body when no article or main elements exist.</p>
        </body></html>"#;

        let result = extract(html);
        assert!(result.body_text.contains("plain page"));
    }

    #[test]
    fn empty_html_returns_empty() {
        let result = extract("<html><body></body></html>");
        assert!(result.body_text.is_empty() || result.body_text.len() < 10);
    }
}
