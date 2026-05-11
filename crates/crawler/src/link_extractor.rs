use scraper::{Html, Selector};
use url::Url;

/// A link extracted from a page.
#[derive(Debug, Clone)]
pub struct ExtractedLink {
    pub url: String,
    pub anchor_text: String,
    pub is_external: bool,
    pub rel: Option<String>,
}

/// Extract all links from HTML content, resolving relative URLs.
pub fn extract_links(html: &str, base_url: &str) -> Vec<ExtractedLink> {
    let document = Html::parse_document(html);
    let selector = Selector::parse("a[href]").unwrap();
    let base = match Url::parse(base_url) {
        Ok(u) => u,
        Err(_) => return vec![],
    };
    let base_domain = base.host_str().unwrap_or("");

    let mut links = Vec::new();

    for element in document.select(&selector) {
        let href = match element.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        // Skip non-HTTP links
        if href.starts_with("javascript:")
            || href.starts_with("mailto:")
            || href.starts_with("tel:")
            || href.starts_with("data:")
            || href == "#"
        {
            continue;
        }

        // Resolve relative URLs
        let resolved = match base.join(href) {
            Ok(u) => u,
            Err(_) => continue,
        };

        // Only keep http(s) URLs
        if resolved.scheme() != "http" && resolved.scheme() != "https" {
            continue;
        }

        let anchor_text = element.text().collect::<String>().trim().to_string();
        let link_domain = resolved.host_str().unwrap_or("");
        let is_external = link_domain != base_domain;
        let rel = element.value().attr("rel").map(|r| r.to_string());

        links.push(ExtractedLink {
            url: resolved.to_string(),
            anchor_text,
            is_external,
            rel,
        });
    }

    links
}

/// Find the "next page" link for pagination (rel="next" or common patterns).
pub fn find_next_page_link(html: &str, base_url: &str) -> Option<String> {
    let document = Html::parse_document(html);
    let base = Url::parse(base_url).ok()?;

    // 1. Check for rel="next" link
    let next_sel = Selector::parse(r#"link[rel="next"], a[rel="next"]"#).unwrap();
    for el in document.select(&next_sel) {
        if let Some(href) = el.value().attr("href") {
            if let Ok(resolved) = base.join(href) {
                return Some(resolved.to_string());
            }
        }
    }

    // 2. Check for common "next" patterns in anchors
    let a_sel = Selector::parse("a[href]").unwrap();
    for el in document.select(&a_sel) {
        let text = el.text().collect::<String>().to_lowercase();
        let classes = el.value().attr("class").unwrap_or("");
        let aria = el.value().attr("aria-label").unwrap_or("");

        let is_next = text.contains("next")
            || text.contains("›")
            || text.contains("»")
            || classes.contains("next")
            || aria.to_lowercase().contains("next");

        if is_next {
            if let Some(href) = el.value().attr("href") {
                if let Ok(resolved) = base.join(href) {
                    return Some(resolved.to_string());
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_absolute_links() {
        let html = r#"<html><body>
            <a href="https://example.com/page1">Page 1</a>
            <a href="https://other.com/page2">Other</a>
        </body></html>"#;

        let links = extract_links(html, "https://example.com/");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].url, "https://example.com/page1");
        assert!(!links[0].is_external);
        assert!(links[1].is_external);
    }

    #[test]
    fn resolve_relative_links() {
        let html = r#"<a href="/about">About</a><a href="sub/page">Sub</a>"#;
        let links = extract_links(html, "https://example.com/dir/");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].url, "https://example.com/about");
        assert_eq!(links[1].url, "https://example.com/dir/sub/page");
    }

    #[test]
    fn skip_non_http_links() {
        let html = r#"
            <a href="javascript:void(0)">JS</a>
            <a href="mailto:x@y.com">Email</a>
            <a href="tel:+1234">Phone</a>
            <a href="https://real.com">Real</a>
        "#;
        let links = extract_links(html, "https://example.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].url, "https://real.com/");
    }

    #[test]
    fn extract_anchor_text() {
        let html = r#"<a href="/page">  Click Here  </a>"#;
        let links = extract_links(html, "https://example.com/");
        assert_eq!(links[0].anchor_text, "Click Here");
    }

    #[test]
    fn find_rel_next() {
        let html = r#"<html><head><link rel="next" href="/page/2"></head><body></body></html>"#;
        let next = find_next_page_link(html, "https://example.com/page/1");
        assert_eq!(next.unwrap(), "https://example.com/page/2");
    }

    #[test]
    fn find_next_by_text() {
        let html = r#"<a href="/page/3">Next ›</a>"#;
        let next = find_next_page_link(html, "https://example.com/page/2");
        assert_eq!(next.unwrap(), "https://example.com/page/3");
    }

    #[test]
    fn no_next_page() {
        let html = r#"<a href="/about">About</a>"#;
        let next = find_next_page_link(html, "https://example.com/");
        assert!(next.is_none());
    }
}
