use scraper::{Html, Selector};

/// Parse a sitemap.xml and extract page URLs.
///
/// Handles both regular sitemaps and sitemap index files.
pub fn parse_sitemap(xml: &str) -> Vec<String> {
    let mut urls = Vec::new();

    // Try parsing as XML-like with scraper (handles both HTML and XML)
    let doc = Html::parse_document(xml);

    // Regular sitemap: <url><loc>...</loc></url>
    let loc_sel = Selector::parse("loc").unwrap();
    for el in doc.select(&loc_sel) {
        let text = el.text().collect::<String>().trim().to_string();
        if text.starts_with("http") {
            urls.push(text);
        }
    }

    // If no <loc> found, try plain text (one URL per line)
    if urls.is_empty() {
        for line in xml.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("http") && !trimmed.contains('<') {
                urls.push(trimmed.to_string());
            }
        }
    }

    urls
}

/// Check if a sitemap URL points to a sitemap index (contains other sitemaps).
pub fn is_sitemap_index(xml: &str) -> bool {
    xml.contains("<sitemapindex") || xml.contains("sitemap>")
}

/// Extract sitemap URLs from a sitemap index file.
pub fn parse_sitemap_index(xml: &str) -> Vec<String> {
    let doc = Html::parse_document(xml);
    let loc_sel = Selector::parse("loc").unwrap();

    doc.select(&loc_sel)
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|u| u.starts_with("http") && (u.contains("sitemap") || u.ends_with(".xml")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_regular_sitemap() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>"#;

        let urls = parse_sitemap(xml);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://example.com/page1");
    }

    #[test]
    fn parse_sitemap_index_file() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
            <sitemap><loc>https://example.com/sitemap2.xml</loc></sitemap>
        </sitemapindex>"#;

        assert!(is_sitemap_index(xml));
        let sitemaps = parse_sitemap_index(xml);
        assert_eq!(sitemaps.len(), 2);
    }

    #[test]
    fn detect_sitemap_index() {
        assert!(is_sitemap_index("<sitemapindex>...</sitemapindex>"));
        assert!(!is_sitemap_index("<urlset><url><loc>http://a.com</loc></url></urlset>"));
    }
}
