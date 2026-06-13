use scraper::{Html, Selector};

/// One entry from a Google News sitemap (`xmlns:news=
/// "http://www.google.com/schemas/sitemap-news/0.9"`).
///
/// Ground truth: <https://developers.google.com/search/docs/crawling-indexing/sitemaps/news-sitemap>
/// A news sitemap is a normal `<urlset>` whose `<url>` entries additionally carry a
/// `<news:news>` block with `<news:publication><news:name>`, `<news:publication_date>`
/// (W3C/ISO-8601) and `<news:title>`. We surface these so the R2 alternative-surface
/// rung can use a site's news feed of fresh article URLs + titles as a low-ban-risk
/// discovery surface (ADR 0003 §3.1 R2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewsEntry {
    /// The article URL (`<loc>`).
    pub loc: String,
    /// `<news:title>` if present.
    pub title: Option<String>,
    /// `<news:publication_date>` (raw string, ISO-8601) if present.
    pub publication_date: Option<String>,
    /// `<news:name>` (publication name) if present.
    pub publication_name: Option<String>,
}

/// True iff the document looks like a Google News sitemap (carries the news
/// namespace or any `<news:...>` element).
pub fn is_news_sitemap(xml: &str) -> bool {
    xml.contains("sitemap-news/0.9") || xml.contains("<news:news") || xml.contains(":news>")
}

/// Parse a Google News sitemap into per-article entries.
///
/// `scraper`'s html5ever-based parser does not reliably expose XML-namespaced tag
/// names (`news:title` etc.), so the namespaced news fields are extracted with a
/// tolerant regex scan per `<url>` block while `<loc>` reuses the same text the
/// regular parser sees. Entries without a `<loc>` are skipped. Order is preserved.
pub fn parse_news_sitemap(xml: &str) -> Vec<NewsEntry> {
    use regex::Regex;
    // Compile once per call (call sites are infrequent — one per fetched sitemap).
    let url_block = Regex::new(r"(?is)<url\b[^>]*>(.*?)</url>").unwrap();
    let loc_re = Regex::new(r"(?is)<loc>\s*(.*?)\s*</loc>").unwrap();
    let title_re = Regex::new(r"(?is)<news:title>\s*(.*?)\s*</news:title>").unwrap();
    let date_re =
        Regex::new(r"(?is)<news:publication_date>\s*(.*?)\s*</news:publication_date>").unwrap();
    let name_re = Regex::new(r"(?is)<news:name>\s*(.*?)\s*</news:name>").unwrap();

    let strip = |s: &str| -> String {
        // Unwrap a possible CDATA section and trim.
        let s = s.trim();
        let s = s
            .strip_prefix("<![CDATA[")
            .and_then(|s| s.strip_suffix("]]>"))
            .unwrap_or(s);
        s.trim().to_string()
    };

    let mut out = Vec::new();
    for block in url_block.captures_iter(xml) {
        let inner = &block[1];
        let loc = match loc_re.captures(inner) {
            Some(c) => {
                let l = strip(&c[1]);
                if l.starts_with("http") {
                    l
                } else {
                    continue;
                }
            }
            None => continue,
        };
        out.push(NewsEntry {
            loc,
            title: title_re.captures(inner).map(|c| strip(&c[1])).filter(|s| !s.is_empty()),
            publication_date: date_re.captures(inner).map(|c| strip(&c[1])).filter(|s| !s.is_empty()),
            publication_name: name_re.captures(inner).map(|c| strip(&c[1])).filter(|s| !s.is_empty()),
        });
    }
    out
}

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

    #[test]
    fn detect_news_sitemap() {
        let news = r#"<urlset xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"></urlset>"#;
        assert!(is_news_sitemap(news));
        assert!(!is_news_sitemap("<urlset><url><loc>http://a.com</loc></url></urlset>"));
    }

    #[test]
    fn parse_news_sitemap_entries() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
                xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
          <url>
            <loc>https://news.example.com/article-1</loc>
            <news:news>
              <news:publication>
                <news:name>Example Times</news:name>
                <news:language>en</news:language>
              </news:publication>
              <news:publication_date>2026-06-12T08:00:00Z</news:publication_date>
              <news:title>Breaking: Something Happened</news:title>
            </news:news>
          </url>
          <url>
            <loc>https://news.example.com/article-2</loc>
            <news:news>
              <news:publication><news:name>Example Times</news:name></news:publication>
              <news:publication_date>2026-06-11T20:30:00Z</news:publication_date>
              <news:title><![CDATA[A Title With & Ampersand]]></news:title>
            </news:news>
          </url>
        </urlset>"#;

        let entries = parse_news_sitemap(xml);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].loc, "https://news.example.com/article-1");
        assert_eq!(entries[0].title.as_deref(), Some("Breaking: Something Happened"));
        assert_eq!(entries[0].publication_date.as_deref(), Some("2026-06-12T08:00:00Z"));
        assert_eq!(entries[0].publication_name.as_deref(), Some("Example Times"));
        // CDATA-wrapped title is unwrapped.
        assert_eq!(entries[1].title.as_deref(), Some("A Title With & Ampersand"));
    }

    #[test]
    fn news_sitemap_skips_locless_blocks() {
        // A <url> with no <loc> is skipped, not panicked on.
        let xml = r#"<urlset><url><news:news><news:title>orphan</news:title></news:news></url>
        <url><loc>https://a.com/x</loc></url></urlset>"#;
        let entries = parse_news_sitemap(xml);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].loc, "https://a.com/x");
        assert!(entries[0].title.is_none());
    }
}
