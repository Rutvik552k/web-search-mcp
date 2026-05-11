use chrono::{DateTime, NaiveDate, Utc};
use scraper::{Html, Selector};
use web_search_common::models::{Link, OpenGraphData, Table};

/// Extract structured metadata from HTML.
pub struct MetadataResult {
    pub author: Option<String>,
    pub published_date: Option<DateTime<Utc>>,
    pub language: Option<String>,
    pub description: Option<String>,
    pub json_ld: Option<serde_json::Value>,
    pub open_graph: Option<OpenGraphData>,
    pub links: Vec<Link>,
    pub tables: Vec<Table>,
}

/// Extract all metadata from HTML content.
pub fn extract_metadata(html: &str, base_url: &str) -> MetadataResult {
    let document = Html::parse_document(html);

    MetadataResult {
        author: extract_author(&document),
        published_date: extract_date(&document),
        language: extract_language(&document),
        description: extract_description(&document),
        json_ld: extract_json_ld(&document),
        open_graph: extract_open_graph(&document),
        links: extract_links(&document, base_url),
        tables: extract_tables(&document),
    }
}

fn extract_author(doc: &Html) -> Option<String> {
    // Try meta author
    let sel = Selector::parse(r#"meta[name="author"]"#).unwrap();
    if let Some(el) = doc.select(&sel).next() {
        if let Some(content) = el.value().attr("content") {
            let author = content.trim();
            if !author.is_empty() {
                return Some(author.to_string());
            }
        }
    }

    // Try article:author
    let sel = Selector::parse(r#"meta[property="article:author"]"#).unwrap();
    if let Some(el) = doc.select(&sel).next() {
        if let Some(content) = el.value().attr("content") {
            return Some(content.trim().to_string());
        }
    }

    // Try .author, .byline selectors
    let sel = Selector::parse(".author, .byline, [rel='author']").unwrap();
    if let Some(el) = doc.select(&sel).next() {
        let text = el.text().collect::<String>().trim().to_string();
        if !text.is_empty() && text.len() < 100 {
            return Some(text);
        }
    }

    None
}

fn extract_date(doc: &Html) -> Option<DateTime<Utc>> {
    // Try article:published_time
    let selectors = [
        r#"meta[property="article:published_time"]"#,
        r#"meta[name="date"]"#,
        r#"meta[name="DC.date"]"#,
        r#"meta[name="publish-date"]"#,
    ];

    for sel_str in &selectors {
        let sel = Selector::parse(sel_str).unwrap();
        if let Some(el) = doc.select(&sel).next() {
            if let Some(content) = el.value().attr("content") {
                if let Some(dt) = parse_date(content.trim()) {
                    return Some(dt);
                }
            }
        }
    }

    // Try <time> element
    let sel = Selector::parse("time[datetime]").unwrap();
    if let Some(el) = doc.select(&sel).next() {
        if let Some(dt_str) = el.value().attr("datetime") {
            if let Some(dt) = parse_date(dt_str.trim()) {
                return Some(dt);
            }
        }
    }

    None
}

fn parse_date(s: &str) -> Option<DateTime<Utc>> {
    // Try ISO 8601
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }

    // Try common formats
    let formats = ["%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%d %B %Y"];
    for fmt in &formats {
        if let Ok(nd) = NaiveDate::parse_from_str(s, fmt) {
            return Some(nd.and_hms_opt(0, 0, 0)?.and_utc());
        }
    }

    None
}

fn extract_language(doc: &Html) -> Option<String> {
    // Try html lang attribute
    let sel = Selector::parse("html[lang]").unwrap();
    if let Some(el) = doc.select(&sel).next() {
        if let Some(lang) = el.value().attr("lang") {
            return Some(lang.to_string());
        }
    }

    // Try meta language
    let sel = Selector::parse(r#"meta[http-equiv="content-language"]"#).unwrap();
    if let Some(el) = doc.select(&sel).next() {
        if let Some(content) = el.value().attr("content") {
            return Some(content.to_string());
        }
    }

    None
}

fn extract_description(doc: &Html) -> Option<String> {
    let sel = Selector::parse(r#"meta[name="description"]"#).unwrap();
    if let Some(el) = doc.select(&sel).next() {
        if let Some(content) = el.value().attr("content") {
            let desc = content.trim();
            if !desc.is_empty() {
                return Some(desc.to_string());
            }
        }
    }

    // Fallback to og:description
    let sel = Selector::parse(r#"meta[property="og:description"]"#).unwrap();
    if let Some(el) = doc.select(&sel).next() {
        if let Some(content) = el.value().attr("content") {
            return Some(content.trim().to_string());
        }
    }

    None
}

fn extract_json_ld(doc: &Html) -> Option<serde_json::Value> {
    let sel = Selector::parse(r#"script[type="application/ld+json"]"#).unwrap();
    for el in doc.select(&sel) {
        let text = el.text().collect::<String>();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) {
            return Some(val);
        }
    }
    None
}

fn extract_open_graph(doc: &Html) -> Option<OpenGraphData> {
    let get_og = |property: &str| -> Option<String> {
        let sel = Selector::parse(&format!(r#"meta[property="{}"]"#, property)).unwrap();
        doc.select(&sel)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };

    let og = OpenGraphData {
        og_title: get_og("og:title"),
        og_description: get_og("og:description"),
        og_image: get_og("og:image"),
        og_type: get_og("og:type"),
        og_site_name: get_og("og:site_name"),
    };

    // Only return if at least one field is set
    if og.og_title.is_some() || og.og_description.is_some() {
        Some(og)
    } else {
        None
    }
}

fn extract_links(doc: &Html, base_url: &str) -> Vec<Link> {
    let sel = Selector::parse("a[href]").unwrap();
    let base = match url::Url::parse(base_url) {
        Ok(u) => u,
        Err(_) => return vec![],
    };
    let base_domain = base.host_str().unwrap_or("");

    doc.select(&sel)
        .filter_map(|el| {
            let href = el.value().attr("href")?;
            let resolved = base.join(href).ok()?;
            if resolved.scheme() != "http" && resolved.scheme() != "https" {
                return None;
            }
            let anchor = el.text().collect::<String>().trim().to_string();
            let link_domain = resolved.host_str().unwrap_or("");
            Some(Link {
                url: resolved.to_string(),
                anchor_text: anchor,
                is_external: link_domain != base_domain,
            })
        })
        .collect()
}

fn extract_tables(doc: &Html) -> Vec<Table> {
    let table_sel = Selector::parse("table").unwrap();
    let tr_sel = Selector::parse("tr").unwrap();
    let th_sel = Selector::parse("th").unwrap();
    let td_sel = Selector::parse("td").unwrap();

    let mut tables = Vec::new();

    for table_el in doc.select(&table_sel) {
        let mut headers = Vec::new();
        let mut rows = Vec::new();

        for (_i, row) in table_el.select(&tr_sel).enumerate() {
            let ths: Vec<String> = row
                .select(&th_sel)
                .map(|c| c.text().collect::<String>().trim().to_string())
                .collect();

            if !ths.is_empty() && headers.is_empty() {
                headers = ths;
                continue;
            }

            let tds: Vec<String> = row
                .select(&td_sel)
                .map(|c| c.text().collect::<String>().trim().to_string())
                .collect();

            if !tds.is_empty() {
                rows.push(tds);
            }
        }

        if !rows.is_empty() {
            tables.push(Table { headers, rows });
        }
    }

    tables
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_og_data() {
        let html = r#"<html><head>
            <meta property="og:title" content="My Article">
            <meta property="og:description" content="An article about things">
            <meta property="og:image" content="https://example.com/image.jpg">
            <meta property="og:type" content="article">
        </head><body></body></html>"#;

        let result = extract_metadata(html, "https://example.com/");
        let og = result.open_graph.unwrap();
        assert_eq!(og.og_title.as_deref(), Some("My Article"));
        assert_eq!(og.og_type.as_deref(), Some("article"));
    }

    #[test]
    fn extract_json_ld_data() {
        let html = r#"<html><head>
            <script type="application/ld+json">
            {"@type": "Article", "headline": "Test", "author": {"name": "John"}}
            </script>
        </head><body></body></html>"#;

        let result = extract_metadata(html, "https://example.com/");
        let ld = result.json_ld.unwrap();
        assert_eq!(ld["@type"], "Article");
        assert_eq!(ld["headline"], "Test");
    }

    #[test]
    fn extract_author_from_meta() {
        let html = r#"<html><head><meta name="author" content="Jane Doe"></head><body></body></html>"#;
        let result = extract_metadata(html, "https://example.com/");
        assert_eq!(result.author.as_deref(), Some("Jane Doe"));
    }

    #[test]
    fn extract_date_iso() {
        let html = r#"<html><head><meta property="article:published_time" content="2025-03-15T10:30:00Z"></head><body></body></html>"#;
        let result = extract_metadata(html, "https://example.com/");
        assert!(result.published_date.is_some());
    }

    #[test]
    fn extract_language() {
        let html = r#"<html lang="en-US"><body></body></html>"#;
        let result = extract_metadata(html, "https://example.com/");
        assert_eq!(result.language.as_deref(), Some("en-US"));
    }

    #[test]
    fn extract_description() {
        let html = r#"<html><head><meta name="description" content="A page about stuff"></head><body></body></html>"#;
        let result = extract_metadata(html, "https://example.com/");
        assert_eq!(result.description.as_deref(), Some("A page about stuff"));
    }

    #[test]
    fn extract_table() {
        let html = r#"<html><body>
            <table>
                <tr><th>Name</th><th>Value</th></tr>
                <tr><td>Alpha</td><td>1</td></tr>
                <tr><td>Beta</td><td>2</td></tr>
            </table>
        </body></html>"#;

        let result = extract_metadata(html, "https://example.com/");
        assert_eq!(result.tables.len(), 1);
        assert_eq!(result.tables[0].headers, vec!["Name", "Value"]);
        assert_eq!(result.tables[0].rows.len(), 2);
        assert_eq!(result.tables[0].rows[0], vec!["Alpha", "1"]);
    }

    #[test]
    fn extract_links_resolved() {
        let html = r#"<a href="/about">About</a><a href="https://other.com">External</a>"#;
        let result = extract_metadata(html, "https://example.com/");
        assert_eq!(result.links.len(), 2);
        assert_eq!(result.links[0].url, "https://example.com/about");
        assert!(!result.links[0].is_external);
        assert!(result.links[1].is_external);
    }
}
