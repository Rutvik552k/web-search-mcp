use regex::Regex;
use url::Url;

/// Detected pagination pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum PaginationPattern {
    /// ?page=N or &page=N
    QueryParam { param: String, start: u32 },
    /// ?offset=N (step by page_size)
    OffsetParam { param: String, step: u32 },
    /// /page/N/ in URL path
    PathSegment { prefix: String },
    /// Cursor-based (?cursor=xxx) — detected but can't auto-generate
    Cursor,
    /// No pagination detected
    None,
}

/// Detect the pagination pattern used by a URL.
pub fn detect_pagination(url: &str, html: &str) -> PaginationPattern {
    // Check for common query param patterns
    if let Ok(parsed) = Url::parse(url) {
        for (key, value) in parsed.query_pairs() {
            let key_lower = key.to_lowercase();
            if matches!(key_lower.as_str(), "page" | "p" | "pg" | "pagenum") {
                if let Ok(n) = value.parse::<u32>() {
                    return PaginationPattern::QueryParam {
                        param: key.to_string(),
                        start: n,
                    };
                }
            }
            if matches!(key_lower.as_str(), "offset" | "start" | "skip") {
                if let Ok(_n) = value.parse::<u32>() {
                    // Try to guess step size from page content
                    let step = guess_page_size(html);
                    return PaginationPattern::OffsetParam {
                        param: key.to_string(),
                        step,
                    };
                }
            }
            if matches!(key_lower.as_str(), "cursor" | "after" | "next_token") {
                return PaginationPattern::Cursor;
            }
        }
    }

    // Check for /page/N pattern in path
    let page_path_re = Regex::new(r"/page/(\d+)").unwrap();
    if let Some(caps) = page_path_re.captures(url) {
        let prefix = &url[..caps.get(0).unwrap().start()];
        return PaginationPattern::PathSegment {
            prefix: prefix.to_string(),
        };
    }

    PaginationPattern::None
}

/// Generate the next page URL from a detected pattern.
pub fn next_page_url(base_url: &str, pattern: &PaginationPattern, current_page: u32) -> Option<String> {
    match pattern {
        PaginationPattern::QueryParam { param, .. } => {
            let mut parsed = Url::parse(base_url).ok()?;
            let pairs: Vec<(String, String)> = parsed
                .query_pairs()
                .map(|(k, v)| {
                    if k == param.as_str() {
                        (k.to_string(), (current_page + 1).to_string())
                    } else {
                        (k.to_string(), v.to_string())
                    }
                })
                .collect();
            parsed.query_pairs_mut().clear();
            for (k, v) in &pairs {
                parsed.query_pairs_mut().append_pair(k, v);
            }
            Some(parsed.to_string())
        }
        PaginationPattern::OffsetParam { param, step } => {
            let mut parsed = Url::parse(base_url).ok()?;
            let new_offset = (current_page + 1) * step;
            let pairs: Vec<(String, String)> = parsed
                .query_pairs()
                .map(|(k, v)| {
                    if k == param.as_str() {
                        (k.to_string(), new_offset.to_string())
                    } else {
                        (k.to_string(), v.to_string())
                    }
                })
                .collect();
            parsed.query_pairs_mut().clear();
            for (k, v) in &pairs {
                parsed.query_pairs_mut().append_pair(k, v);
            }
            Some(parsed.to_string())
        }
        PaginationPattern::PathSegment { prefix } => {
            Some(format!("{}/page/{}", prefix, current_page + 1))
        }
        PaginationPattern::Cursor | PaginationPattern::None => None,
    }
}

/// Heuristic: guess page size from content (count items/articles/rows).
fn guess_page_size(html: &str) -> u32 {
    let lower = html.to_lowercase();
    // Count article-like elements
    let articles = lower.matches("<article").count()
        + lower.matches(r#"class="item"#).count()
        + lower.matches(r#"class="result"#).count()
        + lower.matches("<li").count() / 2; // rough guess

    if articles > 5 && articles < 100 {
        articles as u32
    } else {
        20 // default page size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_query_page_param() {
        let p = detect_pagination("https://example.com/search?q=rust&page=1", "");
        assert!(matches!(p, PaginationPattern::QueryParam { param, start } if param == "page" && start == 1));
    }

    #[test]
    fn detect_offset_param() {
        let p = detect_pagination("https://example.com/api?offset=0&limit=20", "");
        assert!(matches!(p, PaginationPattern::OffsetParam { param, .. } if param == "offset"));
    }

    #[test]
    fn detect_path_segment() {
        let p = detect_pagination("https://blog.com/articles/page/3", "");
        assert!(matches!(p, PaginationPattern::PathSegment { .. }));
    }

    #[test]
    fn detect_cursor() {
        let p = detect_pagination("https://api.com/data?cursor=abc123", "");
        assert_eq!(p, PaginationPattern::Cursor);
    }

    #[test]
    fn detect_none() {
        let p = detect_pagination("https://example.com/about", "");
        assert_eq!(p, PaginationPattern::None);
    }

    #[test]
    fn generate_next_page_query() {
        let pattern = PaginationPattern::QueryParam {
            param: "page".to_string(),
            start: 1,
        };
        let next = next_page_url("https://example.com/search?q=rust&page=1", &pattern, 1).unwrap();
        assert!(next.contains("page=2"));
        assert!(next.contains("q=rust"));
    }

    #[test]
    fn generate_next_page_path() {
        let pattern = PaginationPattern::PathSegment {
            prefix: "https://blog.com/articles".to_string(),
        };
        let next = next_page_url("https://blog.com/articles/page/1", &pattern, 1).unwrap();
        assert_eq!(next, "https://blog.com/articles/page/2");
    }
}
