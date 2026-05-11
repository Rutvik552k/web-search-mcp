use scraper::{Html, Selector};
use std::collections::HashSet;

/// A parsed search result from a search engine results page.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    pub snippet: String,
}

/// Detect if a URL is a search engine results page and extract result links.
///
/// Supports: Brave Search, Mojeek, DuckDuckGo, Wikipedia Search, ArXiv, Reddit Search.
/// Returns None if the URL is not a recognized search page.
pub fn parse_search_results(url: &str, html: &str) -> Option<Vec<SearchResult>> {
    let domain = url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_lowercase()))?;

    match domain.as_str() {
        d if d.contains("search.brave.com") => Some(parse_brave(html)),
        d if d.contains("mojeek.com") => Some(parse_mojeek(html)),
        d if d.contains("duckduckgo.com") => Some(parse_duckduckgo(html)),
        d if d.contains("wikipedia.org") && url.contains("search") => Some(parse_wikipedia_search(html)),
        d if d.contains("arxiv.org") && url.contains("search") => Some(parse_arxiv(html)),
        d if d.contains("reddit.com") && url.contains("search") => Some(parse_reddit_search(html)),
        d if d.contains("hn.algolia.com") => Some(parse_hackernews(html)),
        d if d.contains("scholar.google.com") => Some(parse_google_scholar(html)),
        d if d.contains("pubmed.ncbi.nlm.nih.gov") => Some(parse_pubmed(html)),
        _ => None,
    }
}

/// Parse Brave Search results.
/// Result links use svelte classes, target external URLs.
fn parse_brave(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    // Brave result links: <a href="..." class="...l1...">
    // They link to external sites, not brave.com
    let a_sel = Selector::parse("a[href]").unwrap();

    for el in doc.select(&a_sel) {
        let href = match el.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        // Skip brave internal links
        if href.starts_with("/") || href.contains("brave.com") || href.contains("bravesoftware") {
            continue;
        }
        if !href.starts_with("http") {
            continue;
        }

        // Skip social/video/shopping
        if is_noise_url(href) {
            continue;
        }

        let classes = el.value().attr("class").unwrap_or("");
        // Brave uses svelte-generated classes. Result links have "l1" class
        if !classes.contains("l1") && !classes.contains("title") {
            continue;
        }

        if !seen.insert(href.to_string()) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();

        results.push(SearchResult {
            url: href.to_string(),
            title,
            snippet: String::new(),
        });
    }

    // If class-based extraction failed, fallback to all external links
    if results.is_empty() {
        results = extract_all_external_links(&doc, "search.brave.com", &mut seen);
    }

    results.truncate(15);
    results
}

/// Parse Mojeek results.
/// Result links use class="title".
fn parse_mojeek(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    let title_sel = Selector::parse("a.title[href]").unwrap();
    for el in doc.select(&title_sel) {
        let href = match el.value().attr("href") {
            Some(h) if h.starts_with("http") => h,
            _ => continue,
        };

        if !seen.insert(href.to_string()) || is_noise_url(href) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();

        results.push(SearchResult {
            url: href.to_string(),
            title,
            snippet: String::new(),
        });
    }

    if results.is_empty() {
        results = extract_all_external_links(&doc, "mojeek.com", &mut seen);
    }

    results.truncate(15);
    results
}

/// Parse DuckDuckGo HTML results.
fn parse_duckduckgo(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    // DDG HTML: <a class="result__a" href="...">
    let result_sel = Selector::parse("a.result__a[href], .result__url[href]").unwrap();
    for el in doc.select(&result_sel) {
        let href = match el.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        // DDG wraps URLs in redirect: //duckduckgo.com/l/?uddg=ENCODED_URL
        let actual_url = if href.contains("uddg=") {
            href.split("uddg=")
                .nth(1)
                .and_then(|u| urlencoding_decode(u))
                .unwrap_or_else(|| href.to_string())
        } else if href.starts_with("http") {
            href.to_string()
        } else {
            continue;
        };

        if !seen.insert(actual_url.clone()) || is_noise_url(&actual_url) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();

        results.push(SearchResult {
            url: actual_url,
            title,
            snippet: String::new(),
        });
    }

    if results.is_empty() {
        results = extract_all_external_links(&doc, "duckduckgo.com", &mut seen);
    }

    results.truncate(15);
    results
}

/// Parse Wikipedia search results page.
/// Extract article links from search results.
fn parse_wikipedia_search(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    let a_sel = Selector::parse("a[href]").unwrap();
    for el in doc.select(&a_sel) {
        let href = match el.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        // Only article links: /wiki/Actual_Article
        if !href.starts_with("/wiki/") {
            continue;
        }

        // Skip meta pages
        let skip_prefixes = [
            "/wiki/Special:", "/wiki/Wikipedia:", "/wiki/Help:",
            "/wiki/Talk:", "/wiki/User:", "/wiki/Portal:",
            "/wiki/File:", "/wiki/Template:", "/wiki/Category:",
            "/wiki/MediaWiki:", "/wiki/Main_Page",
        ];
        if skip_prefixes.iter().any(|p| href.starts_with(p)) {
            continue;
        }

        let full_url = format!("https://en.wikipedia.org{href}");
        if !seen.insert(full_url.clone()) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();
        if title.is_empty() || title.len() < 2 {
            continue;
        }

        results.push(SearchResult {
            url: full_url,
            title,
            snippet: String::new(),
        });
    }

    // Deduplicate, keep first occurrence
    results.truncate(10);
    results
}

/// Parse ArXiv search results.
fn parse_arxiv(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    // ArXiv result links: <a href="/abs/XXXX.XXXXX">
    let a_sel = Selector::parse("a[href]").unwrap();
    for el in doc.select(&a_sel) {
        let href = match el.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        if !href.contains("/abs/") {
            continue;
        }

        let full_url = if href.starts_with("http") {
            href.to_string()
        } else {
            format!("https://arxiv.org{href}")
        };

        if !seen.insert(full_url.clone()) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();

        results.push(SearchResult {
            url: full_url,
            title,
            snippet: String::new(),
        });
    }

    results.truncate(10);
    results
}

/// Parse Reddit search results (old.reddit.com).
fn parse_reddit_search(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    let a_sel = Selector::parse("a[href]").unwrap();
    for el in doc.select(&a_sel) {
        let href = match el.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        // Reddit comment threads
        if !href.contains("/comments/") {
            continue;
        }

        let full_url = if href.starts_with("http") {
            href.to_string()
        } else {
            format!("https://old.reddit.com{href}")
        };

        if !seen.insert(full_url.clone()) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();
        if title.len() < 5 {
            continue;
        }

        results.push(SearchResult {
            url: full_url,
            title,
            snippet: String::new(),
        });
    }

    results.truncate(10);
    results
}

/// Parse Hacker News (Algolia) results.
fn parse_hackernews(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    extract_all_external_links(&doc, "hn.algolia.com", &mut HashSet::new())
}

/// Parse Google Scholar results.
fn parse_google_scholar(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    extract_all_external_links(&doc, "scholar.google.com", &mut HashSet::new())
}

/// Parse PubMed search results.
fn parse_pubmed(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    let a_sel = Selector::parse("a[href]").unwrap();
    for el in doc.select(&a_sel) {
        let href = match el.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        // PubMed article links: /NNNNNNNN/ (numeric ID)
        if !href.starts_with("/") || href.len() < 3 {
            continue;
        }
        let path = href.trim_matches('/');
        if !path.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        let full_url = format!("https://pubmed.ncbi.nlm.nih.gov{href}");
        if !seen.insert(full_url.clone()) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();
        if title.len() < 5 {
            continue;
        }

        results.push(SearchResult {
            url: full_url,
            title,
            snippet: String::new(),
        });
    }

    results.truncate(10);
    results
}

/// Fallback: extract all external links from a search page.
fn extract_all_external_links(
    doc: &Html,
    own_domain: &str,
    seen: &mut HashSet<String>,
) -> Vec<SearchResult> {
    let a_sel = Selector::parse("a[href]").unwrap();
    let mut results = Vec::new();

    for el in doc.select(&a_sel) {
        let href = match el.value().attr("href") {
            Some(h) if h.starts_with("http") => h,
            _ => continue,
        };

        if href.contains(own_domain) || is_noise_url(href) {
            continue;
        }

        if !seen.insert(href.to_string()) {
            continue;
        }

        let title = el.text().collect::<String>().trim().to_string();
        if title.len() < 3 {
            continue;
        }

        results.push(SearchResult {
            url: href.to_string(),
            title,
            snippet: String::new(),
        });
    }

    results.truncate(15);
    results
}

/// Filter out noise URLs (social, video, shopping, etc.)
fn is_noise_url(url: &str) -> bool {
    let noise = [
        "youtube.com/watch", "facebook.com", "twitter.com/", "x.com/",
        "instagram.com", "tiktok.com", "pinterest.com",
        "amazon.com/dp/", "amazon.com/gp/", "ebay.com",
        "play.google.com", "apps.apple.com",
        "linkedin.com/in/", "linkedin.com/company/",
    ];
    noise.iter().any(|n| url.contains(n))
}

/// Simple URL decoding for DDG redirect URLs.
fn urlencoding_decode(s: &str) -> Option<String> {
    let decoded = s
        .replace("%3A", ":")
        .replace("%2F", "/")
        .replace("%3F", "?")
        .replace("%3D", "=")
        .replace("%26", "&")
        .replace("%23", "#")
        .replace("%25", "%");
    // Take URL up to first & (strip DDG tracking params)
    Some(decoded.split('&').next()?.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_brave_results() {
        let html = r#"<html><body>
            <a href="https://rust-lang.org/" class="svelte-14r20fy l1">Rust Language</a>
            <a href="https://en.wikipedia.org/wiki/Rust" class="svelte-14r20fy l1">Rust - Wikipedia</a>
            <a href="/settings" class="l1">Settings</a>
            <a href="https://brave.com/about">About Brave</a>
        </body></html>"#;

        let results = parse_brave(html);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].url, "https://rust-lang.org/");
        assert_eq!(results[0].title, "Rust Language");
        assert_eq!(results[1].url, "https://en.wikipedia.org/wiki/Rust");
    }

    #[test]
    fn parse_mojeek_results() {
        let html = r#"<html><body>
            <a class="title" href="https://rust-lang.org/">Rust</a>
            <a class="title" href="https://doc.rust-lang.org/book/">The Book</a>
            <a href="https://mojeek.com/about">About</a>
        </body></html>"#;

        let results = parse_mojeek(html);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].url, "https://rust-lang.org/");
    }

    #[test]
    fn parse_wikipedia_search_results() {
        let html = r#"<html><body>
            <a href="/wiki/Quantum_computing">Quantum computing</a>
            <a href="/wiki/Qubit">Qubit</a>
            <a href="/wiki/Special:Search">Search</a>
            <a href="/wiki/Wikipedia:About">About</a>
            <a href="/wiki/Category:Physics">Physics Cat</a>
        </body></html>"#;

        let results = parse_wikipedia_search(html);
        assert_eq!(results.len(), 2); // Only real articles, not Special/Wikipedia/Category
        assert!(results[0].url.contains("Quantum_computing"));
        assert!(results[1].url.contains("Qubit"));
    }

    #[test]
    fn parse_arxiv_results() {
        let html = r#"<html><body>
            <a href="/abs/2301.12345">Paper Title One</a>
            <a href="/abs/2302.67890">Paper Title Two</a>
            <a href="/help/about">About ArXiv</a>
        </body></html>"#;

        let results = parse_arxiv(html);
        assert_eq!(results.len(), 2);
        assert!(results[0].url.contains("abs/2301.12345"));
    }

    #[test]
    fn detect_search_page() {
        assert!(parse_search_results("https://search.brave.com/search?q=test", "<html></html>").is_some());
        assert!(parse_search_results("https://www.mojeek.com/search?q=test", "<html></html>").is_some());
        assert!(parse_search_results("https://example.com/article", "<html></html>").is_none());
    }

    #[test]
    fn noise_urls_filtered() {
        assert!(is_noise_url("https://youtube.com/watch?v=abc"));
        assert!(is_noise_url("https://facebook.com/page"));
        assert!(!is_noise_url("https://nature.com/article"));
    }

    #[test]
    fn dedup_results() {
        let html = r#"<html><body>
            <a href="https://example.com/page" class="svelte-14r20fy l1">Link 1</a>
            <a href="https://example.com/page" class="svelte-14r20fy l1">Link 1 again</a>
        </body></html>"#;

        let results = parse_brave(html);
        assert_eq!(results.len(), 1); // deduped
    }
}
