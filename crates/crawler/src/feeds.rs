//! R2 — Alternative-surface FEED probe (ADR 0003 Rung R2 / TASKS.md 1.5.4).
//!
//! Syndication feeds (RSS 2.0, Atom 1.0, JSON Feed 1.1) are static, rarely
//! WAF-challenged surfaces. When the HTML article page is blocked, the same
//! content is frequently still reachable through the site's feed — which often
//! carries the full article body (`content:encoded` / Atom `<content>` /
//! JSON Feed `content_html`) or at least a usable summary. This module discovers
//! a site's feeds, parses them, and recovers the feed item matching the page we
//! failed to fetch.
//!
//! ## Ground-truth specs (verified before implementation, Rule 1)
//!
//! - **RSS 2.0**: <https://www.rssboard.org/rss-specification>. An `<item>` may
//!   carry `<title>`, `<link>`, `<description>`, `<pubDate>` (RFC 822 date).
//!   At least one of `<title>`/`<description>` must be present; all are
//!   otherwise optional.
//! - **`content:encoded`**: from the RSS 1.0 Content module, namespace URI
//!   `http://purl.org/rss/1.0/modules/content/`
//!   (<https://www.rssboard.org/rss-profile#namespace-elements-content>). When
//!   present it carries the FULL HTML body; `<description>` is then the summary,
//!   ordered before `content:encoded` within the item.
//! - **Atom 1.0**: RFC 4287 <https://www.rfc-editor.org/rfc/rfc4287>. A `<feed>`
//!   holds `<entry>` elements with `<title>`, `<link href="..." rel="alternate">`
//!   (the canonical link is the `href` of the `rel="alternate"` link; `rel`
//!   defaults to `"alternate"` when omitted, per RFC 4287 §4.2.7.2),
//!   `<summary>`, `<content>`, and `<updated>` (RFC 3339 date).
//! - **JSON Feed 1.1**: <https://www.jsonfeed.org/version/1.1/>. Top-level
//!   `{ "version", "items": [...] }`; each item has `id`, optional `url`,
//!   `title`, `content_html`, `content_text`, `summary`, `date_published`
//!   (RFC 3339). At least one of `content_html`/`content_text` must be present.
//! - **HTML autodiscovery**: <https://www.rssboard.org/rss-autodiscovery> and
//!   WHATWG. A `<link>` in `<head>` with `rel="alternate"` and a feed `type`
//!   announces a feed. Verified MIME types:
//!   `application/rss+xml`, `application/atom+xml`, `application/feed+json`.
//!   `href` may be relative and is resolved against the page URL.
//! - **Well-known paths**: common conventional feed locations
//!   (<https://www.rssboard.org/rss-autodiscovery>, and the petefreitag /
//!   roboleary autodiscovery guides): `/feed`, `/rss`, `/rss.xml`, `/atom.xml`,
//!   `/feed.json` (plus `/feed.xml`, `/index.xml`). We probe a small set as a
//!   fallback when autodiscovery yields nothing.
//!
//! ## Scraper namespace note (verified)
//!
//! `scraper` is built on html5ever (lenient HTML parsing) and exposes elements
//! via CSS `Selector`s. A CSS selector cannot target a `:` inside a tag name —
//! the colon is the CSS pseudo-class delimiter, so `content:encoded` (and the
//! escaped `content\:encoded`) cannot be selected cleanly. html5ever also folds
//! the namespaced tag into something CSS can't address. Per the module spec we
//! therefore extract `content:encoded` with a dedicated regex over the raw item
//! XML, and document the reason here. Plain tags (`item`, `title`, `link`,
//! `description`, `entry`, `summary`, `content`, `updated`, `pubDate`) are
//! selected normally via `scraper`, matching the existing `sitemap.rs` pattern.

use scraper::{Html, Selector};
use serde::Deserialize;

/// Which syndication format a discovered feed link points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedKind {
    /// RSS 2.0 (`application/rss+xml`).
    Rss,
    /// Atom 1.0 / RFC 4287 (`application/atom+xml`).
    Atom,
    /// JSON Feed 1.1 (`application/feed+json`).
    JsonFeed,
}

/// A feed URL discovered in page HTML, with its declared format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedLink {
    /// Absolute feed URL (relative hrefs resolved against the page URL).
    pub url: String,
    /// Declared feed format from the link's `type` attribute.
    pub kind: FeedKind,
}

/// A single parsed feed entry. Fields are normalized across RSS/Atom/JSON Feed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FeedItem {
    /// Entry title (plain text). Empty string if the feed omitted it.
    pub title: String,
    /// Canonical entry link (RSS `<link>`, Atom `rel="alternate"` href,
    /// JSON Feed `url`). Empty string if absent.
    pub link: String,
    /// Full HTML body when the feed carried one (RSS `content:encoded`,
    /// Atom `<content>`, JSON Feed `content_html`).
    pub content_html: Option<String>,
    /// Shorter summary/description (RSS `<description>`, Atom `<summary>`,
    /// JSON Feed `summary` or `content_text`).
    pub summary: Option<String>,
    /// Publication date as the feed declared it (RSS `pubDate` RFC 822,
    /// Atom `updated` / JSON Feed `date_published` RFC 3339). Not parsed —
    /// callers normalize if they need a `chrono` value.
    pub published: Option<String>,
}

// ---------------------------------------------------------------------------
// Feed autodiscovery (pure)
// ---------------------------------------------------------------------------

/// Discover syndication feeds declared in a page's `<head>`.
///
/// Scans `<link rel="alternate">` tags, classifies each by its `type` MIME
/// string, and resolves relative `href`s against `base_url` via
/// [`url::Url::join`]. Unknown/non-feed `type`s are ignored. Deterministic and
/// network-free.
///
/// Ground truth: <https://www.rssboard.org/rss-autodiscovery>.
pub fn discover_feeds(html: &str, base_url: &str) -> Vec<FeedLink> {
    let doc = Html::parse_document(html);
    // `link[rel~="alternate"]` matches rel token lists like rel="alternate".
    // We also accept any <link> with a feed `type`, since some sites omit rel.
    let link_sel = match Selector::parse("link") {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let base = url::Url::parse(base_url).ok();
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for el in doc.select(&link_sel) {
        let ty = el.value().attr("type").unwrap_or("").trim().to_ascii_lowercase();
        let kind = match ty.as_str() {
            "application/rss+xml" => FeedKind::Rss,
            // RSS 1.0/RDF feeds are served as rss+xml too; treat as Rss.
            "application/rdf+xml" => FeedKind::Rss,
            "application/atom+xml" => FeedKind::Atom,
            "application/feed+json" | "application/json" => FeedKind::JsonFeed,
            _ => continue,
        };

        // rel="alternate" is the autodiscovery rel. Be lenient: accept missing
        // rel (some feeds use rel="alternate home" or omit it), but reject
        // rel values that clearly aren't alternate (e.g. "stylesheet").
        let rel = el.value().attr("rel").unwrap_or("alternate").to_ascii_lowercase();
        if !rel.split_whitespace().any(|t| t == "alternate") && !rel.trim().is_empty() {
            // A feed type with a non-alternate rel (e.g. preload) — skip.
            if !rel.split_whitespace().any(|t| t == "alternate") {
                continue;
            }
        }

        let href = match el.value().attr("href") {
            Some(h) if !h.trim().is_empty() => h.trim(),
            _ => continue,
        };

        let abs = match &base {
            Some(b) => match b.join(href) {
                Ok(u) => u.to_string(),
                Err(_) => continue,
            },
            // No valid base: only keep already-absolute hrefs.
            None => {
                if href.starts_with("http://") || href.starts_with("https://") {
                    href.to_string()
                } else {
                    continue;
                }
            }
        };

        if seen.insert(abs.clone()) {
            out.push(FeedLink { url: abs, kind });
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Feed parsing (pure)
// ---------------------------------------------------------------------------

/// Parse a feed body of a known [`FeedKind`] into normalized [`FeedItem`]s.
///
/// Deterministic and network-free. Returns an empty `Vec` on unparseable input
/// rather than erroring (this is a non-fatal salvage path).
pub fn parse_feed(body: &str, kind: FeedKind) -> Vec<FeedItem> {
    match kind {
        FeedKind::Rss => parse_rss(body),
        FeedKind::Atom => parse_atom(body),
        FeedKind::JsonFeed => parse_json_feed(body),
    }
}

/// Auto-detect the feed format from the body and parse it.
///
/// Used when we fetch a discovered URL without trusting its declared `type`
/// (the server's `Content-Type` and the autodiscovery `type` can lie). Sniffs:
/// a leading `{` (after whitespace) → JSON Feed; `<feed` present → Atom;
/// otherwise → RSS. Deterministic and network-free.
pub fn sniff_and_parse(body: &str) -> Vec<FeedItem> {
    // Strip a UTF-8 BOM FIRST (it may precede whitespace), then trim, so the
    // first-char sniff is reliable regardless of BOM/whitespace ordering.
    let trimmed = body.strip_prefix('\u{feff}').unwrap_or(body).trim_start();
    let trimmed = trimmed.strip_prefix('\u{feff}').unwrap_or(trimmed);
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return parse_json_feed(body);
    }
    // XML: distinguish Atom (<feed) from RSS/RDF. Atom's root is <feed>.
    let lowered = trimmed.to_ascii_lowercase();
    if lowered.contains("<feed") && !lowered.contains("<rss") {
        return parse_atom(body);
    }
    parse_rss(body)
}

/// Strip a single wrapping CDATA section, if present, and trim.
fn unwrap_cdata(s: &str) -> String {
    let t = s.trim();
    if let Some(inner) = t.strip_prefix("<![CDATA[").and_then(|x| x.strip_suffix("]]>")) {
        inner.trim().to_string()
    } else {
        t.to_string()
    }
}

/// Decode the small set of mandatory XML predefined entities (RFC 4287 / XML
/// 1.0 §4.6). Applied to escaped text fields (RSS `<description>`, Atom
/// `<content type="html">`). Order matters: `&amp;` last so we don't double-decode.
fn decode_xml_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Normalize whitespace-only strings to `None`.
fn non_empty(s: String) -> Option<String> {
    if s.trim().is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Extract the first occurrence of `<tag ...>inner</tag>` from `xml` and return
/// the inner content (CDATA unwrapped, trimmed). Tag-name match is
/// case-insensitive and tolerant of attributes and a namespace prefix already
/// embedded in `tag` (e.g. `content:encoded`). Returns `None` if absent.
///
/// We parse RSS/Atom over the RAW XML with regex rather than `scraper` because
/// `scraper`/html5ever parse in HTML mode and CORRUPT XML feeds two ways
/// (verified empirically, 2026-06): (1) `<![CDATA[..]]>` is reinterpreted as a
/// bogus HTML comment, mangling `content:encoded` bodies; (2) `<link>` is a void
/// element in HTML, so its text URL is dropped and the closing tag is lost. Raw
/// regex extraction avoids both. Cited in the module-level note.
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    // (?is) case-insensitive + dotall. Match `<tag` then either attributes-then-`>`
    // or a bare `>`, capture lazily up to the matching closing tag.
    let pattern = format!(r"(?is)<{tag}(?:\s[^>]*)?>(.*?)</{tag}\s*>", tag = regex::escape(tag));
    let re = regex::Regex::new(&pattern).ok()?;
    re.captures(xml)
        .and_then(|c| c.get(1))
        .map(|m| unwrap_cdata(m.as_str()))
}

/// Split a feed's raw XML into per-record blocks for `tag` (`item` or `entry`),
/// returning the inner XML of each. Case-insensitive, attribute-tolerant.
fn split_records(xml: &str, tag: &str) -> Vec<String> {
    let pattern = format!(r"(?is)<{tag}(?:\s[^>]*)?>(.*?)</{tag}\s*>", tag = regex::escape(tag));
    match regex::Regex::new(&pattern) {
        Ok(re) => re
            .captures_iter(xml)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Parse RSS 2.0 (<https://www.rssboard.org/rss-specification>) over raw XML.
///
/// `<item>` → `title`/`link`/`description`/`pubDate`; full body from
/// `content:encoded` (RSS 1.0 Content module namespace
/// `http://purl.org/rss/1.0/modules/content/`). Raw-XML regex parse — see
/// [`extract_tag`] for why `scraper` is unsafe for XML feeds.
fn parse_rss(xml: &str) -> Vec<FeedItem> {
    let mut items = Vec::new();
    for block in split_records(xml, "item") {
        let title = extract_tag(&block, "title").unwrap_or_default();
        // RSS <link> carries the URL as the element's text (no attributes).
        let link = extract_tag(&block, "link").unwrap_or_default();
        // <description> is the summary; it may carry escaped HTML — leave as-is
        // for the caller, but decode entities so it is human-readable.
        let summary = extract_tag(&block, "description")
            .map(|s| decode_xml_entities(&s))
            .and_then(non_empty);
        let published = extract_tag(&block, "pubDate").and_then(non_empty);
        // Full HTML body when present (namespaced content:encoded).
        let content_html = extract_tag(&block, "content:encoded").and_then(non_empty);

        items.push(FeedItem {
            title,
            link,
            content_html,
            summary,
            published,
        });
    }
    items
}

/// Extract the canonical entry link from an Atom `<entry>` block.
///
/// Prefers the `<link>` with `rel="alternate"`; `rel` defaults to `"alternate"`
/// when the attribute is absent (RFC 4287 §4.2.7.2). Falls back to the first
/// `<link>` with an `href` if no explicit/implicit alternate is found. Raw-XML
/// regex (Atom `<link>` is attribute-based; see [`extract_tag`] rationale).
fn extract_atom_link(entry_xml: &str) -> String {
    // Match each self-closing or paired <link ...> and inspect its attributes.
    let link_re = match regex::Regex::new(r"(?is)<link\b([^>]*?)/?>") {
        Ok(r) => r,
        Err(_) => return String::new(),
    };
    let href_re = regex::Regex::new(r#"(?is)\bhref\s*=\s*["']([^"']*)["']"#).unwrap();
    let rel_re = regex::Regex::new(r#"(?is)\brel\s*=\s*["']([^"']*)["']"#).unwrap();

    let mut first_href = String::new();
    for caps in link_re.captures_iter(entry_xml) {
        let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let href = href_re
            .captures(attrs)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();
        if href.is_empty() {
            continue;
        }
        if first_href.is_empty() {
            first_href = href.clone();
        }
        // rel defaults to "alternate" when absent.
        let rel = rel_re
            .captures(attrs)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_ascii_lowercase())
            .unwrap_or_else(|| "alternate".to_string());
        if rel == "alternate" {
            return href;
        }
    }
    first_href
}

/// Parse Atom 1.0 / RFC 4287 (<https://www.rfc-editor.org/rfc/rfc4287>) over raw XML.
///
/// `<entry>` → `title`/`summary`/`content`/`updated`. The link is the `href` of
/// the `rel="alternate"` `<link>`. Raw-XML regex parse — see [`extract_tag`] for
/// why `scraper` is unsafe for XML feeds.
fn parse_atom(xml: &str) -> Vec<FeedItem> {
    let mut items = Vec::new();
    for block in split_records(xml, "entry") {
        let title = extract_tag(&block, "title").unwrap_or_default();
        let link = extract_atom_link(&block);
        let summary = extract_tag(&block, "summary")
            .map(|s| decode_xml_entities(&s))
            .and_then(non_empty);
        // <content> may be type="html" (escaped) or type="xhtml" (raw markup).
        // Decode entities so an escaped html body is returned as real HTML.
        let content_html = extract_tag(&block, "content")
            .map(|s| decode_xml_entities(&s))
            .and_then(non_empty);
        let published = extract_tag(&block, "updated").and_then(non_empty);

        items.push(FeedItem {
            title,
            link,
            content_html,
            summary,
            published,
        });
    }
    items
}

// --- JSON Feed 1.1 (https://www.jsonfeed.org/version/1.1/) ---

#[derive(Deserialize)]
struct JsonFeedDoc {
    #[serde(default)]
    items: Vec<JsonFeedItem>,
}

#[derive(Deserialize)]
struct JsonFeedItem {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content_html: Option<String>,
    #[serde(default)]
    content_text: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    date_published: Option<String>,
}

/// Parse JSON Feed 1.1 (<https://www.jsonfeed.org/version/1.1/>).
///
/// Maps `items[].{url,title,content_html,summary|content_text,date_published}`
/// onto [`FeedItem`]. Returns empty on invalid JSON (non-fatal salvage path).
fn parse_json_feed(body: &str) -> Vec<FeedItem> {
    // serde_json rejects a leading UTF-8 BOM, so strip it before parsing.
    let body = body.strip_prefix('\u{feff}').unwrap_or(body).trim_start();
    let doc: JsonFeedDoc = match serde_json::from_str(body) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    doc.items
        .into_iter()
        .map(|it| {
            // summary falls back to content_text (a plain-text alternative body).
            let summary = it.summary.and_then(non_empty).or_else(|| it.content_text.and_then(non_empty));
            FeedItem {
                title: it.title.unwrap_or_default(),
                link: it.url.unwrap_or_default(),
                content_html: it.content_html.and_then(non_empty),
                summary,
                published: it.date_published.and_then(non_empty),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Item matching (pure)
// ---------------------------------------------------------------------------

/// Normalize a URL for loose equality: lowercase scheme/host, collapse
/// http/https, drop a single trailing slash, drop a leading `www.`.
fn normalize_url(u: &str) -> String {
    let mut s = u.trim().to_string();
    // Strip scheme (treat http/https as equal).
    if let Some(rest) = s.strip_prefix("https://") {
        s = rest.to_string();
    } else if let Some(rest) = s.strip_prefix("http://") {
        s = rest.to_string();
    }
    if let Some(rest) = s.strip_prefix("www.") {
        s = rest.to_string();
    }
    // Drop a fragment; queries are kept (they can disambiguate articles).
    if let Some(idx) = s.find('#') {
        s.truncate(idx);
    }
    // Drop a single trailing slash.
    if s.ends_with('/') {
        s.pop();
    }
    s.to_ascii_lowercase()
}

/// Find the feed item whose `link` matches `target_url` under loose URL
/// normalization (scheme-agnostic, trailing-slash- and `www.`-insensitive).
///
/// This is how R2 recovers the body for the *specific* blocked page: the feed
/// lists many entries; we want the one whose link equals the URL we failed to
/// fetch. Returns `None` if no item matches. Deterministic.
pub fn find_item_for_url<'a>(items: &'a [FeedItem], target_url: &str) -> Option<&'a FeedItem> {
    let target = normalize_url(target_url);
    items.iter().find(|it| !it.link.is_empty() && normalize_url(&it.link) == target)
}

// ---------------------------------------------------------------------------
// Network probe (async, non-fatal)
// ---------------------------------------------------------------------------

/// Conventional well-known feed paths, tried in order as a fallback when HTML
/// autodiscovery finds nothing. Ground truth:
/// <https://www.rssboard.org/rss-autodiscovery> and common autodiscovery guides.
const WELL_KNOWN_PATHS: &[(&str, FeedKind)] = &[
    ("/feed", FeedKind::Rss),
    ("/rss", FeedKind::Rss),
    ("/rss.xml", FeedKind::Rss),
    ("/feed.xml", FeedKind::Rss),
    ("/index.xml", FeedKind::Rss),
    ("/atom.xml", FeedKind::Atom),
    ("/feed.json", FeedKind::JsonFeed),
];

/// R2 probe: try to recover `page_url`'s body via the site's feeds.
///
/// 1. Discover feeds declared in `page_html` (autodiscovery).
/// 2. Add conventional well-known paths (`/feed`, `/rss.xml`, …) resolved
///    against `page_url`'s origin, up to `max_probes` total candidate fetches.
/// 3. Fetch each candidate with a per-request `timeout_ms` budget, parse via
///    [`sniff_and_parse`] (declared `type` is not trusted), and return the
///    [`FeedItem`] whose link matches `page_url`. If no exact match is found in
///    any feed, return the first item of the first non-empty feed (the newest
///    entry, by feed convention) as a best-effort fallback.
///
/// Non-fatal by contract: any network/parse error yields `None`; never panics,
/// never propagates an error upward. This is a salvage rung, not a hard
/// dependency.
pub async fn probe_feeds(
    client: &reqwest::Client,
    page_url: &str,
    page_html: &str,
    timeout_ms: u64,
    max_probes: usize,
) -> Option<FeedItem> {
    if max_probes == 0 {
        return None;
    }

    // Build the ordered, de-duplicated candidate list: discovered feeds first
    // (most authoritative), then well-known conventional paths.
    let mut candidates: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for f in discover_feeds(page_html, page_url) {
        if seen.insert(f.url.clone()) {
            candidates.push(f.url);
        }
    }

    if let Ok(base) = url::Url::parse(page_url) {
        for (path, _kind) in WELL_KNOWN_PATHS {
            if let Ok(u) = base.join(path) {
                let s = u.to_string();
                if seen.insert(s.clone()) {
                    candidates.push(s);
                }
            }
        }
    }

    candidates.truncate(max_probes);

    let timeout = std::time::Duration::from_millis(timeout_ms);
    let mut newest_fallback: Option<FeedItem> = None;

    for url in candidates {
        // Per-candidate timeout. tokio::time::timeout wraps the whole fetch.
        let body = match tokio::time::timeout(timeout, fetch_text(client, &url)).await {
            Ok(Some(b)) => b,
            // Timed out, or the fetch returned None (network/HTTP error).
            _ => {
                tracing::debug!(feed_url = %url, "feed probe fetch failed or timed out");
                continue;
            }
        };

        let items = sniff_and_parse(&body);
        if items.is_empty() {
            continue;
        }

        // Exact match wins immediately.
        if let Some(hit) = find_item_for_url(&items, page_url) {
            tracing::debug!(feed_url = %url, "feed probe matched target page");
            return Some(hit.clone());
        }

        // Remember the first (newest) item of the first feed that had any.
        if newest_fallback.is_none() {
            newest_fallback = items.into_iter().next();
        }
    }

    newest_fallback
}

/// Fetch a URL as text, returning `None` on any error or non-success status.
/// Kept separate so the timeout in [`probe_feeds`] wraps the entire fetch
/// (connect + body read). Never errors upward.
async fn fetch_text(client: &reqwest::Client, url: &str) -> Option<String> {
    let resp = client.get(url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.text().await.ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- discover_feeds ---

    #[test]
    fn discover_rss_atom_json_with_relative_hrefs() {
        let html = r#"<!doctype html><html><head>
            <link rel="alternate" type="application/rss+xml" href="/feed.xml">
            <link rel="alternate" type="application/atom+xml" href="atom/">
            <link rel="alternate" type="application/feed+json" href="https://cdn.example.com/feed.json">
            <link rel="stylesheet" type="text/css" href="/style.css">
            </head><body></body></html>"#;
        let feeds = discover_feeds(html, "https://example.com/blog/post");
        assert_eq!(feeds.len(), 3, "should find 3 feeds, ignore the stylesheet");
        assert_eq!(feeds[0], FeedLink { url: "https://example.com/feed.xml".into(), kind: FeedKind::Rss });
        assert_eq!(feeds[1], FeedLink { url: "https://example.com/blog/atom/".into(), kind: FeedKind::Atom });
        assert_eq!(feeds[2], FeedLink { url: "https://cdn.example.com/feed.json".into(), kind: FeedKind::JsonFeed });
    }

    #[test]
    fn discover_ignores_non_feed_and_dedups() {
        let html = r#"<head>
            <link rel="alternate" type="application/rss+xml" href="https://a.com/f">
            <link rel="alternate" type="application/rss+xml" href="https://a.com/f">
            <link rel="preload" type="application/atom+xml" href="https://a.com/atom">
            <link rel="icon" href="/favicon.ico">
        </head>"#;
        let feeds = discover_feeds(html, "https://a.com/");
        assert_eq!(feeds.len(), 1, "dedup identical + drop preload/icon");
        assert_eq!(feeds[0].url, "https://a.com/f");
    }

    #[test]
    fn discover_no_feeds_returns_empty() {
        let html = "<head><title>nothing here</title></head>";
        assert!(discover_feeds(html, "https://x.com").is_empty());
    }

    // --- parse_rss ---

    #[test]
    fn parse_rss_with_content_encoded() {
        let xml = r#"<?xml version="1.0"?>
        <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
          <channel>
            <title>Example Blog</title>
            <item>
              <title>First Post</title>
              <link>https://example.com/posts/first</link>
              <description>A short summary.</description>
              <content:encoded><![CDATA[<p>Full <b>HTML</b> body here.</p>]]></content:encoded>
              <pubDate>Mon, 06 Sep 2021 16:20:00 +0000</pubDate>
            </item>
            <item>
              <title>Second Post</title>
              <link>https://example.com/posts/second</link>
              <description>Only a description, no content:encoded.</description>
            </item>
          </channel>
        </rss>"#;
        let items = parse_feed(xml, FeedKind::Rss);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].title, "First Post");
        assert_eq!(items[0].link, "https://example.com/posts/first");
        assert_eq!(items[0].summary.as_deref(), Some("A short summary."));
        assert_eq!(items[0].content_html.as_deref(), Some("<p>Full <b>HTML</b> body here.</p>"));
        assert_eq!(items[0].published.as_deref(), Some("Mon, 06 Sep 2021 16:20:00 +0000"));
        // Second item: no content:encoded → None, but description present.
        assert!(items[1].content_html.is_none());
        assert_eq!(items[1].summary.as_deref(), Some("Only a description, no content:encoded."));
        assert!(items[1].published.is_none());
    }

    #[test]
    fn parse_rss_decodes_escaped_description_and_handles_no_link() {
        let xml = r#"<rss version="2.0"><channel><item>
            <title>Escaped</title>
            <description>Read &lt;b&gt;more&lt;/b&gt; &amp; enjoy</description>
        </item></channel></rss>"#;
        let items = parse_feed(xml, FeedKind::Rss);
        assert_eq!(items.len(), 1);
        // Entities decoded; missing <link> → empty string (not a panic).
        assert_eq!(items[0].summary.as_deref(), Some("Read <b>more</b> & enjoy"));
        assert_eq!(items[0].link, "");
        assert!(items[0].content_html.is_none());
    }

    // --- parse_atom ---

    #[test]
    fn parse_atom_entry() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Example Feed</title>
          <entry>
            <title>Atom-Powered Robots Run Amok</title>
            <link rel="self" href="https://example.org/self"/>
            <link rel="alternate" href="https://example.org/2003/12/13/atom03"/>
            <updated>2003-12-13T18:30:02Z</updated>
            <summary>Some text.</summary>
            <content type="html">&lt;p&gt;Body markup.&lt;/p&gt;</content>
          </entry>
        </feed>"#;
        let items = parse_feed(xml, FeedKind::Atom);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].title, "Atom-Powered Robots Run Amok");
        // Must pick rel="alternate", not rel="self".
        assert_eq!(items[0].link, "https://example.org/2003/12/13/atom03");
        assert_eq!(items[0].summary.as_deref(), Some("Some text."));
        assert_eq!(items[0].published.as_deref(), Some("2003-12-13T18:30:02Z"));
        assert!(items[0].content_html.as_ref().unwrap().contains("Body markup."));
    }

    #[test]
    fn parse_atom_link_defaults_to_alternate_when_rel_absent() {
        let xml = r#"<feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>No rel</title>
            <link href="https://example.org/no-rel"/>
            <updated>2024-01-01T00:00:00Z</updated>
          </entry>
        </feed>"#;
        let items = parse_feed(xml, FeedKind::Atom);
        assert_eq!(items[0].link, "https://example.org/no-rel");
    }

    // --- parse_json_feed ---

    #[test]
    fn parse_json_feed_items() {
        let body = r#"{
          "version": "https://jsonfeed.org/version/1.1",
          "title": "My Example Feed",
          "items": [
            {
              "id": "2",
              "url": "https://example.org/second-item",
              "title": "Second item",
              "content_html": "<p>Hello from JSON Feed.</p>",
              "summary": "A summary.",
              "date_published": "2024-02-19T14:58:55-05:00"
            },
            {
              "id": "1",
              "url": "https://example.org/first-item",
              "content_text": "Plain text body only."
            }
          ]
        }"#;
        let items = parse_feed(body, FeedKind::JsonFeed);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].title, "Second item");
        assert_eq!(items[0].link, "https://example.org/second-item");
        assert_eq!(items[0].content_html.as_deref(), Some("<p>Hello from JSON Feed.</p>"));
        assert_eq!(items[0].summary.as_deref(), Some("A summary."));
        assert_eq!(items[0].published.as_deref(), Some("2024-02-19T14:58:55-05:00"));
        // Second item: no summary → falls back to content_text; no title → "".
        assert_eq!(items[1].title, "");
        assert!(items[1].content_html.is_none());
        assert_eq!(items[1].summary.as_deref(), Some("Plain text body only."));
    }

    #[test]
    fn parse_invalid_returns_empty() {
        assert!(parse_feed("not json at all", FeedKind::JsonFeed).is_empty());
        assert!(parse_feed("<html><body>not a feed</body></html>", FeedKind::Rss).is_empty());
    }

    // --- sniff_and_parse ---

    #[test]
    fn sniff_detects_json_atom_rss() {
        let json = r#"{"version":"https://jsonfeed.org/version/1.1","items":[{"id":"1","url":"https://x.com/a","title":"J"}]}"#;
        assert_eq!(sniff_and_parse(json).len(), 1);
        assert_eq!(sniff_and_parse(json)[0].title, "J");

        let atom = r#"<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>A</title><link rel="alternate" href="https://x.com/a"/></entry></feed>"#;
        let a = sniff_and_parse(atom);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].title, "A");
        assert_eq!(a[0].link, "https://x.com/a");

        let rss = r#"<rss version="2.0"><channel><item><title>R</title><link>https://x.com/r</link></item></channel></rss>"#;
        let r = sniff_and_parse(rss);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].title, "R");
    }

    #[test]
    fn sniff_handles_bom_and_leading_whitespace() {
        let json = "\u{feff}  \n {\"items\":[{\"id\":\"1\",\"url\":\"https://x.com/a\",\"title\":\"T\"}]}";
        let items = sniff_and_parse(json);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].title, "T");
    }

    // --- find_item_for_url ---

    #[test]
    fn find_item_matches_with_normalization() {
        let items = vec![
            FeedItem { link: "https://example.com/posts/first/".into(), title: "first".into(), ..Default::default() },
            FeedItem { link: "https://example.com/posts/second".into(), title: "second".into(), ..Default::default() },
        ];
        // http vs https, trailing slash mismatch, www. prefix → still matches.
        let hit = find_item_for_url(&items, "http://www.example.com/posts/first");
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().title, "first");

        let hit2 = find_item_for_url(&items, "https://example.com/posts/second/");
        assert_eq!(hit2.unwrap().title, "second");

        // No match.
        assert!(find_item_for_url(&items, "https://example.com/posts/missing").is_none());
    }

    #[test]
    fn find_item_ignores_empty_links() {
        let items = vec![FeedItem { link: "".into(), title: "no-link".into(), ..Default::default() }];
        assert!(find_item_for_url(&items, "https://example.com/").is_none());
    }

    // --- probe_feeds (no real network; max_probes=0 short-circuit) ---

    #[tokio::test]
    async fn probe_feeds_zero_probes_returns_none() {
        let client = reqwest::Client::new();
        let out = probe_feeds(&client, "https://example.com/page", "<html></html>", 100, 0).await;
        assert!(out.is_none());
    }
}
