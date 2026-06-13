//! In-band hydration-state extraction (ADR 0002, technique 1 — "Rung -1" IN-band).
//!
//! Parses embedded application state that SSR/SSG frameworks ship inside the
//! *already-fetched* HTML body. Zero extra network requests. Used to (a) enrich
//! extraction metadata and (b) SALVAGE pages where the rendered DOM is thin
//! (soft-blocked / JS-shell) but a hydration blob still carries the real content.
//!
//! Verified marker shapes (web-research 2026-06, cited inline):
//! - Pages Router: `<script id="__NEXT_DATA__" type="application/json">{…}</script>`
//!   with the record under `props.pageProps`.
//!   https://github.com/vercel/next.js/discussions/15117
//!   https://brightdata.com/blog/how-tos/web-scraping-with-next-js
//! - App Router (Next 13+): React Server Component "flight" data pushed via
//!   `self.__next_f.push([n, "chunk"])` where each push is a 2-tuple `[0..=3, string]`.
//!   Full wire-format decoding is out of scope (njsparser is the reference); we do a
//!   conservative, high-precision targeted-key scan over the concatenated chunks.
//!   https://www.trickster.dev/post/scraping-nextjs-web-sites-in-2025/
//!   https://github.com/novitae/njsparser
//! - Nuxt 3: `<script type="application/json" id="__NUXT_DATA__">[…devalue array…]</script>`
//!   (devalue is reference-based but valid JSON; readable strings are plain array
//!   elements, so a string-harvest recovers content). Legacy/Nuxt 2 use a
//!   `window.__NUXT__ = {…}` assignment.
//!   https://deepwiki.com/nuxt/nuxt/6.2-payload-system
//! - Generic SSR state: `window.__INITIAL_STATE__`, `window.__APOLLO_STATE__`,
//!   `window.__PRELOADED_STATE__`, `window.__REDUX_STATE__` assignments.
//!   https://www.apollographql.com/docs/react/features/server-side-rendering/

use regex::Regex;
use scraper::{Html, Selector};
use serde_json::Value;

/// Maximum recursion depth when walking a parsed state tree (guards against
/// pathological / deeply-nested or cyclic-by-reference blobs).
const MAX_DEPTH: usize = 12;
/// Minimum body length (chars) for a hydration hit to count as "meaningful"
/// content on its own (vs. only carrying a title/description).
const MIN_BODY: usize = 200;

// Field keys are matched case-insensitively after stripping non-alphanumerics,
// so `articleBody`, `article_body`, and `article-body` all normalize equal.
const TITLE_KEYS: &[&str] = &["headline", "title", "seotitle", "metatitle", "ogtitle", "pagetitle"];
const BODY_KEYS: &[&str] = &[
    "articlebody", "bodyhtml", "bodytext", "body", "contenthtml", "content",
    "fulltext", "richtext", "maincontent", "story", "text",
];
const DESC_KEYS: &[&str] =
    &["metadescription", "description", "excerpt", "summary", "subtitle", "dek", "standfirst", "abstract"];
const AUTHOR_KEYS: &[&str] = &["authorname", "author", "byline", "creator", "writer"];
const DATE_KEYS: &[&str] = &[
    "datepublished", "publishedat", "publisheddate", "published", "firstpublished",
    "publishtime", "pubdate", "date",
];

/// Human-readable content recovered from a hydration blob.
#[derive(Debug, Clone, Default)]
pub struct HydrationContent {
    pub title: Option<String>,
    pub body_text: String,
    pub description: Option<String>,
    pub author: Option<String>,
    /// ISO-8601-ish date string; parsed downstream by `metadata::parse_date`.
    pub published: Option<String>,
    /// Which marker produced this hit (for tracing/tests).
    pub source: &'static str,
}

/// Try every known hydration marker in cost/reliability order and return the
/// first that yields meaningful content. `None` when no usable blob is present
/// (pure-CSR page, or the soft-block replaced the body with a challenge shell).
pub fn extract(html: &str) -> Option<HydrationContent> {
    // Quick reject: avoid parsing if none of the marker tokens are present.
    if !html.contains("__NEXT_DATA__")
        && !html.contains("__NUXT_DATA__")
        && !html.contains("__NUXT__")
        && !html.contains("__INITIAL_STATE__")
        && !html.contains("__APOLLO_STATE__")
        && !html.contains("__PRELOADED_STATE__")
        && !html.contains("__REDUX_STATE__")
        && !html.contains("__next_f")
    {
        return None;
    }

    let document = Html::parse_document(html);

    if let Some(h) = from_script_json(&document, "__NEXT_DATA__", "next_data", true) {
        if meaningful(&h) {
            return Some(h);
        }
    }
    if let Some(h) = from_script_json(&document, "__NUXT_DATA__", "nuxt_data", false) {
        if meaningful(&h) {
            return Some(h);
        }
    }
    if let Some(h) = from_window_assignments(html) {
        if meaningful(&h) {
            return Some(h);
        }
    }
    if let Some(h) = from_rsc_flight(html) {
        if meaningful(&h) {
            return Some(h);
        }
    }
    None
}

fn meaningful(h: &HydrationContent) -> bool {
    h.body_text.chars().count() >= MIN_BODY || (h.title.is_some() && h.description.is_some())
}

/// Parse a `<script id="…" type="application/json">` JSON blob.
/// `prefer_pageprops` digs into `props.pageProps` first (Next.js Pages Router).
fn from_script_json(
    doc: &Html,
    id: &str,
    source: &'static str,
    prefer_pageprops: bool,
) -> Option<HydrationContent> {
    let sel = Selector::parse(&format!(r#"script#{id}"#)).ok()?;
    let el = doc.select(&sel).next()?;
    let txt = el.text().collect::<String>();
    let value: Value = serde_json::from_str(txt.trim()).ok()?;

    let mut h = if prefer_pageprops {
        let target = value
            .get("props")
            .and_then(|p| p.get("pageProps"))
            .unwrap_or(&value);
        let from_target = harvest(target);
        // Fall back to whole document if pageProps had no body.
        if from_target.body_text.is_empty() && from_target.title.is_none() {
            harvest(&value)
        } else {
            from_target
        }
    } else {
        harvest(&value)
    };
    h.source = source;
    Some(h)
}

/// Scan for `window.__X__ = {…}` / `= [...]` assignment blobs and harvest them.
fn from_window_assignments(html: &str) -> Option<HydrationContent> {
    // Ordered: app-specific first, then generic Redux/Apollo stores.
    let markers = [
        ("__NUXT__", "nuxt_window"),
        ("__INITIAL_STATE__", "initial_state"),
        ("__APOLLO_STATE__", "apollo_state"),
        ("__PRELOADED_STATE__", "preloaded_state"),
        ("__REDUX_STATE__", "redux_state"),
    ];

    for (marker, source) in markers {
        let mut search_from = 0;
        while let Some(rel) = html[search_from..].find(marker) {
            let after_marker = search_from + rel + marker.len();
            search_from = after_marker;

            let rest = html[after_marker..].trim_start();
            // Must be an assignment (`marker = …`); skip incidental matches such
            // as the `id="__NUXT_DATA__"` attribute or references in comments.
            let Some(after_eq) = rest.strip_prefix('=') else {
                continue;
            };
            let after_eq = after_eq.trim_start();
            // A function-wrapped payload (`= (function(){…})(…)`, devalue) is not
            // plain JSON — bail rather than mis-parse.
            let Some(start) = after_eq.find(['{', '[']) else {
                continue;
            };
            // Only accept when the bracket is the very next token (allow a leading
            // `(` for `=({…})`). Reject if real code precedes it.
            let preamble = after_eq[..start].trim();
            if !preamble.is_empty() && preamble != "(" {
                continue;
            }
            let Some(json_slice) = balanced_extract(&after_eq[start..]) else {
                continue;
            };
            if let Ok(value) = serde_json::from_str::<Value>(json_slice) {
                let mut h = harvest(&value);
                if !h.body_text.is_empty() || h.title.is_some() {
                    h.source = source;
                    return Some(h);
                }
            }
        }
    }
    None
}

/// Best-effort App-Router RSC flight extraction. Concatenates the string payloads
/// from `self.__next_f.push([n, "chunk"])` and runs a high-precision targeted-key
/// scan (we do NOT attempt full React-wire decoding — see module docs).
fn from_rsc_flight(html: &str) -> Option<HydrationContent> {
    const PUSH: &str = "self.__next_f.push(";
    let mut concat = String::new();
    let mut search_from = 0;
    while let Some(rel) = html[search_from..].find(PUSH) {
        let open = search_from + rel + PUSH.len() - 1; // points at '('
        search_from = open + 1;
        let Some(arg) = balanced_extract_delims(&html[open..], '(', ')') else {
            continue;
        };
        // `arg` includes the parens: (…). Strip them to get the JS array literal.
        let inner = &arg[1..arg.len() - 1];
        if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(inner) {
            if let Some(Value::String(chunk)) = items.get(1) {
                concat.push_str(chunk);
            }
        }
    }
    if concat.is_empty() {
        return None;
    }

    let mut h = harvest_targeted(&concat);
    h.source = "rsc_flight";
    if h.body_text.is_empty() && h.title.is_none() {
        None
    } else {
        Some(h)
    }
}

// ── Harvesting ───────────────────────────────────────────────────────────────

#[derive(Default)]
struct Harvest {
    title: Option<String>,
    body: String,
    description: Option<String>,
    author: Option<String>,
    published: Option<String>,
}

/// Recursively walk a parsed state tree, collecting human-readable fields by key.
fn harvest(value: &Value) -> HydrationContent {
    let mut acc = Harvest::default();
    walk(value, 0, &mut acc);
    HydrationContent {
        title: acc.title,
        body_text: acc.body,
        description: acc.description,
        author: acc.author,
        published: acc.published,
        source: "",
    }
}

fn walk(value: &Value, depth: usize, acc: &mut Harvest) {
    if depth > MAX_DEPTH {
        return;
    }
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                if let Value::String(s) = v {
                    apply_field(&normalize_key(k), s, acc);
                }
                walk(v, depth + 1, acc);
            }
        }
        Value::Array(items) => {
            for v in items {
                walk(v, depth + 1, acc);
            }
        }
        _ => {}
    }
}

/// Targeted, precision-first harvest over a raw text blob (used for RSC flight),
/// matching `"key":"value"` pairs without parsing the surrounding wire format.
fn harvest_targeted(blob: &str) -> HydrationContent {
    let mut acc = Harvest::default();
    for key in BODY_KEYS {
        if let Some(s) = json_string_after(blob, key) {
            apply_field(key, &s, &mut acc);
        }
    }
    for key in TITLE_KEYS {
        if let Some(s) = json_string_after(blob, key) {
            apply_field(key, &s, &mut acc);
        }
    }
    for key in DESC_KEYS {
        if let Some(s) = json_string_after(blob, key) {
            apply_field(key, &s, &mut acc);
        }
    }
    for key in AUTHOR_KEYS {
        if let Some(s) = json_string_after(blob, key) {
            apply_field(key, &s, &mut acc);
        }
    }
    for key in DATE_KEYS {
        if let Some(s) = json_string_after(blob, key) {
            apply_field(key, &s, &mut acc);
        }
    }
    HydrationContent {
        title: acc.title,
        body_text: acc.body,
        description: acc.description,
        author: acc.author,
        published: acc.published,
        source: "",
    }
}

fn apply_field(norm_key: &str, raw: &str, acc: &mut Harvest) {
    if BODY_KEYS.contains(&norm_key) {
        let cleaned = strip_html_tags(raw);
        if cleaned.chars().count() > acc.body.chars().count() {
            acc.body = cleaned;
        }
    } else if TITLE_KEYS.contains(&norm_key) {
        if acc.title.is_none() {
            let t = raw.trim();
            if (5..=250).contains(&t.chars().count()) {
                acc.title = Some(t.to_string());
            }
        }
    } else if DESC_KEYS.contains(&norm_key) {
        if acc.description.is_none() {
            let d = strip_html_tags(raw);
            if (20..=2000).contains(&d.chars().count()) {
                acc.description = Some(d);
            }
        }
    } else if AUTHOR_KEYS.contains(&norm_key) {
        if acc.author.is_none() {
            let a = raw.trim();
            if (2..=120).contains(&a.chars().count()) {
                acc.author = Some(a.to_string());
            }
        }
    } else if DATE_KEYS.contains(&norm_key)
        && acc.published.is_none()
        && raw.chars().any(|c| c.is_ascii_digit())
        && (4..=40).contains(&raw.trim().chars().count())
    {
        acc.published = Some(raw.trim().to_string());
    }
}

fn normalize_key(k: &str) -> String {
    k.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

// ── Low-level helpers ────────────────────────────────────────────────────────

/// Extract a balanced `{…}` or `[…]` slice; `s` must start at the opening bracket.
fn balanced_extract(s: &str) -> Option<&str> {
    let open = s.as_bytes().first().copied()?;
    let close = match open {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    };
    balanced_extract_delims(s, open as char, close as char)
}

/// Extract a balanced delimited slice (`open`/`close`), respecting JSON string
/// literals and backslash escapes. `s` must start at the opening delimiter.
fn balanced_extract_delims(s: &str, open: char, close: char) -> Option<&str> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let (ob, cb) = (open as u8, close as u8);

    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            x if x == ob => depth += 1,
            x if x == cb => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Find `"key":"value"` in a raw blob and return the JSON-unescaped value.
/// Precision-first: only matches a string value immediately after the key.
/// Key matching is ASCII-case-insensitive (JSON keys are camelCase in the wild,
/// e.g. `articleBody`); `key` is expected already-lowercase. ASCII lowercasing
/// preserves byte length, so positions found in the lowered copy index the original.
fn json_string_after(haystack: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let lowered = haystack.to_ascii_lowercase();
    let mut from = 0;
    while let Some(rel) = lowered[from..].find(&needle) {
        let after = from + rel + needle.len();
        from = after;
        let rest = haystack[after..].trim_start();
        let Some(rest) = rest.strip_prefix(':') else {
            continue;
        };
        let rest = rest.trim_start();
        if !rest.starts_with('"') {
            continue;
        }
        // Walk to the closing quote honoring escapes, then JSON-decode the literal.
        let bytes = rest.as_bytes();
        let mut escaped = false;
        for i in 1..bytes.len() {
            let b = bytes[i];
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                let literal = &rest[..=i];
                if let Ok(s) = serde_json::from_str::<String>(literal) {
                    return Some(s);
                }
                break;
            }
        }
    }
    None
}

/// Strip HTML tags and decode a handful of common entities, then collapse spaces.
fn strip_html_tags(s: &str) -> String {
    let re = Regex::new(r"(?s)<[^>]+>").unwrap();
    let no_tags = re.replace_all(s, " ");
    let decoded = no_tags
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_next_data_pageprops() {
        // Real-world-shaped Next.js Pages Router blob (props.pageProps.article).
        let html = r#"<html><head>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"article":{
            "headline":"Quantum chips reach new milestone",
            "articleBody":"Researchers announced a breakthrough in error correction today. The new chip sustains coherence far longer than previous designs, a result that could accelerate practical quantum computing by years according to the team behind it.",
            "description":"A breakthrough in quantum error correction.",
            "author":{"name":"Dana Lee"},
            "datePublished":"2026-05-01T08:00:00Z"
        }}},"buildId":"abc123","page":"/article"}
        </script></head><body><div id="root"></div></body></html>"#;

        let h = extract(html).expect("should find __NEXT_DATA__");
        assert_eq!(h.source, "next_data");
        assert_eq!(h.title.as_deref(), Some("Quantum chips reach new milestone"));
        assert!(h.body_text.contains("error correction"));
        assert_eq!(h.published.as_deref(), Some("2026-05-01T08:00:00Z"));
    }

    #[test]
    fn parses_nuxt_data_array() {
        // Nuxt 3 devalue-style array (valid JSON array; strings are plain elements).
        let body = "The council approved the new transit plan after months of debate. \
        The plan expands light rail to three additional districts and adds protected \
        bike lanes along the riverfront corridor, funded by a regional bond measure.";
        let html = format!(
            r#"<html><body><div id="__nuxt"></div>
            <script type="application/json" id="__NUXT_DATA__">
            [{{"data":1}},{{"title":"Transit plan approved","articleBody":"{body}"}}]
            </script></body></html>"#
        );

        let h = extract(&html).expect("should find __NUXT_DATA__");
        assert_eq!(h.source, "nuxt_data");
        assert!(h.body_text.contains("light rail"));
        assert_eq!(h.title.as_deref(), Some("Transit plan approved"));
    }

    #[test]
    fn parses_window_initial_state_assignment() {
        let html = r#"<html><body><script>
        window.__INITIAL_STATE__ = {"page":{"content":"This is the full recovered article body that the rendered DOM never showed because the page shipped only a JavaScript shell to the browser on first paint. The real content lived entirely in the Redux preload state, which is exactly what this in-band salvage path is designed to recover.","title":"Recovered Story"}};
        </script></body></html>"#;

        let h = extract(html).expect("should find __INITIAL_STATE__");
        assert_eq!(h.source, "initial_state");
        assert!(h.body_text.contains("full recovered article body"));
    }

    #[test]
    fn parses_rsc_flight_targeted() {
        // App Router flight: 2-tuple pushes [n, "chunk"]; chunk carries embedded JSON.
        let html = r#"<html><body>
        <script>self.__next_f.push([0])</script>
        <script>self.__next_f.push([1,"{\"headline\":\"Flight headline works\",\"articleBody\":\"This article body is embedded inside the React Server Component flight payload and must be recovered by the targeted key scan even though we do not decode the full React wire format here. Recovering it proves the App Router salvage path works end to end.\"}"])</script>
        </body></html>"#;

        let h = extract(html).expect("should find flight data");
        assert_eq!(h.source, "rsc_flight");
        assert_eq!(h.title.as_deref(), Some("Flight headline works"));
        assert!(h.body_text.contains("React Server Component flight"));
    }

    #[test]
    fn body_html_is_stripped() {
        let html = r#"<html><head><script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"post":{"title":"Tagged body","bodyHtml":"<p>First para with <strong>bold</strong> text.</p><p>Second paragraph continues the story with considerably more detail so that the stripped body comfortably crosses the minimum length threshold required for a meaningful hydration salvage hit, while also exercising the HTML tag stripping path thoroughly.</p>"}}}}
        </script></head><body></body></html>"#;

        let h = extract(html).expect("should parse");
        assert!(!h.body_text.contains('<'));
        assert!(h.body_text.contains("First para with bold text"));
    }

    #[test]
    fn no_blob_returns_none() {
        let html = r#"<html><head><title>Plain Page</title></head>
        <body><article><p>Just ordinary server-rendered HTML with no hydration state of any kind.</p></article></body></html>"#;
        assert!(extract(html).is_none());
    }

    #[test]
    fn csr_shell_without_content_returns_none() {
        // Marker present but no recoverable human-readable content (pure CSR).
        let html = r#"<html><body><div id="__next"></div>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{}},"buildId":"x","page":"/"}
        </script></body></html>"#;
        assert!(extract(html).is_none());
    }

    #[test]
    fn balanced_extract_handles_nested_and_strings() {
        let s = r#"{"a":{"b":"}}}"},"c":[1,2]}TRAILING"#;
        let got = balanced_extract(s).unwrap();
        assert_eq!(got, r#"{"a":{"b":"}}}"},"c":[1,2]}"#);
    }
}
