//! Tavily keyed-API `SearchSource` (ADR 0004 §4.1 + Addendum A.4).
//!
//! Optional, zero-IP-cost augmentation. The default product path does NOT use
//! this — it is only built when an operator sets `search_api_key_env` to a SET
//! env var and `search_api_provider == "tavily"` (see `selector.rs`).
//!
//! # Security posture (A.4 — binding)
//!
//! - The API key is held as a plain `String`, read from the environment by the
//!   *caller* (never inside this client — ASR-SC3 / api-security SECRETS), and
//!   is serialized ONLY into the request body sent over TLS. It is NEVER logged
//!   and NEVER appears in any Debug/Serialize output: this struct deliberately
//!   does NOT derive `Debug`/`Serialize`, and errors are logged as a single
//!   generic WARN with no body and no key.
//! - Transport is locked down: HTTPS only, host HARDCODED to
//!   `https://api.tavily.com/search` (no config-settable base URL — prevents
//!   key-exfil to an attacker endpoint), invalid certs rejected, redirects
//!   disabled. The `#[cfg(test)] with_base` seam is test-only and can NEVER be
//!   reached from config or any runtime path (A.6.3).
//! - Any error (401 bad key / 429 quota / timeout / transport / parse) is
//!   non-fatal: log ONE warn and return `Ok(vec![])` so the selector falls
//!   through to the next source. A bad key MUST NEVER break search (ASR-SC3).
//!
//! # Unverified schema seam (Rule 1)
//!
//! VERIFY request/response schema against docs.tavily.com before production
//! (ADR §11.3). The request/response field names live in two PURE functions
//! (`build_body`, `parse_results`) so a vendor terms change is a one-function
//! edit, not a rewrite. No live call is made here (testing rule: no real
//! external services) — the pure functions are unit-tested over fixtures.

use async_trait::async_trait;
use serde_json::json;

use crate::search_source::{SearchHit, SearchSource};

/// Hardcoded Tavily endpoint (A.4 — NO config-settable base URL).
const TAVILY_URL: &str = "https://api.tavily.com/search";

/// Upper bound on title/snippet length before they enter the shared index
/// (A.4 sanitization — untrusted text persists in the shared session index and
/// can surface under unrelated later queries, KI-1).
const MAX_FIELD_LEN: usize = 2000;

/// Tavily keyed search source. Holds the API key; intentionally NOT `Debug`.
pub struct TavilySource {
    http: reqwest::Client,
    /// API key STRING — read from env by the caller, never by this client.
    /// Serialized only into the TLS request body; never logged/printed.
    api_key: String,
    /// Endpoint base. Hardcoded to `TAVILY_URL` in `new`; only the
    /// `#[cfg(test)]` `with_base` seam can change it.
    base: String,
}

impl TavilySource {
    /// Construct from the API key STRING (the caller reads it from the env —
    /// this client never touches `std::env`) and the per-request timeout.
    ///
    /// The reqwest client is transport-locked per A.4: HTTPS, valid certs only,
    /// no redirects, bounded timeout.
    pub fn new(api_key: String, timeout_secs: u64) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            // A.4: be explicit even though this is the default.
            .danger_accept_invalid_certs(false)
            // A.4: a keyed request must never be redirected (no key-exfil hop).
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_default();
        Self {
            http,
            api_key,
            base: TAVILY_URL.to_string(),
        }
    }

    /// Point the client at a local stub (unit tests only). MUST be
    /// `#[cfg(test)]`-only — never reachable from config or runtime (A.6.3),
    /// which is what lets A.4 hardcode the production host safely.
    #[cfg(test)]
    pub fn with_base(mut self, base: String) -> Self {
        self.base = base;
        self
    }

    /// PURE request-body builder (unit-tested).
    ///
    /// VERIFY request/response schema against docs.tavily.com before production
    /// (ADR §11.3). `search_depth: "basic"` = 1 credit [TAVILY-2].
    fn build_body(api_key: &str, query: &str, limit: usize) -> serde_json::Value {
        json!({
            "api_key": api_key,
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
        })
    }

    /// PURE response parser (unit-tested). Maps `results[].url/.title/.content`
    /// to `SearchHit`, skipping non-http(s) URLs and sanitizing text (A.4).
    ///
    /// VERIFY request/response schema against docs.tavily.com before production
    /// (ADR §11.3).
    fn parse_results(json: &serde_json::Value) -> Vec<SearchHit> {
        let items = match json["results"].as_array() {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut hits = Vec::with_capacity(items.len());
        for item in items {
            let url = match item["url"].as_str() {
                // SSRF note: scheme is gated here, but the FETCH path (Wave 2b)
                // still owns full SSRF validation (A.3) before opening a socket.
                // FIX #7: strip control chars from the url too (not just
                // title/snippet) before it reaches logs/index.
                Some(u) if u.starts_with("http://") || u.starts_with("https://") => sanitize_url(u),
                _ => continue,
            };
            let title = sanitize(item["title"].as_str().unwrap_or(""));
            let snippet = sanitize(item["content"].as_str().unwrap_or(""));
            hits.push(SearchHit { url, title, snippet });
        }
        hits
    }
}

/// A.4: bound length and strip ASCII control chars (keeps `\t`/`\n`/`\r` out
/// too — they become spaces) before untrusted text enters the shared index.
pub(crate) fn sanitize(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.chars().count() > MAX_FIELD_LEN {
        trimmed.chars().take(MAX_FIELD_LEN).collect()
    } else {
        trimmed.to_string()
    }
}

/// FIX #7: control-char strip for the `url` field. Same A.4 intent as
/// [`sanitize`] (untrusted text from a search hit reaches logs + the index), but
/// control chars are DROPPED rather than replaced with spaces — a space inside a
/// URL is itself invalid, so substituting one would corrupt the URL, whereas a
/// raw `\n`/`\r` is a log/index-injection vector. Length is bounded like
/// `sanitize`. The caller still scheme-gates (http(s) only) before this.
pub(crate) fn sanitize_url(s: &str) -> String {
    let cleaned: String = s.chars().filter(|c| !c.is_control()).collect();
    let trimmed = cleaned.trim();
    if trimmed.chars().count() > MAX_FIELD_LEN {
        trimmed.chars().take(MAX_FIELD_LEN).collect()
    } else {
        trimmed.to_string()
    }
}

#[async_trait]
impl SearchSource for TavilySource {
    fn name(&self) -> &str {
        "tavily"
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        let body = Self::build_body(&self.api_key, query, limit);
        // On ANY error: ONE generic warn (no key, no body, no url query) +
        // Ok(vec![]) so the selector falls through (ASR-SC3).
        let resp = match self.http.post(&self.base).json(&body).send().await {
            Ok(r) => r,
            Err(_) => {
                tracing::warn!(source = "tavily", "keyed search request failed; falling through");
                return Ok(Vec::new());
            }
        };
        if !resp.status().is_success() {
            // 401 (bad key) / 429 (quota) / 5xx — non-fatal. Log status code
            // only (never the key/body).
            tracing::warn!(
                source = "tavily",
                status = resp.status().as_u16(),
                "keyed search non-success; falling through"
            );
            return Ok(Vec::new());
        }
        let json: serde_json::Value = match resp.json().await {
            Ok(j) => j,
            Err(_) => {
                tracing::warn!(source = "tavily", "keyed search response parse failed; falling through");
                return Ok(Vec::new());
            }
        };
        Ok(Self::parse_results(&json))
    }

    fn spends_ip_reputation(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_body_has_expected_fields() {
        let body = TavilySource::build_body("SECRET123", "rust async", 7);
        assert_eq!(body["api_key"], "SECRET123");
        assert_eq!(body["query"], "rust async");
        assert_eq!(body["max_results"], 7);
        assert_eq!(body["search_depth"], "basic");
    }

    #[test]
    fn parse_results_maps_url_title_content() {
        let json = json!({
            "results": [
                { "url": "https://a.com", "title": "A", "content": "alpha" },
                { "url": "https://b.com", "title": "B", "content": "beta" },
            ]
        });
        let hits = TavilySource::parse_results(&json);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], SearchHit {
            url: "https://a.com".into(),
            title: "A".into(),
            snippet: "alpha".into(),
        });
        assert_eq!(hits[1].url, "https://b.com");
        assert_eq!(hits[1].snippet, "beta");
    }

    #[test]
    fn parse_results_skips_non_http_urls() {
        let json = json!({
            "results": [
                { "url": "ftp://x.com", "title": "X", "content": "x" },
                { "url": "javascript:alert(1)", "title": "J", "content": "j" },
                { "url": "https://ok.com", "title": "OK", "content": "ok" },
            ]
        });
        let hits = TavilySource::parse_results(&json);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].url, "https://ok.com");
    }

    #[test]
    fn parse_results_empty_or_missing_array_is_empty() {
        assert!(TavilySource::parse_results(&json!({})).is_empty());
        assert!(TavilySource::parse_results(&json!({ "results": [] })).is_empty());
    }

    #[test]
    fn sanitize_strips_control_chars() {
        let dirty = "hello\u{0007}\u{0000}world\nline";
        let clean = sanitize(dirty);
        assert!(!clean.contains('\u{0007}'));
        assert!(!clean.contains('\u{0000}'));
        assert!(!clean.contains('\n'));
        assert!(clean.contains("hello"));
        assert!(clean.contains("world"));
    }

    #[test]
    fn sanitize_bounds_length() {
        let long = "x".repeat(5000);
        let clean = sanitize(&long);
        assert_eq!(clean.chars().count(), MAX_FIELD_LEN);
    }

    #[test]
    fn sanitize_url_drops_control_chars_and_bounds_length() {
        // FIX #7: control chars (CR/LF/NUL — log/index-injection vectors) are
        // DROPPED from the url, not turned into spaces (a space would corrupt it).
        let dirty = "https://ex.com/a\r\n\u{0000}b";
        let clean = sanitize_url(dirty);
        assert_eq!(clean, "https://ex.com/ab");
        assert!(!clean.contains('\n') && !clean.contains('\r') && !clean.contains('\u{0000}'));
        // Length bounded like sanitize.
        let long = format!("https://ex.com/{}", "x".repeat(5000));
        assert_eq!(sanitize_url(&long).chars().count(), MAX_FIELD_LEN);
    }

    #[test]
    fn parse_results_sanitizes_url_control_chars() {
        // A url carrying an embedded CRLF must be stripped before it reaches the
        // index/logs (still scheme-gated http(s)).
        let json = json!({
            "results": [
                { "url": "https://a.com/x\r\ny", "title": "t", "content": "c" },
            ]
        });
        let hits = TavilySource::parse_results(&json);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].url, "https://a.com/xy");
    }

    #[test]
    fn parse_results_sanitizes_title_and_snippet() {
        let json = json!({
            "results": [
                { "url": "https://a.com", "title": "ti\u{0007}tle", "content": "snip\u{0000}pet" },
            ]
        });
        // Control chars become spaces (not deleted) to avoid gluing tokens.
        let hits = TavilySource::parse_results(&json);
        assert_eq!(hits[0].title, "ti tle");
        assert_eq!(hits[0].snippet, "snip pet");
    }

    #[test]
    fn metadata_is_keyless_safe() {
        let src = TavilySource::new("KEY".into(), 5);
        assert_eq!(src.name(), "tavily");
        assert!(!src.spends_ip_reputation());
    }
}
