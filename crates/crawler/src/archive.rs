//! R3 — Internet Archive (Wayback) CDX fallback rung.
//!
//! ADR 0003 (`docs/adr/0003-single-ip-escalation-ladder.md`) Rung R3.
//!
//! This rung NEVER touches the live target origin. It only queries
//! `web.archive.org`, so it carries **zero ban risk** against the target site.
//! When the live origin is hostile/blocked (CAPTCHA wall, 403, Cloudflare,
//! geo-block), we go around it by fetching the newest archived snapshot from the
//! Wayback Machine and handing that HTML to the extractor instead.
//!
//! # Ground truth (verified per CLAUDE.md Rule 1, June 2026)
//!
//! 1. **CDX query API** — `GET http://web.archive.org/cdx/search/cdx`
//!    Parameters used:
//!    - `url=<target>` — the URL to look up.
//!    - `output=json` — return a JSON **array of arrays**. The FIRST inner array
//!      is a header row of field names; every following inner array is one
//!      capture's values, positionally aligned to the header.
//!      The default public field set is:
//!      `["urlkey","timestamp","original","mimetype","statuscode","digest","length"]`
//!    - `filter=statuscode:200` — only keep captures that archived a 200 response.
//!    - `collapse=digest` — dedup adjacent identical captures (same content hash).
//!    - `limit=-1` — a **negative** limit returns captures from the END of the
//!      list, i.e. the most recent. `limit=-1` => the single newest snapshot.
//!    Docs: <https://github.com/internetarchive/wayback/blob/master/wayback-cdx-server/README.md>
//!          <https://archive.org/developers/wayback-cdx-server.html>
//!
//! 2. **Timestamp format** — the `timestamp` field is a 14-digit
//!    `YYYYMMDDhhmmss` string in UTC.
//!    Docs: <https://archive.org/developers/wayback-cdx-server.html>
//!
//! 3. **Raw snapshot fetch** — `GET http://web.archive.org/web/<ts>id_/<original>`
//!    The `id_` modifier returns the raw, ORIGINAL archived bytes with **no**
//!    Wayback toolbar injection and **no** link/resource rewriting — i.e. the
//!    page exactly as archived. (Contrast: `if_` renders without the toolbar but
//!    STILL rewrites links; default has both toolbar + rewriting.) We want `id_`
//!    so the extractor sees the real page markup.
//!    Docs: <https://archive.org/post/1008502/faq-on-id_-wayback-toolbar-removal>
//!          <https://en.wikipedia.org/wiki/Help:Using_the_Wayback_Machine>
//!
//! 4. **Rate-limit / etiquette** — IA treats 429/500/502/503/504 as retryable and
//!    may send a `Retry-After` header; recommended ~1 req/sec with a custom
//!    polite User-Agent so IA can identify the client. This rung is a
//!    best-effort, non-fatal fallback: on ANY 503/timeout/error it simply
//!    returns `None` and the caller falls through to the next rung — it does not
//!    retry aggressively or surface errors upward.
//!    Docs: <https://wayback.readthedocs.io/en/stable/_modules/wayback/_client.html>

use std::time::Duration;

use chrono::{TimeZone, Utc};
use serde_json::Value;
use tokio::time::timeout;

/// Configuration for the R3 Archive fallback rung.
///
/// Field names mirror ADR 0003 §6.2 (`archive_*` keys) so the orchestrator can
/// map config 1:1.
#[derive(Debug, Clone)]
pub struct ArchiveConfig {
    /// CDX search endpoint base, e.g. `http://web.archive.org/cdx/search/cdx`.
    /// (ADR key: `archive_cdx_endpoint`.)
    pub cdx_endpoint: String,
    /// Per-request timeout in milliseconds, applied to BOTH the CDX query and the
    /// snapshot fetch. (ADR key: `archive_timeout_ms`.)
    pub timeout_ms: u64,
    /// Reject snapshots older than this many days. `None` = accept any age.
    /// (ADR key: `archive_max_snapshot_age_days`.)
    pub max_snapshot_age_days: Option<u32>,
    /// Polite User-Agent so IA can identify the client. `None` => a sane default.
    /// (ADR key: `archive_user_agent`.)
    pub user_agent: Option<String>,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            cdx_endpoint: "http://web.archive.org/cdx/search/cdx".to_string(),
            timeout_ms: 10_000,
            max_snapshot_age_days: None,
            user_agent: None,
        }
    }
}

/// Default polite User-Agent used when `ArchiveConfig::user_agent` is `None`.
const DEFAULT_UA: &str =
    "jarvis-web-search-mcp/0.6 (+https://github.com/Rutvik552k/web_search_mcp) archive-fallback";

/// One CDX capture row, narrowed to the fields this rung uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CdxSnapshot {
    /// 14-digit `YYYYMMDDhhmmss` UTC capture timestamp.
    pub timestamp: String,
    /// The original (pre-archive) URL of the captured resource.
    pub original: String,
    /// Archived HTTP status code (we only keep 200s).
    pub status: u16,
}

/// A successfully fetched archived page, ready to hand to the extractor.
#[derive(Debug, Clone)]
pub struct ArchivedPage {
    /// The Wayback `…id_/…` URL the body was fetched from.
    pub final_url: String,
    /// The raw archived HTML body.
    pub body: String,
    /// HTTP status of the snapshot fetch (typically 200).
    pub status: u16,
}

/// Parse a CDX `timestamp` (`YYYYMMDDhhmmss`, 14 digits, UTC) into a unix epoch
/// second count. Returns `None` if the string is malformed or not a valid date.
///
/// Pure + deterministic — takes no clock.
fn cdx_timestamp_to_unix(ts: &str) -> Option<u64> {
    // Must be exactly 14 ASCII digits.
    if ts.len() != 14 || !ts.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let year: i32 = ts[0..4].parse().ok()?;
    let month: u32 = ts[4..6].parse().ok()?;
    let day: u32 = ts[6..8].parse().ok()?;
    let hour: u32 = ts[8..10].parse().ok()?;
    let min: u32 = ts[10..12].parse().ok()?;
    let sec: u32 = ts[12..14].parse().ok()?;

    // `with_ymd_and_hms` validates ranges and rejects impossible dates.
    match Utc.with_ymd_and_hms(year, month, day, hour, min, sec) {
        chrono::LocalResult::Single(dt) => {
            let secs = dt.timestamp();
            if secs < 0 {
                None
            } else {
                Some(secs as u64)
            }
        }
        // Ambiguous/None never happens for fixed-offset UTC, but treat as invalid.
        _ => None,
    }
}

/// Parse the CDX JSON response and return the NEWEST qualifying snapshot.
///
/// Behavior:
/// - Expects a JSON **array of arrays**. The first inner array is the header row
///   of field names; it is skipped.
/// - Locates the `timestamp`, `original`, and `statuscode` columns BY NAME from
///   the header, so the parse is robust to field-order or field-set changes.
/// - Keeps only rows whose `statuscode == 200`.
/// - If `max_age_days` is `Some`, rejects rows older than that relative to
///   `now_unix` (a parameter, so this fn stays clock-free and deterministic).
/// - Returns the row with the LARGEST (most recent) timestamp, or `None` if no
///   row qualifies.
///
/// `now_unix` is the current time in unix epoch seconds, supplied by the caller.
/// Pure + deterministic.
pub fn parse_cdx_newest(json: &str, max_age_days: Option<u32>, now_unix: u64) -> Option<CdxSnapshot> {
    let value: Value = serde_json::from_str(json).ok()?;
    let rows = value.as_array()?;

    // First row is the header of field names; without it we cannot resolve columns.
    let header = rows.first()?.as_array()?;
    let mut idx_timestamp: Option<usize> = None;
    let mut idx_original: Option<usize> = None;
    let mut idx_status: Option<usize> = None;
    for (i, field) in header.iter().enumerate() {
        match field.as_str() {
            Some("timestamp") => idx_timestamp = Some(i),
            Some("original") => idx_original = Some(i),
            Some("statuscode") => idx_status = Some(i),
            _ => {}
        }
    }
    let idx_timestamp = idx_timestamp?;
    let idx_original = idx_original?;
    let idx_status = idx_status?;

    // Optional age cutoff: reject snapshots whose capture unix < cutoff.
    let cutoff_unix: Option<u64> = max_age_days.map(|days| {
        let window = (days as u64).saturating_mul(86_400);
        now_unix.saturating_sub(window)
    });

    let mut best: Option<(u64, CdxSnapshot)> = None;

    for row in rows.iter().skip(1) {
        let cols = match row.as_array() {
            Some(c) => c,
            None => continue,
        };

        // CDX JSON values are strings even for numeric fields.
        let timestamp = match cols.get(idx_timestamp).and_then(Value::as_str) {
            Some(s) => s,
            None => continue,
        };
        let original = match cols.get(idx_original).and_then(Value::as_str) {
            Some(s) => s,
            None => continue,
        };
        let status_str = match cols.get(idx_status).and_then(Value::as_str) {
            Some(s) => s,
            None => continue,
        };

        // Only archived 200s — a captured 404/redirect is useless to the extractor.
        let status: u16 = match status_str.parse() {
            Ok(s) => s,
            Err(_) => continue,
        };
        if status != 200 {
            continue;
        }

        let ts_unix = match cdx_timestamp_to_unix(timestamp) {
            Some(u) => u,
            None => continue,
        };

        if let Some(cutoff) = cutoff_unix {
            if ts_unix < cutoff {
                continue; // too old
            }
        }

        let is_newer = match &best {
            Some((best_unix, _)) => ts_unix > *best_unix,
            None => true,
        };
        if is_newer {
            best = Some((
                ts_unix,
                CdxSnapshot {
                    timestamp: timestamp.to_string(),
                    original: original.to_string(),
                    status,
                },
            ));
        }
    }

    best.map(|(_, snap)| snap)
}

/// Build the raw-snapshot Wayback URL: `<base>/web/<timestamp>id_/<original_url>`.
///
/// The `id_` modifier yields the ORIGINAL archived bytes with no toolbar and no
/// link rewriting (verified ground truth, see module docs).
///
/// `cdx_endpoint_or_base` may be either the CDX endpoint
/// (`http://web.archive.org/cdx/search/cdx`) or any URL on the same host; only
/// the scheme + host are used to derive `http(s)://web.archive.org`. If the
/// scheme/host cannot be derived, falls back to `http://web.archive.org`.
///
/// Pure + deterministic.
pub fn snapshot_raw_url(cdx_endpoint_or_base: &str, timestamp: &str, original_url: &str) -> String {
    let host_base = derive_host_base(cdx_endpoint_or_base);
    format!("{host_base}/web/{timestamp}id_/{original_url}")
}

/// Derive `scheme://host[:port]` from an arbitrary URL on the Wayback host.
/// Falls back to `http://web.archive.org` if parsing fails.
fn derive_host_base(any_url: &str) -> String {
    match url::Url::parse(any_url) {
        Ok(u) => match (u.scheme(), u.host_str()) {
            (scheme, Some(host)) => match u.port() {
                Some(port) => format!("{scheme}://{host}:{port}"),
                None => format!("{scheme}://{host}"),
            },
            _ => "http://web.archive.org".to_string(),
        },
        Err(_) => "http://web.archive.org".to_string(),
    }
}

/// Build the full CDX query URL for `url` requesting the single newest archived
/// 200 snapshot.
///
/// Pure + deterministic.
fn cdx_query_url(cdx_endpoint: &str, target_url: &str) -> String {
    // Percent-encode the target URL into the `url=` query value so reserved
    // characters (`?`, `&`, `#`, `=`) in the target don't corrupt the query.
    let encoded = encode_query_value(target_url);
    format!(
        "{cdx_endpoint}?url={encoded}&output=json&filter=statuscode:200&collapse=digest&limit=-1"
    )
}

/// Minimal percent-encoding for a query-string VALUE. Encodes everything that is
/// not an unreserved URL char (`ALPHA / DIGIT / -._~`). Conservative on purpose:
/// over-encoding is harmless to the CDX server, under-encoding corrupts the query.
fn encode_query_value(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push('%');
                out.push(hex_nibble(b >> 4));
                out.push(hex_nibble(b & 0x0F));
            }
        }
    }
    out
}

#[inline]
fn hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'A' + (n - 10)) as char,
    }
}

/// R3: fetch the newest archived snapshot of `url` from the Wayback Machine.
///
/// Pipeline:
/// 1. GET the CDX query (newest 200 capture) with a timeout.
/// 2. `parse_cdx_newest` → newest qualifying [`CdxSnapshot`] (honoring max age).
/// 3. Build the `id_` raw-snapshot URL.
/// 4. GET the snapshot body with a timeout.
/// 5. Return [`ArchivedPage`].
///
/// **Non-fatal by contract**: returns `None` on ANY miss, timeout, network
/// error, non-2xx, or empty body. It never returns an error and never panics —
/// the caller simply falls through to the next rung.
///
/// `client` is reused from the caller so connection pooling/config is shared.
pub async fn fetch_archived(
    client: &reqwest::Client,
    url: &str,
    cfg: &ArchiveConfig,
) -> Option<ArchivedPage> {
    let ua = cfg.user_agent.as_deref().unwrap_or(DEFAULT_UA);
    let req_timeout = Duration::from_millis(cfg.timeout_ms);

    // Clock read happens HERE (in the async, side-effecting fn), never inside the
    // pure parse fn — keeps `parse_cdx_newest` deterministically testable.
    let now_unix = Utc::now().timestamp().max(0) as u64;

    // --- Step 1: CDX query ---
    let query_url = cdx_query_url(&cfg.cdx_endpoint, url);
    let cdx_json = match timeout(req_timeout, async {
        let resp = client
            .get(&query_url)
            .header(reqwest::header::USER_AGENT, ua)
            .send()
            .await
            .ok()?;
        // IA returns 503/429 under load — treat any non-success as a miss.
        if !resp.status().is_success() {
            tracing::debug!(target: "archive", status = %resp.status(), "CDX query non-success");
            return None;
        }
        resp.text().await.ok()
    })
    .await
    {
        Ok(Some(body)) => body,
        Ok(None) => return None,
        Err(_) => {
            tracing::debug!(target: "archive", url, "CDX query timed out");
            return None;
        }
    };

    // --- Step 2: parse newest qualifying snapshot ---
    let snapshot = parse_cdx_newest(&cdx_json, cfg.max_snapshot_age_days, now_unix)?;
    tracing::debug!(
        target: "archive",
        timestamp = %snapshot.timestamp,
        original = %snapshot.original,
        "archive: selected newest snapshot"
    );

    // --- Step 3: build raw `id_` snapshot URL ---
    let raw_url = snapshot_raw_url(&cfg.cdx_endpoint, &snapshot.timestamp, &snapshot.original);

    // --- Step 4: fetch the snapshot body ---
    let page = timeout(req_timeout, async {
        let resp = client
            .get(&raw_url)
            .header(reqwest::header::USER_AGENT, ua)
            .send()
            .await
            .ok()?;
        let status = resp.status();
        if !status.is_success() {
            tracing::debug!(target: "archive", status = %status, "snapshot fetch non-success");
            return None;
        }
        let body = resp.text().await.ok()?;
        if body.is_empty() {
            return None;
        }
        Some(ArchivedPage {
            final_url: raw_url.clone(),
            body,
            status: status.as_u16(),
        })
    })
    .await;

    match page {
        Ok(p) => p,
        Err(_) => {
            tracing::debug!(target: "archive", url = %raw_url, "snapshot fetch timed out");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A realistic CDX `output=json` fixture: header row first, then capture rows.
    // Fields: urlkey, timestamp, original, mimetype, statuscode, digest, length.
    const SAMPLE_CDX: &str = r#"[
        ["urlkey","timestamp","original","mimetype","statuscode","digest","length"],
        ["com,example)/","20180101120000","http://example.com/","text/html","200","AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","1024"],
        ["com,example)/","20230615093000","http://example.com/","text/html","200","BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB","2048"],
        ["com,example)/","20220301080000","http://example.com/","text/html","200","CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC","1536"]
    ]"#;

    // now_unix anchored to 2024-01-01T00:00:00Z (1704067200) for deterministic
    // age-filter tests.
    const NOW_2024: u64 = 1_704_067_200;

    #[test]
    fn timestamp_parses_to_unix() {
        // 2023-06-15T09:30:00Z == 1686821400
        assert_eq!(cdx_timestamp_to_unix("20230615093000"), Some(1_686_821_400));
        // 1970 epoch
        assert_eq!(cdx_timestamp_to_unix("19700101000000"), Some(0));
    }

    #[test]
    fn timestamp_rejects_malformed() {
        assert_eq!(cdx_timestamp_to_unix(""), None);
        assert_eq!(cdx_timestamp_to_unix("2023"), None); // too short
        assert_eq!(cdx_timestamp_to_unix("2023061509300"), None); // 13 digits
        assert_eq!(cdx_timestamp_to_unix("202306150930000"), None); // 15 digits
        assert_eq!(cdx_timestamp_to_unix("2023XX15093000"), None); // non-digit
        assert_eq!(cdx_timestamp_to_unix("20231315093000"), None); // month 13
        assert_eq!(cdx_timestamp_to_unix("20230632093000"), None); // day 32
    }

    #[test]
    fn parse_cdx_picks_newest() {
        let snap = parse_cdx_newest(SAMPLE_CDX, None, NOW_2024).expect("should find a snapshot");
        // Newest of the three 200s is 2023-06-15.
        assert_eq!(snap.timestamp, "20230615093000");
        assert_eq!(snap.original, "http://example.com/");
        assert_eq!(snap.status, 200);
    }

    #[test]
    fn parse_cdx_resolves_columns_by_name_not_position() {
        // Reordered header: original BEFORE timestamp, status first.
        let json = r#"[
            ["statuscode","original","timestamp"],
            ["200","http://a.test/","20210101000000"],
            ["200","http://a.test/","20240101000000"]
        ]"#;
        let snap = parse_cdx_newest(json, None, NOW_2024 + 86_400 * 400).unwrap();
        assert_eq!(snap.timestamp, "20240101000000");
        assert_eq!(snap.original, "http://a.test/");
    }

    #[test]
    fn parse_cdx_skips_non_200() {
        let json = r#"[
            ["urlkey","timestamp","original","mimetype","statuscode","digest","length"],
            ["x)/","20240101000000","http://x.test/","text/html","404","D","10"],
            ["x)/","20230101000000","http://x.test/","text/html","200","E","20"]
        ]"#;
        let snap = parse_cdx_newest(json, None, NOW_2024 + 86_400 * 400).unwrap();
        // The newer row is a 404 and must be skipped; the 200 from 2023 wins.
        assert_eq!(snap.timestamp, "20230101000000");
        assert_eq!(snap.status, 200);
    }

    #[test]
    fn parse_cdx_age_filter_rejects_old() {
        // Relative to 2024-01-01, allow only snapshots <= 365 days old.
        // The newest 200 in SAMPLE_CDX is 2023-06-15 (~200 days before NOW_2024)
        // => within 365 days, accepted.
        let snap = parse_cdx_newest(SAMPLE_CDX, Some(365), NOW_2024).unwrap();
        assert_eq!(snap.timestamp, "20230615093000");

        // Now tighten to 30 days: 2023-06-15 is ~200 days old => rejected, and
        // every other row is older still => no snapshot at all.
        assert_eq!(parse_cdx_newest(SAMPLE_CDX, Some(30), NOW_2024), None);
    }

    #[test]
    fn parse_cdx_empty_or_header_only_is_none() {
        assert_eq!(parse_cdx_newest("[]", None, NOW_2024), None);
        let header_only = r#"[["urlkey","timestamp","original","mimetype","statuscode","digest","length"]]"#;
        assert_eq!(parse_cdx_newest(header_only, None, NOW_2024), None);
    }

    #[test]
    fn parse_cdx_invalid_json_is_none() {
        assert_eq!(parse_cdx_newest("not json", None, NOW_2024), None);
        assert_eq!(parse_cdx_newest("{\"a\":1}", None, NOW_2024), None); // object, not array
        assert_eq!(parse_cdx_newest("", None, NOW_2024), None);
    }

    #[test]
    fn parse_cdx_missing_required_column_is_none() {
        // Header lacks `original`.
        let json = r#"[
            ["urlkey","timestamp","statuscode"],
            ["x)/","20240101000000","200"]
        ]"#;
        assert_eq!(parse_cdx_newest(json, None, NOW_2024 + 86_400 * 400), None);
    }

    #[test]
    fn snapshot_raw_url_uses_id_modifier() {
        let u = snapshot_raw_url(
            "http://web.archive.org/cdx/search/cdx",
            "20230615093000",
            "http://example.com/",
        );
        assert_eq!(
            u,
            "http://web.archive.org/web/20230615093000id_/http://example.com/"
        );
    }

    #[test]
    fn snapshot_raw_url_preserves_https_host() {
        let u = snapshot_raw_url(
            "https://web.archive.org/cdx/search/cdx",
            "20200101000000",
            "https://foo.test/bar?q=1",
        );
        assert_eq!(
            u,
            "https://web.archive.org/web/20200101000000id_/https://foo.test/bar?q=1"
        );
    }

    #[test]
    fn snapshot_raw_url_falls_back_on_garbage_base() {
        let u = snapshot_raw_url("not a url", "20200101000000", "http://x.test/");
        assert_eq!(
            u,
            "http://web.archive.org/web/20200101000000id_/http://x.test/"
        );
    }

    #[test]
    fn cdx_query_url_is_well_formed_and_encodes_target() {
        let q = cdx_query_url(
            "http://web.archive.org/cdx/search/cdx",
            "http://example.com/a?b=c&d=e",
        );
        assert!(q.starts_with("http://web.archive.org/cdx/search/cdx?url="));
        assert!(q.contains("&output=json"));
        assert!(q.contains("&filter=statuscode:200"));
        assert!(q.contains("&collapse=digest"));
        assert!(q.contains("&limit=-1"));
        // The target's reserved chars must be percent-encoded, not raw, so they
        // don't leak into the CDX query string.
        assert!(q.contains("url=http%3A%2F%2Fexample.com%2Fa%3Fb%3Dc%26d%3De"));
        assert!(!q.contains("url=http://example.com/a?b=c&d=e"));
    }

    #[test]
    fn encode_query_value_leaves_unreserved_alone() {
        assert_eq!(encode_query_value("aZ09-._~"), "aZ09-._~");
        assert_eq!(encode_query_value("a b"), "a%20b");
        assert_eq!(encode_query_value("/"), "%2F");
    }

    #[test]
    fn default_config_targets_wayback() {
        let cfg = ArchiveConfig::default();
        assert_eq!(cfg.cdx_endpoint, "http://web.archive.org/cdx/search/cdx");
        assert_eq!(cfg.timeout_ms, 10_000);
        assert_eq!(cfg.max_snapshot_age_days, None);
        assert_eq!(cfg.user_agent, None);
    }
}
