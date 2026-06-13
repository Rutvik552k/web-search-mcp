//! Block-detection classifier (Design 0005 §1, ADR 0001 §4).
//!
//! A PURE, DETERMINISTIC, NO-I/O function over a raw HTTP response. It emits a
//! [`Verdict`] (a [`BlockClass`] plus the firing signal for observability) that
//! the escalation controller (`fetcher.rs`) uses to route to a rung.
//!
//! # Priority order (load-bearing — first match wins)
//!
//! `Captcha → JsChallenge → Cloudflare403 → RateLimited → SoftBlock → RealContent`
//!
//! A CAPTCHA page is *also* a 403 with a Cloudflare marker, so CAPTCHA MUST be
//! checked before the generic-403 branch or it would misroute to R1 instead of
//! R5. The classifier tests prove this ordering (`captcha_beats_cf_challenge`).
//!
//! # Markers come from config, not hard-coded literals
//!
//! [`ClassifierConfig`] carries the marker lists, seeded with the ADR 0001 §4
//! grounded defaults via [`ClassifierConfig::seeded`]. Vendor marker churn is a
//! config change, not a code change (ADR 0001 §4 confidence/tuning note). The
//! struct is owned by THIS module (not `CrawlerConfig`) so adding it does not
//! touch `config.rs` beyond the additive `soft_block_min_bytes` field.

use reqwest::header::HeaderMap;

use crate::captcha::CaptchaKind as SolverCaptchaKind;

/// Verdict over a raw HTTP response. Pure, deterministic, no I/O (ADR 0001 C1).
///
/// `RealContent` is the ONLY class the controller caches (Design 0005 H5 / ADR
/// 0001 C6). `is_spa` is carried by the `FetchResult`, not here — the controller
/// reads both to decide between an R4 *render* (SPA) and a challenge solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockClass {
    /// 2xx, no block marker, visible text >= threshold.
    RealContent,
    /// 403 + CF/DataDome fingerprint but NO JS-challenge or CAPTCHA body
    /// markers. IP/edge reputation block.
    Cloudflare403,
    /// Cloudflare IUAM / DataDome interstitial -> route to R4 (stealth headless).
    JsChallenge { vendor: ChallengeVendor },
    /// CAPTCHA present -> route to R5 (solver) when enabled + budget OK.
    Captcha { kind: CaptchaKind },
    /// 200 (or non-CF non-429 error) but empty/placeholder/block phrase.
    SoftBlock,
    /// 429 / vendor rate signal.
    RateLimited { retry_after_secs: u64 },
}

/// Challenge vendor. Akamai/PerimeterX remain stubs (ADR 0001 §4(2)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeVendor {
    Cloudflare,
    DataDome,
    /// Stub (ADR 0001 §4(2)) — markers not grounded; never produced today.
    Akamai,
    Unknown,
}

/// The classifier's narrower CAPTCHA set (ADR 0001 §4). This is a SEPARATE type
/// from the R5 solver's [`crate::captcha::CaptchaKind`] (which splits reCAPTCHA
/// into v2/v3). The controller maps classifier-kind → solver-kind at the R5
/// boundary via [`CaptchaKind::to_solver_kind`]; the two enums are deliberately
/// not unified (avoids a cross-module coupling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptchaKind {
    Recaptcha,
    Hcaptcha,
    Turnstile,
}

impl CaptchaKind {
    /// Map the classifier kind to the solver's kind. reCAPTCHA defaults to v2;
    /// the actual extract step ([`crate::captcha::extract::detect_captcha`]) may
    /// refine v2 vs v3 from the markup, so this is only the coarse default.
    pub fn to_solver_kind(self) -> SolverCaptchaKind {
        match self {
            CaptchaKind::Recaptcha => SolverCaptchaKind::RecaptchaV2,
            CaptchaKind::Hcaptcha => SolverCaptchaKind::Hcaptcha,
            CaptchaKind::Turnstile => SolverCaptchaKind::Turnstile,
        }
    }
}

/// Which signal fired — emitted in the tracing event (ADR 0001 ASR-5). Returned
/// alongside the class so the controller logs it without re-deriving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub class: BlockClass,
    pub signal: &'static str,
}

/// CAPTCHA marker strings (config-tunable). Seeded with ADR 0001 §4(1).
#[derive(Debug, Clone)]
pub struct CaptchaMarkers {
    pub turnstile: Vec<String>,
    pub hcaptcha: Vec<String>,
    pub recaptcha: Vec<String>,
}

/// JS-challenge marker strings (config-tunable). Seeded with ADR 0001 §4(2).
#[derive(Debug, Clone)]
pub struct ChallengeMarkers {
    /// Cloudflare body markers (the `cf-mitigated` header is checked separately
    /// as it is authoritative).
    pub cloudflare_body: Vec<String>,
    /// DataDome body markers.
    pub datadome_body: Vec<String>,
}

/// Classifier configuration. Owned by this module; built once from
/// `CrawlerConfig` fields plus the §4 default marker lists.
#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    /// Min visible-text chars for RealContent vs SoftBlock (default ~500).
    pub soft_block_min_bytes: usize,
    /// Default retry-after seconds when a 429 carries no `Retry-After` (30).
    pub default_retry_after_secs: u64,
    pub captcha_markers: CaptchaMarkers,
    pub challenge_markers: ChallengeMarkers,
    /// Lowercase placeholder/block phrases that mark a soft block.
    pub soft_block_phrases: Vec<String>,
    /// Lowercase `<title>` substrings that mark a soft block.
    pub soft_block_title_markers: Vec<String>,
}

impl ClassifierConfig {
    /// Build with the ADR 0001 §4 grounded default marker lists.
    /// `soft_block_min_bytes` comes from `CrawlerConfig`.
    pub fn seeded(soft_block_min_bytes: usize, default_retry_after_secs: u64) -> Self {
        Self {
            soft_block_min_bytes,
            default_retry_after_secs,
            captcha_markers: CaptchaMarkers {
                // ADR 0001 §4(1) Turnstile markers.
                turnstile: vec![
                    "challenges.cloudflare.com/turnstile/v0/api.js".into(),
                    "cf-turnstile".into(),
                ],
                // hCaptcha markers.
                hcaptcha: vec![
                    "js.hcaptcha.com/1/api.js".into(),
                    "hcaptcha.com/captcha".into(),
                    "h-captcha".into(),
                ],
                // reCAPTCHA markers.
                recaptcha: vec![
                    "www.google.com/recaptcha/api.js".into(),
                    "www.gstatic.com/recaptcha".into(),
                    "g-recaptcha".into(),
                ],
            },
            challenge_markers: ChallengeMarkers {
                // ADR 0001 §4(2) Cloudflare IUAM body markers.
                cloudflare_body: vec![
                    "_cf_chl_opt".into(),
                    "/cdn-cgi/challenge-platform/".into(),
                    "challenges.cloudflare.com".into(),
                    "just a moment".into(),
                    "checking your browser".into(),
                    "cf-browser-verification".into(),
                ],
                // DataDome body markers.
                datadome_body: vec![
                    "geo.captcha-delivery.com".into(),
                    "datadome".into(),
                ],
            },
            soft_block_phrases: vec![
                "please enable javascript".into(),
                "enable javascript".into(),
                "access denied".into(),
                "pardon our interruption".into(),
                "attention required".into(),
            ],
            soft_block_title_markers: vec![
                "access denied".into(),
                "attention required".into(),
                "just a moment".into(),
            ],
        }
    }
}

/// Pure, deterministic, no I/O. Priority order is enforced INSIDE this fn:
/// Captcha → JsChallenge → Cloudflare403 → RateLimited → SoftBlock → RealContent.
pub fn classify(
    status: u16,
    headers: &HeaderMap,
    final_url: &str,
    body: &str,
    cfg: &ClassifierConfig,
) -> Verdict {
    let body_lower = body.to_lowercase();
    let url_lower = final_url.to_lowercase();

    // (1) CAPTCHA — status NOT required (can appear inline on a 200 form).
    if let Some(v) = classify_captcha(&body_lower, cfg) {
        return v;
    }

    // (2) JsChallenge — interstitial, not a CAPTCHA.
    if let Some(v) = classify_js_challenge(status, headers, &body_lower, &url_lower, cfg) {
        return v;
    }

    // (3) Cloudflare403 — 403 + CF/DataDome fingerprint, NO body markers.
    if status == 403 && has_waf_fingerprint(headers) {
        return Verdict { class: BlockClass::Cloudflare403, signal: "cf-403-fingerprint" };
    }

    // (4) RateLimited — 429 (+ Retry-After).
    if status == 429 {
        let secs = parse_retry_after(headers, cfg.default_retry_after_secs);
        return Verdict {
            class: BlockClass::RateLimited { retry_after_secs: secs },
            signal: "http-429",
        };
    }

    // (5) SoftBlock — block phrase / title marker / thin body.
    if let Some(v) = classify_soft_block(status, body, &body_lower, cfg) {
        return v;
    }

    // (6) RealContent — fall-through.
    Verdict { class: BlockClass::RealContent, signal: "real-content" }
}

/// Step 1: CAPTCHA detection. Checks Turnstile, then hCaptcha, then reCAPTCHA.
fn classify_captcha(body_lower: &str, cfg: &ClassifierConfig) -> Option<Verdict> {
    if cfg.captcha_markers.turnstile.iter().any(|m| body_lower.contains(&m.to_lowercase())) {
        return Some(Verdict {
            class: BlockClass::Captcha { kind: CaptchaKind::Turnstile },
            signal: "cf-turnstile",
        });
    }
    if cfg.captcha_markers.hcaptcha.iter().any(|m| body_lower.contains(&m.to_lowercase())) {
        return Some(Verdict {
            class: BlockClass::Captcha { kind: CaptchaKind::Hcaptcha },
            signal: "h-captcha",
        });
    }
    if cfg.captcha_markers.recaptcha.iter().any(|m| body_lower.contains(&m.to_lowercase())) {
        return Some(Verdict {
            class: BlockClass::Captcha { kind: CaptchaKind::Recaptcha },
            signal: "g-recaptcha",
        });
    }
    None
}

/// Step 2: JS-challenge detection (Cloudflare / DataDome interstitial).
fn classify_js_challenge(
    _status: u16,
    headers: &HeaderMap,
    body_lower: &str,
    url_lower: &str,
    cfg: &ClassifierConfig,
) -> Option<Verdict> {
    // Authoritative Cloudflare header: `cf-mitigated: challenge` (only valid
    // value is "challenge"). ADR 0001 §4(2).
    if let Some(val) = header_str(headers, "cf-mitigated") {
        if val.to_ascii_lowercase().contains("challenge") {
            return Some(Verdict {
                class: BlockClass::JsChallenge { vendor: ChallengeVendor::Cloudflare },
                signal: "cf-mitigated",
            });
        }
    }

    // Cloudflare body / URL markers.
    if cfg
        .challenge_markers
        .cloudflare_body
        .iter()
        .any(|m| { let ml = m.to_lowercase(); body_lower.contains(&ml) || url_lower.contains(&ml) })
    {
        return Some(Verdict {
            class: BlockClass::JsChallenge { vendor: ChallengeVendor::Cloudflare },
            signal: "cf-body-marker",
        });
    }

    // DataDome header.
    if header_str(headers, "x-datadome").is_some()
        || header_str(headers, "x-datadome-cid").is_some()
    {
        return Some(Verdict {
            class: BlockClass::JsChallenge { vendor: ChallengeVendor::DataDome },
            signal: "x-datadome",
        });
    }

    // DataDome `Set-Cookie` token or body marker.
    if let Some(sc) = header_str(headers, "set-cookie") {
        if sc.to_ascii_lowercase().contains("datadome") {
            return Some(Verdict {
                class: BlockClass::JsChallenge { vendor: ChallengeVendor::DataDome },
                signal: "datadome-cookie",
            });
        }
    }
    if cfg
        .challenge_markers
        .datadome_body
        .iter()
        .any(|m| body_lower.contains(&m.to_lowercase()))
    {
        return Some(Verdict {
            class: BlockClass::JsChallenge { vendor: ChallengeVendor::DataDome },
            signal: "datadome-body",
        });
    }

    None
}

/// Step 5: SoftBlock detection.
fn classify_soft_block(
    status: u16,
    body: &str,
    body_lower: &str,
    cfg: &ClassifierConfig,
) -> Option<Verdict> {
    // A block phrase anywhere in the body wins regardless of size (disambiguates
    // from a legit SPA, which has no block phrase — ADR 0001 §4(5)).
    if cfg.soft_block_phrases.iter().any(|p| body_lower.contains(&p.to_lowercase())) {
        return Some(Verdict { class: BlockClass::SoftBlock, signal: "soft:phrase" });
    }

    // `<title>` block marker.
    if let Some(title) = extract_title_lower(body_lower) {
        if cfg.soft_block_title_markers.iter().any(|m| title.contains(&m.to_lowercase())) {
            return Some(Verdict { class: BlockClass::SoftBlock, signal: "soft:title" });
        }
    }

    // Non-CF non-429 error statuses (4xx/5xx) with no marker => SoftBlock so the
    // controller can still salvage (Design 0005 §2.4). A plain 200 below the
    // visible-text threshold is only a soft block when it ALSO has no real text;
    // otherwise we let it fall through to RealContent (the SPA path handles thin
    // 200s downstream via `is_spa`).
    let is_error = (400..600).contains(&status);
    if is_error {
        return Some(Verdict { class: BlockClass::SoftBlock, signal: "soft:error-status" });
    }

    // 200 but essentially empty (below threshold AND no scripts that would make
    // it a SPA shell). We do NOT classify a script-heavy thin body as SoftBlock
    // — that is the SPA case (RealContent + is_spa). Only a truly empty/tiny
    // body with no scripts is a soft block.
    if status == 200 {
        let text_len = visible_text_len(body);
        let has_scripts = body_lower.contains("<script");
        if text_len < cfg.soft_block_min_bytes && !has_scripts && body.len() < 5000 {
            return Some(Verdict { class: BlockClass::SoftBlock, signal: "soft:thin" });
        }
    }

    None
}

// -- helpers ------------------------------------------------------------------

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

fn has_waf_fingerprint(headers: &HeaderMap) -> bool {
    let server_is_cf = header_str(headers, "server")
        .map(|s| s.to_ascii_lowercase().contains("cloudflare"))
        .unwrap_or(false);
    server_is_cf
        || headers.get("cf-ray").is_some()
        || headers.get("x-datadome").is_some()
        || headers.get("x-datadome-cid").is_some()
}

/// Parse `Retry-After` (seconds or HTTP-date) → secs, with a default fallback.
fn parse_retry_after(headers: &HeaderMap, default_secs: u64) -> u64 {
    let Some(raw) = header_str(headers, "retry-after") else {
        return default_secs;
    };
    let raw = raw.trim();
    // Delta-seconds form.
    if let Ok(secs) = raw.parse::<u64>() {
        return secs;
    }
    // HTTP-date form (RFC 7231). Compute seconds until that instant; clamp to >=0.
    if let Ok(when) = httpdate_to_unix(raw) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let delta = when - now;
        if delta > 0 {
            return delta as u64;
        }
        return 0;
    }
    default_secs
}

/// Minimal RFC 1123 / RFC 850 / asctime HTTP-date → unix-seconds parser. We only
/// need it to honor `Retry-After` dates; on any parse failure the caller falls
/// back to the default. Avoids pulling in a date dependency for this one use.
fn httpdate_to_unix(s: &str) -> Result<i64, ()> {
    // Expect e.g. "Wed, 21 Oct 2026 07:28:00 GMT". Parse the numeric fields.
    // Strip the leading weekday + comma if present.
    let s = s.trim();
    let rest = match s.find(',') {
        Some(i) => s[i + 1..].trim(),
        None => s,
    };
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() < 5 {
        return Err(());
    }
    let day: i64 = parts[0].parse().map_err(|_| ())?;
    let month = match parts[1].to_ascii_lowercase().as_str() {
        "jan" => 1, "feb" => 2, "mar" => 3, "apr" => 4, "may" => 5, "jun" => 6,
        "jul" => 7, "aug" => 8, "sep" => 9, "oct" => 10, "nov" => 11, "dec" => 12,
        _ => return Err(()),
    };
    let year: i64 = parts[2].parse().map_err(|_| ())?;
    let time: Vec<&str> = parts[3].split(':').collect();
    if time.len() != 3 {
        return Err(());
    }
    let hh: i64 = time[0].parse().map_err(|_| ())?;
    let mm: i64 = time[1].parse().map_err(|_| ())?;
    let ss: i64 = time[2].parse().map_err(|_| ())?;
    // days since unix epoch (civil-from-days algorithm, Howard Hinnant).
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Ok(days * 86400 + hh * 3600 + mm * 60 + ss)
}

/// Extract a lowercase `<title>` content from an already-lowercased body.
fn extract_title_lower(body_lower: &str) -> Option<String> {
    let start = body_lower.find("<title")?;
    let after_open = body_lower[start..].find('>')? + start + 1;
    let end_rel = body_lower[after_open..].find("</title>")?;
    Some(body_lower[after_open..after_open + end_rel].trim().to_string())
}

/// Visible-text length estimate (strip tags). Mirrors the fetcher.rs heuristic
/// (kept here so the classifier module is self-contained and pure).
fn visible_text_len(html: &str) -> usize {
    let mut in_tag = false;
    let mut text_len = 0;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => {
                if !ch.is_whitespace() {
                    text_len += 1;
                }
            }
            _ => {}
        }
    }
    text_len
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    fn cfg() -> ClassifierConfig {
        ClassifierConfig::seeded(500, 30)
    }

    fn hdr(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                reqwest::header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    // -- CAPTCHA (each kind) --------------------------------------------------

    #[test]
    fn captcha_turnstile() {
        let body = r#"<div class="cf-turnstile" data-sitekey="0xABC"></div>"#;
        let v = classify(403, &hdr(&[]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::Captcha { kind: CaptchaKind::Turnstile });
        assert_eq!(v.signal, "cf-turnstile");
    }

    #[test]
    fn captcha_recaptcha() {
        let body = r#"<script src="https://www.google.com/recaptcha/api.js"></script>
                      <div class="g-recaptcha" data-sitekey="k"></div>"#;
        let v = classify(200, &hdr(&[]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::Captcha { kind: CaptchaKind::Recaptcha });
    }

    #[test]
    fn captcha_hcaptcha() {
        let body = r#"<div class="h-captcha" data-sitekey="k"></div>"#;
        let v = classify(200, &hdr(&[]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::Captcha { kind: CaptchaKind::Hcaptcha });
    }

    /// PRIORITY PROOF: a 403 with BOTH `cf-mitigated: challenge` AND a turnstile
    /// element must classify as Captcha, NOT JsChallenge (order is load-bearing).
    #[test]
    fn captcha_beats_cf_challenge() {
        let body = r#"<div class="cf-turnstile" data-sitekey="k"></div>
                      <script>window._cf_chl_opt={}</script>"#;
        let v = classify(403, &hdr(&[("cf-mitigated", "challenge")]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::Captcha { kind: CaptchaKind::Turnstile });
    }

    // -- JsChallenge ----------------------------------------------------------

    #[test]
    fn js_challenge_cf_mitigated_header_only() {
        let v = classify(403, &hdr(&[("cf-mitigated", "challenge")]), "https://x.com", "<html></html>", &cfg());
        assert_eq!(v.class, BlockClass::JsChallenge { vendor: ChallengeVendor::Cloudflare });
        assert_eq!(v.signal, "cf-mitigated");
    }

    #[test]
    fn js_challenge_cf_body_marker_403() {
        let body = r#"<html><head><title>Just a moment...</title></head>
                      <body><script>window._cf_chl_opt={}</script></body></html>"#;
        let v = classify(403, &hdr(&[("server", "cloudflare")]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::JsChallenge { vendor: ChallengeVendor::Cloudflare });
    }

    #[test]
    fn js_challenge_datadome_header() {
        let body = r#"<html><body>var dd={};</body></html>"#;
        let v = classify(403, &hdr(&[("x-datadome", "protected")]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::JsChallenge { vendor: ChallengeVendor::DataDome });
    }

    /// 503 IUAM body markers must classify as JsChallenge (proves NO hard 503
    /// dependency — we route on the markers, not the status).
    #[test]
    fn js_challenge_503_iuam() {
        let body = r#"<html><body><div>checking your browser</div>
                      <script src="/cdn-cgi/challenge-platform/h/b/orchestrate"></script></body></html>"#;
        let v = classify(503, &hdr(&[("server", "cloudflare")]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::JsChallenge { vendor: ChallengeVendor::Cloudflare });
    }

    // -- Cloudflare403 --------------------------------------------------------

    /// 403 + cf-ray, NO body markers → Cloudflare403 (must NOT misroute to
    /// JsChallenge or Captcha).
    #[test]
    fn cloudflare_403_no_markers() {
        let body = "<html><body><h1>403 Forbidden</h1></body></html>";
        let v = classify(403, &hdr(&[("cf-ray", "abc123-LAX")]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::Cloudflare403);
        assert_eq!(v.signal, "cf-403-fingerprint");
    }

    // -- RateLimited ----------------------------------------------------------

    #[test]
    fn rate_limited_with_retry_after() {
        let v = classify(429, &hdr(&[("retry-after", "120")]), "https://x.com", "slow down", &cfg());
        assert_eq!(v.class, BlockClass::RateLimited { retry_after_secs: 120 });
    }

    #[test]
    fn rate_limited_default_when_absent() {
        let v = classify(429, &hdr(&[]), "https://x.com", "slow down", &cfg());
        assert_eq!(v.class, BlockClass::RateLimited { retry_after_secs: 30 });
    }

    #[test]
    fn rate_limited_http_date() {
        // A far-future fixed date must parse to a positive number of seconds.
        let v = classify(
            429,
            &hdr(&[("retry-after", "Wed, 21 Oct 2099 07:28:00 GMT")]),
            "https://x.com",
            "slow down",
            &cfg(),
        );
        match v.class {
            BlockClass::RateLimited { retry_after_secs } => assert!(retry_after_secs > 0),
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    // -- SoftBlock ------------------------------------------------------------

    /// "Access Denied" matches BOTH a body phrase and a title marker; the phrase
    /// check runs first by design, so the signal is `soft:phrase`.
    #[test]
    fn soft_block_access_denied_is_phrase() {
        let body = "<html><head><title>Access Denied</title></head><body>nope</body></html>";
        let v = classify(200, &hdr(&[]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::SoftBlock);
        assert_eq!(v.signal, "soft:phrase");
    }

    /// A title-only block marker ("just a moment") with no body phrase fires the
    /// title branch. (A real CF challenge would already be JsChallenge by its
    /// markers; this isolates the title-marker path with a plain body.)
    #[test]
    fn soft_block_title_marker_only() {
        let body = "<html><head><title>Just a moment</title></head><body>x</body></html>";
        // Strip CF body markers so we exercise SoftBlock title path, not JsChallenge.
        let mut c = cfg();
        c.challenge_markers.cloudflare_body.clear();
        let v = classify(200, &hdr(&[]), "https://x.com", body, &c);
        assert_eq!(v.class, BlockClass::SoftBlock);
        assert_eq!(v.signal, "soft:title");
    }

    #[test]
    fn soft_block_phrase_in_tiny_body() {
        let body = "<html><body>Please enable JavaScript to continue.</body></html>";
        let v = classify(200, &hdr(&[]), "https://x.com", body, &cfg());
        assert_eq!(v.class, BlockClass::SoftBlock);
        assert_eq!(v.signal, "soft:phrase");
    }

    // -- RealContent ----------------------------------------------------------

    #[test]
    fn real_content_full_article() {
        let body = format!(
            "<html><body><article><h1>Title</h1><p>{}</p></article></body></html>",
            "Real article body content. ".repeat(60)
        );
        let v = classify(200, &hdr(&[]), "https://x.com", &body, &cfg());
        assert_eq!(v.class, BlockClass::RealContent);
    }

    /// A SPA shell (empty root + bundle) must classify as RealContent (NOT
    /// SoftBlock) — the SPA/SoftBlock disambiguation. `is_spa` is handled
    /// downstream by the controller.
    #[test]
    fn real_content_spa_shell_not_soft_block() {
        let body = format!(
            r#"<html><body><div id="root"></div><script src="bundle.js"></script>{}</body></html>"#,
            " ".repeat(6000)
        );
        let v = classify(200, &hdr(&[]), "https://x.com", &body, &cfg());
        assert_eq!(v.class, BlockClass::RealContent);
    }

    // -- Determinism ----------------------------------------------------------

    #[test]
    fn determinism_same_input_same_verdict() {
        let body = r#"<div class="cf-turnstile" data-sitekey="k"></div>"#;
        let h = hdr(&[("cf-mitigated", "challenge")]);
        let first = classify(403, &h, "https://x.com", body, &cfg());
        for _ in 0..50 {
            assert_eq!(classify(403, &h, "https://x.com", body, &cfg()), first);
        }
    }

    // -- Config-driven markers ------------------------------------------------

    #[test]
    fn config_override_marker_fires() {
        let mut c = cfg();
        c.challenge_markers.cloudflare_body.push("my-custom-wall-marker".into());
        let body = "<html><body>my-custom-wall-marker present</body></html>";
        let v = classify(200, &hdr(&[]), "https://x.com", body, &c);
        assert_eq!(v.class, BlockClass::JsChallenge { vendor: ChallengeVendor::Cloudflare });
    }

    #[test]
    fn config_removed_marker_does_not_fire() {
        let mut c = cfg();
        c.captcha_markers.turnstile.clear(); // remove turnstile markers
        let body = r#"<div class="cf-turnstile" data-sitekey="k"></div>"#;
        let v = classify(200, &hdr(&[]), "https://x.com", body, &c);
        // With turnstile markers removed and no other marker, this is RealContent
        // (the body is short but script-free and tiny → could be soft:thin).
        assert_ne!(v.class, BlockClass::Captcha { kind: CaptchaKind::Turnstile });
    }
}
