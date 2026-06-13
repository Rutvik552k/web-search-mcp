//! CAPTCHA sitekey / action / cData extraction from page HTML (Design 0004
//! §2.4 step 1). PURE and UNIT-TESTABLE — no network.
//!
//! Markers (verified ground truth, 2026-06-11 — Design 0004 §6):
//!   - reCAPTCHA: `data-sitekey` on `.g-recaptcha`, or `render=<sitekey>` in an
//!     `api.js?render=` script src (v3 invisible). reCAPTCHA token field is
//!     `g-recaptcha-response`.
//!   - hCaptcha:  `data-sitekey` on `.h-captcha`.
//!   - Turnstile: `data-sitekey` on `.cf-turnstile` (+ optional `data-action`
//!     and `data-cdata`).
//!
//! We use `scraper` for attribute reads (robust against attribute ordering and
//! quoting) and a single `regex` only for the `api.js?render=` src form, which
//! is not an element attribute. We deliberately do NOT try to distinguish
//! reCAPTCHA v2 vs v3 from markup alone — the `.g-recaptcha` element is used by
//! both. We classify `api.js?render=<key>` (the invisible/v3 bootstrap) as
//! `RecaptchaV3` and a bare `.g-recaptcha[data-sitekey]` as `RecaptchaV2`; the
//! caller may override `kind` if it has better signal (e.g. the classifier).

use std::sync::LazyLock;

use regex::Regex;
use scraper::{Html, Selector};

use super::CaptchaKind;

/// The detection result: kind, site key, optional action (reCAPTCHA v3 /
/// Turnstile), optional cData (Turnstile).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detected {
    pub kind: CaptchaKind,
    pub site_key: String,
    pub action: Option<String>,
    pub cdata: Option<String>,
}

// `render=` query param on a google recaptcha api.js src → v3/invisible sitekey.
// e.g. <script src="https://www.google.com/recaptcha/api.js?render=6Lc_aXkUAA...">
static RECAPTCHA_RENDER_SRC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"recaptcha/api\.js\?[^"'>\s]*\brender=([A-Za-z0-9_\-]+)"#)
        .expect("static recaptcha render regex is valid")
});

fn sel(s: &str) -> Selector {
    // All selectors here are static, valid CSS — unwrap is safe.
    Selector::parse(s).expect("static selector is valid")
}

/// Detect the first CAPTCHA on the page, if any. Returns `None` when no marker
/// is present. Priority order is deterministic: Turnstile, then hCaptcha, then
/// reCAPTCHA (most specific markers first; reCAPTCHA is the most generic).
pub fn detect_captcha(html: &str) -> Option<Detected> {
    let doc = Html::parse_document(html);

    // -- Turnstile: .cf-turnstile[data-sitekey] (+ data-action / data-cdata) --
    let turnstile_sel = sel(".cf-turnstile");
    if let Some(el) = doc.select(&turnstile_sel).next() {
        if let Some(key) = el.value().attr("data-sitekey").map(str::trim).filter(|s| !s.is_empty()) {
            return Some(Detected {
                kind: CaptchaKind::Turnstile,
                site_key: key.to_string(),
                action: el
                    .value()
                    .attr("data-action")
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
                cdata: el
                    .value()
                    .attr("data-cdata")
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            });
        }
    }

    // -- hCaptcha: .h-captcha[data-sitekey] -----------------------------------
    let hcaptcha_sel = sel(".h-captcha");
    if let Some(el) = doc.select(&hcaptcha_sel).next() {
        if let Some(key) = el.value().attr("data-sitekey").map(str::trim).filter(|s| !s.is_empty()) {
            return Some(Detected {
                kind: CaptchaKind::Hcaptcha,
                site_key: key.to_string(),
                action: None,
                cdata: None,
            });
        }
    }

    // -- reCAPTCHA v2: .g-recaptcha[data-sitekey] -----------------------------
    let grecaptcha_sel = sel(".g-recaptcha");
    if let Some(el) = doc.select(&grecaptcha_sel).next() {
        if let Some(key) = el.value().attr("data-sitekey").map(str::trim).filter(|s| !s.is_empty()) {
            return Some(Detected {
                kind: CaptchaKind::RecaptchaV2,
                site_key: key.to_string(),
                // .g-recaptcha may carry data-action for v3-on-element usage.
                action: el
                    .value()
                    .attr("data-action")
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
                cdata: None,
            });
        }
    }

    // -- reCAPTCHA v3 (invisible) via api.js?render=<sitekey> ------------------
    // Only the regex form is left for v3-without-element. Run it over the raw
    // HTML (the src is a script attribute, easiest matched as text).
    if let Some(caps) = RECAPTCHA_RENDER_SRC.captures(html) {
        let key = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if !key.is_empty() {
            return Some(Detected {
                kind: CaptchaKind::RecaptchaV3,
                site_key: key.to_string(),
                action: None,
                cdata: None,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_turnstile_with_action_and_cdata() {
        let html = r#"
            <html><body>
              <form>
                <div class="cf-turnstile"
                     data-sitekey="0x4AAAAAAADnPIDROzd"
                     data-action="login"
                     data-cdata="sessabc"></div>
              </form>
            </body></html>"#;
        let got = detect_captcha(html).expect("turnstile detected");
        assert_eq!(got.kind, CaptchaKind::Turnstile);
        assert_eq!(got.site_key, "0x4AAAAAAADnPIDROzd");
        assert_eq!(got.action.as_deref(), Some("login"));
        assert_eq!(got.cdata.as_deref(), Some("sessabc"));
    }

    #[test]
    fn detects_turnstile_minimal() {
        let html = r#"<div class="cf-turnstile" data-sitekey="0xABC"></div>"#;
        let got = detect_captcha(html).unwrap();
        assert_eq!(got.kind, CaptchaKind::Turnstile);
        assert_eq!(got.site_key, "0xABC");
        assert_eq!(got.action, None);
        assert_eq!(got.cdata, None);
    }

    #[test]
    fn detects_hcaptcha() {
        let html = r#"
            <div class="h-captcha" data-sitekey="10000000-ffff-ffff-ffff-000000000001"></div>"#;
        let got = detect_captcha(html).unwrap();
        assert_eq!(got.kind, CaptchaKind::Hcaptcha);
        assert_eq!(got.site_key, "10000000-ffff-ffff-ffff-000000000001");
    }

    #[test]
    fn detects_recaptcha_v2_data_sitekey() {
        let html = r#"
            <div class="g-recaptcha" data-sitekey="6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe"></div>
            <script src="https://www.google.com/recaptcha/api.js"></script>"#;
        let got = detect_captcha(html).unwrap();
        assert_eq!(got.kind, CaptchaKind::RecaptchaV2);
        assert_eq!(got.site_key, "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe");
    }

    #[test]
    fn detects_recaptcha_v3_render_src() {
        let html = r#"
            <html><head>
              <script src="https://www.google.com/recaptcha/api.js?render=6Lc_v3KEYabc-DEF_123"></script>
            </head><body>no element, invisible v3</body></html>"#;
        let got = detect_captcha(html).unwrap();
        assert_eq!(got.kind, CaptchaKind::RecaptchaV3);
        assert_eq!(got.site_key, "6Lc_v3KEYabc-DEF_123");
    }

    #[test]
    fn turnstile_takes_priority_over_recaptcha_when_both_present() {
        // Deterministic priority: Turnstile first.
        let html = r#"
            <div class="g-recaptcha" data-sitekey="recap-key"></div>
            <div class="cf-turnstile" data-sitekey="turn-key"></div>"#;
        let got = detect_captcha(html).unwrap();
        assert_eq!(got.kind, CaptchaKind::Turnstile);
        assert_eq!(got.site_key, "turn-key");
    }

    #[test]
    fn no_captcha_returns_none() {
        let html = "<html><body><h1>Just an article</h1><p>content</p></body></html>";
        assert!(detect_captcha(html).is_none());
    }

    #[test]
    fn empty_sitekey_is_ignored() {
        let html = r#"<div class="g-recaptcha" data-sitekey="  "></div>"#;
        assert!(detect_captcha(html).is_none());
    }

    #[test]
    fn missing_sitekey_attr_is_ignored() {
        let html = r#"<div class="g-recaptcha"></div>"#;
        assert!(detect_captcha(html).is_none());
    }

    #[test]
    fn render_src_with_extra_query_params() {
        let html = r#"<script src="/recaptcha/api.js?onload=cb&render=KEY-with_dash"></script>"#;
        let got = detect_captcha(html).unwrap();
        assert_eq!(got.kind, CaptchaKind::RecaptchaV3);
        assert_eq!(got.site_key, "KEY-with_dash");
    }
}
