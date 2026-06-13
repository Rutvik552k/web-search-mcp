//! Token injection helpers (Design 0004 §2.4 step 3). PURE and UNIT-TESTABLE —
//! these build strings; they do NOT touch a browser or the network.
//!
//! # Two paths, two realities
//!
//! **Browser path (preferred, R5-after-R4):** the solved token must be written
//! into the page's hidden response field (`g-recaptcha-response` /
//! `h-captcha-response` / `cf-turnstile-response`) and the page's callback /
//! form submit triggered. [`build_injection_js`] returns the JS string to run
//! via CDP `Runtime.evaluate` on the stealth `EvalContext`. **The actual CDP
//! execution happens at hybrid-wiring time, not here** — we only produce the
//! script so it is unit-testable and the wiring step is mechanical.
//!
//! **HTTP-only path:** there is no in-page JS to run; the token must be placed
//! into the form POST body under the same field name. [`http_form_field`]
//! returns the `(name, value)` pair to add to the form encoding.
//!
//! ## HTTP-only-path LIMITATION (documented, load-bearing — Design 0004 §2.4)
//!
//! reCAPTCHA v3 and Turnstile frequently require the token to be **consumed
//! in-page by JS within its ~2-minute TTL**, so a pure-HTTP form POST often
//! FAILS for those families — the server expects the token to have travelled
//! through the page's own callback. HTTP-only is reliable mainly for classic
//! reCAPTCHA **v2** form posts. Prefer the browser path whenever R4 has a live
//! page open. [`http_only_is_reliable`] encodes this so callers can decide.

use super::{CaptchaKind, CaptchaToken};

/// Build the JS to inject a solved token into the page and nudge the host form
/// to proceed (browser path). The returned string is intended for CDP
/// `Runtime.evaluate`.
///
/// What it does, defensively (each step independently try/caught so one failure
/// does not abort the rest):
///   1. Set every existing hidden `<textarea>/<input>` named for this family's
///      response field to the token value, dispatching `input`/`change` so
///      frameworks observe it.
///   2. If none exists, create a hidden `<textarea>` with that name so a plain
///      form POST still carries it.
///   3. For reCAPTCHA, attempt to invoke the global `___grecaptcha_cfg` /
///      page-defined callback if discoverable (best-effort; many sites wire
///      their own callback we cannot name from here).
///
/// SECURITY: the token is a single-use secret-equivalent — it is embedded in the
/// JS but this string is NEVER logged by this layer (the caller must not log it
/// either). The field name comes from a fixed enum mapping, not user input, so
/// there is no injection vector via the field name. The token value is
/// JSON-string-escaped so it cannot break out of the JS string literal.
pub fn build_injection_js(kind: CaptchaKind, token: &CaptchaToken) -> String {
    let field = kind.response_field();
    // Escape the token as a JSON string literal so arbitrary characters in the
    // provider token cannot terminate the JS string or inject code. serde_json
    // produces a quoted, fully-escaped literal (e.g. "\"abc\\n\"").
    let token_lit = serde_json::to_string(&token.token)
        .unwrap_or_else(|_| "\"\"".to_string());
    // The field name is from a static enum mapping; still emit it as a JSON
    // literal for uniformity.
    let field_lit = serde_json::to_string(field).unwrap_or_else(|_| "\"\"".to_string());

    format!(
        r#"(() => {{
  const FIELD = {field_lit};
  const TOKEN = {token_lit};
  // 1. Populate any existing response field(s).
  try {{
    const els = document.getElementsByName(FIELD);
    if (els && els.length) {{
      for (const el of els) {{
        el.value = TOKEN;
        try {{ el.dispatchEvent(new Event('input',  {{ bubbles: true }})); }} catch (e) {{}}
        try {{ el.dispatchEvent(new Event('change', {{ bubbles: true }})); }} catch (e) {{}}
      }}
    }} else {{
      // 2. None present — create a hidden textarea so a form POST carries it.
      const ta = document.createElement('textarea');
      ta.name = FIELD;
      ta.style.display = 'none';
      ta.value = TOKEN;
      (document.forms && document.forms.length ? document.forms[0] : document.body).appendChild(ta);
    }}
  }} catch (e) {{}}
  // 3. Best-effort: fire a page-defined reCAPTCHA callback if one is named on
  //    the .g-recaptcha element's data-callback attribute.
  try {{
    const holder = document.querySelector('.g-recaptcha, .h-captcha, .cf-turnstile');
    const cbName = holder && holder.getAttribute('data-callback');
    if (cbName && typeof window[cbName] === 'function') {{
      window[cbName](TOKEN);
    }}
  }} catch (e) {{}}
  return true;
}})();"#
    )
}

/// The `(field_name, token_value)` pair to place in a form POST body for the
/// HTTP-only path. The caller is responsible for URL-encoding it into the body.
pub fn http_form_field<'a>(kind: CaptchaKind, token: &'a CaptchaToken) -> (&'static str, &'a str) {
    (kind.response_field(), token.token.as_str())
}

/// Whether the HTTP-only (no-browser) form-POST path is *reliable* for this
/// family. `true` only for classic reCAPTCHA v2; v3/Turnstile/hCaptcha usually
/// need in-page JS consumption within the token TTL (Design 0004 §2.4
/// limitation), so callers should prefer the browser path for those.
pub fn http_only_is_reliable(kind: CaptchaKind) -> bool {
    matches!(kind, CaptchaKind::RecaptchaV2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: &str) -> CaptchaToken {
        CaptchaToken { token: s.to_string(), user_agent: None }
    }

    #[test]
    fn js_uses_correct_field_for_each_kind() {
        let js2 = build_injection_js(CaptchaKind::RecaptchaV2, &tok("T"));
        assert!(js2.contains("\"g-recaptcha-response\""));
        let jsh = build_injection_js(CaptchaKind::Hcaptcha, &tok("T"));
        assert!(jsh.contains("\"h-captcha-response\""));
        let jst = build_injection_js(CaptchaKind::Turnstile, &tok("T"));
        assert!(jst.contains("\"cf-turnstile-response\""));
    }

    #[test]
    fn js_embeds_escaped_token() {
        let js = build_injection_js(CaptchaKind::Turnstile, &tok("abc123"));
        assert!(js.contains("\"abc123\""));
    }

    #[test]
    fn js_escapes_quotes_and_breaks_to_prevent_breakout() {
        // A hostile/odd token with a quote, backslash and newline must be
        // JSON-escaped so it cannot terminate the JS string literal.
        let nasty = "\";alert(1);//\n\\end";
        let js = build_injection_js(CaptchaKind::RecaptchaV2, &tok(nasty));
        // The raw unescaped breakout sequence must NOT appear verbatim.
        assert!(!js.contains("\";alert(1);//\n\\end"));
        // The escaped form must appear (serde_json escapes " \ and \n).
        assert!(js.contains(r#"\";alert(1);//\n\\end"#));
        // Sanity: still a self-invoking function.
        assert!(js.trim_start().starts_with("(() =>"));
    }

    #[test]
    fn js_is_self_contained_iife() {
        let js = build_injection_js(CaptchaKind::Hcaptcha, &tok("T"));
        assert!(js.trim_start().starts_with("(() =>"));
        assert!(js.trim_end().ends_with("})();"));
        // Each DOM step is independently guarded.
        assert!(js.matches("try {").count() >= 3);
    }

    #[test]
    fn http_form_field_pairs_name_and_value() {
        let t = tok("the-token");
        let (name, value) = http_form_field(CaptchaKind::RecaptchaV2, &t);
        assert_eq!(name, "g-recaptcha-response");
        assert_eq!(value, "the-token");
    }

    #[test]
    fn http_only_reliable_only_for_v2() {
        assert!(http_only_is_reliable(CaptchaKind::RecaptchaV2));
        assert!(!http_only_is_reliable(CaptchaKind::RecaptchaV3));
        assert!(!http_only_is_reliable(CaptchaKind::Turnstile));
        assert!(!http_only_is_reliable(CaptchaKind::Hcaptcha));
    }
}
