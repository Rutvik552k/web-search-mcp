//! R5 — commercial CAPTCHA solver abstraction (Design 0004 Part 2, ADR 0003 R5).
//!
//! # LEGAL / COMPLIANCE — READ BEFORE ENABLING
//!
//! Automated CAPTCHA solving is **ToS-sensitive**. This entire module is
//! **DEFAULT-OFF** and is the GOAL.md §2 / ADR 0001 Phase-5 *legal-gated* tier:
//! it may only be enabled in a shipped config **after** the Phase-5 compliance
//! review authorizes it, and only against owned/permitted targets. A
//! permanently-denylisted domain (ADR 0003 §4.2) must never reach R5. Nothing
//! here is wired into the live fetch path — building the off-by-default module
//! (trait + clients + extract/inject/cost flow) is the goal; the hybrid wiring
//! into `fetcher.rs` is a separate, later, gated step.
//!
//! # What this layer is (and is not)
//!
//! It is a thin **provider abstraction** over commercial solver HTTP APIs
//! (CapSolver, 2Captcha). It is NOT a self-hosted solver — OSS solvers are dead
//! and audio-CAPTCHA SOTA fails 93%+ (Design 0004 §2.0). No mature Rust crate
//! exists (the `capsolver` crate is empty/unreliable), so the clients call the
//! provider HTTP APIs directly with the existing `reqwest` dependency.
//!
//! # Ground truth (verified primary-source, 2026-06-11 — see Design 0004 §6)
//!
//! - CapSolver + 2Captcha share a **two-call async shape**: POST `createTask`
//!   (`clientKey` + `task{type, websiteURL, websiteKey, [action], [cdata]}`) →
//!   `taskId`; then POLL `getTaskResult` (`clientKey` + `taskId`) until
//!   `status == "ready"` (statuses idle/processing/ready), then read the token.
//!   Sources: docs.capsolver.com/en/api/,
//!   docs.capsolver.com/en/guide/captcha/cloudflare_turnstile/,
//!   2captcha.com/api-docs/recaptcha-v2.
//! - Token field: reCAPTCHA → `gRecaptchaResponse`; Turnstile/hCaptcha → `token`.
//! - The response also returns a **`userAgent` the caller MUST match** on the
//!   follow-up request — we surface it on [`CaptchaToken`].
//! - Tokens are **single-use, ~2-minute TTL** → NO caching/reuse; solve on
//!   demand. reCAPTCHA `siteverify` `remoteip` is OPTIONAL → tokens are largely
//!   IP-portable (good for single-IP). Source:
//!   developers.google.com/recaptcha/docs/verify.
//! - reCAPTCHA v3 is **score-based (0.0–1.0)**; proxyless solvers struggle →
//!   treat v3 as LOW-confidence (Design 0004 §2.7).
//!
//! # Hard gating / security (api-security + llm-safety + cost-tracking rules)
//!
//! - **Default OFF.** [`build_solver`] returns `Ok(None)` (disabled) unless
//!   `enable_captcha_solver` AND a known provider AND a SET env var are all
//!   present; otherwise it fails fast with a clear, secret-free error — never a
//!   panic.
//! - **API key by env-var NAME only** (`captcha_api_key_env`), read via
//!   `std::env::var`. Never a config literal; never logged (we log the provider
//!   name only).
//! - **Cost**: every solve goes through a [`CostMeter`]; on reaching the session
//!   cap the meter hard-halts (cost-tracking BUDGET GUARD).

use async_trait::async_trait;

pub mod capsolver;
pub mod cost;
pub mod extract;
pub mod inject;
pub mod twocaptcha;

pub use capsolver::CapSolverClient;
pub use cost::{CostMeter, CostMeterHandle};
pub use twocaptcha::TwoCaptchaClient;

/// The CAPTCHA families this layer can request a token for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptchaKind {
    /// reCAPTCHA v2 (checkbox / invisible). Token field `gRecaptchaResponse`.
    RecaptchaV2,
    /// reCAPTCHA v3. SCORE-based (0.0–1.0); proxyless solves are low-confidence
    /// (Design 0004 §2.7). Token field `gRecaptchaResponse`.
    RecaptchaV3,
    /// hCaptcha. Token field `token`.
    Hcaptcha,
    /// Cloudflare Turnstile. Token field `token`.
    Turnstile,
}

impl CaptchaKind {
    /// The hidden-field name the solved token must be injected into on the page
    /// (browser path) or posted as (HTTP-only path). Used by [`inject`].
    pub fn response_field(&self) -> &'static str {
        match self {
            CaptchaKind::RecaptchaV2 | CaptchaKind::RecaptchaV3 => "g-recaptcha-response",
            CaptchaKind::Hcaptcha => "h-captcha-response",
            CaptchaKind::Turnstile => "cf-turnstile-response",
        }
    }
}

/// A request to solve one CAPTCHA instance found on a page.
#[derive(Clone, Debug)]
pub struct CaptchaRequest {
    pub kind: CaptchaKind,
    /// The site's public key (`data-sitekey`).
    pub site_key: String,
    /// The page the CAPTCHA is on (provider `websiteURL`).
    pub page_url: String,
    /// reCAPTCHA v3 `action` / page-action, when present.
    pub action: Option<String>,
    /// Turnstile `cData`, when present.
    pub cdata: Option<String>,
}

/// A solved CAPTCHA token. **Single-use, ~2-minute TTL — never cache it.**
#[derive(Clone, Debug)]
pub struct CaptchaToken {
    /// The solved token value (`gRecaptchaResponse` or `token`). NEVER logged.
    pub token: String,
    /// The User-Agent the provider solved under. If present, the caller MUST
    /// replay it verbatim on the follow-up request (Design 0004 §2.3) or the
    /// token may be rejected.
    pub user_agent: Option<String>,
}

/// Typed solver error. Distinguishes **retriable** vs **non-retriable** so the
/// caller (and the bounded poll loop) can decide whether to back off and retry
/// or give up (error-handling TYPED EXCEPTIONS rule). Carries NO secrets.
#[derive(thiserror::Error, Debug)]
pub enum CaptchaError {
    /// Provider returned an application-level error (bad key, unsupported task,
    /// zero balance, …). Non-retriable. `code`/`message` come from the provider
    /// response — never the API key.
    #[error("provider error {code}: {message}")]
    Provider { code: String, message: String },

    /// Bounded total wait elapsed before the task became `ready`. Non-retriable
    /// at this layer (the budget is spent).
    #[error("timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// The session USD cost cap would be exceeded by this solve. Non-retriable.
    #[error("cost cap would be exceeded")]
    CostCapExceeded,

    /// Solver is disabled by config (default state). Non-retriable.
    #[error("captcha solver disabled by config")]
    Disabled,

    /// Network/transport failure talking to the provider. **Retriable** (bounded
    /// by the poll budget). The string is a sanitized description, not the key.
    #[error("transport: {0}")]
    Transport(String),

    /// The provider returned a token whose family we did not request, or a
    /// missing/empty token. Non-retriable (a protocol mismatch).
    #[error("unexpected provider response: {0}")]
    BadResponse(String),
}

impl CaptchaError {
    /// Whether a bounded retry is sensible. Only transport failures are
    /// retriable; everything else is terminal for this solve.
    pub fn is_retriable(&self) -> bool {
        matches!(self, CaptchaError::Transport(_))
    }

    /// Map to the workspace error at the public boundary, attaching the provider
    /// name. The `reason` is already secret-free by construction.
    pub fn into_common(self, provider: &str) -> web_search_common::Error {
        web_search_common::Error::Captcha {
            provider: provider.to_string(),
            reason: self.to_string(),
        }
    }
}

/// The provider-agnostic solver contract. Implemented by [`CapSolverClient`] and
/// [`TwoCaptchaClient`]. Both implement the same two-call async shape
/// (createTask → poll getTaskResult → token).
#[async_trait]
pub trait CaptchaSolver: Send + Sync {
    /// Solve one CAPTCHA. On success returns a single-use token (and the UA to
    /// replay, if the provider supplied one).
    async fn solve(&self, req: &CaptchaRequest) -> Result<CaptchaToken, CaptchaError>;

    /// Stable provider identifier for logging/metering (never the key).
    fn provider(&self) -> &'static str;
}

/// Parameters needed to construct a solver from config — kept as a small,
/// borrow-only struct so the factory does not depend on the whole
/// `CrawlerConfig` shape (and so it is trivially unit-testable).
#[derive(Clone, Copy, Debug)]
pub struct SolverParams<'a> {
    pub enable_captcha_solver: bool,
    pub provider: Option<&'a str>,
    pub api_key_env: Option<&'a str>,
    pub timeout_secs: u64,
    pub session_cost_cap_usd: f64,
}

/// Outcome of the gating decision, separated from construction so the gate logic
/// is pure and unit-testable without reading the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    /// Solver is off by config (the default). Construction returns `Ok(None)`.
    Disabled,
    /// All switches are on and a known provider is named — proceed to read the
    /// env var and construct.
    Enabled(Provider),
}

/// Known providers. Unknown strings are rejected at the gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    CapSolver,
    TwoCaptcha,
}

impl Provider {
    fn parse(s: &str) -> Option<Provider> {
        match s.trim().to_ascii_lowercase().as_str() {
            "capsolver" => Some(Provider::CapSolver),
            "2captcha" | "twocaptcha" => Some(Provider::TwoCaptcha),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Provider::CapSolver => "capsolver",
            Provider::TwoCaptcha => "2captcha",
        }
    }
}

/// Pure gating decision (Design 0004 §2.5). Default OFF: returns
/// [`Gate::Disabled`] unless the master switch is on AND a *known* provider is
/// named. An enabled-but-misconfigured state (unknown/missing provider, or a
/// non-positive cost cap) is a hard `Err`, not a silent disable, so the operator
/// learns about the misconfiguration at startup (config-management VALIDATE AT
/// STARTUP). This function does NOT read the environment — the env-var lookup
/// happens in [`build_solver`] so the gate stays pure.
pub fn decide_gate(p: &SolverParams) -> Result<Gate, web_search_common::Error> {
    if !p.enable_captcha_solver {
        return Ok(Gate::Disabled);
    }
    // Enabled — everything below must be valid or we fail fast.
    let provider_str = p.provider.map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| {
        web_search_common::Error::Config(
            "enable_captcha_solver=true but captcha_provider is unset \
             (expected \"capsolver\" or \"2captcha\")"
                .to_string(),
        )
    })?;
    let provider = Provider::parse(provider_str).ok_or_else(|| {
        web_search_common::Error::Config(format!(
            "unknown captcha_provider {provider_str:?} (expected \"capsolver\" or \"2captcha\")"
        ))
    })?;
    if !(p.session_cost_cap_usd > 0.0) {
        return Err(web_search_common::Error::Config(format!(
            "captcha_session_cost_cap_usd must be > 0 when the solver is enabled (got {})",
            p.session_cost_cap_usd
        )));
    }
    if p.api_key_env.map(str::trim).filter(|s| !s.is_empty()).is_none() {
        return Err(web_search_common::Error::Config(
            "enable_captcha_solver=true but captcha_api_key_env (the NAME of the env var \
             holding the API key) is unset"
                .to_string(),
        ));
    }
    Ok(Gate::Enabled(provider))
}

/// Construct a solver from config. Returns:
/// - `Ok(None)` when the solver is OFF (the default) — callers treat this as
///   "R5 unavailable" and give up gracefully.
/// - `Ok(Some(_))` when fully configured and the named env var is SET.
/// - `Err(_)` when enabled-but-misconfigured (unknown provider, non-positive
///   cap, env var unset/empty) — fail fast, never panic.
///
/// The API key is read from `std::env::var(api_key_env)` ONLY and is moved into
/// the client; it is never logged (we log the provider name only).
pub fn build_solver(
    p: &SolverParams,
    meter: CostMeterHandle,
) -> Result<Option<Box<dyn CaptchaSolver>>, web_search_common::Error> {
    let provider = match decide_gate(p)? {
        Gate::Disabled => return Ok(None),
        Gate::Enabled(prov) => prov,
    };

    // SAFETY/SECURITY: read the key by env-var NAME only; never a literal. If the
    // named var is unset or empty we fail fast (Design 0004 §2.5 startup
    // validation) rather than constructing a solver that would 401 on first use.
    let env_name = p.api_key_env.unwrap_or_default().trim();
    let api_key = match std::env::var(env_name) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            return Err(web_search_common::Error::Config(format!(
                "enable_captcha_solver=true but env var {env_name:?} is unset or empty \
                 (it must hold the {} API key)",
                provider.name()
            )));
        }
    };

    let timeout = std::time::Duration::from_secs(p.timeout_secs.max(1));
    // Log the provider only — NEVER the key (api-security NO SECRETS IN LOGS).
    tracing::info!(
        provider = provider.name(),
        timeout_secs = p.timeout_secs,
        cost_cap_usd = p.session_cost_cap_usd,
        "captcha solver enabled (R5) — legal/compliance gate is the operator's responsibility"
    );

    let solver: Box<dyn CaptchaSolver> = match provider {
        Provider::CapSolver => Box::new(CapSolverClient::new(api_key, timeout, meter)),
        Provider::TwoCaptcha => Box::new(TwoCaptchaClient::new(api_key, timeout, meter)),
    };
    Ok(Some(solver))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meter() -> CostMeterHandle {
        CostMeter::new(5.0).handle()
    }

    #[test]
    fn default_off_returns_disabled_gate() {
        let p = SolverParams {
            enable_captcha_solver: false,
            provider: Some("capsolver"),
            api_key_env: Some("SOME_KEY"),
            timeout_secs: 120,
            session_cost_cap_usd: 5.0,
        };
        assert_eq!(decide_gate(&p).unwrap(), Gate::Disabled);
    }

    #[test]
    fn enabled_with_known_provider_is_enabled() {
        let p = SolverParams {
            enable_captcha_solver: true,
            provider: Some("CapSolver"), // case-insensitive
            api_key_env: Some("KEY_ENV"),
            timeout_secs: 120,
            session_cost_cap_usd: 5.0,
        };
        assert_eq!(decide_gate(&p).unwrap(), Gate::Enabled(Provider::CapSolver));

        let p2 = SolverParams { provider: Some("2captcha"), ..p };
        assert_eq!(decide_gate(&p2).unwrap(), Gate::Enabled(Provider::TwoCaptcha));
    }

    #[test]
    fn enabled_with_unknown_provider_fails_fast() {
        let p = SolverParams {
            enable_captcha_solver: true,
            provider: Some("deathbycaptcha"),
            api_key_env: Some("KEY_ENV"),
            timeout_secs: 120,
            session_cost_cap_usd: 5.0,
        };
        assert!(decide_gate(&p).is_err());
    }

    #[test]
    fn enabled_with_missing_provider_fails_fast() {
        let p = SolverParams {
            enable_captcha_solver: true,
            provider: None,
            api_key_env: Some("KEY_ENV"),
            timeout_secs: 120,
            session_cost_cap_usd: 5.0,
        };
        assert!(decide_gate(&p).is_err());
    }

    #[test]
    fn enabled_with_nonpositive_cap_fails_fast() {
        let p = SolverParams {
            enable_captcha_solver: true,
            provider: Some("capsolver"),
            api_key_env: Some("KEY_ENV"),
            timeout_secs: 120,
            session_cost_cap_usd: 0.0,
        };
        assert!(decide_gate(&p).is_err());
    }

    #[test]
    fn enabled_with_empty_api_key_env_fails_fast() {
        let p = SolverParams {
            enable_captcha_solver: true,
            provider: Some("capsolver"),
            api_key_env: Some("   "),
            timeout_secs: 120,
            session_cost_cap_usd: 5.0,
        };
        assert!(decide_gate(&p).is_err());
    }

    #[test]
    fn build_solver_disabled_returns_none() {
        let p = SolverParams {
            enable_captcha_solver: false,
            provider: None,
            api_key_env: None,
            timeout_secs: 120,
            session_cost_cap_usd: 5.0,
        };
        let got = build_solver(&p, meter()).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn build_solver_enabled_but_env_unset_fails_fast() {
        // Use a var name that is overwhelmingly unlikely to be set anywhere.
        // (We do NOT mutate the process env — `remove_var` is unsafe in edition
        // 2024 and racy under parallel tests.)
        let env_name = "WEB_SEARCH_MCP_CAPTCHA_KEY_DEFINITELY_UNSET_8F3A21";
        assert!(std::env::var(env_name).is_err(), "test precondition: env var must be unset");
        let p = SolverParams {
            enable_captcha_solver: true,
            provider: Some("capsolver"),
            api_key_env: Some(env_name),
            timeout_secs: 120,
            session_cost_cap_usd: 5.0,
        };
        let got = build_solver(&p, meter());
        assert!(got.is_err(), "missing env var must fail fast, not build a solver");
    }

    #[test]
    fn response_field_mapping() {
        assert_eq!(CaptchaKind::RecaptchaV2.response_field(), "g-recaptcha-response");
        assert_eq!(CaptchaKind::RecaptchaV3.response_field(), "g-recaptcha-response");
        assert_eq!(CaptchaKind::Hcaptcha.response_field(), "h-captcha-response");
        assert_eq!(CaptchaKind::Turnstile.response_field(), "cf-turnstile-response");
    }

    #[test]
    fn error_retriability() {
        assert!(CaptchaError::Transport("conn reset".into()).is_retriable());
        assert!(!CaptchaError::Timeout(std::time::Duration::from_secs(60)).is_retriable());
        assert!(!CaptchaError::CostCapExceeded.is_retriable());
        assert!(!CaptchaError::Disabled.is_retriable());
        assert!(!CaptchaError::Provider {
            code: "ERROR_ZERO_BALANCE".into(),
            message: "no funds".into()
        }
        .is_retriable());
    }
}
