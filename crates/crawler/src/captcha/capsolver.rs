//! CapSolver client (Design 0004 §2.3). Calls the CapSolver HTTP API directly
//! with the workspace `reqwest` dep — no mature Rust crate exists (the
//! `capsolver` crate is empty/unreliable, ground truth 2026-06-11).
//!
//! Two-call async shape (docs.capsolver.com/en/api/):
//!   1. POST `{base}/createTask`    {{ clientKey, task: {{...}} }}  -> {{ taskId }}
//!   2. POLL `{base}/getTaskResult` {{ clientKey, taskId }}         -> {{ status, solution }}
//!      until `status == "ready"`, bounded by `timeout`.
//!
//! kind -> CapSolver task `type` (verified, Design 0004 §2.3):
//!   RecaptchaV2 -> ReCaptchaV2TaskProxyLess
//!   RecaptchaV3 -> ReCaptchaV3TaskProxyLess (with pageAction)
//!   Hcaptcha    -> HCaptchaTaskProxyLess
//!   Turnstile   -> AntiTurnstileTaskProxyLess  (token minted WITHOUT our IP)
//!
//! Token field: reCAPTCHA -> `gRecaptchaResponse`; Turnstile/hCaptcha -> `token`.
//! The `solution` also carries a `userAgent` the caller MUST replay (§2.3).
//!
//! SECURITY: the `clientKey` is held in this struct and serialized only into the
//! request body sent over TLS to the provider. It is NEVER logged and never
//! appears in any `CaptchaError` (api-security NO SECRETS IN LOGS).

use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{CaptchaError, CaptchaKind, CaptchaRequest, CaptchaSolver, CaptchaToken, CostMeterHandle};

const DEFAULT_BASE: &str = "https://api.capsolver.com";
/// Poll interval between `getTaskResult` calls. CapSolver recommends ~1–3s; we
/// use 3s to be polite and to bound total request volume (Design 0004 §2.3).
const POLL_INTERVAL: Duration = Duration::from_secs(3);
/// Conservative per-solve cost estimate (USD) reserved against the cost cap
/// BEFORE the paid call. CapSolver Turnstile/reCAPTCHA are roughly < $0.001–$0.003
/// per solve depending on type/volume (UNVERIFIED pricing — Design 0004 §7 R-4;
/// re-verify before enabling in prod). We reserve a deliberately high estimate
/// so the cap is conservative, not optimistic.
const EST_COST_PER_SOLVE_USD: f64 = 0.003;

pub struct CapSolverClient {
    http: reqwest::Client,
    client_key: String,
    base: String,
    timeout: Duration,
    meter: CostMeterHandle,
}

impl CapSolverClient {
    pub fn new(client_key: String, timeout: Duration, meter: CostMeterHandle) -> Self {
        Self {
            http: reqwest::Client::new(),
            client_key,
            base: DEFAULT_BASE.to_string(),
            timeout,
            meter,
        }
    }

    /// Override the base URL (used by unit tests; never logged).
    #[cfg(test)]
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    /// The CapSolver `type` string for a kind.
    fn task_type(kind: CaptchaKind) -> &'static str {
        match kind {
            CaptchaKind::RecaptchaV2 => "ReCaptchaV2TaskProxyLess",
            CaptchaKind::RecaptchaV3 => "ReCaptchaV3TaskProxyLess",
            CaptchaKind::Hcaptcha => "HCaptchaTaskProxyLess",
            CaptchaKind::Turnstile => "AntiTurnstileTaskProxyLess",
        }
    }

    /// Build the `task` object for a request (pure; unit-tested).
    fn build_task(req: &CaptchaRequest) -> serde_json::Value {
        let mut task = json!({
            "type": Self::task_type(req.kind),
            "websiteURL": req.page_url,
            "websiteKey": req.site_key,
        });
        let obj = task.as_object_mut().expect("task is an object");
        // reCAPTCHA v3 needs pageAction; CapSolver also accepts pageAction for
        // v3-on-element. Turnstile accepts action/cdata via metadata.
        if let Some(action) = req.action.as_deref().filter(|s| !s.is_empty()) {
            match req.kind {
                CaptchaKind::RecaptchaV3 => {
                    obj.insert("pageAction".into(), json!(action));
                    // v3 is score-based; default a reasonable minScore. Proxyless
                    // solves are low-confidence (Design 0004 §2.7).
                    obj.insert("minScore".into(), json!(0.7));
                }
                CaptchaKind::Turnstile => {
                    obj.insert(
                        "metadata".into(),
                        json!({ "action": action }),
                    );
                }
                _ => {}
            }
        }
        if let (CaptchaKind::Turnstile, Some(cdata)) =
            (req.kind, req.cdata.as_deref().filter(|s| !s.is_empty()))
        {
            // Merge cdata into Turnstile metadata.
            let meta = obj
                .entry("metadata")
                .or_insert_with(|| json!({}));
            if let Some(m) = meta.as_object_mut() {
                m.insert("cdata".into(), json!(cdata));
            }
        }
        task
    }

    /// Build the full createTask request body (pure; unit-tested).
    fn create_task_body(&self, req: &CaptchaRequest) -> serde_json::Value {
        json!({
            "clientKey": self.client_key,
            "task": Self::build_task(req),
        })
    }
}

// -- CapSolver wire types -----------------------------------------------------

#[derive(Debug, Deserialize)]
struct CreateTaskResp {
    #[serde(rename = "errorId")]
    error_id: i64,
    #[serde(rename = "errorCode")]
    error_code: Option<String>,
    #[serde(rename = "errorDescription")]
    error_description: Option<String>,
    #[serde(rename = "taskId")]
    task_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskResultResp {
    #[serde(rename = "errorId")]
    error_id: i64,
    #[serde(rename = "errorCode")]
    error_code: Option<String>,
    #[serde(rename = "errorDescription")]
    error_description: Option<String>,
    /// "idle" | "processing" | "ready"
    status: Option<String>,
    solution: Option<Solution>,
}

#[derive(Debug, Deserialize)]
struct Solution {
    #[serde(rename = "gRecaptchaResponse")]
    g_recaptcha_response: Option<String>,
    token: Option<String>,
    #[serde(rename = "userAgent")]
    user_agent: Option<String>,
}

#[derive(Debug, Serialize)]
struct GetResultBody<'a> {
    #[serde(rename = "clientKey")]
    client_key: &'a str,
    #[serde(rename = "taskId")]
    task_id: &'a str,
}

impl Solution {
    /// Pick the token field per family and return `(token, user_agent)`.
    fn into_token(self, kind: CaptchaKind) -> Result<CaptchaToken, CaptchaError> {
        let token = match kind {
            CaptchaKind::RecaptchaV2 | CaptchaKind::RecaptchaV3 => self.g_recaptcha_response,
            CaptchaKind::Hcaptcha | CaptchaKind::Turnstile => self.token,
        }
        .filter(|t| !t.is_empty());
        match token {
            Some(t) => Ok(CaptchaToken { token: t, user_agent: self.user_agent }),
            None => Err(CaptchaError::BadResponse(
                "provider returned ready status but no token for the requested kind".into(),
            )),
        }
    }
}

#[async_trait]
impl CaptchaSolver for CapSolverClient {
    async fn solve(&self, req: &CaptchaRequest) -> Result<CaptchaToken, CaptchaError> {
        // COST GATE — reserve BEFORE any paid call (Design 0004 §2.5). Hard-halts
        // at the session cap. Refunded if the create call fails before billing.
        self.meter.try_reserve(EST_COST_PER_SOLVE_USD)?;

        // 1) createTask
        let create_body = self.create_task_body(req);
        let create_url = format!("{}/createTask", self.base);
        let resp = self
            .http
            .post(&create_url)
            .json(&create_body)
            .send()
            .await
            .map_err(|e| {
                self.meter.refund(EST_COST_PER_SOLVE_USD);
                CaptchaError::Transport(sanitize(&e))
            })?;
        let create: CreateTaskResp = resp.json().await.map_err(|e| {
            self.meter.refund(EST_COST_PER_SOLVE_USD);
            CaptchaError::Transport(sanitize(&e))
        })?;
        if create.error_id != 0 {
            self.meter.refund(EST_COST_PER_SOLVE_USD);
            return Err(CaptchaError::Provider {
                code: create.error_code.unwrap_or_else(|| "UNKNOWN".into()),
                message: create.error_description.unwrap_or_default(),
            });
        }
        let task_id = create.task_id.ok_or_else(|| {
            self.meter.refund(EST_COST_PER_SOLVE_USD);
            CaptchaError::BadResponse("createTask returned no taskId".into())
        })?;

        // 2) poll getTaskResult until ready or the bounded deadline elapses.
        let result_url = format!("{}/getTaskResult", self.base);
        let deadline = Instant::now() + self.timeout;
        loop {
            if Instant::now() >= deadline {
                // Budget spent: the solve did not complete in time. Keep the
                // reservation (we did consume provider work).
                return Err(CaptchaError::Timeout(self.timeout));
            }
            let body = GetResultBody { client_key: &self.client_key, task_id: &task_id };
            let resp = self
                .http
                .post(&result_url)
                .json(&body)
                .send()
                .await
                .map_err(|e| CaptchaError::Transport(sanitize(&e)))?;
            let result: TaskResultResp =
                resp.json().await.map_err(|e| CaptchaError::Transport(sanitize(&e)))?;
            if result.error_id != 0 {
                return Err(CaptchaError::Provider {
                    code: result.error_code.unwrap_or_else(|| "UNKNOWN".into()),
                    message: result.error_description.unwrap_or_default(),
                });
            }
            match result.status.as_deref() {
                Some("ready") => {
                    let sol = result
                        .solution
                        .ok_or_else(|| CaptchaError::BadResponse("ready without solution".into()))?;
                    return sol.into_token(req.kind);
                }
                // idle / processing / unknown -> keep polling.
                _ => {
                    tokio::time::sleep(POLL_INTERVAL).await;
                }
            }
        }
    }

    fn provider(&self) -> &'static str {
        "capsolver"
    }
}

/// Turn a reqwest error into a short, SECRET-FREE description. reqwest error
/// Display can include the URL but never our request body (where the key lives),
/// so this is safe; we additionally strip any query string defensively.
fn sanitize(e: &reqwest::Error) -> String {
    let s = e.to_string();
    // Defensive: drop anything after a '?' in case a URL with a query leaked in.
    match s.split_once('?') {
        Some((head, _)) => head.to_string(),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meter() -> CostMeterHandle {
        super::super::CostMeter::new(5.0).handle()
    }

    fn req(kind: CaptchaKind) -> CaptchaRequest {
        CaptchaRequest {
            kind,
            site_key: "SITEKEY".into(),
            page_url: "https://example.com/login".into(),
            action: None,
            cdata: None,
        }
    }

    #[test]
    fn task_type_mapping() {
        assert_eq!(CapSolverClient::task_type(CaptchaKind::RecaptchaV2), "ReCaptchaV2TaskProxyLess");
        assert_eq!(CapSolverClient::task_type(CaptchaKind::RecaptchaV3), "ReCaptchaV3TaskProxyLess");
        assert_eq!(CapSolverClient::task_type(CaptchaKind::Hcaptcha), "HCaptchaTaskProxyLess");
        assert_eq!(CapSolverClient::task_type(CaptchaKind::Turnstile), "AntiTurnstileTaskProxyLess");
    }

    #[test]
    fn create_body_has_clientkey_and_task_fields() {
        let c = CapSolverClient::new("SECRET_KEY".into(), Duration::from_secs(60), meter());
        let body = c.create_task_body(&req(CaptchaKind::RecaptchaV2));
        assert_eq!(body["clientKey"], "SECRET_KEY");
        assert_eq!(body["task"]["type"], "ReCaptchaV2TaskProxyLess");
        assert_eq!(body["task"]["websiteURL"], "https://example.com/login");
        assert_eq!(body["task"]["websiteKey"], "SITEKEY");
    }

    #[test]
    fn v3_task_includes_page_action_and_minscore() {
        let mut r = req(CaptchaKind::RecaptchaV3);
        r.action = Some("submit".into());
        let task = CapSolverClient::build_task(&r);
        assert_eq!(task["type"], "ReCaptchaV3TaskProxyLess");
        assert_eq!(task["pageAction"], "submit");
        assert!(task["minScore"].is_number());
    }

    #[test]
    fn turnstile_task_carries_action_and_cdata_metadata() {
        let mut r = req(CaptchaKind::Turnstile);
        r.action = Some("managed".into());
        r.cdata = Some("CDATA123".into());
        let task = CapSolverClient::build_task(&r);
        assert_eq!(task["type"], "AntiTurnstileTaskProxyLess");
        assert_eq!(task["metadata"]["action"], "managed");
        assert_eq!(task["metadata"]["cdata"], "CDATA123");
    }

    #[test]
    fn solution_picks_grecaptcha_for_recaptcha() {
        let sol = Solution {
            g_recaptcha_response: Some("RECAP_TOKEN".into()),
            token: None,
            user_agent: Some("UA/1".into()),
        };
        let t = sol.into_token(CaptchaKind::RecaptchaV2).unwrap();
        assert_eq!(t.token, "RECAP_TOKEN");
        assert_eq!(t.user_agent.as_deref(), Some("UA/1"));
    }

    #[test]
    fn solution_picks_token_for_turnstile() {
        let sol = Solution {
            g_recaptcha_response: None,
            token: Some("TURN_TOKEN".into()),
            user_agent: None,
        };
        let t = sol.into_token(CaptchaKind::Turnstile).unwrap();
        assert_eq!(t.token, "TURN_TOKEN");
        assert_eq!(t.user_agent, None);
    }

    #[test]
    fn solution_missing_token_is_bad_response() {
        let sol = Solution { g_recaptcha_response: None, token: None, user_agent: None };
        assert!(matches!(
            sol.into_token(CaptchaKind::Hcaptcha),
            Err(CaptchaError::BadResponse(_))
        ));
    }

    #[test]
    fn create_resp_deserializes_error_shape() {
        let raw = r#"{"errorId":1,"errorCode":"ERROR_KEY_DENIED_ACCESS","errorDescription":"bad key"}"#;
        let r: CreateTaskResp = serde_json::from_str(raw).unwrap();
        assert_eq!(r.error_id, 1);
        assert_eq!(r.error_code.as_deref(), Some("ERROR_KEY_DENIED_ACCESS"));
        assert!(r.task_id.is_none());
    }

    #[test]
    fn result_resp_deserializes_ready_shape() {
        let raw = r#"{"errorId":0,"status":"ready","solution":{"token":"abc","userAgent":"UA/9"}}"#;
        let r: TaskResultResp = serde_json::from_str(raw).unwrap();
        assert_eq!(r.error_id, 0);
        assert_eq!(r.status.as_deref(), Some("ready"));
        let sol = r.solution.unwrap();
        assert_eq!(sol.token.as_deref(), Some("abc"));
        assert_eq!(sol.user_agent.as_deref(), Some("UA/9"));
    }

    #[tokio::test]
    async fn solve_halts_when_cost_cap_exhausted() {
        // Cap so low that the up-front reservation fails — no network call made.
        let m = super::super::CostMeter::new(0.0);
        let c = CapSolverClient::new("KEY".into(), Duration::from_secs(1), m.handle());
        let err = c.solve(&req(CaptchaKind::RecaptchaV2)).await.unwrap_err();
        assert!(matches!(err, CaptchaError::CostCapExceeded));
        assert_eq!(m.spent_usd(), 0.0);
    }

    #[test]
    fn provider_name_is_capsolver() {
        let c = CapSolverClient::new("KEY".into(), Duration::from_secs(1), meter());
        assert_eq!(c.provider(), "capsolver");
    }
}
