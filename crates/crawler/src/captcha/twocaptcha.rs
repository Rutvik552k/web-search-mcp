//! 2Captcha client (Design 0004 §2.3). Calls the 2Captcha JSON API directly via
//! the workspace `reqwest` dep — same two-call async shape as CapSolver, in
//! 2Captcha's task-type vocabulary.
//!
//! 2Captcha JSON API (api.2captcha.com — 2captcha.com/api-docs/recaptcha-v2):
//!   1. POST `{base}/createTask`    {{ clientKey, task: {{...}} }}  -> {{ taskId }}
//!   2. POLL `{base}/getTaskResult` {{ clientKey, taskId }}         -> {{ status, solution }}
//!      until `status == "ready"`, bounded by `timeout`.
//!
//! kind -> 2Captcha task `type` (2Captcha JSON API vocabulary):
//!   RecaptchaV2 -> RecaptchaV2TaskProxyless
//!   RecaptchaV3 -> RecaptchaV3TaskProxyless (with pageAction + minScore)
//!   Hcaptcha    -> HCaptchaTaskProxyless
//!   Turnstile   -> TurnstileTaskProxyless
//!
//! Token field: reCAPTCHA -> `gRecaptchaResponse`; Turnstile/hCaptcha -> `token`.
//! The `solution` may carry a `userAgent` to replay (§2.3).
//!
//! SECURITY: the `clientKey` (2Captcha API key) is serialized only into the TLS
//! request body — never logged, never in any `CaptchaError`.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{CaptchaError, CaptchaKind, CaptchaRequest, CaptchaSolver, CaptchaToken, CostMeterHandle};

const DEFAULT_BASE: &str = "https://api.2captcha.com";
const POLL_INTERVAL: Duration = Duration::from_secs(3);
/// Conservative per-solve estimate reserved before the paid call. 2Captcha
/// reCAPTCHA/Turnstile run roughly $0.001–$0.003/solve (UNVERIFIED pricing —
/// Design 0004 §7 R-4; re-verify before prod).
const EST_COST_PER_SOLVE_USD: f64 = 0.003;

pub struct TwoCaptchaClient {
    http: reqwest::Client,
    client_key: String,
    base: String,
    timeout: Duration,
    meter: CostMeterHandle,
}

impl TwoCaptchaClient {
    pub fn new(client_key: String, timeout: Duration, meter: CostMeterHandle) -> Self {
        Self {
            http: reqwest::Client::new(),
            client_key,
            base: DEFAULT_BASE.to_string(),
            timeout,
            meter,
        }
    }

    #[cfg(test)]
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    fn task_type(kind: CaptchaKind) -> &'static str {
        match kind {
            CaptchaKind::RecaptchaV2 => "RecaptchaV2TaskProxyless",
            CaptchaKind::RecaptchaV3 => "RecaptchaV3TaskProxyless",
            CaptchaKind::Hcaptcha => "HCaptchaTaskProxyless",
            CaptchaKind::Turnstile => "TurnstileTaskProxyless",
        }
    }

    /// Build the `task` object (pure; unit-tested).
    fn build_task(req: &CaptchaRequest) -> serde_json::Value {
        let mut task = json!({
            "type": Self::task_type(req.kind),
            "websiteURL": req.page_url,
            "websiteKey": req.site_key,
        });
        let obj = task.as_object_mut().expect("task is an object");
        if let Some(action) = req.action.as_deref().filter(|s| !s.is_empty()) {
            match req.kind {
                CaptchaKind::RecaptchaV3 => {
                    obj.insert("pageAction".into(), json!(action));
                    obj.insert("minScore".into(), json!(0.7));
                }
                CaptchaKind::Turnstile => {
                    // 2Captcha Turnstile takes action/data at the task root.
                    obj.insert("action".into(), json!(action));
                }
                _ => {}
            }
        }
        if let (CaptchaKind::Turnstile, Some(cdata)) =
            (req.kind, req.cdata.as_deref().filter(|s| !s.is_empty()))
        {
            obj.insert("data".into(), json!(cdata));
        }
        task
    }

    fn create_task_body(&self, req: &CaptchaRequest) -> serde_json::Value {
        json!({
            "clientKey": self.client_key,
            "task": Self::build_task(req),
        })
    }
}

#[derive(Debug, Deserialize)]
struct CreateTaskResp {
    #[serde(rename = "errorId")]
    error_id: i64,
    #[serde(rename = "errorCode")]
    error_code: Option<String>,
    #[serde(rename = "errorDescription")]
    error_description: Option<String>,
    #[serde(rename = "taskId")]
    task_id: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct TaskResultResp {
    #[serde(rename = "errorId")]
    error_id: i64,
    #[serde(rename = "errorCode")]
    error_code: Option<String>,
    #[serde(rename = "errorDescription")]
    error_description: Option<String>,
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

/// 2Captcha returns `taskId` as either a number or a string depending on
/// endpoint version — normalize to a string for the follow-up poll.
fn task_id_to_string(v: serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) if !s.is_empty() => Some(s),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

#[async_trait]
impl CaptchaSolver for TwoCaptchaClient {
    async fn solve(&self, req: &CaptchaRequest) -> Result<CaptchaToken, CaptchaError> {
        self.meter.try_reserve(EST_COST_PER_SOLVE_USD)?;

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
        let task_id = create.task_id.and_then(task_id_to_string).ok_or_else(|| {
            self.meter.refund(EST_COST_PER_SOLVE_USD);
            CaptchaError::BadResponse("createTask returned no taskId".into())
        })?;

        let result_url = format!("{}/getTaskResult", self.base);
        let deadline = Instant::now() + self.timeout;
        loop {
            if Instant::now() >= deadline {
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
                _ => {
                    tokio::time::sleep(POLL_INTERVAL).await;
                }
            }
        }
    }

    fn provider(&self) -> &'static str {
        "2captcha"
    }
}

fn sanitize(e: &reqwest::Error) -> String {
    let s = e.to_string();
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
        assert_eq!(TwoCaptchaClient::task_type(CaptchaKind::RecaptchaV2), "RecaptchaV2TaskProxyless");
        assert_eq!(TwoCaptchaClient::task_type(CaptchaKind::RecaptchaV3), "RecaptchaV3TaskProxyless");
        assert_eq!(TwoCaptchaClient::task_type(CaptchaKind::Hcaptcha), "HCaptchaTaskProxyless");
        assert_eq!(TwoCaptchaClient::task_type(CaptchaKind::Turnstile), "TurnstileTaskProxyless");
    }

    #[test]
    fn create_body_has_clientkey_and_task() {
        let c = TwoCaptchaClient::new("SECRET_KEY".into(), Duration::from_secs(60), meter());
        let body = c.create_task_body(&req(CaptchaKind::Hcaptcha));
        assert_eq!(body["clientKey"], "SECRET_KEY");
        assert_eq!(body["task"]["type"], "HCaptchaTaskProxyless");
        assert_eq!(body["task"]["websiteURL"], "https://example.com/login");
        assert_eq!(body["task"]["websiteKey"], "SITEKEY");
    }

    #[test]
    fn turnstile_task_uses_action_and_data_at_root() {
        let mut r = req(CaptchaKind::Turnstile);
        r.action = Some("managed".into());
        r.cdata = Some("CDATA".into());
        let task = TwoCaptchaClient::build_task(&r);
        assert_eq!(task["type"], "TurnstileTaskProxyless");
        assert_eq!(task["action"], "managed");
        assert_eq!(task["data"], "CDATA");
    }

    #[test]
    fn v3_includes_pageaction_and_minscore() {
        let mut r = req(CaptchaKind::RecaptchaV3);
        r.action = Some("verify".into());
        let task = TwoCaptchaClient::build_task(&r);
        assert_eq!(task["pageAction"], "verify");
        assert!(task["minScore"].is_number());
    }

    #[test]
    fn task_id_normalizes_number_and_string() {
        assert_eq!(task_id_to_string(json!(123456789_i64)).as_deref(), Some("123456789"));
        assert_eq!(task_id_to_string(json!("abc")).as_deref(), Some("abc"));
        assert_eq!(task_id_to_string(json!("")), None);
        assert_eq!(task_id_to_string(json!(null)), None);
    }

    #[test]
    fn result_resp_ready_deserializes() {
        let raw = r#"{"errorId":0,"status":"ready","solution":{"gRecaptchaResponse":"tok"}}"#;
        let r: TaskResultResp = serde_json::from_str(raw).unwrap();
        let sol = r.solution.unwrap();
        assert_eq!(sol.g_recaptcha_response.as_deref(), Some("tok"));
    }

    #[tokio::test]
    async fn solve_halts_when_cost_cap_exhausted() {
        let m = super::super::CostMeter::new(0.0);
        let c = TwoCaptchaClient::new("KEY".into(), Duration::from_secs(1), m.handle());
        let err = c.solve(&req(CaptchaKind::Turnstile)).await.unwrap_err();
        assert!(matches!(err, CaptchaError::CostCapExceeded));
        assert_eq!(m.spent_usd(), 0.0);
    }

    #[test]
    fn provider_name_is_2captcha() {
        let c = TwoCaptchaClient::new("KEY".into(), Duration::from_secs(1), meter());
        assert_eq!(c.provider(), "2captcha");
    }
}
