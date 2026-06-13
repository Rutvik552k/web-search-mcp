//! Minimal reputation governor + permanent denylist (Design 0005 §5, ADR 0003
//! §4). Single-IP safety: the load-bearing piece is the PERMANENT, file-backed,
//! restart-surviving hard-stop denylist — a confirmed hard ban is terminal for a
//! domain and is never retried (ADR 0003 §4.2/§4.3 / GOAL.md §2).
//!
//! # Scope (this iteration — Design 0005 §5.1)
//!
//! Built: a coarse per-domain pacer (halve-on-soft-signal / nudge-up-on-success)
//! with jitter, a soft cooldown breaker, a per-domain request budget, the pure
//! [`breaker_verdict`] helper, and the file-backed permanent denylist.
//!
//! Deferred (Design 0005 §5.2 — TODO): fine-grained continuous AIMD knobs
//! (`aimd_increase_step_rps` / `aimd_decrease_factor` / `per_domain_max_concurrency`)
//! and the `denylist_blocks_archive` nuance. We use a single coarse delay value
//! per domain instead of a full rate/concurrency controller.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::Mutex;

use crate::classifier::BlockClass;

/// Outcome of an admission check before a live-origin rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// The rung may proceed (pace + jitter already waited).
    Proceed,
    /// Skip the live rung this iteration (budget exhausted or soft breaker open).
    SkipLive(SkipReason),
    /// The domain is on the permanent denylist — never contact the origin.
    DeniedPermanent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    BudgetExhausted,
    BreakerCooldown,
}

/// The rung an outcome was observed at (for tracing / breaker logic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rung {
    R0,
    R1,
    /// R2 alternative-surface probe (feeds/sitemap — same origin, governor-gated).
    R2,
    /// R3 Internet Archive CDX fallback (zero-ban-risk; never touches the origin).
    R3,
    R4,
    R5,
}

/// Breaker trip mode for a give-up verdict (Design 0005 §5.1, ADR 0003 §4.2).
/// PURE / stateless decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerTrip {
    None,
    SoftCooldown,
    HardPermanent,
}

/// Tuning for the governor. Off-safe; derived from `CrawlerConfig`.
#[derive(Debug, Clone)]
pub struct GovernorConfig {
    pub per_domain_request_budget: u32,
    pub soft_breaker_fail_threshold: u32,
    pub domain_breaker_cooldown_secs: u64,
    /// Pacing jitter ratio (±). Clamped to [0, 1) by the constructor.
    pub pacing_jitter_ratio: f64,
    /// Base pacing delay per domain (ms). Coarse single value (Design 0005 §5.1).
    pub base_pace_ms: u64,
    /// Floor / ceiling for the coarse pace (ms).
    pub min_pace_ms: u64,
    pub max_pace_ms: u64,
    /// Persistent denylist path. None => in-memory only (startup WARN).
    pub permanent_denylist_path: Option<PathBuf>,
}

impl GovernorConfig {
    pub fn from_parts(
        per_domain_request_budget: u32,
        soft_breaker_fail_threshold: u32,
        domain_breaker_cooldown_secs: u64,
        pacing_jitter_ratio: f64,
        permanent_denylist_path: Option<PathBuf>,
    ) -> Self {
        Self {
            per_domain_request_budget,
            soft_breaker_fail_threshold,
            domain_breaker_cooldown_secs,
            pacing_jitter_ratio: pacing_jitter_ratio.clamp(0.0, 0.999),
            base_pace_ms: 500,
            min_pace_ms: 50,
            max_pace_ms: 30_000,
            permanent_denylist_path,
        }
    }
}

/// The single-IP reputation governor.
pub trait ReputationGovernor: Send + Sync {
    /// Called before a live-origin rung. Waits out the pace + jitter, then
    /// returns whether the rung may proceed. NOTE: this is sync (the caller can
    /// `tokio::time::sleep` on the returned pace) — keeping the trait sync makes
    /// the pure decision unit-testable without a runtime. See [`MinimalGovernor::admit_pace`].
    fn admit(&self, domain: &str) -> Admission;

    /// The pace delay the caller should sleep before the live request (after a
    /// `Proceed`). Separated so tests don't actually sleep.
    fn pace_delay(&self, domain: &str) -> Duration;

    /// Feed the classifier verdict back so the pacer adjusts and the breaker may
    /// trip. `r4_attempted_and_blocked` drives the HARD-trip decision.
    fn record(&self, domain: &str, outcome: &BlockClass, rung: Rung, r4_attempted_and_blocked: bool);

    /// True iff the domain is on the persistent permanent denylist.
    fn is_permanently_denied(&self, domain: &str) -> bool;
}

/// PURE breaker-trip decision (Design 0005 §5.1 / ADR 0003 §4.2). HardPermanent
/// iff a CF/DataDome/JS block persisted AFTER R4 was attempted and still blocked
/// (we presented a real browser and were still blocked => IP reputation, not a
/// solvable challenge). A single block with R4 untried is SoftCooldown, never
/// Hard. RealContent / RateLimited never trip the breaker here (RateLimited is a
/// pacing decrease, not a breaker trip).
pub fn breaker_verdict(
    class: &BlockClass,
    r4_attempted_and_blocked: bool,
    _cfg: &GovernorConfig,
) -> BreakerTrip {
    match class {
        BlockClass::Cloudflare403 | BlockClass::JsChallenge { .. } => {
            if r4_attempted_and_blocked {
                BreakerTrip::HardPermanent
            } else {
                BreakerTrip::SoftCooldown
            }
        }
        BlockClass::SoftBlock | BlockClass::Captcha { .. } => BreakerTrip::SoftCooldown,
        BlockClass::RealContent | BlockClass::RateLimited { .. } => BreakerTrip::None,
    }
}

/// Per-domain mutable state (behind a `Mutex` inside the `DashMap`).
#[derive(Debug)]
struct DomainState {
    /// Coarse pace delay (ms).
    pace_ms: u64,
    /// Live-origin requests spent this session.
    requests_spent: u32,
    /// Soft-signal counter for the breaker.
    soft_fails: u32,
    /// When the soft breaker opened (if open).
    breaker_open_until: Option<Instant>,
}

/// A denylist entry. JSON-lines on disk (Design 0005 H4 / ADR 0003 §4.3). No
/// secrets — domain / reason / verdict / timestamp only.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DenylistEntry {
    pub domain: String,
    pub reason: String,
    pub classifier_verdict: String,
    pub first_seen_iso: String,
}

/// Minimal governor impl.
pub struct MinimalGovernor {
    cfg: GovernorConfig,
    domains: DashMap<String, Mutex<DomainState>>,
    /// In-memory mirror of the permanent denylist (always populated; the file is
    /// the durable backing when a path is configured).
    denied: DashMap<String, DenylistEntry>,
    /// Serializes appends to the denylist file.
    file_lock: Mutex<()>,
}

impl MinimalGovernor {
    /// Construct, loading any existing denylist file. Emits a startup WARN when
    /// no path is configured (a hard ban won't survive restart — ADR 0003 §6).
    pub fn new(cfg: GovernorConfig) -> Arc<Self> {
        let denied = DashMap::new();
        match &cfg.permanent_denylist_path {
            Some(path) => {
                if let Err(e) = load_denylist(path, &denied) {
                    tracing::warn!(error = %e, path = %path.display(), "governor: failed to load denylist (starting empty)");
                }
                tracing::info!(path = %path.display(), entries = denied.len(), "governor: permanent denylist loaded");
            }
            None => {
                tracing::warn!(
                    "governor: permanent_denylist_path is unset — a hard ban will NOT survive \
                     restart (ADR 0003 §6). Set permanent_denylist_path to persist hard-stops."
                );
            }
        }
        Arc::new(Self {
            cfg,
            domains: DashMap::new(),
            denied,
            file_lock: Mutex::new(()),
        })
    }

    fn state_for(&self, domain: &str) -> dashmap::mapref::one::Ref<'_, String, Mutex<DomainState>> {
        if !self.domains.contains_key(domain) {
            self.domains.entry(domain.to_string()).or_insert_with(|| {
                Mutex::new(DomainState {
                    pace_ms: self.cfg.base_pace_ms,
                    requests_spent: 0,
                    soft_fails: 0,
                    breaker_open_until: None,
                })
            });
        }
        self.domains.get(domain).expect("just inserted")
    }

    /// Hard-trip: write the domain to the permanent denylist (memory + file) and
    /// emit a distinct alert-worthy event (Design 0005 H7). Idempotent.
    pub fn record_hard_ban(&self, domain: &str, reason: &str, classifier_verdict: &str) {
        if self.denied.contains_key(domain) {
            return;
        }
        let entry = DenylistEntry {
            domain: domain.to_string(),
            reason: reason.to_string(),
            classifier_verdict: classifier_verdict.to_string(),
            first_seen_iso: now_iso(),
        };
        self.denied.insert(domain.to_string(), entry.clone());

        // Distinct, alert-worthy event (compliance-audit artifact). No secrets.
        tracing::error!(
            domain = %entry.domain,
            reason = %entry.reason,
            classifier_verdict = %entry.classifier_verdict,
            "governor: PERMANENT hard-stop — domain denylisted (single-IP safety, ADR 0003 §4.2)"
        );

        if let Some(path) = &self.cfg.permanent_denylist_path {
            let _guard = self.file_lock.lock();
            if let Err(e) = append_denylist(path, &entry) {
                tracing::error!(error = %e, path = %path.display(), "governor: FAILED to persist hard-ban (in-memory only)");
            }
        }
    }
}

impl ReputationGovernor for MinimalGovernor {
    fn admit(&self, domain: &str) -> Admission {
        if self.is_permanently_denied(domain) {
            return Admission::DeniedPermanent;
        }
        let state = self.state_for(domain);
        let mut s = state.lock();

        // Soft breaker open?
        if let Some(until) = s.breaker_open_until {
            if Instant::now() < until {
                return Admission::SkipLive(SkipReason::BreakerCooldown);
            }
            // cooldown elapsed -> half-open (clear, keep decreased pace).
            s.breaker_open_until = None;
            s.soft_fails = 0;
        }

        // Budget.
        if s.requests_spent >= self.cfg.per_domain_request_budget {
            return Admission::SkipLive(SkipReason::BudgetExhausted);
        }
        s.requests_spent += 1;
        Admission::Proceed
    }

    fn pace_delay(&self, domain: &str) -> Duration {
        let state = self.state_for(domain);
        let s = state.lock();
        let base = s.pace_ms as f64;
        // Deterministic-ish jitter from a cheap PRNG seed (no rand dep); the
        // exact value does not matter, only that it is within ±ratio.
        let j = self.cfg.pacing_jitter_ratio;
        let frac = pseudo_jitter(domain, s.requests_spent);
        let factor = 1.0 + (frac * 2.0 - 1.0) * j; // in [1-j, 1+j]
        Duration::from_millis((base * factor).max(0.0) as u64)
    }

    fn record(&self, domain: &str, outcome: &BlockClass, _rung: Rung, _r4_blocked: bool) {
        let state = self.state_for(domain);
        let mut s = state.lock();
        match outcome {
            BlockClass::RealContent => {
                // additive-increase of rate == nudge the delay DOWN.
                s.pace_ms = (s.pace_ms.saturating_sub(self.cfg.base_pace_ms / 4))
                    .max(self.cfg.min_pace_ms);
                s.soft_fails = 0;
            }
            BlockClass::RateLimited { .. }
            | BlockClass::SoftBlock
            | BlockClass::JsChallenge { .. }
            | BlockClass::Cloudflare403
            | BlockClass::Captcha { .. } => {
                // multiplicative-decrease of rate == DOUBLE the delay.
                s.pace_ms = (s.pace_ms.saturating_mul(2)).min(self.cfg.max_pace_ms);
                s.soft_fails += 1;
                if s.soft_fails >= self.cfg.soft_breaker_fail_threshold {
                    s.breaker_open_until = Some(
                        Instant::now()
                            + Duration::from_secs(self.cfg.domain_breaker_cooldown_secs),
                    );
                }
            }
        }
    }

    fn is_permanently_denied(&self, domain: &str) -> bool {
        self.denied.contains_key(domain)
    }
}

// -- file persistence ---------------------------------------------------------

fn load_denylist(path: &Path, into: &DashMap<String, DenylistEntry>) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let content = std::fs::read_to_string(path)?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<DenylistEntry>(line) {
            Ok(entry) => {
                into.insert(entry.domain.clone(), entry);
            }
            Err(e) => {
                tracing::warn!(error = %e, "governor: skipping malformed denylist line");
            }
        }
    }
    Ok(())
}

fn append_denylist(path: &Path, entry: &DenylistEntry) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(entry).map_err(std::io::Error::other)?;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    f.flush()?;
    f.sync_all()?; // fsync (Design 0005 H4)
    Ok(())
}

// -- helpers ------------------------------------------------------------------

/// A tiny deterministic [0,1) "jitter" derived from the domain + counter. Avoids
/// a `rand` dependency; the only requirement (resilience rule) is that the pace
/// is not a perfectly periodic signature.
fn pseudo_jitter(domain: &str, counter: u32) -> f64 {
    let mut h: u64 = 1469598103934665603; // FNV offset basis
    for b in domain.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h ^= counter as u64;
    h = h.wrapping_mul(1099511628211);
    // top 24 bits → [0,1)
    ((h >> 40) as f64) / (1u64 << 24) as f64
}

/// Minimal ISO-8601 UTC timestamp without a date dependency.
fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // civil-from-days (Howard Hinnant), inverse of the classifier's date parser.
    let days = (secs / 86400) as i64;
    let rem = (secs % 86400) as i64;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::ChallengeVendor;

    fn gcfg(path: Option<PathBuf>) -> GovernorConfig {
        GovernorConfig::from_parts(3, 2, 300, 0.3, path)
    }

    #[test]
    fn breaker_verdict_single_block_is_soft() {
        let c = gcfg(None);
        assert_eq!(breaker_verdict(&BlockClass::Cloudflare403, false, &c), BreakerTrip::SoftCooldown);
        assert_eq!(
            breaker_verdict(&BlockClass::JsChallenge { vendor: ChallengeVendor::Cloudflare }, false, &c),
            BreakerTrip::SoftCooldown
        );
    }

    #[test]
    fn breaker_verdict_block_after_r4_is_hard() {
        let c = gcfg(None);
        assert_eq!(breaker_verdict(&BlockClass::Cloudflare403, true, &c), BreakerTrip::HardPermanent);
        assert_eq!(
            breaker_verdict(&BlockClass::JsChallenge { vendor: ChallengeVendor::DataDome }, true, &c),
            BreakerTrip::HardPermanent
        );
    }

    #[test]
    fn breaker_verdict_real_content_never_trips() {
        let c = gcfg(None);
        assert_eq!(breaker_verdict(&BlockClass::RealContent, true, &c), BreakerTrip::None);
        assert_eq!(
            breaker_verdict(&BlockClass::RateLimited { retry_after_secs: 30 }, true, &c),
            BreakerTrip::None
        );
    }

    #[test]
    fn pace_decreases_on_success_increases_on_soft() {
        let g = MinimalGovernor::new(gcfg(None));
        let d = "example.com";
        // success nudges down
        g.record(d, &BlockClass::RealContent, Rung::R1, false);
        let after_success = { g.state_for(d).lock().pace_ms };
        assert!(after_success <= g.cfg.base_pace_ms);
        // soft signal doubles
        g.record(d, &BlockClass::SoftBlock, Rung::R1, false);
        let after_soft = { g.state_for(d).lock().pace_ms };
        assert!(after_soft > after_success);
    }

    #[test]
    fn budget_exhaustion_skips_live() {
        let g = MinimalGovernor::new(gcfg(None)); // budget = 3
        let d = "budget.com";
        assert_eq!(g.admit(d), Admission::Proceed);
        assert_eq!(g.admit(d), Admission::Proceed);
        assert_eq!(g.admit(d), Admission::Proceed);
        assert_eq!(g.admit(d), Admission::SkipLive(SkipReason::BudgetExhausted));
    }

    #[test]
    fn soft_breaker_opens_after_threshold() {
        let g = MinimalGovernor::new(gcfg(None)); // threshold = 2
        let d = "breaker.com";
        g.record(d, &BlockClass::SoftBlock, Rung::R1, false);
        g.record(d, &BlockClass::SoftBlock, Rung::R1, false);
        // breaker should now be open -> admit skips live
        assert_eq!(g.admit(d), Admission::SkipLive(SkipReason::BreakerCooldown));
    }

    #[test]
    fn permanent_denylist_blocks_admit() {
        let g = MinimalGovernor::new(gcfg(None));
        let d = "banned.com";
        assert!(!g.is_permanently_denied(d));
        g.record_hard_ban(d, "hard-ban", "JsChallenge{Cloudflare}");
        assert!(g.is_permanently_denied(d));
        assert_eq!(g.admit(d), Admission::DeniedPermanent);
    }

    #[test]
    fn denylist_persist_and_reload_roundtrip() {
        // temp path
        let mut path = std::env::temp_dir();
        path.push(format!("wsmcp_denylist_test_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);

        // write via one governor
        {
            let g = MinimalGovernor::new(gcfg(Some(path.clone())));
            g.record_hard_ban("evil.example", "hard-ban", "Cloudflare403");
            assert!(g.is_permanently_denied("evil.example"));
        }

        // a FRESH governor loading the same file must see the entry (restart-survival)
        {
            let g2 = MinimalGovernor::new(gcfg(Some(path.clone())));
            assert!(g2.is_permanently_denied("evil.example"), "denylist must survive reload");
            assert_eq!(g2.admit("evil.example"), Admission::DeniedPermanent);
        }

        // the file must contain no secrets — just the documented schema fields.
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("evil.example"));
        assert!(content.contains("hard-ban"));
        assert!(content.contains("first_seen_iso") || content.contains("Cloudflare403"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn jitter_within_bounds() {
        let g = MinimalGovernor::new(gcfg(None));
        let d = "jitter.com";
        let base = g.cfg.base_pace_ms as f64;
        for _ in 0..100 {
            // force counter increments
            g.admit(d);
            let delay = g.pace_delay(d).as_millis() as f64;
            assert!(delay >= base * (1.0 - 0.3) - 1.0);
            assert!(delay <= base * (1.0 + 0.3) + 1.0);
        }
    }
}
