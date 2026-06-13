//! Per-session USD cost metering with a hard cap (Design 0004 §2.5; ADR 0001 §6;
//! user cost-tracking rules: WARN path + hard HALT).
//!
//! Every paid solve must pass through this meter:
//!   1. [`CostMeterHandle::try_reserve`] BEFORE calling the provider — fails with
//!      [`super::CaptchaError::CostCapExceeded`] if the estimated cost would push
//!      the cumulative session total over the cap. This is the **hard halt**.
//!   2. After a successful solve, the reservation is the actual spend (these
//!      providers bill per solve at a known unit price; we reserve the estimate
//!      up front so concurrent solves cannot collectively blow the cap).
//!
//! The meter is cheap, `Clone`-able (it's an `Arc` inside), and `Send + Sync` so
//! a single session-wide meter can be shared across all solver calls. No
//! secrets pass through it — it records only USD amounts.

use std::sync::Arc;

use parking_lot::Mutex;

/// User cost-tracking rule thresholds. The meter emits a WARN once cumulative
/// spend crosses [`WARN_THRESHOLD_USD`]; the configured session cap is the hard
/// halt. (ADR 0001 §6 / cost-tracking BUDGET GUARD.)
pub const WARN_THRESHOLD_USD: f64 = 5.0;

#[derive(Debug)]
struct Inner {
    /// Hard per-session cap. A reservation that would exceed this is refused.
    cap_usd: f64,
    /// Cumulative reserved/spent USD this session.
    spent_usd: f64,
    /// Whether we've already emitted the one-time WARN crossing.
    warned: bool,
}

/// Session cost meter. Construct once per session; share the [`CostMeterHandle`]
/// (via [`CostMeter::handle`]) with every solver.
#[derive(Debug)]
pub struct CostMeter {
    inner: Arc<Mutex<Inner>>,
}

/// A cloneable handle to a [`CostMeter`]. This is what solvers hold.
#[derive(Clone, Debug)]
pub struct CostMeterHandle {
    inner: Arc<Mutex<Inner>>,
}

impl CostMeter {
    /// Create a meter with the given hard session cap (USD). A non-positive cap
    /// is clamped to 0.0, which makes every paid reservation fail — that is the
    /// safe default for a misconfigured cap (the config layer rejects `<= 0`
    /// when the solver is enabled, so this clamp is belt-and-suspenders).
    pub fn new(cap_usd: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                cap_usd: cap_usd.max(0.0),
                spent_usd: 0.0,
                warned: false,
            })),
        }
    }

    /// A shareable handle to this meter.
    pub fn handle(&self) -> CostMeterHandle {
        CostMeterHandle { inner: Arc::clone(&self.inner) }
    }

    /// Cumulative spend so far (USD).
    pub fn spent_usd(&self) -> f64 {
        self.inner.lock().spent_usd
    }
}

impl CostMeterHandle {
    /// Reserve `estimated_usd` against the session cap BEFORE making a paid call.
    ///
    /// On success the amount is added to the cumulative total and `Ok(())` is
    /// returned. If the reservation would push the cumulative total **over** the
    /// cap, nothing is recorded and [`super::CaptchaError::CostCapExceeded`] is
    /// returned — the hard halt. A non-positive estimate is treated as 0 (free).
    pub fn try_reserve(&self, estimated_usd: f64) -> Result<(), super::CaptchaError> {
        let est = estimated_usd.max(0.0);
        let mut g = self.inner.lock();
        let next = g.spent_usd + est;
        // Strictly greater-than the cap is refused; spending exactly up to the
        // cap is allowed.
        if next > g.cap_usd {
            tracing::warn!(
                cap_usd = g.cap_usd,
                spent_usd = g.spent_usd,
                attempted_usd = est,
                "captcha cost cap reached — HALTING paid solve (cost-tracking BUDGET GUARD)"
            );
            return Err(super::CaptchaError::CostCapExceeded);
        }
        g.spent_usd = next;
        // One-time WARN when cumulative spend crosses the warn threshold (user
        // rule: warn > $5). No secrets — USD amounts only.
        if !g.warned && g.spent_usd >= WARN_THRESHOLD_USD {
            g.warned = true;
            tracing::warn!(
                spent_usd = g.spent_usd,
                cap_usd = g.cap_usd,
                "captcha cumulative session spend crossed ${WARN_THRESHOLD_USD} (cost-tracking WARN)"
            );
        }
        // Per-solve + cumulative metering event (Design 0004 §2.5 / ADR 0001
        // C5). No secrets — provider/kind are attached by the caller's own span.
        tracing::debug!(
            reserved_usd = est,
            cumulative_usd = g.spent_usd,
            cap_usd = g.cap_usd,
            "captcha cost reserved"
        );
        Ok(())
    }

    /// Refund a previously reserved amount (e.g. the provider call failed before
    /// billing). Never drives the total below zero.
    pub fn refund(&self, usd: f64) {
        let amt = usd.max(0.0);
        let mut g = self.inner.lock();
        g.spent_usd = (g.spent_usd - amt).max(0.0);
    }

    /// Cumulative spend so far (USD).
    pub fn spent_usd(&self) -> f64 {
        self.inner.lock().spent_usd
    }

    /// Remaining headroom before the cap (USD, never negative).
    pub fn remaining_usd(&self) -> f64 {
        let g = self.inner.lock();
        (g.cap_usd - g.spent_usd).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_up_to_cap_then_halts() {
        let m = CostMeter::new(1.0);
        let h = m.handle();
        // Three solves at $0.30 each = $0.90 — all fit under $1.00.
        assert!(h.try_reserve(0.30).is_ok());
        assert!(h.try_reserve(0.30).is_ok());
        assert!(h.try_reserve(0.30).is_ok());
        assert!((m.spent_usd() - 0.90).abs() < 1e-9);
        // Fourth would be $1.20 > $1.00 → hard halt, and nothing recorded.
        let err = h.try_reserve(0.30).unwrap_err();
        assert!(matches!(err, super::super::CaptchaError::CostCapExceeded));
        assert!((m.spent_usd() - 0.90).abs() < 1e-9, "refused reservation must not be recorded");
    }

    #[test]
    fn exact_cap_is_allowed() {
        let m = CostMeter::new(1.0);
        let h = m.handle();
        assert!(h.try_reserve(1.0).is_ok());
        assert!((m.spent_usd() - 1.0).abs() < 1e-9);
        // Any further positive reservation now exceeds.
        assert!(h.try_reserve(0.01).is_err());
    }

    #[test]
    fn zero_or_negative_estimate_is_free() {
        let m = CostMeter::new(0.50);
        let h = m.handle();
        assert!(h.try_reserve(0.0).is_ok());
        assert!(h.try_reserve(-5.0).is_ok());
        assert_eq!(m.spent_usd(), 0.0);
    }

    #[test]
    fn nonpositive_cap_refuses_all_paid_reservations() {
        let m = CostMeter::new(0.0);
        let h = m.handle();
        assert!(h.try_reserve(0.0).is_ok()); // free is fine
        assert!(h.try_reserve(0.01).is_err()); // any spend exceeds a 0 cap
    }

    #[test]
    fn refund_restores_headroom() {
        let m = CostMeter::new(1.0);
        let h = m.handle();
        assert!(h.try_reserve(0.80).is_ok());
        assert!((h.remaining_usd() - 0.20).abs() < 1e-9);
        h.refund(0.50);
        assert!((h.remaining_usd() - 0.70).abs() < 1e-9);
        // Refund never goes below zero.
        h.refund(100.0);
        assert_eq!(m.spent_usd(), 0.0);
    }

    #[test]
    fn handles_share_one_total() {
        let m = CostMeter::new(1.0);
        let a = m.handle();
        let b = m.handle();
        assert!(a.try_reserve(0.60).is_ok());
        // b sees a's spend — concurrent solvers cannot collectively blow the cap.
        assert!(b.try_reserve(0.60).is_err());
        assert!(b.try_reserve(0.40).is_ok());
    }
}
