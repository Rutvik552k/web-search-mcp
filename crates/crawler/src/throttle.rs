use dashmap::DashMap;
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;
use std::sync::Arc;

type DomainLimiter = RateLimiter<
    governor::state::NotKeyed,
    governor::state::InMemoryState,
    governor::clock::DefaultClock,
>;

/// Per-domain rate throttler.
///
/// Enforces requests-per-second limits per domain.
/// Auto-creates limiter on first request to a domain.
pub struct Throttle {
    limiters: DashMap<String, Arc<DomainLimiter>>,
    default_rps: f64,
}

impl Throttle {
    pub fn new(default_rps: f64) -> Self {
        Self {
            limiters: DashMap::new(),
            default_rps: default_rps.max(0.1),
        }
    }

    /// Wait until the domain's rate limiter allows a request.
    pub async fn wait(&self, domain: &str) {
        let limiter = self.get_or_create(domain);
        limiter.until_ready().await;
    }

    /// Set a custom rate limit for a domain (e.g., from robots.txt Crawl-delay).
    pub fn set_domain_rps(&self, domain: &str, rps: f64) {
        let rps = rps.max(0.1);
        let quota = Self::rps_to_quota(rps);
        let limiter = RateLimiter::direct(quota);
        self.limiters.insert(domain.to_string(), Arc::new(limiter));
    }

    /// Temporarily increase backoff for a domain (e.g., after 429).
    pub fn apply_backoff(&self, domain: &str) {
        // Reduce rate to 1 request per 10 seconds
        self.set_domain_rps(domain, 0.1);
    }

    fn get_or_create(&self, domain: &str) -> Arc<DomainLimiter> {
        self.limiters
            .entry(domain.to_string())
            .or_insert_with(|| {
                let quota = Self::rps_to_quota(self.default_rps);
                Arc::new(RateLimiter::direct(quota))
            })
            .value()
            .clone()
    }

    fn rps_to_quota(rps: f64) -> Quota {
        if rps >= 1.0 {
            Quota::per_second(NonZeroU32::new(rps as u32).unwrap_or(NonZeroU32::new(1).unwrap()))
        } else {
            // For sub-1 RPS, use per-minute or longer intervals
            let per_min = (rps * 60.0).ceil() as u32;
            Quota::per_minute(NonZeroU32::new(per_min.max(1)).unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn throttle_allows_requests() {
        let throttle = Throttle::new(100.0); // very high RPS for test
        // Should not block
        throttle.wait("example.com").await;
        throttle.wait("example.com").await;
    }

    #[test]
    fn custom_domain_rps() {
        let throttle = Throttle::new(2.0);
        throttle.set_domain_rps("slow.com", 0.5);
        // Just verify it doesn't panic
        assert!(throttle.limiters.contains_key("slow.com"));
    }
}
