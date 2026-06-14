//! SSRF guard for the crawler fetch path (ADR 0004 §13 Addendum A.3).
//!
//! Every search source (`SearXNG results[].url`, SERP-HTML parser output, Tavily
//! `results[].url`) emits attacker-influenceable URLs that flow straight into
//! `client.get(url)`. Without a guard, an `https://` seed can 302 to
//! `http://169.254.169.254/…` (cloud metadata) or `http://127.0.0.1:<port>`, and
//! a hostile A-record can point a benign-looking hostname at a link-local IP
//! (DNS rebinding). This module supplies:
//!
//!   * [`scheme_is_allowed`] — http(s)-only scheme allowlist (pure).
//!   * [`ip_is_denied`]      — RFC1918 / loopback / link-local / ULA / unspecified
//!                             deny predicate (pure, unit-tested over literals).
//!   * [`SsrfResolver`]      — a `reqwest::dns::Resolve` impl wired via
//!                             `ClientBuilder::dns_resolver`. It runs for the
//!                             initial host AND every redirect hop (reqwest
//!                             resolves per connection), so filtering denied IPs
//!                             here gives **resolve-then-pin + per-hop
//!                             revalidation for free**: the IP the deny check sees
//!                             is exactly the IP the socket connects to (closes
//!                             the A.3 TOCTOU/rebinding re-gate note).
//!   * [`SsrfPolicy`]        — strict in production; a test-only permissive mode
//!                             (`allow_loopback`) lets local mock-server tests opt
//!                             in. There is intentionally NO production config flag
//!                             that disables SSRF (that would be a footgun).
//!
//! Verified against reqwest 0.12.28 (Cargo.lock): `reqwest::dns::Resolve` is
//! `fn resolve(&self, name: Name) -> Resolving` where
//! `Resolving = Pin<Box<dyn Future<Output = Result<Addrs, BoxError>> + Send>>`,
//! `Addrs = Box<dyn Iterator<Item = SocketAddr> + Send>`, `Name::as_str() -> &str`;
//! the builder method is `dns_resolver<R: Resolve + 'static>(Arc<R>)`. The private
//! `Name.0` (`HyperName`) is not accessible, so resolution is delegated to
//! `tokio::net::lookup_host` rather than the GAI resolver, then filtered.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

/// SSRF policy for the fetch client. Strict by default (production). The only
/// relaxation is `allow_loopback`, reachable ONLY from test constructors — never
/// from any production config field (ADR 0004 A.3: do not add a prod SSRF kill
/// switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsrfPolicy {
    /// When true, loopback (`127.0.0.0/8`, `::1`) is allowed so local mock-server
    /// tests can dial `127.0.0.1`. RFC1918 / link-local / ULA / metadata stay
    /// denied even in permissive mode — only loopback is relaxed.
    pub allow_loopback: bool,
}

impl SsrfPolicy {
    /// Production policy: deny ALL internal address ranges incl. loopback.
    pub const fn strict() -> Self {
        Self { allow_loopback: false }
    }

    /// Test-only policy that permits loopback (local mock servers). Not used on
    /// any production path; `Fetcher::new` always uses [`SsrfPolicy::strict`].
    #[cfg(test)]
    pub const fn permissive_loopback() -> Self {
        Self { allow_loopback: true }
    }

    /// Whether an already-resolved IP is denied under this policy.
    pub fn ip_denied(&self, ip: IpAddr) -> bool {
        if self.allow_loopback && ip.is_loopback() {
            return false;
        }
        ip_is_denied(ip)
    }
}

/// Pure scheme allowlist: only `http` and `https` may be fetched. Everything else
/// (`file`, `ftp`, `gopher`, `data`, `dict`, …) is refused before any socket.
pub fn scheme_is_allowed(url: &str) -> bool {
    match url::Url::parse(url) {
        Ok(u) => matches!(u.scheme(), "http" | "https"),
        Err(_) => false,
    }
}

/// If the URL's host is an IP literal (`http://169.254.169.254/`,
/// `http://[::1]/`), return that `IpAddr` so the caller can deny it synchronously
/// without DNS. Returns `None` for hostnames (those go through the resolver) or
/// unparseable URLs.
pub fn host_ip_literal(url: &str) -> Option<IpAddr> {
    let u = url::Url::parse(url).ok()?;
    match u.host()? {
        url::Host::Ipv4(v4) => Some(IpAddr::V4(v4)),
        url::Host::Ipv6(v6) => Some(IpAddr::V6(v6)),
        url::Host::Domain(_) => None,
    }
}

/// Pure deny predicate over a single resolved IP. Denies:
///   * loopback           — `127.0.0.0/8`, `::1`
///   * RFC1918 private    — `10/8`, `172.16/12`, `192.168/16`
///   * link-local         — `169.254.0.0/16` (incl. metadata `169.254.169.254`),
///                          `fe80::/10`
///   * ULA                — `fc00::/7`
///   * unspecified        — `0.0.0.0`, `::`
///   * IPv4-mapped IPv6   — re-checked against the embedded v4 address
///
/// Public/global IPs return `false` (allowed).
pub fn ip_is_denied(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_is_denied(v4),
        IpAddr::V6(v6) => {
            // Unwrap IPv4-mapped (::ffff:a.b.c.d) so a v4 internal target cannot
            // be smuggled through a v6 literal. (`to_ipv4_mapped` is the
            // non-deprecated, mapped-only form — `to_ipv4` is deprecated.)
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return v4_is_denied(mapped);
            }
            v6_is_denied(v6)
        }
    }
}

fn v4_is_denied(ip: Ipv4Addr) -> bool {
    ip.is_loopback()            // 127.0.0.0/8
        || ip.is_private()      // 10/8, 172.16/12, 192.168/16
        || ip.is_link_local()   // 169.254.0.0/16 (incl. 169.254.169.254 metadata)
        || ip.is_unspecified()  // 0.0.0.0
        || ip.is_broadcast()    // 255.255.255.255
        // Carrier-grade NAT 100.64.0.0/10 — internal-ish, deny to be safe.
        || (ip.octets()[0] == 100 && (ip.octets()[1] & 0xc0) == 64)
}

fn v6_is_denied(ip: Ipv6Addr) -> bool {
    // FIX #2: before the pure-v6 ranges, unwrap the v6 forms that EMBED a v4
    // address so a v4 internal target cannot be smuggled through them. Each
    // extracts the embedded v4 and re-runs the v4 deny predicate.
    if let Some(v4) = embedded_v4(ip) {
        return v4_is_denied(v4);
    }
    ip.is_loopback()                 // ::1
        || ip.is_unspecified()       // ::
        || is_unique_local(ip)       // fc00::/7 (ULA)
        || is_unicast_link_local(ip) // fe80::/10
}

/// Extract a v4 address embedded in a v6 literal, for the transition/compat
/// forms that carry a v4 target (FIX #2). Returns `None` for ordinary v6.
/// (`::ffff:a.b.c.d` IPv4-MAPPED is handled separately in [`ip_is_denied`].)
///
///   * IPv4-COMPATIBLE `::a.b.c.d` — high 6 segments zero, low 32 bits are v4.
///     (Excludes `::` and `::1`, which are not embedded-v4 targets.)
///   * NAT64 well-known `64:ff9b::/96` — embedded v4 in the low 32 bits.
///   * 6to4 `2002::/16` — embedded v4 in segments[1..=2].
fn embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let seg = ip.segments();
    let low_v4 = || Ipv4Addr::new(
        (seg[6] >> 8) as u8, (seg[6] & 0xff) as u8,
        (seg[7] >> 8) as u8, (seg[7] & 0xff) as u8,
    );

    // NAT64 64:ff9b::/96 — embedded v4 in the trailing 32 bits.
    if seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2..6].iter().all(|&s| s == 0) {
        return Some(low_v4());
    }

    // 6to4 2002::/16 — v4 in segments[1..=2].
    if seg[0] == 0x2002 {
        return Some(Ipv4Addr::new(
            (seg[1] >> 8) as u8, (seg[1] & 0xff) as u8,
            (seg[2] >> 8) as u8, (seg[2] & 0xff) as u8,
        ));
    }

    // IPv4-COMPATIBLE ::a.b.c.d — high 6 segments zero. Exclude ::(unspecified)
    // and ::1(loopback), which the pure-v6 checks already cover and are not
    // "embedded v4 targets".
    if seg[0..6].iter().all(|&s| s == 0) {
        let v4 = low_v4();
        if !v4.is_unspecified() && low_v4() != Ipv4Addr::new(0, 0, 0, 1) {
            return Some(v4);
        }
    }

    None
}

/// `fc00::/7` — unique local addresses.
fn is_unique_local(ip: Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00
}

/// `fe80::/10` — link-local unicast.
fn is_unicast_link_local(ip: Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80
}

/// Async SSRF pre-check for a navigation target that does NOT go through the
/// reqwest [`SsrfResolver`] — notably the headless-browser path, where Chrome
/// uses its OWN network stack and the reqwest `dns_resolver` hook never runs
/// (FIX #1). Call this immediately before any `page.goto(url)` / `new_page(url)`
/// navigation to an untrusted target.
///
/// Fails CLOSED, mirroring the reqwest path's deny semantics:
///   1. Non-http(s) scheme → denied ([`scheme_is_allowed`]).
///   2. Host is an IP literal → denied iff [`SsrfPolicy::ip_denied`].
///   3. Host is a hostname → resolve via `tokio::net::lookup_host`; denied if ANY
///      resolved IP is denied, OR if it does not resolve at all (no allowed
///      address — same rule as [`SsrfResolver`]).
///
/// On denial returns the fixed-form [`SsrfDenied`] so no internal resolution
/// detail leaks (same constraint as the resolver). Reuses the SAME predicates
/// (`scheme_is_allowed`, `host_ip_literal`, `SsrfPolicy::ip_denied`) as the
/// reqwest path — the deny logic is defined once.
pub async fn precheck_navigation(url: &str, policy: SsrfPolicy) -> Result<(), SsrfDenied> {
    // 1. Scheme allowlist (also rejects unparseable URLs).
    if !scheme_is_allowed(url) {
        return Err(SsrfDenied("blocked: scheme not allowed"));
    }

    // 2. IP-literal host → decide synchronously, no DNS.
    if let Some(ip) = host_ip_literal(url) {
        if policy.ip_denied(ip) {
            return Err(SsrfDenied("blocked: internal address"));
        }
        return Ok(());
    }

    // 3. Hostname → resolve and deny if ANY resolved IP is denied (fail closed).
    let host = match url::Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_string)) {
        Some(h) => h,
        None => return Err(SsrfDenied("blocked: no host")),
    };
    let lookup = format!("{host}:0");
    let resolved = match tokio::net::lookup_host(lookup).await {
        Ok(it) => it,
        // Did not resolve at all → no allowed address (same as resolver).
        Err(_) => return Err(SsrfDenied("blocked: no allowed address")),
    };

    let mut saw_any = false;
    for sa in resolved {
        saw_any = true;
        if policy.ip_denied(sa.ip()) {
            // Fail closed: a single denied IP blocks the whole navigation so a
            // rebinding host that returns one internal A-record cannot slip.
            return Err(SsrfDenied("blocked: internal address"));
        }
    }
    if !saw_any {
        return Err(SsrfDenied("blocked: no allowed address"));
    }
    Ok(())
}

/// A `reqwest::dns::Resolve` that resolves via `tokio::net::lookup_host` and then
/// drops every denied IP. If, after filtering, NO addresses remain (i.e. the host
/// ONLY resolves to internal IPs), it returns an error so reqwest never opens a
/// socket to an internal address — for the initial host and for EVERY redirect
/// hop (reqwest re-resolves per connection). This is the resolve-then-pin /
/// per-hop revalidation seam from ADR 0004 A.3.
#[derive(Debug)]
pub struct SsrfResolver {
    policy: SsrfPolicy,
}

impl SsrfResolver {
    pub fn new(policy: SsrfPolicy) -> Self {
        Self { policy }
    }
}

impl Resolve for SsrfResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let policy = self.policy;
        let host = name.as_str().to_string();
        Box::pin(async move {
            // `Resolving`'s error type is reqwest's crate-private
            // `BoxError = Box<dyn Error + Send + Sync>`; we can build that boxed
            // type directly without naming the alias.
            type DnErr = Box<dyn std::error::Error + Send + Sync>;

            // Port is irrelevant for the deny decision; reqwest overrides the port
            // from the URL afterward. Use 0 — lookup_host needs host:port form.
            let lookup = format!("{host}:0");
            let resolved = tokio::net::lookup_host(lookup)
                .await
                .map_err(|e| Box::new(e) as DnErr)?;

            let mut allowed: Vec<SocketAddr> = Vec::new();
            let mut saw_denied = false;
            for sa in resolved {
                if policy.ip_denied(sa.ip()) {
                    saw_denied = true;
                    continue;
                }
                allowed.push(sa);
            }

            if allowed.is_empty() {
                // Either the host only resolves to internal IPs (rebinding /
                // metadata / loopback) or it didn't resolve at all. Refuse —
                // sanitized message, no internal resolution details leaked.
                let reason = if saw_denied {
                    "blocked: internal address"
                } else {
                    "blocked: no allowed address"
                };
                return Err(Box::new(SsrfDenied(reason)) as DnErr);
            }

            let addrs: Addrs = Box::new(allowed.into_iter());
            Ok(addrs)
        })
    }
}

/// Boxed error returned to reqwest when a host resolves only to denied IPs. The
/// `Display` text is the only thing that can surface to a caller via
/// `reqwest::Error` — kept to the fixed "blocked: internal address" form so no
/// internal resolution detail (the actual IP) leaks (ADR 0004 §13 constraint).
#[derive(Debug)]
pub struct SsrfDenied(pub &'static str);

impl std::fmt::Display for SsrfDenied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for SsrfDenied {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn ip(s: &str) -> IpAddr {
        IpAddr::from_str(s).unwrap()
    }

    #[test]
    fn ip_denied_loopback() {
        assert!(ip_is_denied(ip("127.0.0.1")));
        assert!(ip_is_denied(ip("127.255.255.254")));
        assert!(ip_is_denied(ip("::1")));
    }

    #[test]
    fn ip_denied_rfc1918() {
        assert!(ip_is_denied(ip("10.0.0.1")));
        assert!(ip_is_denied(ip("10.255.255.255")));
        assert!(ip_is_denied(ip("172.16.0.1")));
        assert!(ip_is_denied(ip("172.31.255.255")));
        assert!(ip_is_denied(ip("192.168.1.1")));
    }

    #[test]
    fn ip_denied_link_local_and_metadata() {
        assert!(ip_is_denied(ip("169.254.0.1")));
        // The cloud metadata endpoint specifically.
        assert!(ip_is_denied(ip("169.254.169.254")));
        assert!(ip_is_denied(ip("fe80::1")));
    }

    #[test]
    fn ip_denied_ula_and_unspecified() {
        assert!(ip_is_denied(ip("fc00::1")));
        assert!(ip_is_denied(ip("fd12:3456::1")));
        assert!(ip_is_denied(ip("0.0.0.0")));
        assert!(ip_is_denied(ip("::")));
    }

    #[test]
    fn ip_denied_ipv4_mapped_smuggling() {
        // ::ffff:169.254.169.254 must be caught via the embedded v4 address.
        assert!(ip_is_denied(ip("::ffff:169.254.169.254")));
        assert!(ip_is_denied(ip("::ffff:127.0.0.1")));
        assert!(ip_is_denied(ip("::ffff:10.0.0.1")));
    }

    #[test]
    fn public_ips_allowed() {
        assert!(!ip_is_denied(ip("1.1.1.1")));
        assert!(!ip_is_denied(ip("8.8.8.8")));
        assert!(!ip_is_denied(ip("93.184.216.34"))); // example.com
        assert!(!ip_is_denied(ip("2606:4700:4700::1111"))); // cloudflare v6
    }

    #[test]
    fn ip_denied_v6_embedded_v4_smuggling() {
        // FIX #2: v6 transition/compat forms that embed an internal v4 target.
        // IPv4-compatible ::a.b.c.d (high 6 segments zero).
        assert!(ip_is_denied(ip("::169.254.169.254")), "IPv4-compat metadata");
        assert!(ip_is_denied(ip("::127.0.0.1")), "IPv4-compat loopback");
        assert!(ip_is_denied(ip("::10.0.0.1")), "IPv4-compat RFC1918");
        // NAT64 well-known prefix 64:ff9b::/96 wrapping the metadata IP.
        assert!(ip_is_denied(ip("64:ff9b::a9fe:a9fe")), "NAT64 metadata");
        assert!(ip_is_denied(ip("64:ff9b::7f00:1")), "NAT64 loopback 127.0.0.1");
        // 6to4 2002::/16 wrapping the metadata IP (a9fe:a9fe == 169.254.169.254).
        assert!(ip_is_denied(ip("2002:a9fe:a9fe::")), "6to4 metadata");
        assert!(ip_is_denied(ip("2002:0a00:0001::")), "6to4 RFC1918 10.0.0.1");
    }

    #[test]
    fn ip_allowed_v6_embedded_v4_public_and_normal_v6() {
        // A normal public v6 is still allowed (not an embedded-v4 form).
        assert!(!ip_is_denied(ip("2606:4700::1")), "public v6 allowed");
        // NAT64 / 6to4 / IPv4-compat wrapping a PUBLIC v4 stays allowed.
        assert!(!ip_is_denied(ip("64:ff9b::8080:808")), "NAT64 public 8.8.8.8");
        assert!(!ip_is_denied(ip("2002:0808:0808::")), "6to4 public 8.8.8.8");
        assert!(!ip_is_denied(ip("::8.8.8.8")), "IPv4-compat public 8.8.8.8");
    }

    #[tokio::test]
    async fn precheck_navigation_rejects_internal_and_schemes() {
        let p = SsrfPolicy::strict();
        // Non-http(s) schemes.
        assert!(precheck_navigation("file:///etc/passwd", p).await.is_err());
        assert!(precheck_navigation("ftp://example.com/", p).await.is_err());
        // IP-literal internal targets — synchronous deny, no DNS.
        assert!(precheck_navigation("http://169.254.169.254/latest/meta-data/", p).await.is_err());
        assert!(precheck_navigation("http://127.0.0.1/admin", p).await.is_err());
        assert!(precheck_navigation("http://[::1]:8080/", p).await.is_err());
        assert!(precheck_navigation("http://10.0.0.5/", p).await.is_err());
        // v6 embedded-v4 smuggling literal also rejected pre-navigation.
        assert!(precheck_navigation("http://[::169.254.169.254]/", p).await.is_err());
        // Non-resolvable host → no allowed address → rejected (fail closed).
        assert!(precheck_navigation("http://denied.invalid/", p).await.is_err());
    }

    #[test]
    fn scheme_allowlist() {
        assert!(scheme_is_allowed("http://example.com/"));
        assert!(scheme_is_allowed("https://example.com/path?q=1"));
        assert!(!scheme_is_allowed("file:///etc/passwd"));
        assert!(!scheme_is_allowed("ftp://example.com/x"));
        assert!(!scheme_is_allowed("gopher://example.com/"));
        assert!(!scheme_is_allowed("data:text/plain,hi"));
        assert!(!scheme_is_allowed("not a url"));
    }

    #[test]
    fn policy_loopback_relaxation_is_loopback_only() {
        let strict = SsrfPolicy::strict();
        let permissive = SsrfPolicy::permissive_loopback();

        // strict denies loopback; permissive allows it.
        assert!(strict.ip_denied(ip("127.0.0.1")));
        assert!(!permissive.ip_denied(ip("127.0.0.1")));
        assert!(!permissive.ip_denied(ip("::1")));

        // permissive still denies metadata / RFC1918 / link-local.
        assert!(permissive.ip_denied(ip("169.254.169.254")));
        assert!(permissive.ip_denied(ip("10.0.0.1")));
        assert!(permissive.ip_denied(ip("192.168.0.1")));
        assert!(permissive.ip_denied(ip("fe80::1")));
    }
}
