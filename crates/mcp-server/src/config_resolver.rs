//! Config-path resolution + startup validation (ADR 0004 §7/§8, Addendum A.6.4).
//!
//! Both functions here are PURE: they take the args/env/config as parameters and
//! perform no I/O of their own, so each precedence branch and validation rule is
//! a no-server-boot unit test (mirrors the `decide_gate` pattern in
//! `crawler::captcha`).

use std::path::PathBuf;

use web_search_common::config::Config;

/// Where the config came from. The EXPLICIT-vs-DEFAULT distinction drives the
/// fail-fast rule (ADR 0004 §7): an explicitly-named path that fails to load is
/// an operator error and must NOT silently fall back to defaults; a missing
/// DEFAULT file is the supported keyless standalone case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    /// Operator named this path explicitly via `--config <path>` or
    /// `WEB_SEARCH_MCP_CONFIG`. Missing/unparseable => fail fast.
    Explicit(PathBuf),
    /// No path named; the conventional `config/default.toml` relative to CWD.
    /// Missing => fall back to embedded `Config::default()` (current behavior).
    Default(PathBuf),
}

impl ConfigSource {
    /// The path this source points at, for logging/loading.
    pub fn path(&self) -> &std::path::Path {
        match self {
            ConfigSource::Explicit(p) | ConfigSource::Default(p) => p,
        }
    }

    /// True iff the operator named this path (CLI/env) — drives fail-fast.
    pub fn is_explicit(&self) -> bool {
        matches!(self, ConfigSource::Explicit(_))
    }
}

/// Resolve the config path from CLI args + an injected env lookup (PURE).
///
/// Resolution order (ADR 0004 §7):
/// 1. `--config <path>` arg            → `Explicit`
/// 2. `WEB_SEARCH_MCP_CONFIG` env var  → `Explicit`
/// 3. `config/default.toml` (default)  → `Default`
///
/// `args` is the full argv (including argv[0]); `env` is any closure mapping a
/// var name to its value (inject a fake in tests; pass
/// `|k| std::env::var(k).ok()` at runtime).
pub fn resolve_config_path(args: &[String], env: impl Fn(&str) -> Option<String>) -> ConfigSource {
    // 1. --config <path>  (also accept --config=<path>)
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--config" {
            if let Some(path) = iter.next() {
                if !path.trim().is_empty() {
                    return ConfigSource::Explicit(PathBuf::from(path));
                }
            }
        } else if let Some(rest) = arg.strip_prefix("--config=") {
            if !rest.trim().is_empty() {
                return ConfigSource::Explicit(PathBuf::from(rest));
            }
        }
    }

    // 2. WEB_SEARCH_MCP_CONFIG env var
    if let Some(path) = env("WEB_SEARCH_MCP_CONFIG") {
        if !path.trim().is_empty() {
            return ConfigSource::Explicit(PathBuf::from(path));
        }
    }

    // 3. conventional default path
    ConfigSource::Default(PathBuf::from("config/default.toml"))
}

/// Accepted keyed-search providers (ADR 0004 §8). `None`/empty `search_api_provider`
/// is valid (means "no keyed source"); a *set* value must be one of these.
const ACCEPTED_PROVIDERS: &[&str] = &["tavily", "serper"];

/// Pure startup validation for the self-contained search config (ADR 0004 §8).
/// Mirrors the `decide_gate` shape: no I/O except the injected env lookup.
///
/// Rules:
/// - `search_source == "tavily"` (pinned) but `search_api_key_env` is None OR
///   names an unset/empty env var → Err (fail-fast, actionable).
/// - `search_api_provider` set to an unknown value → Err listing accepted values.
/// - `search_source == "auto"` + no key → Ok (the supported keyless default; the
///   INFO log is emitted by the CALLER, not this pure fn).
///
/// `env` is injected (pass `|k| std::env::var(k).ok()` at runtime) so each branch
/// is unit-testable without mutating the process environment.
pub fn validate_search_config(
    cfg: &Config,
    env: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<()> {
    let c = &cfg.crawler;

    // Unknown provider (when one is named) is always an error.
    if let Some(provider) = c.search_api_provider.as_deref() {
        let p = provider.trim();
        if !p.is_empty() && !ACCEPTED_PROVIDERS.contains(&p.to_ascii_lowercase().as_str()) {
            anyhow::bail!(
                "unknown search_api_provider {p:?} (accepted: {})",
                ACCEPTED_PROVIDERS.join(", ")
            );
        }
    }

    // A pinned keyed source that cannot run is a fail-fast (explicit request for
    // a source with no usable key).
    if c.search_source.trim().eq_ignore_ascii_case("tavily") {
        let key_env = c
            .search_api_key_env
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let key_env = match key_env {
            Some(name) => name,
            None => anyhow::bail!(
                "search_source=\"tavily\" but search_api_key_env is unset \
                 (it must name the env var holding the Tavily API key)"
            ),
        };
        let key_set = env(key_env).map(|v| !v.trim().is_empty()).unwrap_or(false);
        if !key_set {
            anyhow::bail!(
                "search_source=\"tavily\" but env var {key_env:?} is unset or empty \
                 (it must hold the Tavily API key)"
            );
        }
    }

    // search_source=="auto" with no key is the supported default → Ok.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // ---- resolve_config_path: precedence branches ----

    #[test]
    fn cli_config_arg_is_explicit_and_wins() {
        let a = args(&["web-search-mcp", "--config", "/etc/wsm/prod.toml"]);
        // Even with the env set, the CLI arg wins.
        let env = |k: &str| {
            (k == "WEB_SEARCH_MCP_CONFIG").then(|| "/env/path.toml".to_string())
        };
        let got = resolve_config_path(&a, env);
        assert_eq!(got, ConfigSource::Explicit(PathBuf::from("/etc/wsm/prod.toml")));
        assert!(got.is_explicit());
    }

    #[test]
    fn cli_config_eq_form_is_explicit() {
        let a = args(&["web-search-mcp", "--config=/etc/wsm/prod.toml"]);
        let got = resolve_config_path(&a, no_env);
        assert_eq!(got, ConfigSource::Explicit(PathBuf::from("/etc/wsm/prod.toml")));
    }

    #[test]
    fn env_var_is_explicit_when_no_cli_arg() {
        let a = args(&["web-search-mcp"]);
        let env = |k: &str| {
            (k == "WEB_SEARCH_MCP_CONFIG").then(|| "/env/path.toml".to_string())
        };
        let got = resolve_config_path(&a, env);
        assert_eq!(got, ConfigSource::Explicit(PathBuf::from("/env/path.toml")));
        assert!(got.is_explicit());
    }

    #[test]
    fn default_path_when_no_arg_no_env() {
        let a = args(&["web-search-mcp"]);
        let got = resolve_config_path(&a, no_env);
        assert_eq!(got, ConfigSource::Default(PathBuf::from("config/default.toml")));
        assert!(!got.is_explicit());
    }

    #[test]
    fn empty_cli_value_does_not_become_explicit() {
        // `--config` with an empty value falls through to env/default rather than
        // resolving to an empty path.
        let a = args(&["web-search-mcp", "--config", "   "]);
        let got = resolve_config_path(&a, no_env);
        assert_eq!(got, ConfigSource::Default(PathBuf::from("config/default.toml")));
    }

    #[test]
    fn empty_env_value_falls_through_to_default() {
        let a = args(&["web-search-mcp"]);
        let env = |k: &str| (k == "WEB_SEARCH_MCP_CONFIG").then(String::new);
        let got = resolve_config_path(&a, env);
        assert_eq!(got, ConfigSource::Default(PathBuf::from("config/default.toml")));
    }

    // ---- validate_search_config: each branch ----

    fn cfg_with(source: &str, provider: Option<&str>, key_env: Option<&str>) -> Config {
        let mut cfg = Config::default();
        cfg.crawler.search_source = source.to_string();
        cfg.crawler.search_api_provider = provider.map(String::from);
        cfg.crawler.search_api_key_env = key_env.map(String::from);
        cfg
    }

    #[test]
    fn auto_with_no_key_is_ok() {
        let cfg = cfg_with("auto", None, None);
        assert!(validate_search_config(&cfg, no_env).is_ok());
    }

    #[test]
    fn tavily_pinned_without_key_env_fails_fast() {
        let cfg = cfg_with("tavily", Some("tavily"), None);
        assert!(validate_search_config(&cfg, no_env).is_err());
    }

    #[test]
    fn tavily_pinned_with_unset_env_var_fails_fast() {
        let cfg = cfg_with("tavily", Some("tavily"), Some("WSM_TAVILY_KEY"));
        // env closure returns None for everything → key unset.
        assert!(validate_search_config(&cfg, no_env).is_err());
    }

    #[test]
    fn tavily_pinned_with_set_env_var_is_ok() {
        let cfg = cfg_with("tavily", Some("tavily"), Some("WSM_TAVILY_KEY"));
        let env = |k: &str| (k == "WSM_TAVILY_KEY").then(|| "secret-value".to_string());
        assert!(validate_search_config(&cfg, env).is_ok());
    }

    #[test]
    fn tavily_pinned_with_empty_env_value_fails_fast() {
        let cfg = cfg_with("tavily", Some("tavily"), Some("WSM_TAVILY_KEY"));
        let env = |k: &str| (k == "WSM_TAVILY_KEY").then(String::new);
        assert!(validate_search_config(&cfg, env).is_err());
    }

    #[test]
    fn unknown_provider_fails_fast_listing_accepted() {
        let cfg = cfg_with("auto", Some("bing-secret-api"), None);
        let err = validate_search_config(&cfg, no_env).unwrap_err().to_string();
        assert!(err.contains("unknown search_api_provider"), "got: {err}");
        assert!(err.contains("tavily"), "error should list accepted providers: {err}");
    }

    #[test]
    fn known_provider_serper_is_ok() {
        let cfg = cfg_with("auto", Some("serper"), None);
        assert!(validate_search_config(&cfg, no_env).is_ok());
    }

    #[test]
    fn empty_provider_is_treated_as_unset() {
        let cfg = cfg_with("auto", Some(""), None);
        assert!(validate_search_config(&cfg, no_env).is_ok());
    }
}
