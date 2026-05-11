use dashmap::DashMap;
use std::time::Duration;

/// Parsed robots.txt rules for a domain.
#[derive(Debug, Clone)]
pub struct RobotsRules {
    pub disallow: Vec<String>,
    pub allow: Vec<String>,
    pub crawl_delay: Option<Duration>,
    pub sitemaps: Vec<String>,
}

impl RobotsRules {
    /// Check if a path is allowed by these rules.
    pub fn is_allowed(&self, path: &str) -> bool {
        // Allow rules take precedence over disallow for same-length prefix
        // Check most specific (longest) match
        let mut best_disallow = 0;
        let mut best_allow = 0;

        for rule in &self.disallow {
            if path_matches(path, rule) && rule.len() > best_disallow {
                best_disallow = rule.len();
            }
        }

        for rule in &self.allow {
            if path_matches(path, rule) && rule.len() > best_allow {
                best_allow = rule.len();
            }
        }

        // If no disallow matched, allowed
        if best_disallow == 0 {
            return true;
        }

        // Allow wins if more specific (longer or equal match)
        best_allow >= best_disallow
    }
}

/// In-memory cache of robots.txt rules per domain.
pub struct RobotsCache {
    cache: DashMap<String, RobotsRules>,
    user_agent: String,
}

impl RobotsCache {
    pub fn new(user_agent: &str) -> Self {
        Self {
            cache: DashMap::new(),
            user_agent: user_agent.to_lowercase(),
        }
    }

    /// Parse and cache robots.txt content for a domain.
    pub fn parse_and_cache(&self, domain: &str, robots_txt: &str) {
        let rules = parse_robots_txt(robots_txt, &self.user_agent);
        self.cache.insert(domain.to_string(), rules);
    }

    /// Check if a URL is allowed. Returns true if no robots.txt cached for domain.
    pub fn is_allowed(&self, url: &str) -> bool {
        let parsed = match url::Url::parse(url) {
            Ok(u) => u,
            Err(_) => return true,
        };

        let domain = parsed.host_str().unwrap_or("");
        let path = parsed.path();

        match self.cache.get(domain) {
            Some(rules) => rules.is_allowed(path),
            None => true, // no robots.txt = everything allowed
        }
    }

    /// Get crawl delay for domain (if specified in robots.txt).
    pub fn crawl_delay(&self, domain: &str) -> Option<Duration> {
        self.cache.get(domain).and_then(|r| r.crawl_delay)
    }

    /// Get sitemaps for domain.
    pub fn sitemaps(&self, domain: &str) -> Vec<String> {
        self.cache
            .get(domain)
            .map(|r| r.sitemaps.clone())
            .unwrap_or_default()
    }
}

/// Parse robots.txt content for a specific user agent.
fn parse_robots_txt(content: &str, user_agent: &str) -> RobotsRules {
    let mut rules = RobotsRules {
        disallow: vec![],
        allow: vec![],
        crawl_delay: None,
        sitemaps: vec![],
    };

    let mut in_matching_group = false;
    let mut _in_wildcard_group = false;
    let mut found_specific = false;

    for line in content.lines() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse directive
        let (key, value) = match line.split_once(':') {
            Some((k, v)) => (k.trim().to_lowercase(), v.trim().to_string()),
            None => continue,
        };

        match key.as_str() {
            "user-agent" => {
                let ua = value.to_lowercase();
                if ua == "*" {
                    _in_wildcard_group = true;
                    in_matching_group = !found_specific;
                } else if user_agent.contains(&ua) || ua.contains(user_agent) {
                    in_matching_group = true;
                    found_specific = true;
                    // If we found specific match, reset rules (ignore wildcard)
                    rules.disallow.clear();
                    rules.allow.clear();
                    rules.crawl_delay = None;
                } else {
                    in_matching_group = false;
                    _in_wildcard_group = false;
                }
            }
            "disallow" if in_matching_group && !value.is_empty() => {
                rules.disallow.push(value);
            }
            "allow" if in_matching_group && !value.is_empty() => {
                rules.allow.push(value);
            }
            "crawl-delay" if in_matching_group => {
                if let Ok(secs) = value.parse::<f64>() {
                    rules.crawl_delay = Some(Duration::from_secs_f64(secs));
                }
            }
            "sitemap" => {
                // Sitemaps are global, not per-agent
                rules.sitemaps.push(value);
            }
            _ => {}
        }
    }

    rules
}

/// Check if a path matches a robots.txt pattern.
fn path_matches(path: &str, pattern: &str) -> bool {
    if pattern == "/" {
        return true;
    }

    // Handle wildcard *
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        let mut pos = 0;
        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }
            match path[pos..].find(part) {
                Some(found) => {
                    if i == 0 && found != 0 {
                        return false; // first part must match from start
                    }
                    pos += found + part.len();
                }
                None => return false,
            }
        }
        return true;
    }

    // Handle $ anchor
    if pattern.ends_with('$') {
        return path == &pattern[..pattern.len() - 1];
    }

    // Simple prefix match
    path.starts_with(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_robots() {
        let txt = "User-agent: *\nDisallow: /private/\nDisallow: /admin/\nAllow: /admin/public/\n";
        let rules = parse_robots_txt(txt, "mybot");
        assert_eq!(rules.disallow.len(), 2);
        assert_eq!(rules.allow.len(), 1);
    }

    #[test]
    fn disallow_blocks_path() {
        let txt = "User-agent: *\nDisallow: /private/\n";
        let rules = parse_robots_txt(txt, "mybot");
        assert!(!rules.is_allowed("/private/secret"));
        assert!(rules.is_allowed("/public/page"));
    }

    #[test]
    fn allow_overrides_disallow() {
        let txt = "User-agent: *\nDisallow: /admin/\nAllow: /admin/public/\n";
        let rules = parse_robots_txt(txt, "mybot");
        assert!(!rules.is_allowed("/admin/secret"));
        assert!(rules.is_allowed("/admin/public/page"));
    }

    #[test]
    fn crawl_delay_parsed() {
        let txt = "User-agent: *\nCrawl-delay: 2.5\n";
        let rules = parse_robots_txt(txt, "mybot");
        assert_eq!(rules.crawl_delay, Some(Duration::from_secs_f64(2.5)));
    }

    #[test]
    fn sitemaps_extracted() {
        let txt = "Sitemap: https://example.com/sitemap.xml\nSitemap: https://example.com/sitemap2.xml\n";
        let rules = parse_robots_txt(txt, "mybot");
        assert_eq!(rules.sitemaps.len(), 2);
    }

    #[test]
    fn robots_cache_works() {
        let cache = RobotsCache::new("mybot");
        cache.parse_and_cache("example.com", "User-agent: *\nDisallow: /secret/\n");
        assert!(!cache.is_allowed("https://example.com/secret/page"));
        assert!(cache.is_allowed("https://example.com/public/page"));
        // Unknown domain = allowed
        assert!(cache.is_allowed("https://other.com/anything"));
    }

    #[test]
    fn wildcard_pattern_matching() {
        assert!(path_matches("/search?q=test", "/search"));
        assert!(path_matches("/foo/bar.html", "/*.html"));
        assert!(!path_matches("/foo/bar.json", "/*.html"));
    }
}
