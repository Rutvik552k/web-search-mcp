use web_search_common::models::SourceTier;

/// Determine source tier for a domain.
///
/// Tier 1: .gov, .edu, major journals, authoritative institutions
/// Tier 2: Established news, official docs, tech documentation
/// Tier 3: Blogs, forums, wikis, community sites
/// Tier 4: Unknown, user-generated, unverified
pub fn classify_domain(domain: &str) -> SourceTier {
    let d = domain.to_lowercase();

    // Tier 1: government, education, authoritative
    if d.ends_with(".gov")
        || d.ends_with(".edu")
        || d.ends_with(".ac.uk")
        || d.ends_with(".mil")
        || is_tier1_domain(&d)
    {
        return SourceTier::Tier1;
    }

    // Tier 2: established sources
    if is_tier2_domain(&d) {
        return SourceTier::Tier2;
    }

    // Tier 3: community/social
    if is_tier3_domain(&d) {
        return SourceTier::Tier3;
    }

    // Default: Tier 4
    SourceTier::Tier4
}

/// Compute authority boost multiplier for a document.
///
/// score *= (1 + authority_weight * trustrank_estimate)
pub fn authority_boost(tier: SourceTier) -> f32 {
    1.0 + 0.2 * match tier {
        SourceTier::Tier1 => 1.0,
        SourceTier::Tier2 => 0.5,
        SourceTier::Tier3 => 0.0,
        SourceTier::Tier4 => -0.3,
    }
}

fn is_tier1_domain(d: &str) -> bool {
    let tier1 = [
        "nature.com", "science.org", "thelancet.com",
        "who.int", "nasa.gov", "cdc.gov", "nih.gov",
        "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
        "ieee.org", "acm.org", "springer.com",
        "sciencedirect.com", "cell.com", "pnas.org",
        "bmj.com", "nejm.org", "worldbank.org",
        "un.org", "europa.eu",
    ];
    tier1.iter().any(|t| d == *t || d.ends_with(&format!(".{t}")))
}

fn is_tier2_domain(d: &str) -> bool {
    let tier2 = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "nytimes.com", "washingtonpost.com", "theguardian.com",
        "docs.python.org", "docs.rs", "doc.rust-lang.org",
        "developer.mozilla.org", "developer.apple.com",
        "learn.microsoft.com", "cloud.google.com",
        "en.wikipedia.org", "stackoverflow.com",
        "github.com", "docs.github.com",
        "cppreference.com", "man7.org",
    ];
    tier2.iter().any(|t| d == *t || d.ends_with(&format!(".{t}")))
}

fn is_tier3_domain(d: &str) -> bool {
    let tier3 = [
        "medium.com", "dev.to", "reddit.com",
        "quora.com", "news.ycombinator.com",
        "hashnode.dev", "substack.com",
        "wordpress.com", "blogspot.com",
        "tumblr.com",
    ];
    tier3.iter().any(|t| d == *t || d.ends_with(&format!(".{t}")))
}

/// Extract the organization name from a domain for echo chamber detection.
///
/// Groups subdomains under parent org:
/// "blog.example.com" → "example.com"
/// "en.wikipedia.org" → "wikipedia.org"
pub fn domain_to_org(domain: &str) -> String {
    let parts: Vec<&str> = domain.split('.').collect();
    if parts.len() <= 2 {
        return domain.to_string();
    }
    // Special cases: co.uk, com.au, etc.
    let tld_len2 = ["co.uk", "com.au", "co.jp", "co.kr", "com.br", "co.in"];
    let suffix = format!("{}.{}", parts[parts.len() - 2], parts[parts.len() - 1]);
    if tld_len2.iter().any(|t| suffix == *t) && parts.len() > 2 {
        return format!("{}.{}", parts[parts.len() - 3], suffix);
    }
    format!("{}.{}", parts[parts.len() - 2], parts[parts.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_gov_as_tier1() {
        assert_eq!(classify_domain("www.nasa.gov"), SourceTier::Tier1);
        assert_eq!(classify_domain("data.gov"), SourceTier::Tier1);
    }

    #[test]
    fn classify_edu_as_tier1() {
        assert_eq!(classify_domain("mit.edu"), SourceTier::Tier1);
        assert_eq!(classify_domain("cs.stanford.edu"), SourceTier::Tier1);
    }

    #[test]
    fn classify_nature_as_tier1() {
        assert_eq!(classify_domain("nature.com"), SourceTier::Tier1);
        assert_eq!(classify_domain("arxiv.org"), SourceTier::Tier1);
    }

    #[test]
    fn classify_bbc_as_tier2() {
        assert_eq!(classify_domain("bbc.com"), SourceTier::Tier2);
        assert_eq!(classify_domain("stackoverflow.com"), SourceTier::Tier2);
    }

    #[test]
    fn classify_wikipedia_as_tier2() {
        assert_eq!(classify_domain("en.wikipedia.org"), SourceTier::Tier2);
    }

    #[test]
    fn classify_medium_as_tier3() {
        assert_eq!(classify_domain("medium.com"), SourceTier::Tier3);
        assert_eq!(classify_domain("reddit.com"), SourceTier::Tier3);
    }

    #[test]
    fn classify_unknown_as_tier4() {
        assert_eq!(classify_domain("randomsite.xyz"), SourceTier::Tier4);
    }

    #[test]
    fn authority_boost_values() {
        assert!(authority_boost(SourceTier::Tier1) > 1.0);
        assert!(authority_boost(SourceTier::Tier2) > 1.0);
        assert_eq!(authority_boost(SourceTier::Tier3), 1.0);
        assert!(authority_boost(SourceTier::Tier4) < 1.0);
    }

    #[test]
    fn domain_to_org_groups_subdomains() {
        assert_eq!(domain_to_org("blog.example.com"), "example.com");
        assert_eq!(domain_to_org("en.wikipedia.org"), "wikipedia.org");
        assert_eq!(domain_to_org("example.com"), "example.com");
    }

    #[test]
    fn domain_to_org_handles_country_tlds() {
        assert_eq!(domain_to_org("www.bbc.co.uk"), "bbc.co.uk");
        assert_eq!(domain_to_org("news.example.com.au"), "example.com.au");
    }
}
