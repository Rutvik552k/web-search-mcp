// SPDX-License-Identifier: MIT

use crate::authority;

/// Entity-to-official-domain mapping for canonical source prioritization.
///
/// When a query mentions a known entity (company, product, technology),
/// results from that entity's official domain get a significant boost.
/// This ensures e.g. openai.com ranks above Wikipedia for "GPT-5 release".

/// (entity_keyword, official_domains) — keyword matched case-insensitively against query.
/// Order: most specific first (product names before company names).
const ENTITY_DOMAINS: &[(&str, &[&str])] = &[
    // ── AI Companies & Products ──
    ("chatgpt", &["openai.com"]),
    ("gpt-5", &["openai.com"]),
    ("gpt-4", &["openai.com"]),
    ("gpt-4o", &["openai.com"]),
    ("dall-e", &["openai.com"]),
    ("openai", &["openai.com", "platform.openai.com"]),
    ("claude", &["anthropic.com", "docs.anthropic.com"]),
    ("anthropic", &["anthropic.com", "docs.anthropic.com"]),
    ("gemini", &["deepmind.google", "ai.google"]),
    ("deepmind", &["deepmind.google"]),
    ("google ai", &["ai.google", "deepmind.google"]),
    ("bard", &["ai.google"]),
    ("copilot", &["github.com", "copilot.microsoft.com"]),
    ("llama", &["ai.meta.com", "llama.meta.com"]),
    ("meta ai", &["ai.meta.com"]),
    ("mistral", &["mistral.ai"]),
    ("cohere", &["cohere.com"]),
    ("stability ai", &["stability.ai"]),
    ("stable diffusion", &["stability.ai"]),
    ("midjourney", &["midjourney.com"]),
    ("hugging face", &["huggingface.co"]),
    ("huggingface", &["huggingface.co"]),
    ("perplexity", &["perplexity.ai"]),

    // ── GPU / Hardware ──
    ("h100", &["nvidia.com", "developer.nvidia.com"]),
    ("h200", &["nvidia.com", "developer.nvidia.com"]),
    ("b200", &["nvidia.com", "developer.nvidia.com"]),
    ("b100", &["nvidia.com", "developer.nvidia.com"]),
    ("gb200", &["nvidia.com", "developer.nvidia.com"]),
    ("a100", &["nvidia.com", "developer.nvidia.com"]),
    ("rtx 5090", &["nvidia.com"]),
    ("rtx 4090", &["nvidia.com"]),
    ("cuda", &["developer.nvidia.com"]),
    ("tensorrt", &["developer.nvidia.com"]),
    ("nvidia", &["nvidia.com", "developer.nvidia.com"]),
    ("radeon", &["amd.com"]),
    ("ryzen", &["amd.com"]),
    ("epyc", &["amd.com"]),
    ("rocm", &["amd.com"]),
    ("amd", &["amd.com"]),
    ("intel arc", &["intel.com"]),
    ("xeon", &["intel.com"]),
    ("core ultra", &["intel.com"]),
    ("intel", &["intel.com"]),
    ("apple silicon", &["apple.com"]),
    ("m4 chip", &["apple.com"]),
    ("qualcomm", &["qualcomm.com"]),
    ("snapdragon", &["qualcomm.com"]),

    // ── Cloud Providers ──
    ("aws", &["aws.amazon.com", "docs.aws.amazon.com"]),
    ("amazon web services", &["aws.amazon.com"]),
    ("ec2", &["aws.amazon.com"]),
    ("lambda aws", &["aws.amazon.com"]),
    ("s3 bucket", &["aws.amazon.com"]),
    ("azure", &["azure.microsoft.com", "learn.microsoft.com"]),
    ("gcp", &["cloud.google.com"]),
    ("google cloud", &["cloud.google.com"]),
    ("firebase", &["firebase.google.com"]),
    ("vercel", &["vercel.com"]),
    ("netlify", &["netlify.com"]),
    ("cloudflare", &["cloudflare.com"]),
    ("digitalocean", &["digitalocean.com"]),
    ("supabase", &["supabase.com"]),

    // ── Programming Languages ──
    ("rust lang", &["rust-lang.org", "doc.rust-lang.org"]),
    ("rustup", &["rust-lang.org"]),
    ("cargo crate", &["crates.io", "doc.rust-lang.org"]),
    ("python", &["python.org", "docs.python.org"]),
    ("pip install", &["pypi.org"]),
    ("typescript", &["typescriptlang.org"]),
    ("javascript", &["developer.mozilla.org"]),
    ("golang", &["go.dev"]),
    ("kotlin", &["kotlinlang.org"]),
    ("swift lang", &["swift.org", "developer.apple.com"]),
    ("java jdk", &["oracle.com", "openjdk.org"]),
    ("csharp", &["learn.microsoft.com"]),
    ("dotnet", &["dotnet.microsoft.com"]),

    // ── Frameworks & Libraries ──
    ("react", &["react.dev"]),
    ("nextjs", &["nextjs.org"]),
    ("next.js", &["nextjs.org"]),
    ("vue.js", &["vuejs.org"]),
    ("vuejs", &["vuejs.org"]),
    ("angular", &["angular.io", "angular.dev"]),
    ("svelte", &["svelte.dev"]),
    ("django", &["djangoproject.com"]),
    ("flask", &["flask.palletsprojects.com"]),
    ("fastapi", &["fastapi.tiangolo.com"]),
    ("spring boot", &["spring.io"]),
    ("laravel", &["laravel.com"]),
    ("rails", &["rubyonrails.org"]),
    ("express.js", &["expressjs.com"]),
    ("tailwind", &["tailwindcss.com"]),
    ("pytorch", &["pytorch.org"]),
    ("tensorflow", &["tensorflow.org"]),
    ("docker", &["docker.com", "docs.docker.com"]),
    ("kubernetes", &["kubernetes.io"]),
    ("terraform", &["terraform.io"]),
    ("ansible", &["ansible.com"]),

    // ── Databases ──
    ("postgresql", &["postgresql.org"]),
    ("postgres", &["postgresql.org"]),
    ("mysql", &["mysql.com", "dev.mysql.com"]),
    ("mongodb", &["mongodb.com"]),
    ("redis", &["redis.io"]),
    ("elasticsearch", &["elastic.co"]),
    ("sqlite", &["sqlite.org"]),
    ("cockroachdb", &["cockroachlabs.com"]),
    ("planetscale", &["planetscale.com"]),

    // ── Big Tech Products ──
    ("iphone", &["apple.com"]),
    ("ipad", &["apple.com"]),
    ("macbook", &["apple.com"]),
    ("apple", &["apple.com"]),
    ("windows 11", &["microsoft.com"]),
    ("microsoft", &["microsoft.com"]),
    ("google search", &["google.com"]),
    ("chrome", &["google.com", "chromium.org"]),
    ("android", &["developer.android.com", "android.com"]),
    ("pixel phone", &["store.google.com"]),
    ("samsung galaxy", &["samsung.com"]),
    ("samsung", &["samsung.com"]),
    ("tesla", &["tesla.com"]),
    ("spacex", &["spacex.com"]),

    // ── Dev Tools & Platforms ──
    ("github", &["github.com", "docs.github.com"]),
    ("gitlab", &["gitlab.com"]),
    ("bitbucket", &["bitbucket.org"]),
    ("jira", &["atlassian.com"]),
    ("confluence", &["atlassian.com"]),
    ("slack api", &["api.slack.com"]),
    ("discord bot", &["discord.com"]),
    ("figma", &["figma.com"]),
    ("notion", &["notion.so"]),
    ("linear", &["linear.app"]),
    ("stripe", &["stripe.com", "docs.stripe.com"]),
    ("twilio", &["twilio.com"]),
    ("sendgrid", &["sendgrid.com"]),
    ("auth0", &["auth0.com"]),
    ("okta", &["okta.com"]),
    ("datadog", &["datadoghq.com"]),
    ("grafana", &["grafana.com"]),
    ("sentry", &["sentry.io"]),
    ("npm", &["npmjs.com"]),
    ("yarn", &["yarnpkg.com"]),

    // ── Crypto / Web3 ──
    ("ethereum", &["ethereum.org"]),
    ("solana", &["solana.com"]),
    ("bitcoin", &["bitcoin.org"]),
    ("polygon", &["polygon.technology"]),
    ("chainlink", &["chain.link"]),
];

/// Given a query, detect entity mentions and return matching official domains.
///
/// Returns domains sorted and deduplicated. Most specific matches win
/// (product names matched before company names due to ordering).
pub fn detect_official_domains(query: &str) -> Vec<&'static str> {
    let q = query.to_lowercase();
    let mut domains = Vec::new();

    for &(entity, official_domains) in ENTITY_DOMAINS {
        if q.contains(entity) {
            domains.extend_from_slice(official_domains);
        }
    }

    domains.sort_unstable();
    domains.dedup();
    domains
}

/// Check if a result's domain is directly mentioned in the query.
///
/// E.g., query "openai GPT-5 release" mentions "openai" →
/// results from openai.com get a massive boost.
pub fn query_mentions_domain(query: &str, domain: &str) -> bool {
    let q = query.to_lowercase();
    let org = authority::domain_to_org(domain);
    // Extract the name part (before TLD)
    let org_name = org.split('.').next().unwrap_or("");
    // Require at least 3 chars to avoid false matches on "ai", "io", etc.
    org_name.len() >= 3 && q.contains(org_name)
}

/// Check if a domain matches any of the detected official domains for a query.
pub fn is_canonical_domain(domain: &str, official_domains: &[&str]) -> bool {
    let org = authority::domain_to_org(domain);
    official_domains.iter().any(|d| {
        org == *d || domain == *d || domain.ends_with(&format!(".{d}"))
    })
}

/// Compute the primary source boost multiplier for a candidate.
///
/// Returns a multiplier > 1.0 if the candidate is from a canonical/official source.
/// Multipliers stack: domain-in-query (3x) × entity-canonical (2x).
pub fn primary_source_boost(query: &str, domain: &str, official_domains: &[&str]) -> f32 {
    let mut boost = 1.0_f32;

    // Strongest signal: domain is literally mentioned in query
    if query_mentions_domain(query, domain) {
        boost *= 2.5;
    }

    // Second signal: domain matches entity-to-official-domain mapping
    if is_canonical_domain(domain, official_domains) {
        boost *= 1.8;
    }

    boost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_openai_domains() {
        let domains = detect_official_domains("GPT-5 release date");
        assert!(domains.contains(&"openai.com"));
    }

    #[test]
    fn detect_nvidia_domains() {
        let domains = detect_official_domains("NVIDIA H200 GPU specs");
        assert!(domains.contains(&"nvidia.com"));
        assert!(domains.contains(&"developer.nvidia.com"));
    }

    #[test]
    fn detect_multiple_entities() {
        let domains = detect_official_domains("OpenAI vs Anthropic comparison");
        assert!(domains.contains(&"openai.com"));
        assert!(domains.contains(&"anthropic.com"));
    }

    #[test]
    fn no_match_for_generic_query() {
        let domains = detect_official_domains("best programming language 2026");
        assert!(domains.is_empty());
    }

    #[test]
    fn query_mentions_domain_works() {
        assert!(query_mentions_domain("openai GPT-5 release", "openai.com"));
        assert!(query_mentions_domain("nvidia H200 specs", "developer.nvidia.com"));
        assert!(!query_mentions_domain("GPU performance benchmarks", "nvidia.com"));
    }

    #[test]
    fn canonical_domain_match() {
        let official = vec!["openai.com", "platform.openai.com"];
        assert!(is_canonical_domain("openai.com", &official));
        assert!(is_canonical_domain("platform.openai.com", &official));
        assert!(!is_canonical_domain("fakeopenai.com", &official));
    }

    #[test]
    fn primary_source_boost_stacks() {
        let official = vec!["openai.com"];
        // Domain mentioned in query + canonical match = both boosts
        let boost = primary_source_boost("openai gpt-5", "openai.com", &official);
        assert!(boost > 3.0, "Expected stacked boost > 3.0, got {boost}");

        // Only canonical, not in query
        let boost2 = primary_source_boost("gpt-5 release date", "openai.com", &official);
        assert!(boost2 > 1.5 && boost2 < 3.0, "Expected canonical-only boost, got {boost2}");

        // No match at all
        let boost3 = primary_source_boost("gpt-5 release date", "wikipedia.org", &official);
        assert_eq!(boost3, 1.0);
    }

    #[test]
    fn rust_lang_detection() {
        let domains = detect_official_domains("rust lang async tutorial");
        assert!(domains.contains(&"rust-lang.org"));
    }

    #[test]
    fn docker_kubernetes_detection() {
        let domains = detect_official_domains("docker vs kubernetes deployment");
        assert!(domains.contains(&"docker.com"));
        assert!(domains.contains(&"kubernetes.io"));
    }
}
