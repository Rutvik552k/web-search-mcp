//! Regression test for the long-article truncation bug (G4 coverage loss vs
//! Firecrawl, verified 2026-06-18).
//!
//! On the Wikipedia "Rust (programming language)" page, the trafilatura pass
//! captured only a ~2.8k-char infobox fragment yet reported a flat 0.85
//! confidence, beating the readability pass (76k chars, the real body, 0.80) in
//! `extract_base`'s confidence-only tie-break. The cleaned result was 1531 chars
//! and missed "memory safety". The fix prefers a substantially richer body over
//! a higher-but-coarse confidence score.
//!
//! Fixture is a committed snapshot (no live fetch — see the no-real-external-
//! services testing rule): crates/extractor/tests/fixtures/wikipedia_rust.html

use web_search_extractor::extract_page;

const WIKIPEDIA_RUST_HTML: &str = include_str!("fixtures/wikipedia_rust.html");
const URL: &str = "https://en.wikipedia.org/wiki/Rust_(programming_language)";

/// The bug produced exactly 1531 chars; the full article is ~75k. Assert we are
/// well above the buggy ceiling and that the key body phrase is present.
#[test]
fn long_article_extracts_full_body_not_infobox_fragment() {
    let result = extract_page(WIKIPEDIA_RUST_HTML, URL);
    let chars = result.body_text.chars().count();

    assert!(
        chars > 20_000,
        "expected the full article body, got only {chars} chars (regression: the \
         confidence tie-break used to pick a ~1531-char infobox fragment)"
    );
    assert!(
        result.body_text.contains("memory safety"),
        "extracted body is missing the expected phrase 'memory safety'"
    );
}
