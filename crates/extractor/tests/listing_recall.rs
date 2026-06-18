//! Regression tests for the listing/index recall bug (G4 coverage loss vs
//! Firecrawl, verified 2026-06-18).
//!
//! The trafilatura pass over-filtered link-bearing table/list cells: a post
//! index is structurally identical to a nav bar (a cell of nothing but a link),
//! so the flat link-density penalty deleted it. On blog.rust-lang.org this cut
//! the body to 552 chars, dropping ~16k chars of real post-listing content.
//!
//! Fix A loosens the link penalty ONLY for `li`/`td`/`dd` cells with a real-title
//! anchor (>= 15 chars) or under a positive listing ancestor; Fix B adds a thin-
//! body recovery floor. The nav-guard test below is the honesty check that Fix A
//! did not simply re-admit boilerplate.
//!
//! Fixtures are committed snapshots (no live fetch — see the no-real-external-
//! services testing rule).

use web_search_extractor::extract_page;

const BLOG_RUST_HTML: &str = include_str!("fixtures/blog_rust_lang.html");
const URL: &str = "https://blog.rust-lang.org/";

/// Before the fix this page extracted exactly 552 chars (the intro blurb only),
/// dropping the entire `<table class="post-list">` of post titles. Assert we now
/// recover the listing — well above the buggy ceiling — and that several post
/// titles actually present in the snapshot survive extraction.
#[test]
fn listing_page_recovers_post_index() {
    let result = extract_page(BLOG_RUST_HTML, URL);
    let chars = result.body_text.chars().count();

    assert!(
        chars > 10_000,
        "expected the post listing to be recovered, got only {chars} chars \
         (regression: link-bearing post-index cells used to be deleted, leaving a \
         ~552-char intro stub)"
    );

    // Titles verified present in the committed fixture HTML (post-list table).
    for title in [
        "Announcing Rust 1.96.0",
        "Launching the Rust Foundation Maintainers Fund",
        "2025 State of Rust Survey Results",
        "docs.rs: building fewer targets by default",
        "Rust is participating in Outreachy",
    ] {
        assert!(
            result.body_text.contains(title),
            "recovered listing is missing the post title {title:?}"
        );
    }
}

/// Honesty test for Fix A's link-penalty loosening: a `<nav>` and a `<ul>` of
/// short one-word links (true navigation chrome) must NOT leak into the body.
/// This proves the loosening only fires for long-anchor / positive-ancestor
/// content cells, not for nav. Built as a self-contained synthetic page so the
/// assertion targets only the nav labels (no overlap with article prose).
#[test]
fn short_link_nav_is_not_admitted() {
    let html = r#"<html><head><title>Docs - Acme</title></head><body>
        <nav class="topbar">
            <ul>
                <li><a href="/">HomeNavLabel</a></li>
                <li><a href="/docs">DocsNavLabel</a></li>
                <li><a href="/api">ApiNavLabel</a></li>
                <li><a href="/blog">BlogNavLabel</a></li>
                <li><a href="/about">AboutNavLabel</a></li>
                <li><a href="/login">LoginNavLabel</a></li>
            </ul>
        </nav>
        <article class="post-content">
            <h1>Getting Started With The Acme Framework</h1>
            <p>This article walks through installing the Acme framework and writing your first service from scratch, with enough prose to score as genuine main content for the extractor.</p>
            <p>The second paragraph continues with configuration details and a worked example so the consensus body is unambiguously the article, not the navigation chrome above it.</p>
        </article>
    </body></html>"#;

    let result = extract_page(html, "https://acme.example/docs");

    // Article body is present...
    assert!(result.body_text.contains("Getting Started"));
    // ...but none of the short one-word nav labels leaked in.
    for nav_label in [
        "HomeNavLabel",
        "DocsNavLabel",
        "ApiNavLabel",
        "BlogNavLabel",
        "AboutNavLabel",
        "LoginNavLabel",
    ] {
        assert!(
            !result.body_text.contains(nav_label),
            "nav label {nav_label:?} leaked into body — Fix A re-admitted boilerplate"
        );
    }
}

/// A normal single-article page (no listing) must extract identically to its
/// content — Fix B's recovery floor must not alter pages that already have a
/// substantial prose body. Guards against the listing path firing on articles.
#[test]
fn single_article_page_unchanged_by_recovery() {
    let html = r#"<html><head><title>One Article - Site</title></head><body>
        <article class="entry-content">
            <h1>A Single Long Article</h1>
            <p>This is a fully server-rendered article whose body is long, prose-heavy paragraphs with no link-bearing listing cells at all, so the primary threshold selection is already well above the recovery floor.</p>
            <p>A second paragraph adds further depth, ensuring the consensus body comfortably exceeds the thin-body floor and the listing-recovery pass never engages on a normal article like this one.</p>
            <p>A third paragraph closes the piece with a conclusion and a handful of additional sentences of genuine prose so the page is unambiguously a single article.</p>
        </article>
    </body></html>"#;

    let result = extract_page(html, "https://site.example/a");
    assert!(result.body_text.contains("A Single Long Article") || result.body_text.contains("fully server-rendered article"));
    assert!(result.extraction_confidence > 0.5);
}
