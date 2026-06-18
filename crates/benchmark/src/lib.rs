//! Shared library for the web-search benchmark harness binaries.
//!
//! Pure, deterministic scoring + reporting modules live here so both the
//! `benchmark` (G1/G2/G3 coverage+accuracy) and `firecrawl-compare` (G4
//! head-to-head) binaries can reuse the exact same math under unit test — the
//! base of the test pyramid. The binaries do the I/O (MCP / HTTP).

pub mod compare;
pub mod firecrawl;
pub mod metrics;
