# A.7 Perf Gate — CPU neural-inference latency (ADR 0004 §6.3 / A.7)

The `force_cpu` meaning change (skip-neural → neural-on-CPU) means the default
path may now run MiniLM on CPU. ADR §6.3 / A.7 requires measuring CPU
neural-inference latency against a **stated threshold**, committed as a result
file, before the old fast (hash) path is removed. The
`embedder_backend = "hash"` escape hatch preserves the fast capability
regardless, which de-risks this gate.

## Threshold (stated)

- **Metric:** mean wall-clock latency to embed ONE short sentence (single text,
  384-d MiniLM all-MiniLM-L6-v2) on CPU, warm (after a warmup forward).
- **Threshold:** `< 250 ms` mean over 20 runs.
- **Rationale:** the embedder is called per query and per indexed chunk; sub-250ms
  keeps a single embed well under the search-API 5s budget and is acceptable for
  the keyless/CPU default profile. If CPU inference exceeds this, operators keep
  `embedder_backend="hash"` (no semantic matching, but fast) — the capability is
  not silently deleted.

## Method

Test: `crates/embedder/src/lib.rs::tests::a7_perf_gate_cpu_inference`
(`#[cfg(feature="candle")]`, `#[ignore]` — NOT in CI; needs candle + a cached or
fetchable MiniLM model, so the testing-rule "no real external services" keeps it
out of the default run).

Run:

```
cargo test -p web-search-embedder --features candle -- --ignored --nocapture a7_perf_gate_cpu_inference
```

The test forces `force_cpu=true` + `embedder_backend="neural"`, asserts the
NEURAL embedder loaded on CPU (`is_neural()`, `dimensions()==384`,
`model_name()=="sentence-transformers/all-MiniLM-L6-v2"`), warms up once, then
times 20 single-sentence embeds and asserts the mean is below the threshold.

## Reproducibility

- Model: `sentence-transformers/all-MiniLM-L6-v2` (cached in the HF hub cache;
  no network at run time).
- Framework: candle 0.8 (CPU device), Rust 1.85+, Windows 11 (dev box).
- Seed: deterministic forward (no sampling); latency is the measured quantity.
- The exact measured `mean_ms` is printed by the test (`--nocapture`) and should
  be pasted below when the gate is run on the target hardware.

## Measured

- Date: 2026-06-14. Dev box (Windows 11), candle 0.8 CPU, all-MiniLM-L6-v2
  (from HF cache, no network).
- **Build profile: DEBUG (`cargo test`, unoptimized + debuginfo).**
- **mean embed latency: 1972.8 ms over 20 runs (single short sentence).**
- **PASS/FAIL vs 250 ms: FAIL (1972.8 ms >= 250 ms) — gate FAILED in this profile.**

### Interpretation (reported as-is, not reframed)

The gate FAILED as measured. The dominant caveat is the **debug build**: candle's
gemm/BERT forward is ~10-40x slower unoptimized, so this number is an upper bound,
not the production figure. The honest position:

1. The measured number (1972.8 ms debug) does **NOT** clear the threshold.
2. A **release-profile re-measure is REQUIRED** before declaring the gate passed
   and before any future removal of the hash fast path. Run:
   `cargo test -p web-search-embedder --release --features candle -- --ignored --nocapture a7_perf_gate_cpu_inference`
   (left as a separate background task — release-building the candle tree is heavy).
3. Because the gate is currently FAILING (debug) and unverified (release), the
   `embedder_backend="hash"` **escape hatch is load-bearing and MUST stay**. The
   force_cpu meaning change (neural-on-CPU) is shipped, but the fast keyword path
   is explicitly preserved for operators on slow CPUs — exactly the A.7 safeguard.

### TODO (separate background task, flagged)

- [ ] Re-run the perf gate in `--release` on the target hardware; record the
      release mean here. If release CPU latency is still > 250 ms, either raise
      the threshold with justification or keep "hash" as the documented default
      for CPU-only deploys. Do NOT remove the hash path until a release number
      clears a stated threshold (ADR A.7 integrity rule: perf claim → committed
      result).
