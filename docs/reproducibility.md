# Reproducibility and acceptance gates

## Current verified slices

The offline analytic core is reproducible from the repository commit and lockfile. `dfi check` performs no network call and validates:

- the unchanged v2 JSON Schema plus semantic/cross-row invariants;
- exact word-to-multiple-piece mask geometry;
- frozen SHA-256 work IDs under reordered requests;
- CE, entropy, collision, `Delta`, `swapLLR`, drift, and dispersion;
- claim, declared-contrast, and family reductions;
- atomic Parquet plus JSON receipt round-trip;
- immutable cache-shard publication, hydration, readback, and corruption rejection.

The saved marginals are synthetic. This is a CPU correctness gate, not model parity or scientific reproduction. The accepted `fixed-v1` slice reports no NELBO term; an NELBO-like quantity would require a separately declared iid-Bernoulli masking policy.

The real runner is implemented with a revision-pinned LLaDA loader, scalar full-logit reference, global length-bucketed token-budget batching, one runner-wide OOM fallback, resumable local partitions, bounded single-writer Drive publication, and a sealed two-file run bundle. Its CPU/mock integration tests establish runner and cache behavior only. The pinned real tokenizer has also been checked locally, without model weights or network access, against exact token IDs, offsets, word-piece mappings, masks, targets, and work IDs for all eight synthetic fixture claims.

Repository ZIPs must be produced with `git archive`. The exported `.git_archival.txt` records the exact commit through Git's `export-subst` mechanism, allowing runs from an unpacked archive and a non-editable install to retain commit provenance without bundling `.git`.

## Cache contract

The persistent result cache has one explicit root and two content levels:

```text
<root>/<model-and-protocol-fingerprint>/<inference-plan-hash>/
  cache.json
  part-0000-<content-hash>.parquet
```

Shards are immutable. The manifest records a generation; a type-exact fingerprint containing the protocol, model, tokenizer, remote-code, prompt, mask-policy, temperature, dtype, and backend revisions; the plan hash; and each shard's row count, SHA-256, sorted-work-ID digest, and creation environment. Manifest-level schema or semantic mismatches fail closed. A missing, truncated, hash-mismatched, duplicate, out-of-plan, or otherwise invalid physical shard becomes a partition-scoped miss while independently valid shards remain eligible for reuse. Repair is explicit and may replace only a partition already proven invalid; a valid immutable partition is never overwritten. Valid shards are copied and re-hashed on local SSD before use.

Google Drive is the durable cache for the A100 runner, but never a per-batch GPU hot-path filesystem. Publication is single-writer; concurrent writers are unsupported in v0.1. `run.json` records both the current runtime and the creation environments of reused and final cache shards so cross-environment reuse is visible without opening `cache.json`.

## Required manual A100 gate

Before the real runner is accepted, use `GSAI-ML/LLaDA-8B-Base` with literal model, tokenizer, and trusted remote-code commits, BF16, four audited anchors, and frozen real-tokenizer masks. Compare batch-size-one full logits with the global batcher and require:

- identical work IDs, target IDs, strongest-alternative IDs, and cardinality;
- no missing or duplicate rows;
- numeric tolerance frozen before performance results are inspected;
- unchanged claim ordering and matched decisions;
- zero forwards when a fresh local session hydrates all covered Drive shards;
- rejection and recomputation of a deliberately corrupt shard.

The checked-in `a100-smoke.toml` uses four synthetic matched families and therefore cannot satisfy the audited-anchor clause. Its parity selection is deliberately bounded to one prior mask for each of the eight claims. The separate `a100-throughput.toml` expands the same synthetic inputs to exactly 4,096 deterministic physical requests for performance measurement only; it is not scientific evidence.

The fixed throughput gate requires five warm-up batches and three full timed repetitions. Record active wall time, physical requests/second, forward count, peak VRAM, padding ratio, result-write time, and uploader stalls. No absolute speed or accepted batch-token default exists before that receipt. The local development environment has no CUDA/A100, so no A100 receipt is claimed by repository tests.

## Release boundary

No `v0.1.0` tag or scientific result is justified until the A100 parity/cache gate, fixed-hardware receipt, model/data redistribution review, secret/local-path scan, and dataset interpretation gate all pass. The package version remains `0.1.0.dev0` until then.
