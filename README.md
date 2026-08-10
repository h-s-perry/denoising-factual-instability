# Denoising Factual Instability

DFI is a performance-first research scorer for factual-risk signals from masked-token marginals produced by a pinned masked diffusion language model.

## What DFI is—and is not

For an atomic claim, DFI masks exact token pieces and reduces the model posterior into submitted-token cross-entropy, posterior entropy, `Delta = CE - H`, strongest-alternative `swapLLR`, expected one-step drift, and collision dispersion. DFI measures compatibility with one immutable model checkpoint and an explicitly named evidence condition. It is not a trained truth model, a calibrated falsehood decision, or proof that a claim is false.

This repository is the maintained scorer. The separately preserved provenance archive remains unchanged and outside Git; selected archive-relative sources and their recomputed hashes are recorded in the migration document. Runtime code never depends on the archive.

## Offline quickstart

```bash
uv sync --frozen --no-editable
uv run --no-sync dfi check
```

`dfi check` uses synthetic saved marginals. It needs no GPU, model download, network, Drive mount, or credentials. Passing it establishes formula, mask, reduction, cache-contract, and artifact-writing correctness only; it is not scientific evidence and does not reproduce LLaDA.

## Seven-stage pipeline

```text
reviewed v2 JSONL
  -> exact word-to-piece masks
  -> deterministic request plan
  -> global length-bucketed inference
  -> analytic sufficient statistics
  -> claim/pair/family reductions
  -> sealed results.parquet + run.json
```

## First A100 run

The runnable LLaDA path is intentionally not admitted in the offline-core slice. It will use one complete TOML config, three immutable Hugging Face revisions, BF16 on a named A100, a local-SSD hot path, and an explicit Google Drive cache root. Scalar/global-batch parity and a cold-to-warm zero-forward Drive test must pass before that slice is accepted.

## Final run files

A sealed run contains only:

- `results.parquet`: compact per-mask sufficient statistics and identifiers;
- `run.json`: resolved provenance, environment, cache/accounting, result hash, metrics, and the interpretation gate.

Partial partitions, caches, model weights, full datasets, and executed notebook output are never final run files and are excluded from Git.

## Scientific interpretation warning

High DFI means that the submitted tokens are unstable or comparatively disfavored under the declared model and context. It can reflect contradiction, model ignorance, wording difficulty, or evidence mismatch. Aggregate separation does not establish truth, per-claim correctness, checkpoint knowledge, or a `DFI-Known` benchmark. The default `fixed-v1` policy does not report NELBO. A future NELBO-like quantity would require a separately declared iid-Bernoulli masking policy; it would be a masking-channel negative-ELBO estimate, not I-MMSE.

## Focused documentation

- [Scientific definitions and limits](docs/science.md)
- [Reproducibility and acceptance gates](docs/reproducibility.md)
- [Archive-to-repository migration record](docs/migration.md)
