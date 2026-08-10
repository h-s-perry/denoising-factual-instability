# Scientific definitions and limits

## Fixed-mask analytic quantities

For a claim token sequence `x`, optional evidence `e`, word mask `M`, and masked position `i`, the pinned denoiser exposes the full posterior

```text
q_M,i(v) = q_theta(X_i = v | C_M(x), e).
```

The maintained `analytic-v1` protocol computes, in natural-log units and at temperature `T = 1`:

```text
CE_i          = -log q_M,i(x_i)
H_i           = -sum_v q_M,i(v) log q_M,i(v)
H2_i          = -log sum_v q_M,i(v)^2
Delta_i       = CE_i - H_i
swapLLR_i     = log max_(v != x_i) q_M,i(v) - log q_M,i(x_i)
drift_i       = 1 - q_M,i(x_i)
dispersion_i  = 1 - sum_v q_M,i(v)^2
```

All pieces of a selected word are masked and scored. Per-mask values are arithmetic means over masked pieces; claim values are arithmetic means over declared masks. The default `fixed-v1` policy selects a fixed count of content words at a declared band rate. That rate is not an independent inclusion probability for every maskable piece, so `fixed-v1` does not report an NELBO term.

An NELBO-like per-mask contribution belongs only to a separately declared policy that masks each eligible piece independently with probability `t`:

```text
sum_(i in M) CE_i / (t * L_maskable-pieces),
```

Under that distinct iid-Bernoulli policy, the expression is a masked-diffusion negative-ELBO integrand normalized per maskable piece. It must not be attached to `fixed-v1`. It is not I-MMSE or a Gaussian-channel identity.

## Contradiction versus ignorance

High cross-entropy means the submitted token is unlikely under the declared denoising context. High posterior entropy means the model is broadly uncertain. `Delta = CE - H` and strongest-alternative `swapLLR` help distinguish a concentrated preference for another token from diffuse ignorance, but they do not prove falsity.

Expected drift and collision dispersion are exact expectations under the notebook's factorized one-step sampler. They replace reconstruction Monte Carlo for these quantities; they do not reduce the number of independent mask-conditioned forwards when the mask count is unchanged.

## Evaluation units

- Mask rows are repeated measurements, never independent labeled examples.
- Claim AUROC uses only `supported` and `refuted`; `insufficient` and `ambiguous` remain separate.
- Declared contrasts are reduced as matched pairs.
- Families are the split and resampling unit when multiple conditions share one source fact.

The default interpretation gate is false. A passing software check, completed GPU run, or high aggregate metric does not by itself admit a dataset, prove checkpoint knowledge, establish calibration, or give a per-claim guarantee.

## Dataset labels

- `dfi-v2-pilot`: reserved for a reviewed, family-safe target dataset that passes its admission gates.
- `legacy-v3-exploratory`: historical generated 68-claim research fixture.
- `wikipedia-screened-unverified`: historical machine-screened v4-derived material.
- `dfi-synthetic-smoke-v1`: this repository's formula/contract fixture only.
