# Archive migration record

The maintained repository was created beside, not inside, the separately preserved DFI archive. The archive is a provenance-preserving snapshot and was not edited. Its independently validated static-set fingerprint is:

```text
ece5dacbebfbbbc807ca5a1c94014784628427b662213159c4dd10a19d574cb8
```

Paths below are relative to `01_LOCAL_PROJECT_SNAPSHOT/DIFFUSION_PURIFICATION/` unless noted. Hashes were recomputed at implementation start and match the earlier blueprint.

| Canonical source | SHA-256 | v0.1 disposition |
|---|---|---|
| `DFI_RESEARCH_OVERVIEW.md` | `99a471fd7c4bc812024de57b3f286376a0108445f830ad0cd750ac9476fb8c12` | Narrative context; retained in archive |
| `DFI_ANALYTIC_THEORY.md` | `e5c4e8241b828882db63551a4b46cfe2e97644f8782df0250d720f610e3e1878` | Formulas adapted into package/docs |
| `DFI_V2_DATA_SPEC.md` | `5f2d4ef5cc5fa66b3095ee8ed9af46248409f6affc7fb5a809d254f3272cfa1f` | Admission/evaluation rules adapted |
| `notebooks/DFI_ANALYTIC.ipynb` | `12615b933993e1d5289dbfaa6b52b77fb650b4bf7b7ee4e097a402ff8a9b98cb` | Pure analytic behavior extracted; source retained |
| `notebooks/DFI_CANDIDATE_BUILDER.ipynb` | `f0139cfd962cd8a305a4024577ef47ff7387e1c21ee53bd432b18027cc2281fe` | Deferred workflow; retained in archive |
| `notebooks/DFI_KNOWLEDGE_AUDIT.ipynb` | `1bfdb5ce54b038459f73cbe7c4289674ce0ec955f65f7bf241c7394aab8dba8f` | Deferred workflow; retained in archive |
| `notebooks/DFI_EXPLORATION.ipynb` | `d678f21e809de3fafed0733e0be87c1a2302bf7c91ef2b90369ee498971302bc` | Sampled exploratory history; retained in archive |
| `data/dfi_v2/schema.json` | `a454af060830c338e3f96b6ff2a6b0643b5e4f6cd472e9701254242cdfac8b18` | Copied byte-for-byte as `data/dfi-v2.schema.json` |
| `scripts/validate_dfi_v2.py` | `87774d21740e7da98b948140f027a8331f519e9a3b23658fc5c27b172c62c8b4` | Useful semantic/cross-row checks adapted; schema remains authoritative |

Blueprint provenance:

| Planning document | SHA-256 |
|---|---|
| `DFI_REFACTOR_AND_GITHUB_BLUEPRINT.md` | `62bbda28831e0c056956f57983c6d8cd52c49332383cc7bd8dbe509667b64556` |
| `DFI_STREAMLINED_REFACTOR_BLUEPRINT.md` | `0963901085c78fac06655679d81376816529ca5e8ff372a7530e5e44a6b6feeb` |

## Fixture decision

The archived four-candidate file has SHA-256 `ed590ea6f174c46da21b7c37d05288a3683900b907892e3b397f250cb9aebc42`. It is a knowledge-audit candidate format, not schema-valid v2 scored JSONL, and it lacks redistribution-license fields. The archive contains no saved full marginal/logit fixture. Accordingly, PR 1 uses self-authored synthetic v2 claims and synthetic saved marginals with `interpretation_allowed=false`. The four historical candidates remain private provenance for a fresh, revision-pinned A100 parity capture.

## Notebook disposition

Historical notebooks are not copied or edited. Package modules are the sole maintained source of scoring behavior. The eventual walkthrough is a thin, unexecuted client that imports package functions and reads sealed outputs; it contains no installation, Drive discovery, model implementation, or hidden scoring logic.
