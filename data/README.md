# Data contracts and smoke fixtures

`dfi-v2.schema.json` is a byte-for-byte copy of the archive's sole canonical public data contract, originally stored at `data/dfi_v2/schema.json`. Package code adds documented semantic and cross-row checks; it does not define a competing public schema.

`fixtures/anchors.jsonl` contains four self-authored synthetic families (eight supported/refuted claims) under CC0-1.0. `fixtures/fixed-masks.json` uses a tiny synthetic offset tokenizer and deliberately splits every edited word into two pieces. `fixtures/saved-marginals.npz` contains synthetic seven-token posterior distributions and independently frozen expected statistics.

These files test formulas, exact multi-piece geometry, stable work IDs, reductions, Parquet/JSON artifacts, and the cache manifest. They are not produced by LLaDA, do not test checkpoint knowledge, and cannot support a scientific DFI result. The run interpretation gate must remain false.

The archive's four historical audit candidates are intentionally not copied here. They are not v2 scored records, include no redistribution-license field, and only two of four passed the `recall_known` gate. A real model-derived parity fixture must be captured afresh on the pinned A100 path.
