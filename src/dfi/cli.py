"""Thin command routing for the DFI research scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from importlib import resources
from pathlib import Path

import numpy as np

from dfi.data import load_jsonl
from dfi.evaluation import evaluate_rows, reduce_claims
from dfi.masking import (
    MaskChoice,
    align_word_pieces,
    apply_mask_geometry,
    deterministic_masks,
    stable_work_id,
)
from dfi.pipeline import (
    cache_row_from_result,
    hydrate_cache_shards,
    inspect_cache_manifest,
    local_creation_environment,
    publish_cache_shard,
    read_parquet_rows,
    sha256_file,
    validate_cache_manifest,
    write_run_bundle,
)
from dfi.scoring import reduce_positions, score_marginals, stats_as_lists


@contextmanager
def _fixture_path(name: str) -> Iterator[Path]:
    source = Path(__file__).resolve().parents[2] / "data" / "fixtures" / name
    if source.is_file():
        yield source
        return
    packaged = resources.files("dfi").joinpath("fixtures", name)
    with resources.as_file(packaged) as installed:
        yield installed


def _assert_array(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    except AssertionError as exc:
        raise ValueError(f"saved-marginal mismatch for {name}") from exc


def _check_aggregation_contract() -> None:
    """Freeze distinct mask, declared-pair, and family aggregation semantics."""

    def metric_row(
        example_id: str,
        label: str,
        score: float,
        mask_index: int,
        paired_example_id: str | None,
    ) -> dict[str, object]:
        return {
            "example_id": example_id,
            "family_id": "aggregation-probe-family",
            "split": "dev",
            "label": label,
            "paired_example_id": paired_example_id,
            "arm": "prior",
            "mask_index": mask_index,
            "ce_mean": score,
            "entropy_mean": score,
            "collision_entropy_mean": score,
            "delta_mean": score,
            "swap_llr_mean": score,
            "expected_drift_mean": score,
            "expected_dispersion_mean": score,
            "top_mismatch_rate": score,
        }

    rows = [
        metric_row("paired-supported", "supported", 1.0, 0, "paired-refuted"),
        metric_row("paired-supported", "supported", 3.0, 1, "paired-refuted"),
        metric_row("paired-refuted", "refuted", 0.5, 0, "paired-supported"),
        metric_row("paired-refuted", "refuted", 0.5, 1, "paired-supported"),
        metric_row("extra-supported", "supported", 0.0, 0, None),
        metric_row("extra-refuted", "refuted", 10.0, 0, None),
    ]
    claims = reduce_claims(rows)
    supported = next(row for row in claims if row["example_id"] == "paired-supported")
    if supported["n_masks"] != 2 or supported["ce_mean"] != 2.0:
        raise ValueError("claim-level mask aggregation changed")
    metrics = evaluate_rows(
        rows,
        interpretation_allowed=False,
        interpretation_reason="synthetic aggregation correctness probe",
    )
    ce_metrics = metrics["arms"]["prior"]["scores"]["ce_mean"]
    declared = ce_metrics["declared_contrasts"]
    families = ce_metrics["families"]
    if declared["n_declared_contrasts"] != 1 or declared["wins"] != 0:
        raise ValueError("declared-contrast aggregation changed")
    if families["n_binary_families"] != 1 or families["wins"] != 1:
        raise ValueError("family aggregation changed")


def offline_check() -> dict[str, object]:
    """Run the complete CPU-only analytic smoke check and return its summary."""

    with ExitStack() as stack:
        anchors_path = stack.enter_context(_fixture_path("anchors.jsonl"))
        masks_path = stack.enter_context(_fixture_path("fixed-masks.json"))
        marginals_path = stack.enter_context(_fixture_path("saved-marginals.npz"))

        _, examples = load_jsonl(anchors_path)
        by_example = {example.example_id: example for example in examples}
        fixed = json.loads(masks_path.read_text(encoding="utf-8"))
        if fixed.get("fixture_kind") != "synthetic-correctness-smoke":
            raise ValueError("offline fixture must be explicitly synthetic")
        requests = fixed["requests"]
        identity = fixed["work_identity"]
        probe = fixed["mask_policy_probe"]
        probe_choices = deterministic_masks(
            probe["text"],
            example_id=probe["example_id"],
            arm=probe["arm"],
            seed=probe["seed"],
            n_masks=probe["n_masks"],
        )
        observed_probe = [
            {
                "mask_index": choice.mask_index,
                "mask_rate": choice.mask_rate,
                "word_indices": list(choice.word_indices),
            }
            for choice in probe_choices
        ]
        if observed_probe != probe["expected"]:
            raise ValueError("fixed-v1 mask policy changed from its frozen probe")

        with np.load(marginals_path, allow_pickle=False) as saved:
            if saved["fixture_kind"].item() != "synthetic-correctness-smoke":
                raise ValueError("saved marginals are not the declared synthetic fixture")
            request_ids = saved["request_ids"].tolist()
            if request_ids != [request["request_id"] for request in requests]:
                raise ValueError("fixed masks and saved marginals use different request order")
            probabilities = saved["probabilities"]
            saved_targets = saved["target_ids"]

            rows: list[dict[str, object]] = []
            work_ids: dict[str, str] = {}
            for index, request in enumerate(requests):
                example = by_example.get(request["example_id"])
                if example is None or example.claim != request["text"]:
                    raise ValueError(
                        f"fixture request {request['request_id']} has no matching claim"
                    )

                word_pieces = align_word_pieces(request["text"], request["offsets"])
                geometry = apply_mask_geometry(
                    request["token_ids"],
                    word_pieces,
                    MaskChoice(
                        int(request["mask_index"]),
                        float(request["mask_rate"]),
                        tuple(request["mask_word_indices"]),
                    ),
                    mask_token_id=int(fixed["mask_token_id"]),
                )
                if list(geometry.piece_indices) != request["piece_indices"]:
                    raise ValueError(f"{request['request_id']}: piece geometry changed")
                if list(geometry.target_ids) != request["target_ids"]:
                    raise ValueError(f"{request['request_id']}: target IDs changed")
                if list(geometry.masked_input_ids) != request["masked_input_ids"]:
                    raise ValueError(f"{request['request_id']}: masked input changed")

                work_id = stable_work_id(
                    arm=request["arm"],
                    input_ids=geometry.masked_input_ids,
                    attention_mask=[1] * len(geometry.masked_input_ids),
                    masked_positions=geometry.piece_indices,
                    target_ids=geometry.target_ids,
                    mask_rate=request["mask_rate"],
                    **identity,
                )
                if work_id != request["expected_work_id"]:
                    raise ValueError(f"{request['request_id']}: stable work ID changed")
                work_ids[request["request_id"]] = work_id

                if not np.array_equal(saved_targets[index], geometry.target_ids):
                    raise ValueError(f"{request['request_id']}: NPZ target IDs changed")
                stats = score_marginals(probabilities[index], saved_targets[index])
                expected = {
                    "top_ids": saved["expected_top_ids"][index],
                    "top_alternative_ids": saved["expected_top_alternative_ids"][index],
                    "ce": saved["expected_ce"][index],
                    "entropy": saved["expected_entropy"][index],
                    "collision_probability": saved["expected_collision_probability"][index],
                    "collision_entropy": saved["expected_collision_entropy"][index],
                    "delta": saved["expected_delta"][index],
                    "swap_llr": saved["expected_swap_llr"][index],
                    "expected_drift": saved["expected_drift"][index],
                    "expected_dispersion": saved["expected_dispersion"][index],
                    "top_mismatch": saved["expected_top_mismatch"][index],
                }
                for name, values in expected.items():
                    _assert_array(np.asarray(getattr(stats, name)), np.asarray(values), name)

                row: dict[str, object] = {
                    "work_id": work_id,
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "split": example.split,
                    "label": example.label,
                    "paired_example_id": example.paired_example_id,
                    "arm": request["arm"],
                    "mask_index": request["mask_index"],
                    "mask_rate": request["mask_rate"],
                    "masked_positions": request["piece_indices"],
                }
                row.update(stats_as_lists(stats))
                row.update(reduce_positions(stats))
                rows.append(row)

        reversed_ids = {
            request["request_id"]: stable_work_id(
                arm=request["arm"],
                input_ids=request["masked_input_ids"],
                attention_mask=[1] * len(request["masked_input_ids"]),
                masked_positions=request["piece_indices"],
                target_ids=request["target_ids"],
                mask_rate=request["mask_rate"],
                **identity,
            )
            for request in reversed(requests)
        }
        if reversed_ids != work_ids:
            raise ValueError("work IDs changed when request order changed")

        metrics = evaluate_rows(
            rows,
            interpretation_allowed=False,
            interpretation_reason="synthetic saved marginals are a correctness smoke test",
        )
        ce_metrics = metrics["arms"]["prior"]["scores"]["ce_mean"]
        if ce_metrics["claim_auroc"] != 1.0:
            raise ValueError("synthetic claim reduction no longer matches its frozen result")
        if ce_metrics["families"]["wins"] != 4:
            raise ValueError("synthetic family reduction no longer matches its frozen result")
        _check_aggregation_contract()

        with tempfile.TemporaryDirectory(prefix="dfi-check-") as temporary_name:
            temporary = Path(temporary_name)
            bundle = temporary / "run"
            results_path, run_path = write_run_bundle(
                bundle,
                rows,
                receipt={
                    "run_uuid": "offline-check",
                    "fixture_kind": "synthetic-correctness-smoke",
                    "evaluation": metrics,
                    "interpretation_allowed": False,
                    "interpretation_reason": "synthetic correctness fixture",
                },
                expected_work_ids=set(work_ids.values()),
            )
            readback = read_parquet_rows(results_path)
            if len(readback) != len(rows):
                raise ValueError("artifact round-trip changed row count")
            receipt = json.loads(run_path.read_text(encoding="utf-8"))
            if receipt["results"]["sha256"] != sha256_file(results_path):
                raise ValueError("run receipt result hash does not validate")

            plan_hash = hashlib.sha256("\n".join(sorted(work_ids.values())).encode()).hexdigest()
            fingerprint = dict(identity)
            creation_environment = local_creation_environment()
            cache = temporary / "drive-cache" / "synthetic" / plan_hash
            cache_rows = [cache_row_from_result(row) for row in rows]
            publish_cache_shard(
                cache,
                cache_rows[:4],
                partition_index=0,
                plan_hash=plan_hash,
                fingerprint=fingerprint,
                expected_work_ids=set(work_ids.values()),
                creation_environment=creation_environment,
            )
            publish_cache_shard(
                cache,
                cache_rows[4:],
                partition_index=1,
                plan_hash=plan_hash,
                fingerprint=fingerprint,
                expected_work_ids=set(work_ids.values()),
                creation_environment=creation_environment,
            )
            shards = validate_cache_manifest(
                cache / "cache.json",
                expected_plan_hash=plan_hash,
                expected_fingerprint=fingerprint,
                expected_work_ids=set(work_ids.values()),
            )
            hydrated = hydrate_cache_shards(shards, temporary / "local-ssd")
            if [sha256_file(path) for path in hydrated] != [shard.sha256 for shard in shards]:
                raise ValueError("hydrated cache bytes changed")

            corrupt = temporary / "corrupt-cache"
            shutil.copytree(cache, corrupt)
            corrupt_manifest = json.loads((corrupt / "cache.json").read_text(encoding="utf-8"))
            corrupt_shard = corrupt / corrupt_manifest["shards"][0]["name"]
            corrupt_shard.write_bytes(corrupt_shard.read_bytes()[:-8])
            inspection = inspect_cache_manifest(
                corrupt / "cache.json",
                expected_plan_hash=plan_hash,
                expected_fingerprint=fingerprint,
                expected_work_ids=set(work_ids.values()),
            )
            if len(inspection.valid_shards) != 1 or len(inspection.failures) != 1:
                raise ValueError("one corrupt partition did not remain an isolated cache miss")
            publish_cache_shard(
                corrupt,
                cache_rows[:4],
                partition_index=0,
                plan_hash=plan_hash,
                fingerprint=fingerprint,
                expected_work_ids=set(work_ids.values()),
                creation_environment=creation_environment,
                repair_invalid_partition=True,
            )
            repaired = validate_cache_manifest(
                corrupt / "cache.json",
                expected_plan_hash=plan_hash,
                expected_fingerprint=fingerprint,
                expected_work_ids=set(work_ids.values()),
            )
            if len(repaired) != 2:
                raise ValueError("truncated cache shard was silently accepted")

    return {
        "records": len(examples),
        "families": len({example.family_id for example in examples}),
        "mask_rows": len(rows),
        "work_ids": len(work_ids),
        "fixture_kind": "synthetic-correctness-smoke",
        "interpretation_allowed": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dfi", description="Denoising Factual Instability")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="run the offline analytic correctness fixture")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "check":
            summary = offline_check()
            print("DFI offline check: PASS")
            print(
                f"  {summary['records']} records / {summary['families']} families / "
                f"{summary['mask_rows']} mask rows"
            )
            print("  formulas, multi-piece masks, stable IDs, reductions, cache, artifacts: PASS")
            print("  interpretation_allowed: false")
            print(
                "Smoke test only: synthetic marginals are not LLaDA output or scientific evidence."
            )
            return 0
    except (OSError, ValueError) as exc:
        print(f"dfi: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
