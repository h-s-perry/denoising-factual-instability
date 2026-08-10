from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dfi.evaluation import evaluate_rows
from dfi.pipeline import (
    cache_row_from_result,
    hydrate_cache_shards,
    inspect_cache_manifest,
    publish_cache_shard,
    read_parquet_rows,
    sha256_file,
    validate_cache_manifest,
    write_run_bundle,
)
from dfi.scoring import reduce_positions, score_marginals, stats_as_lists

PLAN_HASH = "d" * 64
FINGERPRINT = {
    "scoring_protocol": "analytic-v1",
    "model_repository": "synthetic/model",
    "model_revision": "model-revision",
    "tokenizer_revision": "tokenizer-revision",
    "remote_code_revision": "remote-code-revision",
    "prompt_protocol": "claim-prefix-v1",
    "mask_policy": "fixed-v1",
    "temperature": 1.0,
    "dtype": "float64",
    "inference_backend": "saved-marginals-v1",
}
CREATION_ENVIRONMENT = {
    "python": "3.11-test",
    "platform": "test-platform",
    "pyarrow": "test-version",
}


def _result_row(
    *,
    work_id: str,
    example_id: str,
    family_id: str,
    label: str,
    paired_example_id: str | None,
    probabilities: list[float],
) -> dict[str, object]:
    stats = score_marginals(np.asarray([probabilities], dtype=np.float64), [0])
    row: dict[str, object] = {
        "work_id": work_id,
        "example_id": example_id,
        "family_id": family_id,
        "split": "dev",
        "label": label,
        "paired_example_id": paired_example_id,
        "arm": "prior",
        "mask_index": 0,
        "mask_rate": 0.5,
        "masked_positions": [2],
    }
    row.update(stats_as_lists(stats))
    row.update(reduce_positions(stats))
    return row


def _rows() -> list[dict[str, object]]:
    return [
        _result_row(
            work_id="a" * 64,
            example_id="supported",
            family_id="family",
            label="supported",
            paired_example_id="refuted",
            probabilities=[0.8, 0.15, 0.05],
        ),
        _result_row(
            work_id="b" * 64,
            example_id="refuted",
            family_id="family",
            label="refuted",
            paired_example_id="supported",
            probabilities=[0.1, 0.7, 0.2],
        ),
    ]


def _cache_rows() -> list[dict[str, object]]:
    return [cache_row_from_result(row) for row in _rows()]


def _publish_two_partitions(cache: Path) -> None:
    rows = _cache_rows()
    expected_ids = {row["work_id"] for row in rows}
    publish_cache_shard(
        cache,
        rows[:1],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
    )
    publish_cache_shard(
        cache,
        rows[1:],
        partition_index=1,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
    )


def test_two_file_bundle_round_trip_and_default_gate(tmp_path: Path) -> None:
    rows = _rows()
    metrics = evaluate_rows(rows, interpretation_reason="test fixture")
    expected_ids = {row["work_id"] for row in rows}
    results, receipt = write_run_bundle(
        tmp_path / "run",
        rows,
        receipt={"run_uuid": "test", "evaluation": metrics},
        expected_work_ids=expected_ids,
    )
    assert sorted(path.name for path in results.parent.iterdir()) == [
        "results.parquet",
        "run.json",
    ]
    assert read_parquet_rows(results) == rows
    loaded = json.loads(receipt.read_text(encoding="utf-8"))
    assert loaded["results"]["sha256"] == sha256_file(results)
    assert loaded["results"]["row_count"] == 2
    assert loaded["interpretation_allowed"] is False
    assert loaded["interpretation_reason"]


def test_run_bundle_rejects_duplicates_missing_rows_and_unjustified_gate(
    tmp_path: Path,
) -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate work IDs"):
        write_run_bundle(
            tmp_path / "duplicate",
            [rows[0], dict(rows[0])],
            receipt={"run_uuid": "duplicate"},
            expected_work_ids={"a" * 64},
        )
    with pytest.raises(ValueError, match="do not match the plan"):
        write_run_bundle(
            tmp_path / "missing",
            rows,
            receipt={"run_uuid": "missing"},
            expected_work_ids={"a" * 64, "b" * 64, "c" * 64},
        )
    with pytest.raises(ValueError, match="requires an explicit reason"):
        write_run_bundle(
            tmp_path / "gate",
            rows,
            receipt={"run_uuid": "gate", "interpretation_allowed": True},
            expected_work_ids={"a" * 64, "b" * 64},
        )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "run.partial.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="empty directory"):
        write_run_bundle(
            occupied,
            rows,
            receipt={"run_uuid": "occupied"},
            expected_work_ids={"a" * 64, "b" * 64},
        )
    with pytest.raises(ValueError, match="run_uuid"):
        write_run_bundle(
            tmp_path / "missing-uuid",
            rows,
            receipt={},
            expected_work_ids={"a" * 64, "b" * 64},
        )


def test_cache_hydrates_isolates_corruption_and_repairs_one_partition(
    tmp_path: Path,
) -> None:
    rows = _cache_rows()
    cache = tmp_path / "drive" / "fingerprint" / "plan"
    _publish_two_partitions(cache)
    expected_ids = {row["work_id"] for row in rows}
    shards = validate_cache_manifest(
        cache / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
    )
    hydrated = hydrate_cache_shards(shards, tmp_path / "local")
    assert [shard.sha256 for shard in shards] == [sha256_file(path) for path in hydrated]

    shards[0].path.write_bytes(shards[0].path.read_bytes()[:-8])
    inspection = inspect_cache_manifest(
        cache / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
    )
    assert [shard.partition_index for shard in inspection.valid_shards] == [1]
    assert [failure.partition_index for failure in inspection.failures] == [0]
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_cache_manifest(
            cache / "cache.json",
            expected_plan_hash=PLAN_HASH,
            expected_fingerprint=FINGERPRINT,
        )

    publish_cache_shard(
        cache,
        rows[:1],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
        repair_invalid_partition=True,
    )
    repaired = validate_cache_manifest(
        cache / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
    )
    assert len(repaired) == 2
    manifest = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    assert manifest["generation"] == 3


def test_multiple_corrupt_partitions_can_be_repaired_sequentially(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    rows = _cache_rows()
    expected_ids = {row["work_id"] for row in rows}
    _publish_two_partitions(cache)
    shards = validate_cache_manifest(
        cache / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
    )
    for shard in shards:
        shard.path.write_bytes(shard.path.read_bytes()[:-8])

    publish_cache_shard(
        cache,
        rows[:1],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
        repair_invalid_partition=True,
    )
    halfway = inspect_cache_manifest(
        cache / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
    )
    assert [shard.partition_index for shard in halfway.valid_shards] == [0]
    assert [failure.partition_index for failure in halfway.failures] == [1]

    publish_cache_shard(
        cache,
        rows[1:],
        partition_index=1,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
        repair_invalid_partition=True,
    )
    assert (
        len(
            validate_cache_manifest(
                cache / "cache.json",
                expected_plan_hash=PLAN_HASH,
                expected_fingerprint=FINGERPRINT,
                expected_work_ids=expected_ids,
            )
        )
        == 2
    )


def test_out_of_plan_partition_can_be_replaced_as_a_cache_miss(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    rows = _cache_rows()
    publish_cache_shard(
        cache,
        rows[1:],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids={row["work_id"] for row in rows},
        creation_environment=CREATION_ENVIRONMENT,
    )
    publish_cache_shard(
        cache,
        rows[:1],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids={rows[0]["work_id"]},
        creation_environment=CREATION_ENVIRONMENT,
        repair_invalid_partition=True,
    )
    shards = validate_cache_manifest(
        cache / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids={rows[0]["work_id"]},
    )
    assert shards[0].work_ids == {rows[0]["work_id"]}


def test_cache_rows_exclude_logical_metadata_and_full_logits(tmp_path: Path) -> None:
    result = _rows()[0]
    cache_row = cache_row_from_result(result)
    for field in ("example_id", "family_id", "label", "paired_example_id"):
        assert field not in cache_row
    assert "logits" not in cache_row

    with pytest.raises(ValueError, match="cache row 0 has the wrong schema"):
        publish_cache_shard(
            tmp_path / "cache",
            [result],
            partition_index=0,
            plan_hash=PLAN_HASH,
            fingerprint=FINGERPRINT,
            expected_work_ids={result["work_id"]},
            creation_environment=CREATION_ENVIRONMENT,
        )


def test_exact_result_schema_and_statistical_coherence_are_enforced(tmp_path: Path) -> None:
    row = _rows()[0]
    with_extra = {**row, "logits": [[1.0, 2.0]]}
    with pytest.raises(ValueError, match=r"extra=.*logits"):
        write_run_bundle(
            tmp_path / "extra",
            [with_extra],
            receipt={"run_uuid": "extra"},
            expected_work_ids={"a" * 64},
        )

    incoherent = dict(row)
    incoherent["delta_mean"] = float(incoherent["delta_mean"]) + 1.0
    with pytest.raises(ValueError, match="incoherent reduction delta_mean"):
        write_run_bundle(
            tmp_path / "incoherent",
            [incoherent],
            receipt={"run_uuid": "incoherent"},
            expected_work_ids={"a" * 64},
        )


def test_manifest_fingerprint_comparison_is_type_exact(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    publish_cache_shard(
        cache,
        _cache_rows(),
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids={row["work_id"] for row in _cache_rows()},
        creation_environment=CREATION_ENVIRONMENT,
    )
    with pytest.raises(ValueError, match="fingerprint does not match"):
        validate_cache_manifest(
            cache / "cache.json",
            expected_plan_hash=PLAN_HASH,
            expected_fingerprint={**FINGERPRINT, "temperature": 1},
        )
    incomplete = dict(FINGERPRINT)
    del incomplete["remote_code_revision"]
    with pytest.raises(ValueError, match="exactly these fields"):
        publish_cache_shard(
            tmp_path / "incomplete-fingerprint",
            _cache_rows(),
            partition_index=0,
            plan_hash=PLAN_HASH,
            fingerprint=incomplete,
            expected_work_ids={row["work_id"] for row in _cache_rows()},
            creation_environment=CREATION_ENVIRONMENT,
        )
    with pytest.raises(ValueError, match="must include"):
        publish_cache_shard(
            tmp_path / "incomplete-environment",
            _cache_rows(),
            partition_index=0,
            plan_hash=PLAN_HASH,
            fingerprint=FINGERPRINT,
            expected_work_ids={row["work_id"] for row in _cache_rows()},
            creation_environment={"python": "3.11-test"},
        )


@pytest.mark.parametrize(
    "mutation",
    ["generation-bool", "extra-key", "duplicate-partition", "missing-environment"],
)
def test_malformed_manifest_is_rejected(tmp_path: Path, mutation: str) -> None:
    cache = tmp_path / mutation
    publish_cache_shard(
        cache,
        _cache_rows(),
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids={row["work_id"] for row in _cache_rows()},
        creation_environment=CREATION_ENVIRONMENT,
    )
    manifest_path = cache / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "generation-bool":
        manifest["generation"] = True
    elif mutation == "extra-key":
        manifest["unexpected"] = "schema drift"
    elif mutation == "missing-environment":
        del manifest["shards"][0]["creation_environment"]
    else:
        manifest["shards"].append(dict(manifest["shards"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_cache_manifest(
            manifest_path,
            expected_plan_hash=PLAN_HASH,
            expected_fingerprint=FINGERPRINT,
        )


def test_cache_partition_is_idempotent_but_valid_content_is_immutable(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    rows = _cache_rows()
    first = publish_cache_shard(
        cache,
        rows[:1],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids={row["work_id"] for row in rows},
        creation_environment=CREATION_ENVIRONMENT,
    )
    repeated = publish_cache_shard(
        cache,
        rows[:1],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids={row["work_id"] for row in rows},
        creation_environment=CREATION_ENVIRONMENT,
    )
    assert repeated == first
    with pytest.raises(ValueError, match="other valid content"):
        publish_cache_shard(
            cache,
            rows[1:],
            partition_index=0,
            plan_hash=PLAN_HASH,
            fingerprint=FINGERPRINT,
            expected_work_ids={row["work_id"] for row in rows},
            creation_environment=CREATION_ENVIRONMENT,
            repair_invalid_partition=True,
        )


def test_only_supported_and_refuted_enter_binary_auroc() -> None:
    rows = _rows()
    rows.append(
        _result_row(
            work_id="c" * 64,
            example_id="unknown",
            family_id="unknown-family",
            label="insufficient",
            paired_example_id=None,
            probabilities=[0.001, 0.899, 0.1],
        )
    )
    metrics = evaluate_rows(rows)
    assert metrics["arms"]["prior"]["scores"]["ce_mean"]["claim_auroc"] == 1.0
    assert metrics["arms"]["prior"]["label_counts"]["insufficient"] == 1
