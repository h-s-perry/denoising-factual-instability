from __future__ import annotations

import shutil
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest

import dfi.pipeline as pipeline
from dfi.pipeline import (
    BoundedCacheUploader,
    cache_row_from_result,
    hydrate_cache_shards,
    inspect_cache_manifest,
    partition_work_ids,
    publish_cache_shard,
    read_validated_cache_rows,
    seal_cache_rows,
    upload_sealed_cache_shard,
    validate_cache_manifest,
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


def _cache_row(work_id: str, probabilities: list[float]) -> dict[str, object]:
    stats = score_marginals(np.asarray([probabilities], dtype=np.float64), [0])
    result: dict[str, object] = {
        "work_id": work_id,
        "example_id": f"example-{work_id[0]}",
        "family_id": "family",
        "split": "dev",
        "label": "supported",
        "paired_example_id": None,
        "arm": "prior",
        "mask_index": 0,
        "mask_rate": 0.5,
        "masked_positions": [2],
    }
    result.update(stats_as_lists(stats))
    result.update(reduce_positions(stats))
    return cache_row_from_result(result)


def _rows() -> list[dict[str, object]]:
    return [
        _cache_row("a" * 64, [0.80, 0.15, 0.05]),
        _cache_row("b" * 64, [0.10, 0.70, 0.20]),
        _cache_row("c" * 64, [0.25, 0.25, 0.50]),
        _cache_row("d" * 64, [0.60, 0.10, 0.30]),
    ]


def test_exact_partition_membership_rejects_incomplete_and_wrong_shards(
    tmp_path: Path,
) -> None:
    rows = _rows()
    expected_ids = {str(row["work_id"]) for row in rows}
    partitions = partition_work_ids(sorted(expected_ids), 2)
    assert partitions == {
        0: frozenset({"a" * 64, "b" * 64}),
        1: frozenset({"c" * 64, "d" * 64}),
    }
    with pytest.raises(TypeError, match="ordered iterable"):
        partition_work_ids(expected_ids, 2)
    assert partition_work_ids(["c" * 64, "a" * 64, "b" * 64, "d" * 64], 2) == {
        0: frozenset({"c" * 64, "a" * 64}),
        1: frozenset({"b" * 64, "d" * 64}),
    }

    with pytest.raises(ValueError, match="exactly match expected partition"):
        seal_cache_rows(
            tmp_path / "incomplete-local",
            rows[:1],
            partition_index=0,
            expected_partition_work_ids=set(partitions[0]),
            creation_environment=CREATION_ENVIRONMENT,
        )
    with pytest.raises(ValueError, match="exactly match expected partition"):
        publish_cache_shard(
            tmp_path / "wrong-cache",
            [rows[0], rows[2]],
            partition_index=0,
            plan_hash=PLAN_HASH,
            fingerprint=FINGERPRINT,
            expected_work_ids=expected_ids,
            expected_partitions=partitions,
            creation_environment=CREATION_ENVIRONMENT,
        )
    with pytest.raises(ValueError, match="outside expected_partitions"):
        publish_cache_shard(
            tmp_path / "wrong-index-cache",
            rows[:1],
            partition_index=2,
            plan_hash=PLAN_HASH,
            fingerprint=FINGERPRINT,
            expected_work_ids=expected_ids,
            expected_partitions=partitions,
            creation_environment=CREATION_ENVIRONMENT,
        )

    legacy_cache = tmp_path / "legacy-partial-cache"
    publish_cache_shard(
        legacy_cache,
        rows[:1],
        partition_index=0,
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
    )
    inspection = inspect_cache_manifest(
        legacy_cache / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        expected_partitions=partitions,
    )
    assert not inspection.valid_shards
    assert [failure.partition_index for failure in inspection.failures] == [0]
    assert "incomplete or wrong partition membership" in inspection.failures[0].reason


def test_sealed_upload_hydrates_to_fresh_local_and_reads_exact_physical_rows(
    tmp_path: Path,
) -> None:
    rows = _rows()
    expected_ids = {str(row["work_id"]) for row in rows}
    partitions = partition_work_ids(sorted(expected_ids), 2)
    local = tmp_path / "local-sealed"
    drive = tmp_path / "drive-cache"
    sealed = []
    for partition_index in range(2):
        partition_rows = [row for row in rows if str(row["work_id"]) in partitions[partition_index]]
        shard = seal_cache_rows(
            local,
            partition_rows,
            partition_index=partition_index,
            expected_partition_work_ids=set(partitions[partition_index]),
            creation_environment=CREATION_ENVIRONMENT,
        )
        upload_sealed_cache_shard(
            drive,
            shard,
            plan_hash=PLAN_HASH,
            fingerprint=FINGERPRINT,
            expected_work_ids=expected_ids,
            expected_partitions=partitions,
        )
        sealed.append(shard)

    validated = validate_cache_manifest(
        drive / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        expected_partitions=partitions,
    )
    shutil.rmtree(local)
    fresh = tmp_path / "fresh-local"
    hydrated = hydrate_cache_shards(validated, fresh)
    readback = [
        row
        for shard, path in zip(validated, hydrated, strict=True)
        for row in read_validated_cache_rows(shard, path=path)
    ]
    assert sorted(readback, key=lambda row: row["work_id"]) == sorted(
        rows, key=lambda row: row["work_id"]
    )

    hydrated[0].write_bytes(hydrated[0].read_bytes()[:-8])
    with pytest.raises(ValueError, match="hash differs"):
        read_validated_cache_rows(validated[0], path=hydrated[0])

    validated[0].path.write_bytes(validated[0].path.read_bytes()[:-8])
    inspection = inspect_cache_manifest(
        drive / "cache.json",
        expected_plan_hash=PLAN_HASH,
        expected_fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        expected_partitions=partitions,
    )
    assert [shard.partition_index for shard in inspection.valid_shards] == [1]
    assert [failure.partition_index for failure in inspection.failures] == [0]


def test_bounded_single_writer_drains_and_records_queue_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _rows()[:3]
    expected_ids = {str(row["work_id"]) for row in rows}
    partitions = partition_work_ids(sorted(expected_ids), 1)
    sealed = [
        seal_cache_rows(
            tmp_path / "local",
            [row],
            partition_index=index,
            expected_partition_work_ids=set(partitions[index]),
            creation_environment=CREATION_ENVIRONMENT,
        )
        for index, row in enumerate(rows)
    ]

    original_upload = pipeline.upload_sealed_cache_shard

    def slow_upload(*args: object, **kwargs: object) -> dict[str, object]:
        time.sleep(0.03)
        return original_upload(*args, **kwargs)

    monkeypatch.setattr(pipeline, "upload_sealed_cache_shard", slow_upload)
    uploader = BoundedCacheUploader(
        tmp_path / "drive",
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        expected_partitions=partitions,
        max_queue_size=1,
    )
    with uploader:
        for shard in sealed:
            uploader.submit(shard)

    metrics = uploader.metrics
    assert metrics.submitted_shards == 3
    assert metrics.completed_shards == 3
    assert metrics.failed_shards == 0
    assert metrics.skipped_shards == 0
    assert metrics.uploaded_bytes == sum(shard.size_bytes for shard in sealed)
    assert metrics.queue_stall_seconds >= 0.01
    assert metrics.upload_seconds >= 0.09
    assert uploader.latest_manifest is not None
    assert uploader.latest_manifest["generation"] == 3
    assert (
        len(
            validate_cache_manifest(
                tmp_path / "drive" / "cache.json",
                expected_plan_hash=PLAN_HASH,
                expected_fingerprint=FINGERPRINT,
                expected_work_ids=expected_ids,
                expected_partitions=partitions,
            )
        )
        == 3
    )


def test_bounded_single_writer_propagates_worker_failure_after_final_drain(
    tmp_path: Path,
) -> None:
    work_id = "a" * 64
    expected_ids = {work_id}
    partitions = partition_work_ids(sorted(expected_ids), 1)
    first = seal_cache_rows(
        tmp_path / "local-first",
        [_cache_row(work_id, [0.80, 0.15, 0.05])],
        partition_index=0,
        expected_partition_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
    )
    conflicting = seal_cache_rows(
        tmp_path / "local-conflict",
        [_cache_row(work_id, [0.55, 0.30, 0.15])],
        partition_index=0,
        expected_partition_work_ids=expected_ids,
        creation_environment=CREATION_ENVIRONMENT,
    )
    uploader = BoundedCacheUploader(
        tmp_path / "drive",
        plan_hash=PLAN_HASH,
        fingerprint=FINGERPRINT,
        expected_work_ids=expected_ids,
        expected_partitions=partitions,
        max_queue_size=2,
    )
    uploader.submit(first)
    with suppress(RuntimeError):
        uploader.submit(conflicting)
    with pytest.raises(RuntimeError, match="cache uploader failed"):
        uploader.close()
    metrics = uploader.metrics
    assert metrics.completed_shards == 1
    assert metrics.failed_shards == 1
    assert metrics.completed_shards + metrics.failed_shards + metrics.skipped_shards == 2
    assert (
        len(
            validate_cache_manifest(
                tmp_path / "drive" / "cache.json",
                expected_plan_hash=PLAN_HASH,
                expected_fingerprint=FINGERPRINT,
                expected_work_ids=expected_ids,
                expected_partitions=partitions,
            )
        )
        == 1
    )
