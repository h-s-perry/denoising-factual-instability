"""Atomic artifacts and the immutable cache contract for the offline slice."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from dfi.masking import canonical_json

CACHE_SCHEMA_VERSION = "analytic-cache-stats-v1"
RESULT_SCHEMA_VERSION = "analytic-mask-results-v1"
RUN_SCHEMA_VERSION = "run-v1"
WORK_ID_RE = re.compile(r"[0-9a-f]{64}")
FINGERPRINT_COLUMNS = (
    "scoring_protocol",
    "model_repository",
    "model_revision",
    "tokenizer_revision",
    "remote_code_revision",
    "prompt_protocol",
    "mask_policy",
    "temperature",
    "dtype",
    "inference_backend",
)
FINGERPRINT_REQUIRED_COLUMNS = frozenset(FINGERPRINT_COLUMNS)
CREATION_ENVIRONMENT_REQUIRED_COLUMNS = frozenset({"python", "platform", "pyarrow"})

POSITION_LIST_COLUMNS = (
    "target_ids",
    "top_ids",
    "top_alternative_ids",
    "ce_by_piece",
    "entropy_by_piece",
    "collision_probability_by_piece",
    "collision_entropy_by_piece",
    "delta_by_piece",
    "swap_llr_by_piece",
    "expected_drift_by_piece",
    "expected_dispersion_by_piece",
    "top_mismatch_by_piece",
)
REDUCED_COLUMNS = (
    "ce_mean",
    "entropy_mean",
    "collision_entropy_mean",
    "delta_mean",
    "swap_llr_mean",
    "expected_drift_mean",
    "expected_dispersion_mean",
    "top_mismatch_rate",
)
LOGICAL_RESULT_COLUMNS = (
    "example_id",
    "family_id",
    "split",
    "label",
    "paired_example_id",
    "arm",
    "mask_index",
    "mask_rate",
)
CACHE_COLUMNS = (
    "work_id",
    "masked_positions",
    "n_masked_pieces",
    *POSITION_LIST_COLUMNS,
    *REDUCED_COLUMNS,
)
RESULT_COLUMNS = (
    "work_id",
    *LOGICAL_RESULT_COLUMNS,
    "masked_positions",
    "n_masked_pieces",
    *POSITION_LIST_COLUMNS,
    *REDUCED_COLUMNS,
)
CACHE_REQUIRED_COLUMNS = frozenset(CACHE_COLUMNS)
RESULT_REQUIRED_COLUMNS = frozenset(RESULT_COLUMNS)

_REDUCTION_SOURCES = {
    "ce_mean": "ce_by_piece",
    "entropy_mean": "entropy_by_piece",
    "collision_entropy_mean": "collision_entropy_by_piece",
    "delta_mean": "delta_by_piece",
    "swap_llr_mean": "swap_llr_by_piece",
    "expected_drift_mean": "expected_drift_by_piece",
    "expected_dispersion_mean": "expected_dispersion_by_piece",
    "top_mismatch_rate": "top_mismatch_by_piece",
}


@dataclass(frozen=True)
class ValidatedShard:
    """A cache shard bound to the digest verified with its manifest."""

    path: Path
    sha256: str
    row_count: int
    work_ids: frozenset[str]
    partition_index: int
    creation_environment: dict[str, Any]


@dataclass(frozen=True)
class CacheShardFailure:
    """One partition-scoped cache miss discovered during inspection."""

    partition_index: int
    name: str
    reason: str


@dataclass(frozen=True)
class CacheInspection:
    """Valid cache shards plus isolated partition failures."""

    valid_shards: tuple[ValidatedShard, ...]
    failures: tuple[CacheShardFailure, ...]
    work_ids: frozenset[str]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and WORK_ID_RE.fullmatch(value) is not None


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 string")
    return value


def _require_fingerprint(value: Any, label: str = "fingerprint") -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != FINGERPRINT_REQUIRED_COLUMNS:
        raise ValueError(
            f"{label} must have exactly these fields: {sorted(FINGERPRINT_REQUIRED_COLUMNS)}"
        )
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite canonical JSON values") from exc
    for field in FINGERPRINT_COLUMNS:
        if field == "temperature":
            continue
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{label}.{field} must be a non-empty string")
    temperature = _number(value["temperature"], label=f"{label}.temperature")
    if temperature <= 0.0:
        raise ValueError(f"{label}.temperature must be positive")
    if value["scoring_protocol"] == "analytic-v1" and temperature != 1.0:
        raise ValueError(f"{label}: analytic-v1 requires temperature exactly 1.0")
    return value


def _require_creation_environment(
    value: Any, label: str = "creation_environment"
) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value) >= CREATION_ENVIRONMENT_REQUIRED_COLUMNS:
        raise ValueError(f"{label} must include: {sorted(CREATION_ENVIRONMENT_REQUIRED_COLUMNS)}")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite canonical JSON values") from exc
    for field in CREATION_ENVIRONMENT_REQUIRED_COLUMNS:
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{label}.{field} must be a non-empty string")
    return value


def local_creation_environment() -> dict[str, str]:
    """Return the minimal creation environment recorded for an offline shard."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyarrow": pa.__version__,
    }


def _same_canonical_json(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _validate_physical_row(row: dict[str, Any], index: int, *, kind: str) -> None:
    work_id = row["work_id"]
    if not _is_sha256(work_id):
        raise ValueError(f"{kind} row {index} has an invalid work_id")

    n_pieces = row["n_masked_pieces"]
    if isinstance(n_pieces, bool) or not isinstance(n_pieces, int) or n_pieces < 1:
        raise ValueError(f"{kind} row {index} has an invalid n_masked_pieces")
    positions = row["masked_positions"]
    if (
        not isinstance(positions, list)
        or len(positions) != n_pieces
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in positions
        )
        or positions != sorted(set(positions))
    ):
        raise ValueError(f"{kind} row {index} has invalid masked_positions")

    numeric_lists: dict[str, list[float]] = {}
    integer_lists: dict[str, list[int]] = {}
    for field in POSITION_LIST_COLUMNS:
        values = row[field]
        if not isinstance(values, list) or len(values) != n_pieces:
            raise ValueError(f"{kind} row {index} has invalid {field} cardinality")
        if field in {"target_ids", "top_ids", "top_alternative_ids"}:
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ):
                raise ValueError(f"{kind} row {index} has invalid {field}")
            integer_lists[field] = values
        else:
            numeric_lists[field] = [
                _number(value, label=f"{kind} row {index} {field}") for value in values
            ]

    ce = numeric_lists["ce_by_piece"]
    entropy = numeric_lists["entropy_by_piece"]
    collision = numeric_lists["collision_probability_by_piece"]
    collision_entropy = numeric_lists["collision_entropy_by_piece"]
    delta = numeric_lists["delta_by_piece"]
    drift = numeric_lists["expected_drift_by_piece"]
    dispersion = numeric_lists["expected_dispersion_by_piece"]
    mismatch = numeric_lists["top_mismatch_by_piece"]
    targets = integer_lists["target_ids"]
    top_ids = integer_lists["top_ids"]
    alternatives = integer_lists["top_alternative_ids"]

    for piece in range(n_pieces):
        if ce[piece] < 0.0 or entropy[piece] < 0.0 or collision_entropy[piece] < 0.0:
            raise ValueError(f"{kind} row {index} has a negative information statistic")
        if not 0.0 < collision[piece] <= 1.0:
            raise ValueError(f"{kind} row {index} has invalid collision probability")
        if not 0.0 <= drift[piece] <= 1.0 or not 0.0 <= dispersion[piece] <= 1.0:
            raise ValueError(f"{kind} row {index} has invalid expected-risk probability")
        if mismatch[piece] not in {0.0, 1.0}:
            raise ValueError(f"{kind} row {index} has a non-binary top mismatch")
        if alternatives[piece] == targets[piece]:
            raise ValueError(f"{kind} row {index} does not exclude the target alternative")
        expected_mismatch = float(top_ids[piece] != targets[piece])
        if mismatch[piece] != expected_mismatch:
            raise ValueError(f"{kind} row {index} has incoherent top-mismatch identity")
        if not _close(delta[piece], ce[piece] - entropy[piece]):
            raise ValueError(f"{kind} row {index} has incoherent delta")
        if not _close(collision_entropy[piece], -math.log(collision[piece])):
            raise ValueError(f"{kind} row {index} has incoherent collision entropy")
        if not _close(dispersion[piece], 1.0 - collision[piece]):
            raise ValueError(f"{kind} row {index} has incoherent expected dispersion")
        if not _close(drift[piece], 1.0 - math.exp(-ce[piece])):
            raise ValueError(f"{kind} row {index} has incoherent expected drift")

    for field in REDUCED_COLUMNS:
        value = _number(row[field], label=f"{kind} row {index} {field}")
        source = numeric_lists[_REDUCTION_SOURCES[field]]
        expected = math.fsum(source) / len(source)
        if not _close(value, expected):
            raise ValueError(f"{kind} row {index} has incoherent reduction {field}")


def _validate_cache_rows(rows: list[dict[str, Any]]) -> None:
    """Validate the exact model-output-only cache schema."""

    if not rows:
        raise ValueError("cache table must contain at least one row")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != CACHE_REQUIRED_COLUMNS:
            actual = set(row) if isinstance(row, dict) else set()
            missing = sorted(CACHE_REQUIRED_COLUMNS - actual)
            extra = sorted(actual - CACHE_REQUIRED_COLUMNS)
            raise ValueError(
                f"cache row {index} has the wrong schema: missing={missing}, extra={extra}"
            )
        _validate_physical_row(row, index, kind="cache")


def _validate_result_rows(rows: list[dict[str, Any]]) -> None:
    """Validate the exact logical result schema and all sufficient statistics."""

    if not rows:
        raise ValueError("result table must contain at least one row")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != RESULT_REQUIRED_COLUMNS:
            actual = set(row) if isinstance(row, dict) else set()
            missing = sorted(RESULT_REQUIRED_COLUMNS - actual)
            extra = sorted(actual - RESULT_REQUIRED_COLUMNS)
            raise ValueError(
                f"result row {index} has the wrong schema: missing={missing}, extra={extra}"
            )
        _validate_physical_row(row, index, kind="result")
        for field in ("example_id", "family_id", "split", "label", "arm"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"result row {index} has an invalid {field}")
        paired_id = row["paired_example_id"]
        if paired_id is not None and (not isinstance(paired_id, str) or not paired_id):
            raise ValueError(f"result row {index} has an invalid paired_example_id")
        mask_index = row["mask_index"]
        if isinstance(mask_index, bool) or not isinstance(mask_index, int) or mask_index < 0:
            raise ValueError(f"result row {index} has an invalid mask_index")
        mask_rate = _number(row["mask_rate"], label=f"result row {index} mask_rate")
        if not 0.0 < mask_rate <= 1.0:
            raise ValueError(f"result row {index} has an invalid mask_rate")


def cache_row_from_result(row: dict[str, Any]) -> dict[str, Any]:
    """Project one validated logical result onto reusable physical statistics."""

    _validate_result_rows([row])
    return {column: row[column] for column in CACHE_COLUMNS}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a same-filesystem rename on platforms that support directory fsync."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Write canonical UTF-8 JSON through a same-directory atomic rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _write_parquet_atomic(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    kind: Literal["cache", "result"],
) -> Path:
    materialized = list(rows)
    if kind == "cache":
        _validate_cache_rows(materialized)
        columns = CACHE_COLUMNS
        metadata_key = b"dfi.cache_schema"
        schema_version = CACHE_SCHEMA_VERSION
    else:
        _validate_result_rows(materialized)
        columns = RESULT_COLUMNS
        metadata_key = b"dfi.result_schema"
        schema_version = RESULT_SCHEMA_VERSION
    normalized = [{column: row[column] for column in columns} for row in materialized]

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist(normalized)
        metadata = dict(table.schema.metadata or {})
        metadata[metadata_key] = schema_version.encode("ascii")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, temporary, compression="zstd")
        readback_table = pq.read_table(temporary)
        if readback_table.num_rows != len(normalized):
            raise ValueError("Parquet readback row count differs from the write")
        readback_metadata = readback_table.schema.metadata or {}
        if readback_metadata.get(metadata_key) != schema_version.encode("ascii"):
            raise ValueError("Parquet readback schema marker differs from the write")
        readback = readback_table.to_pylist()
        if canonical_json(readback) != canonical_json(normalized):
            raise ValueError("Parquet content readback differs from the write")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_parquet_atomic(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Atomically write an exact-schema final logical result table."""

    return _write_parquet_atomic(rows, path, kind="result")


def _write_cache_parquet_atomic(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    return _write_parquet_atomic(rows, path, kind="cache")


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a final logical result table."""

    table = pq.read_table(Path(path))
    metadata = table.schema.metadata or {}
    if metadata.get(b"dfi.result_schema") != RESULT_SCHEMA_VERSION.encode("ascii"):
        raise ValueError("result Parquet schema marker is incompatible")
    rows = table.to_pylist()
    _validate_result_rows(rows)
    return rows


def _work_ids_digest(work_ids: Iterable[str]) -> str:
    values = sorted(str(work_id) for work_id in work_ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


_MANIFEST_KEYS = frozenset({"schema_version", "generation", "plan_hash", "fingerprint", "shards"})
_SHARD_ENTRY_KEYS = frozenset(
    {
        "partition_index",
        "name",
        "sha256",
        "row_count",
        "first_work_id",
        "last_work_id",
        "work_ids_sha256",
        "creation_environment",
    }
)


def _read_cache_manifest(
    manifest_path: Path,
    *,
    expected_plan_hash: str,
    expected_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    _require_sha256(expected_plan_hash, "expected_plan_hash")
    _require_fingerprint(expected_fingerprint, "expected_fingerprint")
    if manifest_path.is_symlink():
        raise ValueError("cache manifest must not be a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"cache manifest is missing or malformed: {manifest_path}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("cache manifest has the wrong top-level schema")
    try:
        canonical_json(manifest)
    except (TypeError, ValueError) as exc:
        raise ValueError("cache manifest contains non-canonical numeric values") from exc
    if manifest["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported cache manifest schema")
    _require_sha256(manifest["plan_hash"], "cache plan_hash")
    if manifest["plan_hash"] != expected_plan_hash:
        raise ValueError("cache plan hash does not match")
    _require_fingerprint(manifest["fingerprint"], "cache fingerprint")
    if not _same_canonical_json(manifest["fingerprint"], expected_fingerprint):
        raise ValueError("cache model/protocol fingerprint does not match")
    generation = manifest["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("cache manifest generation is invalid")
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("cache manifest has no shards")

    seen_names: set[str] = set()
    seen_partitions: set[int] = set()
    previous_partition = -1
    for entry in shards:
        if not isinstance(entry, dict) or set(entry) != _SHARD_ENTRY_KEYS:
            raise ValueError("cache shard entry has the wrong schema")
        partition_index = entry["partition_index"]
        if (
            isinstance(partition_index, bool)
            or not isinstance(partition_index, int)
            or partition_index < 0
        ):
            raise ValueError("cache shard partition_index is invalid")
        if partition_index in seen_partitions:
            raise ValueError("cache manifest repeats a partition")
        if partition_index <= previous_partition:
            raise ValueError("cache shard entries must be sorted by partition_index")
        seen_partitions.add(partition_index)
        previous_partition = partition_index

        digest = _require_sha256(entry["sha256"], "cache shard sha256")
        name = entry["name"]
        expected_name = f"part-{partition_index:04d}-{digest}.parquet"
        if not isinstance(name, str) or Path(name).name != name or name != expected_name:
            raise ValueError("cache shard name is unsafe or inconsistent")
        if name in seen_names:
            raise ValueError("cache manifest repeats a shard name")
        seen_names.add(name)
        row_count = entry["row_count"]
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise ValueError(f"cache shard row count is invalid: {name}")
        first = _require_sha256(entry["first_work_id"], "cache first_work_id")
        last = _require_sha256(entry["last_work_id"], "cache last_work_id")
        if first > last:
            raise ValueError(f"cache shard work-ID range is invalid: {name}")
        _require_sha256(entry["work_ids_sha256"], "cache work_ids_sha256")
        _require_creation_environment(
            entry["creation_environment"], f"cache shard {name} creation_environment"
        )
    return manifest


def _validate_cache_shard(
    manifest_path: Path,
    entry: dict[str, Any],
    *,
    cache_root: Path,
    expected_work_ids: set[str] | None,
) -> ValidatedShard:
    partition_index = entry["partition_index"]
    name = entry["name"]
    shard = manifest_path.parent / name
    if shard.is_symlink():
        raise ValueError(f"cache shard must not be a symlink: {name}")
    try:
        shard_stat = shard.stat(follow_symlinks=False)
        resolved_shard = shard.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"cache shard is missing: {name}") from exc
    if not stat.S_ISREG(shard_stat.st_mode) or resolved_shard.parent != cache_root:
        raise ValueError(f"cache shard is not a contained regular file: {name}")
    expected_hash = entry["sha256"]
    if sha256_file(shard) != expected_hash:
        raise ValueError(f"cache shard hash mismatch: {name}")
    try:
        table = pq.read_table(shard)
    except Exception as exc:
        raise ValueError(f"cache shard is not readable Parquet: {name}") from exc
    if table.num_rows != entry["row_count"]:
        raise ValueError(f"cache shard row count mismatch: {name}")
    metadata = table.schema.metadata or {}
    if metadata.get(b"dfi.cache_schema") != CACHE_SCHEMA_VERSION.encode("ascii"):
        raise ValueError(f"cache shard schema is incompatible: {name}")
    if b"dfi.result_schema" in metadata:
        raise ValueError(f"cache shard is mislabeled as a logical result: {name}")
    rows = table.to_pylist()
    try:
        _validate_cache_rows(rows)
    except ValueError as exc:
        raise ValueError(f"cache shard row validation failed: {name}: {exc}") from exc
    row_ids = [row["work_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"cache shard contains duplicate work IDs: {name}")
    if _work_ids_digest(row_ids) != entry["work_ids_sha256"]:
        raise ValueError(f"cache shard work-ID digest mismatch: {name}")
    if min(row_ids) != entry["first_work_id"] or max(row_ids) != entry["last_work_id"]:
        raise ValueError(f"cache shard work-ID range mismatch: {name}")
    row_id_set = set(row_ids)
    if expected_work_ids is not None and not row_id_set <= expected_work_ids:
        raise ValueError(f"cache shard contains work IDs outside the current plan: {name}")
    return ValidatedShard(
        path=shard,
        sha256=expected_hash,
        row_count=len(row_ids),
        work_ids=frozenset(row_ids),
        partition_index=partition_index,
        creation_environment=entry["creation_environment"],
    )


def inspect_cache_manifest(
    manifest_path: str | Path,
    *,
    expected_plan_hash: str,
    expected_fingerprint: dict[str, Any],
    expected_work_ids: set[str] | None = None,
) -> CacheInspection:
    """Inspect one generation, isolating physical shard failures as cache misses."""

    if expected_work_ids is not None:
        if not isinstance(expected_work_ids, set) or not expected_work_ids:
            raise ValueError("expected_work_ids must be a non-empty set")
        for work_id in expected_work_ids:
            _require_sha256(work_id, "expected work ID")
    selected = Path(manifest_path)
    manifest = _read_cache_manifest(
        selected,
        expected_plan_hash=expected_plan_hash,
        expected_fingerprint=expected_fingerprint,
    )
    try:
        cache_root = selected.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("cache directory is missing") from exc

    valid: list[ValidatedShard] = []
    failures: list[CacheShardFailure] = []
    seen_work_ids: set[str] = set()
    for entry in manifest["shards"]:
        try:
            shard = _validate_cache_shard(
                selected,
                entry,
                cache_root=cache_root,
                expected_work_ids=expected_work_ids,
            )
            overlap = seen_work_ids & shard.work_ids
            if overlap:
                raise ValueError("cache shards contain duplicate work IDs")
        except (OSError, ValueError) as exc:
            failures.append(
                CacheShardFailure(
                    partition_index=entry["partition_index"],
                    name=entry["name"],
                    reason=str(exc),
                )
            )
            continue
        valid.append(shard)
        seen_work_ids.update(shard.work_ids)
    return CacheInspection(tuple(valid), tuple(failures), frozenset(seen_work_ids))


def validate_cache_manifest(
    manifest_path: str | Path,
    *,
    expected_plan_hash: str,
    expected_fingerprint: dict[str, Any],
    expected_work_ids: set[str] | None = None,
) -> list[ValidatedShard]:
    """Strictly validate every shard in a cache generation."""

    inspection = inspect_cache_manifest(
        manifest_path,
        expected_plan_hash=expected_plan_hash,
        expected_fingerprint=expected_fingerprint,
        expected_work_ids=expected_work_ids,
    )
    if inspection.failures:
        detail = "; ".join(
            f"partition {failure.partition_index}: {failure.reason}"
            for failure in inspection.failures
        )
        raise ValueError(f"cache validation failed: {detail}")
    return list(inspection.valid_shards)


def _remove_invalid_shard(directory: Path, entry: dict[str, Any]) -> None:
    """Remove only the exact derived shard selected by a controlled repair."""

    target = directory / entry["name"]
    if target.is_symlink():
        target.unlink()
        return
    try:
        metadata = target.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode) or target.resolve().parent != directory.resolve():
        raise ValueError("refusing to repair a cache target that is not a contained file")
    target.unlink()


def publish_cache_shard(
    cache_directory: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    partition_index: int,
    plan_hash: str,
    fingerprint: dict[str, Any],
    expected_work_ids: set[str],
    creation_environment: dict[str, Any],
    repair_invalid_partition: bool = False,
) -> dict[str, Any]:
    """Seal one physical shard and atomically publish a manifest generation.

    Cache rows deliberately omit logical example, label, and family metadata.
    A caller may repair only a partition that the existing generation proves is
    invalid; a valid immutable partition is never overwritten.
    """

    if isinstance(partition_index, bool) or not isinstance(partition_index, int):
        raise ValueError("partition_index must be an integer")
    if partition_index < 0:
        raise ValueError("partition_index must be non-negative")
    if not isinstance(repair_invalid_partition, bool):
        raise ValueError("repair_invalid_partition must be boolean")
    _require_sha256(plan_hash, "plan_hash")
    _require_fingerprint(fingerprint)
    _require_creation_environment(creation_environment)
    if not isinstance(expected_work_ids, set) or not expected_work_ids:
        raise ValueError("expected_work_ids must be a non-empty set")
    for expected_work_id in expected_work_ids:
        _require_sha256(expected_work_id, "expected work ID")
    materialized = list(rows)
    _validate_cache_rows(materialized)
    materialized.sort(key=lambda row: row["work_id"])
    work_ids = [row["work_id"] for row in materialized]
    if len(set(work_ids)) != len(work_ids):
        raise ValueError("cache shard contains duplicate work IDs")
    if not set(work_ids) <= expected_work_ids:
        raise ValueError("cache shard contains work IDs outside the current plan")

    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "cache.json"
    existing_manifest: dict[str, Any] | None = None
    existing_inspection: CacheInspection | None = None
    if manifest_path.exists() or manifest_path.is_symlink():
        existing_manifest = _read_cache_manifest(
            manifest_path,
            expected_plan_hash=plan_hash,
            expected_fingerprint=fingerprint,
        )
        existing_inspection = inspect_cache_manifest(
            manifest_path,
            expected_plan_hash=plan_hash,
            expected_fingerprint=fingerprint,
            expected_work_ids=expected_work_ids,
        )
        target_is_invalid = any(
            failure.partition_index == partition_index for failure in existing_inspection.failures
        )
        other_failures = [
            failure
            for failure in existing_inspection.failures
            if failure.partition_index != partition_index
        ]
        if other_failures and not (repair_invalid_partition and target_is_invalid):
            raise ValueError(
                "existing cache has an invalid partition; repair a failed partition first"
            )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".part-{partition_index:04d}.", suffix=".partial.parquet", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_cache_parquet_atomic(materialized, temporary)
        content_hash = sha256_file(temporary)
        shard_name = f"part-{partition_index:04d}-{content_hash}.parquet"
        entry = {
            "partition_index": partition_index,
            "name": shard_name,
            "sha256": content_hash,
            "row_count": len(materialized),
            "first_work_id": min(work_ids),
            "last_work_id": max(work_ids),
            "work_ids_sha256": _work_ids_digest(work_ids),
            "creation_environment": creation_environment,
        }

        existing_partition: dict[str, Any] | None = None
        partition_is_valid = False
        partition_is_invalid = False
        if existing_manifest is not None and existing_inspection is not None:
            existing_partition = next(
                (
                    candidate
                    for candidate in existing_manifest["shards"]
                    if candidate["partition_index"] == partition_index
                ),
                None,
            )
            partition_is_valid = any(
                shard.partition_index == partition_index
                for shard in existing_inspection.valid_shards
            )
            partition_is_invalid = any(
                failure.partition_index == partition_index
                for failure in existing_inspection.failures
            )

        if existing_partition is not None and partition_is_valid:
            existing_content = {
                key: value
                for key, value in existing_partition.items()
                if key != "creation_environment"
            }
            new_content = {
                key: value for key, value in entry.items() if key != "creation_environment"
            }
            if existing_content == new_content:
                return existing_manifest
            raise ValueError(
                f"cache partition {partition_index} is already sealed with other valid content"
            )
        if existing_partition is not None and partition_is_invalid:
            if not repair_invalid_partition:
                raise ValueError(
                    f"cache partition {partition_index} is invalid; explicit repair is required"
                )
            _remove_invalid_shard(directory, existing_partition)

        destination = directory / shard_name
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or sha256_file(destination) != content_hash:
                if not (partition_is_invalid and repair_invalid_partition):
                    raise ValueError(
                        f"immutable cache target already exists with different bytes: {destination}"
                    )
                _remove_invalid_shard(directory, entry)
            else:
                temporary.unlink(missing_ok=True)
        if temporary.exists():
            os.replace(temporary, destination)
            _fsync_directory(directory)

        cache_root = directory.resolve(strict=True)
        validated = _validate_cache_shard(
            manifest_path,
            entry,
            cache_root=cache_root,
            expected_work_ids=set(work_ids),
        )
        if validated.sha256 != content_hash or validated.row_count != len(materialized):
            raise ValueError("published cache shard failed readback validation")

        if existing_manifest is None:
            shards = [entry]
            generation = 1
        else:
            shards = [
                candidate
                for candidate in existing_manifest["shards"]
                if candidate["partition_index"] != partition_index
            ]
            shards.append(entry)
            generation = existing_manifest["generation"] + 1
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "generation": generation,
            "plan_hash": plan_hash,
            "fingerprint": fingerprint,
            "shards": sorted(shards, key=lambda item: item["partition_index"]),
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def hydrate_cache_shards(
    shards: Iterable[ValidatedShard], local_directory: str | Path
) -> list[Path]:
    """Copy already-validated shards to a local working directory."""

    destination = Path(local_directory)
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("local cache destination must not be a symlink")
    hydrated: list[Path] = []
    for shard in shards:
        if shard.path.is_symlink() or sha256_file(shard.path) != shard.sha256:
            raise ValueError(f"validated cache shard changed before hydration: {shard.path.name}")
        target = destination / shard.path.name
        if target.is_symlink():
            raise ValueError(f"hydration target must not be a symlink: {target.name}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".dfi-hydrate-", suffix=".partial", dir=destination
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(shard.path, temporary)
            if sha256_file(temporary) != shard.sha256:
                raise ValueError(f"cache copy readback failed: {shard.path.name}")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(destination)
        finally:
            temporary.unlink(missing_ok=True)
        if sha256_file(target) != shard.sha256:
            raise ValueError(f"hydrated cache hash changed: {shard.path.name}")
        hydrated.append(target)
    return hydrated


def write_run_bundle(
    run_directory: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    receipt: dict[str, Any],
    expected_work_ids: set[str],
) -> tuple[Path, Path]:
    """Seal the exact two-file final run bundle used by reductions."""

    destination = Path(run_directory)
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink() or any(destination.iterdir()):
            raise ValueError(f"run bundle destination must be an empty directory: {destination}")
    else:
        destination.mkdir(parents=True)
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    if not isinstance(receipt.get("run_uuid"), str) or not receipt["run_uuid"].strip():
        raise ValueError("receipt.run_uuid must be a non-empty string")
    if not isinstance(expected_work_ids, set) or not expected_work_ids:
        raise ValueError("expected_work_ids must be a non-empty set")
    for work_id in expected_work_ids:
        _require_sha256(work_id, "expected work ID")

    materialized = list(rows)
    _validate_result_rows(materialized)
    work_ids = [row["work_id"] for row in materialized]
    if len(work_ids) != len(set(work_ids)):
        raise ValueError("final run contains duplicate work IDs")
    if set(work_ids) != expected_work_ids:
        missing = sorted(expected_work_ids - set(work_ids))
        unexpected = sorted(set(work_ids) - expected_work_ids)
        raise ValueError(
            f"final work IDs do not match the plan: missing={missing}, unexpected={unexpected}"
        )
    materialized.sort(key=lambda row: row["work_id"])

    finalized = dict(receipt)
    interpretation_allowed = finalized.setdefault("interpretation_allowed", False)
    if not isinstance(interpretation_allowed, bool):
        raise ValueError("interpretation_allowed must be boolean")
    if "interpretation_reason" not in finalized:
        if interpretation_allowed:
            raise ValueError("an allowed interpretation requires an explicit reason")
        finalized["interpretation_reason"] = "data and protocol admission gates have not passed"
    if (
        not isinstance(finalized["interpretation_reason"], str)
        or not finalized["interpretation_reason"].strip()
    ):
        raise ValueError("interpretation_reason must be a non-empty string")
    try:
        canonical_json(finalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("receipt must contain finite canonical JSON values") from exc

    results = write_parquet_atomic(materialized, destination / "results.parquet")
    finalized.update(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "complete",
            "results": {
                "path": "results.parquet",
                "sha256": sha256_file(results),
                "schema": RESULT_SCHEMA_VERSION,
                "row_count": pq.read_metadata(results).num_rows,
            },
        }
    )
    run_json = atomic_write_json(destination / "run.json", finalized)
    return results, run_json
