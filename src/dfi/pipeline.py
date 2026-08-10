"""Atomic artifacts and the immutable cache contract for the offline slice."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import queue
import re
import shutil
import stat
import statistics
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from dfi.config import DFIConfig, load_config
from dfi.data import ScoringExample, load_jsonl
from dfi.evaluation import RISK_COLUMNS, evaluate_rows, reduce_claims
from dfi.llada import (
    DTYPE_NAME,
    INFERENCE_BACKEND,
    PROMPT_PROTOCOL,
    SCORING_PROTOCOL,
    TEMPERATURE,
    AnalyticResult,
    LLaDABackend,
    ParityMismatchError,
    WorkRequest,
    plan_length_buckets,
    run_scalar_batch_parity,
)
from dfi.masking import canonical_json, deterministic_masks

CACHE_SCHEMA_VERSION = "analytic-cache-stats-v1"
RESULT_SCHEMA_VERSION = "analytic-mask-results-v1"
RUN_SCHEMA_VERSION = "run-v1"
PARTIAL_RUN_SCHEMA_VERSION = "run-partial-v1"
PLAN_SCHEMA_VERSION = "inference-plan-v1"
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
class SealedCacheShard:
    """One immutable local physical shard ready for cache publication."""

    path: Path
    sha256: str
    size_bytes: int
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


@dataclass(frozen=True)
class CacheUploadMetrics:
    """Thread-safe snapshot of bounded cache-uploader activity."""

    submitted_shards: int
    completed_shards: int
    failed_shards: int
    skipped_shards: int
    uploaded_bytes: int
    queue_stall_seconds: float
    upload_seconds: float


@dataclass(frozen=True)
class _UploadTask:
    shard: SealedCacheShard
    repair_invalid_partition: bool


@dataclass(frozen=True)
class LogicalRequest:
    """One logical example/arm/mask row backed by a physical work request."""

    logical_id: str
    work_id: str
    example_id: str
    family_id: str
    split: str
    label: str
    paired_example_id: str | None
    arm: str
    mask_index: int
    mask_rate: float


@dataclass(frozen=True)
class InferencePlan:
    """Deterministic logical and deduplicated physical work for one run."""

    fingerprint: dict[str, Any]
    fingerprint_hash: str
    plan_hash: str
    logical_requests: tuple[LogicalRequest, ...]
    physical_requests: tuple[WorkRequest, ...]
    execution_order: tuple[str, ...]
    partitions: dict[int, frozenset[str]]

    @property
    def expected_work_ids(self) -> set[str]:
        return {request.work_id for request in self.physical_requests}

    @property
    def expected_logical_ids(self) -> set[str]:
        return {request.logical_id for request in self.logical_requests}


@dataclass
class InferenceTelemetry:
    """Mutable counters for one runner invocation."""

    forwards: int = 0
    successful_batches: int = 0
    completed_requests: int = 0
    real_tokens: int = 0
    padded_tokens: int = 0
    max_batch_rows: int = 0
    max_padded_tokens: int = 0
    oom_fallbacks: int = 0
    fallback_batch_rows: int | None = None
    inference_seconds: float = 0.0
    batch_rows: list[int] = dataclass_field(default_factory=list)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and WORK_ID_RE.fullmatch(value) is not None


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 string")
    return value


def partition_work_ids(
    work_ids: Iterable[str], partition_requests: int
) -> dict[int, frozenset[str]]:
    """Partition a deterministic ordered plan into exact contiguous memberships."""

    if (
        isinstance(partition_requests, bool)
        or not isinstance(partition_requests, int)
        or partition_requests < 1
    ):
        raise ValueError("partition_requests must be a positive integer")
    if isinstance(work_ids, (set, frozenset, Mapping)):
        raise TypeError("work_ids must be an ordered iterable, not a set or mapping")
    values = list(work_ids)
    if not values:
        raise ValueError("work_ids must be non-empty")
    for work_id in values:
        _require_sha256(work_id, "work ID")
    if len(values) != len(set(values)):
        raise ValueError("work_ids must be unique")
    return {
        partition_index: frozenset(values[start : start + partition_requests])
        for partition_index, start in enumerate(range(0, len(values), partition_requests))
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def inference_fingerprint(config: DFIConfig) -> dict[str, Any]:
    """Return the exact score-affecting identity shared by work IDs and cache shards."""

    fingerprint = {
        "scoring_protocol": config.scoring.protocol,
        "model_repository": config.model.repository,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "remote_code_revision": config.model.remote_code_revision,
        "prompt_protocol": config.scoring.prompt_protocol,
        "mask_policy": config.scoring.mask_policy,
        "temperature": config.scoring.temperature,
        "dtype": config.runtime.dtype,
        "inference_backend": INFERENCE_BACKEND,
    }
    return dict(_require_fingerprint(fingerprint))


def _physical_signature(request: WorkRequest) -> tuple[Any, ...]:
    return (
        request.arm,
        request.mask_rate,
        request.input_ids,
        request.attention_mask,
        request.masked_positions,
        request.target_ids,
    )


def build_inference_plan(
    config: DFIConfig,
    examples: Sequence[ScoringExample],
    backend: Any,
) -> InferencePlan:
    """Build the deterministic logical plan and deduplicate identical physical work."""

    if not examples:
        raise ValueError("cannot build an inference plan for an empty dataset")
    if (
        config.scoring.protocol != SCORING_PROTOCOL
        or config.scoring.prompt_protocol != PROMPT_PROTOCOL
        or config.scoring.temperature != TEMPERATURE
        or config.runtime.dtype != DTYPE_NAME
    ):
        raise ValueError("configuration does not match the frozen LLaDA analytic protocol")
    backend_spec = getattr(backend, "spec", None)
    for field_name, expected in (
        ("repository", config.model.repository),
        ("revision", config.model.revision),
        ("tokenizer_revision", config.model.tokenizer_revision),
        ("remote_code_revision", config.model.remote_code_revision),
    ):
        if getattr(backend_spec, field_name, None) != expected:
            raise ValueError(f"loaded backend {field_name} does not match the configuration")

    logical: list[LogicalRequest] = []
    physical_by_id: dict[str, WorkRequest] = {}
    seen_logical_ids: set[str] = set()
    for example in sorted(examples, key=lambda item: item.example_id):
        for arm in config.scoring.arms:
            evidence = None if arm == "prior" else example.evidence_for_arm(arm)
            choices = deterministic_masks(
                example.claim,
                example_id=example.example_id,
                arm=arm,
                seed=config.seed,
                n_masks=config.scoring.n_masks,
            )
            for choice in choices:
                request = backend.prepare_request(
                    example_id=example.example_id,
                    claim=example.claim,
                    evidence=evidence,
                    arm=arm,
                    choice=choice,
                    mask_policy=config.scoring.mask_policy,
                )
                if (
                    request.example_id != example.example_id
                    or request.arm != arm
                    or request.mask_index != choice.mask_index
                    or request.mask_rate != choice.mask_rate
                    or request.word_indices != choice.word_indices
                ):
                    raise ValueError("backend request metadata differs from the planned mask")
                logical_id = _canonical_sha256(
                    {
                        "schema_version": "logical-request-v1",
                        "example_id": example.example_id,
                        "arm": arm,
                        "mask_index": choice.mask_index,
                    }
                )
                if logical_id in seen_logical_ids:
                    raise ValueError("logical request identity is duplicated")
                seen_logical_ids.add(logical_id)
                logical.append(
                    LogicalRequest(
                        logical_id=logical_id,
                        work_id=request.work_id,
                        example_id=example.example_id,
                        family_id=example.family_id,
                        split=example.split,
                        label=example.label,
                        paired_example_id=example.paired_example_id,
                        arm=arm,
                        mask_index=choice.mask_index,
                        mask_rate=choice.mask_rate,
                    )
                )
                previous = physical_by_id.get(request.work_id)
                if previous is None:
                    physical_by_id[request.work_id] = request
                elif _physical_signature(previous) != _physical_signature(request):
                    raise ValueError("one work ID resolved to different physical requests")

    physical = tuple(physical_by_id.values())
    planned_batches = plan_length_buckets(
        physical,
        max_batch_tokens=config.runtime.max_batch_tokens,
        max_batch_rows=config.runtime.max_batch_rows,
    )
    ordered_requests = tuple(request for batch in planned_batches for request in batch)
    execution_order = tuple(request.work_id for request in ordered_requests)
    partitions = partition_work_ids(execution_order, config.runtime.partition_requests)
    partition_slices = [
        list(
            execution_order[
                index * config.runtime.partition_requests : (index + 1)
                * config.runtime.partition_requests
            ]
        )
        for index in range(len(partitions))
    ]
    plan_hash = _canonical_sha256(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "partition_requests": config.runtime.partition_requests,
            "partitions": partition_slices,
        }
    )
    fingerprint = inference_fingerprint(config)
    return InferencePlan(
        fingerprint=fingerprint,
        fingerprint_hash=_canonical_sha256(fingerprint),
        plan_hash=plan_hash,
        logical_requests=tuple(
            sorted(logical, key=lambda item: (item.example_id, item.arm, item.mask_index))
        ),
        physical_requests=ordered_requests,
        execution_order=execution_order,
        partitions=partitions,
    )


def _normalize_expected_partitions(
    expected_partitions: Mapping[int, Iterable[str]] | None,
    *,
    expected_work_ids: set[str] | None,
) -> dict[int, frozenset[str]] | None:
    if expected_partitions is None:
        return None
    if not isinstance(expected_partitions, Mapping) or not expected_partitions:
        raise ValueError("expected_partitions must be a non-empty mapping")

    normalized: dict[int, frozenset[str]] = {}
    all_ids: set[str] = set()
    for partition_index, raw_work_ids in expected_partitions.items():
        if (
            isinstance(partition_index, bool)
            or not isinstance(partition_index, int)
            or partition_index < 0
        ):
            raise ValueError("expected partition indices must be non-negative integers")
        if isinstance(raw_work_ids, (str, bytes)):
            raise ValueError("expected partition membership must be an iterable of work IDs")
        values = list(raw_work_ids)
        if not values:
            raise ValueError(f"expected partition {partition_index} must be non-empty")
        for work_id in values:
            _require_sha256(work_id, f"expected partition {partition_index} work ID")
        members = frozenset(values)
        if len(members) != len(values):
            raise ValueError(f"expected partition {partition_index} repeats a work ID")
        overlap = all_ids & members
        if overlap:
            raise ValueError("expected partitions must not overlap")
        normalized[partition_index] = members
        all_ids.update(members)

    expected_indices = set(range(len(normalized)))
    if set(normalized) != expected_indices:
        raise ValueError("expected partition indices must be contiguous and start at zero")
    if expected_work_ids is not None and all_ids != expected_work_ids:
        missing = len(expected_work_ids - all_ids)
        unexpected = len(all_ids - expected_work_ids)
        raise ValueError(
            "expected partition membership must exactly cover expected_work_ids: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return dict(sorted(normalized.items()))


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


def _read_bound_cache_rows(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_row_count: int,
    expected_work_ids: frozenset[str],
    expected_size_bytes: int | None = None,
) -> list[dict[str, Any]]:
    selected = Path(path)
    _require_sha256(expected_sha256, "expected cache shard sha256")
    if (
        isinstance(expected_row_count, bool)
        or not isinstance(expected_row_count, int)
        or expected_row_count < 1
    ):
        raise ValueError("expected cache shard row count must be a positive integer")
    if not isinstance(expected_work_ids, frozenset) or not expected_work_ids:
        raise ValueError("expected cache shard work IDs must be a non-empty frozenset")
    for work_id in expected_work_ids:
        _require_sha256(work_id, "expected cache shard work ID")
    if expected_size_bytes is not None and (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes < 1
    ):
        raise ValueError("expected cache shard size must be a positive integer")
    if selected.is_symlink():
        raise ValueError("cache shard must not be a symlink")
    try:
        metadata = selected.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"cache shard is missing: {selected}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("cache shard must be a regular file")
    if expected_size_bytes is not None and metadata.st_size != expected_size_bytes:
        raise ValueError("cache shard size differs from its sealed metadata")
    if sha256_file(selected) != expected_sha256:
        raise ValueError("cache shard hash differs from its validated metadata")
    try:
        table = pq.read_table(selected)
    except Exception as exc:
        raise ValueError("cache shard is not readable Parquet") from exc
    metadata_map = table.schema.metadata or {}
    if metadata_map.get(b"dfi.cache_schema") != CACHE_SCHEMA_VERSION.encode("ascii"):
        raise ValueError("cache shard schema marker is incompatible")
    if b"dfi.result_schema" in metadata_map:
        raise ValueError("cache shard is mislabeled as a logical result")
    if table.num_rows != expected_row_count:
        raise ValueError("cache shard row count differs from its validated metadata")
    rows = table.to_pylist()
    _validate_cache_rows(rows)
    row_ids = [row["work_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("cache shard contains duplicate work IDs")
    if frozenset(row_ids) != expected_work_ids:
        raise ValueError("cache shard work IDs differ from its validated metadata")
    after = selected.stat(follow_symlinks=False)
    if not stat.S_ISREG(after.st_mode):
        raise ValueError("cache shard stopped being a regular file during validated read")
    if expected_size_bytes is not None and after.st_size != expected_size_bytes:
        raise ValueError("cache shard size changed during validated read")
    if sha256_file(selected) != expected_sha256:
        raise ValueError("cache shard hash changed during validated read")
    return rows


def read_validated_cache_rows(
    shard: ValidatedShard, *, path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Read physical rows only while they remain bound to a validated shard digest."""

    if not isinstance(shard, ValidatedShard):
        raise TypeError("shard must be a ValidatedShard")
    return _read_bound_cache_rows(
        shard.path if path is None else path,
        expected_sha256=shard.sha256,
        expected_row_count=shard.row_count,
        expected_work_ids=shard.work_ids,
    )


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
    expected_partitions: Mapping[int, frozenset[str]] | None,
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
    if expected_partitions is not None:
        expected_partition = expected_partitions.get(partition_index)
        if expected_partition is None:
            raise ValueError(f"cache shard has an unexpected partition index: {name}")
        if row_id_set != expected_partition:
            missing = len(expected_partition - row_id_set)
            unexpected = len(row_id_set - expected_partition)
            raise ValueError(
                f"cache shard has incomplete or wrong partition membership: {name}: "
                f"missing={missing}, unexpected={unexpected}"
            )
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
    expected_partitions: Mapping[int, Iterable[str]] | None = None,
) -> CacheInspection:
    """Inspect one generation, isolating physical shard failures as cache misses."""

    if expected_work_ids is not None:
        if not isinstance(expected_work_ids, set) or not expected_work_ids:
            raise ValueError("expected_work_ids must be a non-empty set")
        for work_id in expected_work_ids:
            _require_sha256(work_id, "expected work ID")
    normalized_partitions = _normalize_expected_partitions(
        expected_partitions,
        expected_work_ids=expected_work_ids,
    )
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
                expected_partitions=normalized_partitions,
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
    expected_partitions: Mapping[int, Iterable[str]] | None = None,
) -> list[ValidatedShard]:
    """Strictly validate every shard in a cache generation."""

    inspection = inspect_cache_manifest(
        manifest_path,
        expected_plan_hash=expected_plan_hash,
        expected_fingerprint=expected_fingerprint,
        expected_work_ids=expected_work_ids,
        expected_partitions=expected_partitions,
    )
    if inspection.failures:
        detail = "; ".join(
            f"partition {failure.partition_index}: {failure.reason}"
            for failure in inspection.failures
        )
        raise ValueError(f"cache validation failed: {detail}")
    return list(inspection.valid_shards)


def seal_cache_rows(
    local_directory: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    partition_index: int,
    expected_partition_work_ids: set[str],
    creation_environment: dict[str, Any],
) -> SealedCacheShard:
    """Atomically seal exact physical rows on local storage before publication."""

    if (
        isinstance(partition_index, bool)
        or not isinstance(partition_index, int)
        or partition_index < 0
    ):
        raise ValueError("partition_index must be a non-negative integer")
    if not isinstance(expected_partition_work_ids, set) or not expected_partition_work_ids:
        raise ValueError("expected_partition_work_ids must be a non-empty set")
    for work_id in expected_partition_work_ids:
        _require_sha256(work_id, "expected partition work ID")
    environment = dict(_require_creation_environment(creation_environment))

    materialized = list(rows)
    _validate_cache_rows(materialized)
    materialized.sort(key=lambda row: row["work_id"])
    work_ids = [row["work_id"] for row in materialized]
    if len(work_ids) != len(set(work_ids)):
        raise ValueError("cache shard contains duplicate work IDs")
    if set(work_ids) != expected_partition_work_ids:
        missing = len(expected_partition_work_ids - set(work_ids))
        unexpected = len(set(work_ids) - expected_partition_work_ids)
        raise ValueError(
            "cache shard must exactly match expected partition membership: "
            f"missing={missing}, unexpected={unexpected}"
        )

    directory = Path(local_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("local shard directory must be a non-symlink directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".part-{partition_index:04d}.", suffix=".partial.parquet", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_cache_parquet_atomic(materialized, temporary)
        content_hash = sha256_file(temporary)
        size_bytes = temporary.stat(follow_symlinks=False).st_size
        shard_name = f"part-{partition_index:04d}-{content_hash}.parquet"
        destination = directory / shard_name
        sealed = SealedCacheShard(
            path=destination,
            sha256=content_hash,
            size_bytes=size_bytes,
            row_count=len(materialized),
            work_ids=frozenset(work_ids),
            partition_index=partition_index,
            creation_environment=environment,
        )
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise ValueError("sealed local cache target must not be a symlink")
            _read_bound_cache_rows(
                destination,
                expected_sha256=sealed.sha256,
                expected_size_bytes=sealed.size_bytes,
                expected_row_count=sealed.row_count,
                expected_work_ids=sealed.work_ids,
            )
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, destination)
            _fsync_directory(directory)
        _read_bound_cache_rows(
            destination,
            expected_sha256=sealed.sha256,
            expected_size_bytes=sealed.size_bytes,
            expected_row_count=sealed.row_count,
            expected_work_ids=sealed.work_ids,
        )
        return sealed
    finally:
        temporary.unlink(missing_ok=True)


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


def upload_sealed_cache_shard(
    cache_directory: str | Path,
    shard: SealedCacheShard,
    *,
    plan_hash: str,
    fingerprint: dict[str, Any],
    expected_work_ids: set[str],
    expected_partitions: Mapping[int, Iterable[str]] | None = None,
    repair_invalid_partition: bool = False,
) -> dict[str, Any]:
    """Copy one sealed local shard, validate its readback, then publish the manifest."""

    if not isinstance(shard, SealedCacheShard):
        raise TypeError("shard must be a SealedCacheShard")
    if (
        isinstance(shard.partition_index, bool)
        or not isinstance(shard.partition_index, int)
        or shard.partition_index < 0
    ):
        raise ValueError("sealed shard partition_index must be a non-negative integer")
    if not isinstance(repair_invalid_partition, bool):
        raise ValueError("repair_invalid_partition must be boolean")
    _require_sha256(plan_hash, "plan_hash")
    _require_fingerprint(fingerprint)
    _require_creation_environment(shard.creation_environment)
    if not isinstance(expected_work_ids, set) or not expected_work_ids:
        raise ValueError("expected_work_ids must be a non-empty set")
    for expected_work_id in expected_work_ids:
        _require_sha256(expected_work_id, "expected work ID")
    if not shard.work_ids <= expected_work_ids:
        raise ValueError("sealed shard contains work IDs outside the current plan")
    normalized_partitions = _normalize_expected_partitions(
        expected_partitions,
        expected_work_ids=expected_work_ids,
    )
    if normalized_partitions is not None:
        expected_partition = normalized_partitions.get(shard.partition_index)
        if expected_partition is None:
            raise ValueError("sealed shard has an unexpected partition index")
        if shard.work_ids != expected_partition:
            missing = len(expected_partition - shard.work_ids)
            unexpected = len(shard.work_ids - expected_partition)
            raise ValueError(
                "sealed shard must exactly match expected partition membership: "
                f"missing={missing}, unexpected={unexpected}"
            )

    source_name = f"part-{shard.partition_index:04d}-{shard.sha256}.parquet"
    if shard.path.name != source_name:
        raise ValueError("sealed shard filename is inconsistent with its partition and hash")
    _read_bound_cache_rows(
        shard.path,
        expected_sha256=shard.sha256,
        expected_size_bytes=shard.size_bytes,
        expected_row_count=shard.row_count,
        expected_work_ids=shard.work_ids,
    )

    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("cache directory must be a non-symlink directory")
    cache_root = directory.resolve(strict=True)
    source = shard.path.resolve(strict=True)
    if source == cache_root or cache_root in source.parents:
        raise ValueError("sealed local shard must be outside the cache directory")

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
            expected_partitions=normalized_partitions,
        )
        target_is_invalid = any(
            failure.partition_index == shard.partition_index
            for failure in existing_inspection.failures
        )
        other_failures = [
            failure
            for failure in existing_inspection.failures
            if failure.partition_index != shard.partition_index
        ]
        if other_failures and not (repair_invalid_partition and target_is_invalid):
            raise ValueError(
                "existing cache has an invalid partition; repair a failed partition first"
            )

    work_ids = sorted(shard.work_ids)
    entry = {
        "partition_index": shard.partition_index,
        "name": source_name,
        "sha256": shard.sha256,
        "row_count": shard.row_count,
        "first_work_id": min(work_ids),
        "last_work_id": max(work_ids),
        "work_ids_sha256": _work_ids_digest(work_ids),
        "creation_environment": dict(shard.creation_environment),
    }

    existing_partition: dict[str, Any] | None = None
    partition_is_valid = False
    partition_is_invalid = False
    if existing_manifest is not None and existing_inspection is not None:
        existing_partition = next(
            (
                candidate
                for candidate in existing_manifest["shards"]
                if candidate["partition_index"] == shard.partition_index
            ),
            None,
        )
        partition_is_valid = any(
            candidate.partition_index == shard.partition_index
            for candidate in existing_inspection.valid_shards
        )
        partition_is_invalid = any(
            failure.partition_index == shard.partition_index
            for failure in existing_inspection.failures
        )

    if existing_partition is not None and partition_is_valid:
        existing_content = {
            key: value for key, value in existing_partition.items() if key != "creation_environment"
        }
        new_content = {key: value for key, value in entry.items() if key != "creation_environment"}
        if existing_content == new_content:
            return existing_manifest
        raise ValueError(
            f"cache partition {shard.partition_index} is already sealed with other valid content"
        )
    if existing_partition is not None and partition_is_invalid:
        if not repair_invalid_partition:
            raise ValueError(
                f"cache partition {shard.partition_index} is invalid; explicit repair is required"
            )
        _remove_invalid_shard(directory, existing_partition)

    destination = directory / source_name
    if destination.exists() or destination.is_symlink():
        destination_matches = False
        if not destination.is_symlink():
            try:
                _read_bound_cache_rows(
                    destination,
                    expected_sha256=shard.sha256,
                    expected_size_bytes=shard.size_bytes,
                    expected_row_count=shard.row_count,
                    expected_work_ids=shard.work_ids,
                )
                destination_matches = True
            except (OSError, ValueError):
                destination_matches = False
        if not destination_matches:
            if not repair_invalid_partition:
                raise ValueError(
                    f"immutable cache target already exists with different bytes: {destination}"
                )
            _remove_invalid_shard(directory, entry)

    if not destination.exists():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".upload-{shard.partition_index:04d}.",
            suffix=".partial.parquet",
            dir=directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            _read_bound_cache_rows(
                temporary,
                expected_sha256=shard.sha256,
                expected_size_bytes=shard.size_bytes,
                expected_row_count=shard.row_count,
                expected_work_ids=shard.work_ids,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            if destination.exists() or destination.is_symlink():
                raise ValueError("cache target appeared during single-writer publication")
            os.replace(temporary, destination)
            _fsync_directory(directory)
        finally:
            temporary.unlink(missing_ok=True)

    _read_bound_cache_rows(
        source,
        expected_sha256=shard.sha256,
        expected_size_bytes=shard.size_bytes,
        expected_row_count=shard.row_count,
        expected_work_ids=shard.work_ids,
    )
    _read_bound_cache_rows(
        destination,
        expected_sha256=shard.sha256,
        expected_size_bytes=shard.size_bytes,
        expected_row_count=shard.row_count,
        expected_work_ids=shard.work_ids,
    )
    validated = _validate_cache_shard(
        manifest_path,
        entry,
        cache_root=cache_root,
        expected_work_ids=expected_work_ids,
        expected_partitions=normalized_partitions,
    )
    if validated.sha256 != shard.sha256 or validated.row_count != shard.row_count:
        raise ValueError("published cache shard failed readback validation")

    if existing_manifest is None:
        shards = [entry]
        generation = 1
    else:
        shards = [
            candidate
            for candidate in existing_manifest["shards"]
            if candidate["partition_index"] != shard.partition_index
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
    expected_partitions: Mapping[int, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Backward-compatible synchronous seal-and-publish cache helper."""

    if not isinstance(expected_work_ids, set) or not expected_work_ids:
        raise ValueError("expected_work_ids must be a non-empty set")
    for work_id in expected_work_ids:
        _require_sha256(work_id, "expected work ID")
    normalized_partitions = _normalize_expected_partitions(
        expected_partitions,
        expected_work_ids=expected_work_ids,
    )
    materialized = list(rows)
    _validate_cache_rows(materialized)
    row_work_ids = {row["work_id"] for row in materialized}
    if normalized_partitions is None:
        expected_partition_work_ids = row_work_ids
    else:
        expected_partition = normalized_partitions.get(partition_index)
        if expected_partition is None:
            raise ValueError("partition_index is outside expected_partitions")
        expected_partition_work_ids = set(expected_partition)

    with tempfile.TemporaryDirectory(prefix="dfi-cache-seal-") as temporary_directory:
        sealed = seal_cache_rows(
            temporary_directory,
            materialized,
            partition_index=partition_index,
            expected_partition_work_ids=expected_partition_work_ids,
            creation_environment=creation_environment,
        )
        return upload_sealed_cache_shard(
            cache_directory,
            sealed,
            plan_hash=plan_hash,
            fingerprint=fingerprint,
            expected_work_ids=expected_work_ids,
            expected_partitions=normalized_partitions,
            repair_invalid_partition=repair_invalid_partition,
        )


class BoundedCacheUploader:
    """One bounded background writer for already sealed local cache shards."""

    _SENTINEL = object()

    def __init__(
        self,
        cache_directory: str | Path,
        *,
        plan_hash: str,
        fingerprint: dict[str, Any],
        expected_work_ids: set[str],
        expected_partitions: Mapping[int, Iterable[str]] | None = None,
        max_queue_size: int = 2,
    ) -> None:
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size < 1
        ):
            raise ValueError("max_queue_size must be a positive integer")
        _require_sha256(plan_hash, "plan_hash")
        _require_fingerprint(fingerprint)
        if not isinstance(expected_work_ids, set) or not expected_work_ids:
            raise ValueError("expected_work_ids must be a non-empty set")
        for work_id in expected_work_ids:
            _require_sha256(work_id, "expected work ID")

        self._cache_directory = Path(cache_directory)
        self._plan_hash = plan_hash
        self._fingerprint = dict(fingerprint)
        self._expected_work_ids = set(expected_work_ids)
        self._expected_partitions = _normalize_expected_partitions(
            expected_partitions,
            expected_work_ids=self._expected_work_ids,
        )
        self._queue: queue.Queue[_UploadTask | object] = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._accepting = True
        self._closed = False
        self._error: BaseException | None = None
        self._latest_manifest: dict[str, Any] | None = None
        self._submitted_shards = 0
        self._completed_shards = 0
        self._failed_shards = 0
        self._skipped_shards = 0
        self._uploaded_bytes = 0
        self._queue_stall_seconds = 0.0
        self._upload_seconds = 0.0
        self._thread = threading.Thread(
            target=self._worker,
            name="dfi-cache-uploader",
            daemon=True,
        )
        self._thread.start()

    def __enter__(self) -> BoundedCacheUploader:
        with self._lock:
            if self._closed:
                raise RuntimeError("cache uploader is closed")
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        try:
            self.close()
        except RuntimeError as upload_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"cache uploader also failed: {upload_error}")
        return False

    @property
    def metrics(self) -> CacheUploadMetrics:
        with self._lock:
            return CacheUploadMetrics(
                submitted_shards=self._submitted_shards,
                completed_shards=self._completed_shards,
                failed_shards=self._failed_shards,
                skipped_shards=self._skipped_shards,
                uploaded_bytes=self._uploaded_bytes,
                queue_stall_seconds=self._queue_stall_seconds,
                upload_seconds=self._upload_seconds,
            )

    @property
    def latest_manifest(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_manifest is None:
                return None
            return json.loads(canonical_json(self._latest_manifest))

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("cache uploader failed") from error

    def _validate_submission(self, shard: SealedCacheShard) -> None:
        if not isinstance(shard, SealedCacheShard):
            raise TypeError("shard must be a SealedCacheShard")
        if not shard.work_ids <= self._expected_work_ids:
            raise ValueError("sealed shard contains work IDs outside the current plan")
        if self._expected_partitions is not None:
            expected = self._expected_partitions.get(shard.partition_index)
            if expected is None:
                raise ValueError("sealed shard has an unexpected partition index")
            if shard.work_ids != expected:
                raise ValueError("sealed shard does not exactly match its expected partition")

    def submit(
        self,
        shard: SealedCacheShard,
        *,
        repair_invalid_partition: bool = False,
    ) -> None:
        """Queue one immutable shard, blocking only when the bounded queue is full."""

        if not isinstance(repair_invalid_partition, bool):
            raise ValueError("repair_invalid_partition must be boolean")
        self._validate_submission(shard)
        with self._submit_lock:
            with self._lock:
                if not self._accepting:
                    raise RuntimeError("cache uploader is closing or closed")
            self._raise_if_failed()

            task = _UploadTask(shard, repair_invalid_partition)
            started = time.perf_counter()
            while True:
                try:
                    self._queue.put(task, timeout=0.05)
                    break
                except queue.Full:
                    self._raise_if_failed()
                    with self._lock:
                        if not self._accepting:
                            raise RuntimeError("cache uploader is closing or closed") from None
            elapsed = time.perf_counter() - started
            with self._lock:
                self._submitted_shards += 1
                self._queue_stall_seconds += elapsed
            self._raise_if_failed()

    def _destination_already_matches(self, shard: SealedCacheShard) -> bool:
        target = self._cache_directory / shard.path.name
        if target.is_symlink() or not target.is_file():
            return False
        try:
            return (
                target.stat(follow_symlinks=False).st_size == shard.size_bytes
                and sha256_file(target) == shard.sha256
            )
        except OSError:
            return False

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            started = time.perf_counter()
            try:
                if item is self._SENTINEL:
                    return
                if not isinstance(item, _UploadTask):
                    raise TypeError("cache uploader received an invalid internal task")
                with self._lock:
                    prior_error = self._error
                if prior_error is not None:
                    with self._lock:
                        self._skipped_shards += 1
                    continue

                already_present = self._destination_already_matches(item.shard)
                manifest = upload_sealed_cache_shard(
                    self._cache_directory,
                    item.shard,
                    plan_hash=self._plan_hash,
                    fingerprint=self._fingerprint,
                    expected_work_ids=self._expected_work_ids,
                    expected_partitions=self._expected_partitions,
                    repair_invalid_partition=item.repair_invalid_partition,
                )
            except BaseException as exc:
                elapsed = time.perf_counter() - started
                with self._lock:
                    if self._error is None:
                        self._error = exc
                        self._failed_shards += 1
                    self._upload_seconds += elapsed
            else:
                elapsed = time.perf_counter() - started
                with self._lock:
                    self._latest_manifest = manifest
                    self._completed_shards += 1
                    if not already_present:
                        self._uploaded_bytes += item.shard.size_bytes
                    self._upload_seconds += elapsed
            finally:
                self._queue.task_done()

    def drain(self) -> CacheUploadMetrics:
        """Wait for all submitted shards and propagate the first worker failure."""

        self._queue.join()
        self._raise_if_failed()
        return self.metrics

    def close(self) -> CacheUploadMetrics:
        """Stop accepting work, drain once, stop the worker, and propagate failures."""

        with self._close_lock:
            with self._submit_lock, self._lock:
                if self._closed:
                    already_closed = True
                else:
                    already_closed = False
                    self._accepting = False
            if not already_closed:
                self._queue.join()
                self._queue.put(self._SENTINEL)
                self._thread.join()
                with self._lock:
                    self._closed = True
        self._raise_if_failed()
        return self.metrics


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
    expected_logical_ids: set[str] | None = None,
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
    if set(work_ids) != expected_work_ids:
        missing = sorted(expected_work_ids - set(work_ids))
        unexpected = sorted(set(work_ids) - expected_work_ids)
        raise ValueError(
            f"final work IDs do not match the plan: missing={missing}, unexpected={unexpected}"
        )
    if expected_logical_ids is None:
        if len(work_ids) != len(set(work_ids)):
            raise ValueError("final run contains duplicate work IDs")
        materialized.sort(key=lambda row: row["work_id"])
    else:
        if not isinstance(expected_logical_ids, set) or not expected_logical_ids:
            raise ValueError("expected_logical_ids must be a non-empty set")
        for logical_id in expected_logical_ids:
            _require_sha256(logical_id, "expected logical ID")
        logical_ids = [
            _canonical_sha256(
                {
                    "schema_version": "logical-request-v1",
                    "example_id": row["example_id"],
                    "arm": row["arm"],
                    "mask_index": row["mask_index"],
                }
            )
            for row in materialized
        ]
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("final run contains duplicate logical requests")
        if set(logical_ids) != expected_logical_ids:
            missing = sorted(expected_logical_ids - set(logical_ids))
            unexpected = sorted(set(logical_ids) - expected_logical_ids)
            raise ValueError(
                "final logical IDs do not match the plan: "
                f"missing={missing}, unexpected={unexpected}"
            )
        materialized.sort(
            key=lambda row: (row["example_id"], row["arm"], row["mask_index"], row["work_id"])
        )

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


def execute_inference_requests(
    backend: Any,
    requests: Sequence[WorkRequest],
    *,
    max_batch_tokens: int,
    max_batch_rows: int,
    telemetry: InferenceTelemetry | None = None,
) -> tuple[tuple[AnalyticResult, ...], InferenceTelemetry]:
    """Execute global batches with one runner-wide OOM halving fallback."""

    counters = telemetry if telemetry is not None else InferenceTelemetry()
    planned = plan_length_buckets(
        requests,
        max_batch_tokens=max_batch_tokens,
        max_batch_rows=max_batch_rows,
    )
    expected_ids = [request.work_id for request in requests]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("inference requests must have unique work IDs")
    by_work_id: dict[str, AnalyticResult] = {}
    started = time.perf_counter()
    for planned_batch in planned:
        if counters.fallback_batch_rows is None:
            pending = [planned_batch]
        else:
            size = counters.fallback_batch_rows
            pending = [
                planned_batch[index : index + size] for index in range(0, len(planned_batch), size)
            ]
        while pending:
            batch = pending.pop(0)
            counters.forwards += 1
            try:
                results = tuple(backend.infer_batch(batch))
            except BaseException as exc:
                if not backend.is_oom_error(exc):
                    raise
                if counters.oom_fallbacks or len(batch) <= 1:
                    raise RuntimeError(
                        "CUDA OOM recurred after the single permitted batch-halving fallback"
                    ) from exc
                counters.oom_fallbacks = 1
                counters.fallback_batch_rows = max(1, len(batch) // 2)
                size = counters.fallback_batch_rows
                pending = [
                    batch[index : index + size] for index in range(0, len(batch), size)
                ] + pending
                continue

            batch_ids = [request.work_id for request in batch]
            result_ids = [result.work_id for result in results]
            if result_ids != batch_ids:
                raise ValueError(
                    "model batch output IDs or cardinality differ from the request batch"
                )
            for result in results:
                if not isinstance(result, AnalyticResult):
                    raise TypeError("model backend must return AnalyticResult values")
                if result.work_id in by_work_id:
                    raise ValueError("model backend returned a duplicate physical result")
                by_work_id[result.work_id] = result

            padded_tokens = len(batch) * max(request.sequence_length for request in batch)
            real_tokens = sum(request.sequence_length for request in batch)
            counters.successful_batches += 1
            counters.completed_requests += len(batch)
            counters.padded_tokens += padded_tokens
            counters.real_tokens += real_tokens
            counters.max_batch_rows = max(counters.max_batch_rows, len(batch))
            counters.max_padded_tokens = max(counters.max_padded_tokens, padded_tokens)
            counters.batch_rows.append(len(batch))
    counters.inference_seconds += time.perf_counter() - started
    if set(by_work_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_work_id))
        unexpected = sorted(set(by_work_id) - set(expected_ids))
        raise ValueError(
            "model output does not match requested work: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tuple(by_work_id[work_id] for work_id in expected_ids), counters


def _telemetry_receipt(telemetry: InferenceTelemetry) -> dict[str, Any]:
    padded = telemetry.padded_tokens
    padding_ratio = 0.0 if not padded else (padded - telemetry.real_tokens) / padded
    rows = telemetry.batch_rows
    return {
        "strategy": "global-length-bucket-token-budget-v1",
        "successful_batches": telemetry.successful_batches,
        "rows_per_batch": {
            "minimum": min(rows) if rows else 0,
            "median": float(statistics.median(rows)) if rows else 0.0,
            "maximum": max(rows) if rows else 0,
        },
        "real_tokens": telemetry.real_tokens,
        "padded_tokens": telemetry.padded_tokens,
        "padding_ratio": padding_ratio,
        "max_batch_rows_observed": telemetry.max_batch_rows,
        "max_padded_tokens_observed": telemetry.max_padded_tokens,
        "inference_seconds": telemetry.inference_seconds,
        "oom_fallbacks": telemetry.oom_fallbacks,
        "fallback_batch_rows": telemetry.fallback_batch_rows,
    }


_PARTIAL_STATE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "run_uuid",
        "created_at",
        "updated_at",
        "resume_count",
        "configuration_path",
        "configuration_sha256",
        "resolved_configuration",
        "input",
        "fingerprint",
        "fingerprint_hash",
        "plan_hash",
        "source_provenance",
        "acceptance",
        "accounting",
        "completed_partitions",
    }
)
_ACCOUNTING_KEYS = frozenset(
    {
        "forwards",
        "successful_batches",
        "computed_requests",
        "real_tokens",
        "padded_tokens",
        "max_batch_rows",
        "max_padded_tokens",
        "oom_fallbacks",
        "fallback_batch_rows",
        "inference_seconds",
        "batch_rows",
        "result_write_seconds",
        "peak_vram_bytes",
        "active_wall_seconds",
        "cache_uploaded_shards",
        "cache_failed_shards",
        "cache_skipped_shards",
        "cache_uploaded_bytes",
        "cache_queue_stall_seconds",
        "cache_upload_seconds",
    }
)
_PARTIAL_ENTRY_KEYS = frozenset(
    {
        "partition_index",
        "name",
        "sha256",
        "size_bytes",
        "row_count",
        "work_ids_sha256",
        "creation_environment",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _empty_accounting() -> dict[str, Any]:
    return {
        "forwards": 0,
        "successful_batches": 0,
        "computed_requests": 0,
        "real_tokens": 0,
        "padded_tokens": 0,
        "max_batch_rows": 0,
        "max_padded_tokens": 0,
        "oom_fallbacks": 0,
        "fallback_batch_rows": None,
        "inference_seconds": 0.0,
        "batch_rows": [],
        "result_write_seconds": 0.0,
        "peak_vram_bytes": 0,
        "active_wall_seconds": 0.0,
        "cache_uploaded_shards": 0,
        "cache_failed_shards": 0,
        "cache_skipped_shards": 0,
        "cache_uploaded_bytes": 0,
        "cache_queue_stall_seconds": 0.0,
        "cache_upload_seconds": 0.0,
    }


def _validate_accounting(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ACCOUNTING_KEYS:
        raise ValueError("resume accounting has the wrong schema")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("resume accounting is not canonical JSON") from exc
    integer_fields = _ACCOUNTING_KEYS - {
        "fallback_batch_rows",
        "inference_seconds",
        "batch_rows",
        "result_write_seconds",
        "active_wall_seconds",
        "cache_queue_stall_seconds",
        "cache_upload_seconds",
    }
    for name in integer_fields:
        if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 0:
            raise ValueError(f"resume accounting {name} must be a non-negative integer")
    fallback = value["fallback_batch_rows"]
    if fallback is not None and (
        isinstance(fallback, bool) or not isinstance(fallback, int) or fallback < 1
    ):
        raise ValueError("resume accounting fallback_batch_rows is invalid")
    if value["oom_fallbacks"] not in {0, 1}:
        raise ValueError("resume accounting permits at most one OOM fallback")
    if (value["oom_fallbacks"] == 0) != (fallback is None):
        raise ValueError("resume accounting OOM fallback fields are inconsistent")
    for name in (
        "inference_seconds",
        "result_write_seconds",
        "active_wall_seconds",
        "cache_queue_stall_seconds",
        "cache_upload_seconds",
    ):
        _number(value[name], label=f"resume accounting {name}")
        if value[name] < 0:
            raise ValueError(f"resume accounting {name} must be non-negative")
    rows = value["batch_rows"]
    if not isinstance(rows, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in rows
    ):
        raise ValueError("resume accounting batch_rows is invalid")
    return value


def _telemetry_from_accounting(accounting: dict[str, Any]) -> InferenceTelemetry:
    _validate_accounting(accounting)
    return InferenceTelemetry(
        forwards=accounting["forwards"],
        successful_batches=accounting["successful_batches"],
        completed_requests=accounting["computed_requests"],
        real_tokens=accounting["real_tokens"],
        padded_tokens=accounting["padded_tokens"],
        max_batch_rows=accounting["max_batch_rows"],
        max_padded_tokens=accounting["max_padded_tokens"],
        oom_fallbacks=accounting["oom_fallbacks"],
        fallback_batch_rows=accounting["fallback_batch_rows"],
        inference_seconds=float(accounting["inference_seconds"]),
        batch_rows=list(accounting["batch_rows"]),
    )


def _store_accounting(
    accounting: dict[str, Any],
    telemetry: InferenceTelemetry,
    *,
    result_write_seconds: float,
    peak_vram_bytes: int,
    active_wall_seconds: float,
) -> None:
    accounting.update(
        {
            "forwards": telemetry.forwards,
            "successful_batches": telemetry.successful_batches,
            "computed_requests": telemetry.completed_requests,
            "real_tokens": telemetry.real_tokens,
            "padded_tokens": telemetry.padded_tokens,
            "max_batch_rows": telemetry.max_batch_rows,
            "max_padded_tokens": telemetry.max_padded_tokens,
            "oom_fallbacks": telemetry.oom_fallbacks,
            "fallback_batch_rows": telemetry.fallback_batch_rows,
            "inference_seconds": telemetry.inference_seconds,
            "batch_rows": list(telemetry.batch_rows),
            "result_write_seconds": result_write_seconds,
            "peak_vram_bytes": max(accounting["peak_vram_bytes"], peak_vram_bytes),
            "active_wall_seconds": active_wall_seconds,
        }
    )
    _validate_accounting(accounting)


def _add_upload_accounting(accounting: dict[str, Any], metrics: CacheUploadMetrics) -> None:
    accounting["cache_uploaded_shards"] += metrics.completed_shards
    accounting["cache_failed_shards"] += metrics.failed_shards
    accounting["cache_skipped_shards"] += metrics.skipped_shards
    accounting["cache_uploaded_bytes"] += metrics.uploaded_bytes
    accounting["cache_queue_stall_seconds"] += metrics.queue_stall_seconds
    accounting["cache_upload_seconds"] += metrics.upload_seconds
    _validate_accounting(accounting)


def _run_identity(
    config: DFIConfig,
    config_path: Path,
    plan: InferencePlan,
    *,
    input_row_count: int,
    configuration_sha256: str,
    input_sha256: str,
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configuration_path": str(config_path),
        "configuration_sha256": configuration_sha256,
        "resolved_configuration": config.to_receipt_dict(),
        "input": {
            "path": str(config.dataset.path),
            "sha256": input_sha256,
            "row_count": input_row_count,
            "schema_version": "dfi-v2",
        },
        "fingerprint": plan.fingerprint,
        "fingerprint_hash": plan.fingerprint_hash,
        "plan_hash": plan.plan_hash,
        "source_provenance": source_provenance,
    }


def _write_partial_state(run_directory: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    atomic_write_json(run_directory / "run.partial.json", state)


def _open_run_directory(
    config: DFIConfig,
    config_path: Path,
    plan: InferencePlan,
    *,
    input_row_count: int,
    configuration_sha256: str,
    input_sha256: str,
    source_provenance: dict[str, Any],
    resume_directory: str | Path | None,
) -> tuple[Path, Path, dict[str, Any]]:
    identity = _run_identity(
        config,
        config_path,
        plan,
        input_row_count=input_row_count,
        configuration_sha256=configuration_sha256,
        input_sha256=input_sha256,
        source_provenance=source_provenance,
    )
    if resume_directory is None:
        output_root = config.output_root
        output_root.mkdir(parents=True, exist_ok=True)
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError("output_root must be a non-symlink directory")
        run_uuid = str(uuid.uuid4())
        run_directory = output_root / run_uuid
        run_directory.mkdir()
        partial_directory = run_directory / ".partial"
        partial_directory.mkdir()
        now = _utc_now()
        state = {
            "schema_version": PARTIAL_RUN_SCHEMA_VERSION,
            "status": "running",
            "run_uuid": run_uuid,
            "created_at": now,
            "updated_at": now,
            "resume_count": 0,
            **identity,
            "acceptance": {
                "scalar_batch_parity": {"status": "not_run"},
                "fixed_a100_baseline": {"status": "not_run"},
            },
            "accounting": _empty_accounting(),
            "completed_partitions": [],
        }
        _write_partial_state(run_directory, state)
        return run_directory, partial_directory, state

    run_directory = Path(resume_directory).resolve(strict=False)
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise ValueError(f"resume directory is missing or unsafe: {run_directory}")
    if (run_directory / "run.json").exists():
        raise ValueError(f"run is already complete: {run_directory}")
    partial_path = run_directory / "run.partial.json"
    if partial_path.is_symlink():
        raise ValueError("run.partial.json must not be a symlink")
    try:
        state = json.loads(partial_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("resume state is missing or malformed") from exc
    if not isinstance(state, dict) or set(state) != _PARTIAL_STATE_KEYS:
        raise ValueError("resume state has the wrong schema")
    if state["schema_version"] != PARTIAL_RUN_SCHEMA_VERSION or state["status"] != "running":
        raise ValueError("resume state has an incompatible schema or status")
    if not isinstance(state["run_uuid"], str) or not state["run_uuid"]:
        raise ValueError("resume state has an invalid run UUID")
    if isinstance(state["resume_count"], bool) or not isinstance(state["resume_count"], int):
        raise ValueError("resume state has an invalid resume count")
    for key, expected in identity.items():
        if not _same_canonical_json(state.get(key), expected):
            raise ValueError(f"resume state does not match current {key}")
    entries = state["completed_partitions"]
    if not isinstance(entries, list):
        raise ValueError("resume state completed_partitions must be a list")
    _validate_accounting(state["accounting"])
    if not isinstance(state["acceptance"], dict):
        raise ValueError("resume acceptance state must be an object")
    canonical_json(state["acceptance"])
    state["resume_count"] += 1
    partial_directory = run_directory / ".partial"
    if not partial_directory.is_dir() or partial_directory.is_symlink():
        raise ValueError("resume .partial directory is missing or unsafe")
    _write_partial_state(run_directory, state)
    return run_directory, partial_directory, state


def _partial_entry(shard: SealedCacheShard) -> dict[str, Any]:
    return {
        "partition_index": shard.partition_index,
        "name": shard.path.name,
        "sha256": shard.sha256,
        "size_bytes": shard.size_bytes,
        "row_count": shard.row_count,
        "work_ids_sha256": _work_ids_digest(shard.work_ids),
        "creation_environment": shard.creation_environment,
    }


def _remove_local_partial(path: Path, partial_directory: Path) -> None:
    if path.parent != partial_directory or Path(path.name).name != path.name:
        raise ValueError("refusing to remove a file outside the local partial directory")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
        raise ValueError("refusing to remove a non-file local partial artifact")
    path.unlink()


def _load_local_partitions(
    run_directory: Path,
    partial_directory: Path,
    state: dict[str, Any],
    plan: InferencePlan,
) -> tuple[dict[str, dict[str, Any]], dict[int, SealedCacheShard], int]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    sealed_by_partition: dict[int, SealedCacheShard] = {}
    valid_entries: list[dict[str, Any]] = []
    failures = 0
    seen_partitions: set[int] = set()
    for raw_entry in state["completed_partitions"]:
        if not isinstance(raw_entry, dict) or set(raw_entry) != _PARTIAL_ENTRY_KEYS:
            raise ValueError("resume partition entry has the wrong schema")
        partition_index = raw_entry["partition_index"]
        if (
            isinstance(partition_index, bool)
            or not isinstance(partition_index, int)
            or partition_index not in plan.partitions
            or partition_index in seen_partitions
        ):
            raise ValueError("resume partition index is invalid or duplicated")
        seen_partitions.add(partition_index)
        name = raw_entry["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("resume partition filename is unsafe")
        path = partial_directory / name
        try:
            sha = _require_sha256(raw_entry["sha256"], "resume partition sha256")
            size_bytes = raw_entry["size_bytes"]
            row_count = raw_entry["row_count"]
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 1
                or isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count < 1
            ):
                raise ValueError("resume partition size or row count is invalid")
            expected_ids = plan.partitions[partition_index]
            if raw_entry["work_ids_sha256"] != _work_ids_digest(expected_ids):
                raise ValueError("resume partition work-ID digest differs from the plan")
            environment = dict(
                _require_creation_environment(
                    raw_entry["creation_environment"], "resume creation_environment"
                )
            )
            sealed = SealedCacheShard(
                path=path,
                sha256=sha,
                size_bytes=size_bytes,
                row_count=row_count,
                work_ids=expected_ids,
                partition_index=partition_index,
                creation_environment=environment,
            )
            rows = _read_bound_cache_rows(
                path,
                expected_sha256=sha,
                expected_size_bytes=size_bytes,
                expected_row_count=row_count,
                expected_work_ids=expected_ids,
            )
        except (OSError, TypeError, ValueError):
            failures += 1
            _remove_local_partial(path, partial_directory)
            continue
        for row in rows:
            rows_by_id[row["work_id"]] = row
        sealed_by_partition[partition_index] = sealed
        valid_entries.append(raw_entry)
    if valid_entries != state["completed_partitions"]:
        state["completed_partitions"] = sorted(
            valid_entries, key=lambda entry: entry["partition_index"]
        )
        _write_partial_state(run_directory, state)
    return rows_by_id, sealed_by_partition, failures


def _merge_physical_rows(
    destination: dict[str, dict[str, Any]],
    rows: Iterable[dict[str, Any]],
    *,
    source: str,
) -> None:
    for row in rows:
        work_id = row["work_id"]
        previous = destination.get(work_id)
        if previous is not None and not _same_canonical_json(previous, row):
            raise ValueError(
                f"physical result for {work_id} differs between cache sources: {source}"
            )
        destination[work_id] = row


def _logical_rows_for_requests(
    logical_requests: Sequence[LogicalRequest],
    physical_rows: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_work_ids = {request.work_id for request in logical_requests}
    if set(physical_rows) != expected_work_ids:
        missing = sorted(expected_work_ids - set(physical_rows))
        unexpected = sorted(set(physical_rows) - expected_work_ids)
        raise ValueError(
            f"physical result cardinality differs from the plan: missing={missing}, "
            f"unexpected={unexpected}"
        )
    logical_rows: list[dict[str, Any]] = []
    for logical in logical_requests:
        physical = physical_rows[logical.work_id]
        row = {
            "work_id": logical.work_id,
            "example_id": logical.example_id,
            "family_id": logical.family_id,
            "split": logical.split,
            "label": logical.label,
            "paired_example_id": logical.paired_example_id,
            "arm": logical.arm,
            "mask_index": logical.mask_index,
            "mask_rate": logical.mask_rate,
        }
        row.update({column: physical[column] for column in CACHE_COLUMNS if column != "work_id"})
        logical_rows.append(row)
    _validate_result_rows(logical_rows)
    return logical_rows


def _logical_result_rows(
    plan: InferencePlan,
    physical_rows: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return _logical_rows_for_requests(plan.logical_requests, physical_rows)


def _parity_selection(
    plan: InferencePlan,
) -> tuple[tuple[LogicalRequest, ...], tuple[WorkRequest, ...]]:
    logical = tuple(
        request
        for request in plan.logical_requests
        if request.arm == "prior" and request.mask_index == 0
    )
    if not logical:
        raise ValueError("the parity gate requires prior-arm mask_index=0 requests")
    example_ids = [request.example_id for request in logical]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("the parity gate requires exactly one selected request per example")
    physical_by_id = {request.work_id: request for request in plan.physical_requests}
    selected: list[WorkRequest] = []
    seen_work_ids: set[str] = set()
    for request in logical:
        if request.work_id not in seen_work_ids:
            selected.append(physical_by_id[request.work_id])
            seen_work_ids.add(request.work_id)
    return logical, tuple(selected)


def _aggregate_parity_report(
    logical: Sequence[LogicalRequest],
    scalar_results: Sequence[AnalyticResult],
    batched_results: Sequence[AnalyticResult],
) -> dict[str, Any]:
    scalar_rows = _logical_rows_for_requests(
        logical,
        {result.work_id: result.as_cache_row() for result in scalar_results},
    )
    batched_rows = _logical_rows_for_requests(
        logical,
        {result.work_id: result.as_cache_row() for result in batched_results},
    )
    scalar_claims = reduce_claims(scalar_rows)
    batched_claims = reduce_claims(batched_rows)

    def claim_order(rows: Sequence[dict[str, Any]], metric: str) -> list[str]:
        return [
            row["example_id"]
            for row in sorted(rows, key=lambda row: (-float(row[metric]), row["example_id"]))
        ]

    def paired_decisions(rows: Sequence[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
        by_id = {row["example_id"]: row for row in rows}
        consumed: set[frozenset[str]] = set()
        decisions: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: item["example_id"]):
            paired_id = row.get("paired_example_id")
            if not paired_id or paired_id not in by_id:
                continue
            pair = frozenset({row["example_id"], paired_id})
            if len(pair) != 2 or pair in consumed:
                continue
            consumed.add(pair)
            other = by_id[paired_id]
            labels = {row["label"], other["label"]}
            if labels != {"supported", "refuted"}:
                continue
            refuted = row if row["label"] == "refuted" else other
            supported = other if row["label"] == "refuted" else row
            gap = float(refuted[metric]) - float(supported[metric])
            decisions.append(
                {
                    "supported_example_id": supported["example_id"],
                    "refuted_example_id": refuted["example_id"],
                    "decision": "refuted_higher"
                    if gap > 0
                    else "supported_higher"
                    if gap < 0
                    else "tie",
                }
            )
        return decisions

    metrics: dict[str, Any] = {}
    passed = True
    for metric in RISK_COLUMNS:
        scalar_order = claim_order(scalar_claims, metric)
        batched_order = claim_order(batched_claims, metric)
        scalar_decisions = paired_decisions(scalar_claims, metric)
        batched_decisions = paired_decisions(batched_claims, metric)
        order_passed = scalar_order == batched_order
        decisions_passed = scalar_decisions == batched_decisions
        passed = passed and order_passed and decisions_passed
        metrics[metric] = {
            "claim_order": {
                "scalar": scalar_order,
                "batched": batched_order,
                "passed": order_passed,
            },
            "matched_pair_decisions": {
                "scalar": scalar_decisions,
                "batched": batched_decisions,
                "passed": decisions_passed,
            },
        }
    report = {
        "passed": passed,
        "risk_metrics": list(RISK_COLUMNS),
        "metrics": metrics,
    }
    canonical_json(report)
    return report


def _creation_environments(
    shards: Iterable[ValidatedShard | SealedCacheShard],
) -> list[dict[str, Any]]:
    by_digest: dict[str, dict[str, Any]] = {}
    for shard in shards:
        encoded = canonical_json(shard.creation_environment)
        by_digest[hashlib.sha256(encoded).hexdigest()] = shard.creation_environment
    return [by_digest[digest] for digest in sorted(by_digest)]


def _repository_state(anchor: Path) -> dict[str, Any]:
    candidates = [
        anchor.parent if anchor.is_file() else anchor,
        Path(__file__).resolve().parents[2],
    ]

    def git(checkout: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    for candidate in candidates:
        try:
            root_result = git(candidate, "rev-parse", "--show-toplevel")
            if root_result.returncode:
                continue
            checkout = Path(root_result.stdout.strip())
            commit_result = git(checkout, "rev-parse", "HEAD")
            status_result = git(checkout, "status", "--porcelain", "--untracked-files=all")
            if commit_result.returncode or status_result.returncode:
                continue
        except (OSError, subprocess.SubprocessError):
            continue
        return {
            "available": True,
            "commit": commit_result.stdout.strip(),
            "dirty": bool(status_result.stdout.strip()),
        }
    checked_directories: set[Path] = set()
    for candidate in candidates:
        for directory in (candidate, *candidate.parents):
            if directory in checked_directories:
                continue
            checked_directories.add(directory)
            marker = directory / ".git_archival.txt"
            try:
                marker_text = marker.read_text(encoding="utf-8")
            except OSError:
                continue
            match = re.fullmatch(r"commit: ([0-9a-f]{40})\n?", marker_text)
            if match is not None:
                return {"available": True, "commit": match.group(1), "dirty": None}
    environment_commit = os.environ.get("DFI_SOURCE_COMMIT")
    if isinstance(environment_commit, str) and re.fullmatch(r"[0-9a-f]{40}", environment_commit):
        return {"available": True, "commit": environment_commit, "dirty": None}
    return {"available": False, "commit": None, "dirty": None}


def _package_source_hash() -> str:
    package_directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    sources = sorted(package_directory.glob("*.py"), key=lambda path: path.name)
    if not sources:
        raise ValueError("package source files are unavailable")
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_cache_root_mount(root: Path, *, allow_local_cache_for_testing: bool) -> None:
    content_drive = Path("/content/drive")
    try:
        relative_to_drive = root.relative_to(content_drive)
    except ValueError:
        relative_to_drive = None
    if relative_to_drive is None:
        if not allow_local_cache_for_testing:
            raise ValueError("google_drive cache.root must be inside /content/drive")
        return
    if not relative_to_drive.parts:
        raise ValueError("cache.root must name a directory inside the mounted Drive")
    drive_namespace = content_drive / relative_to_drive.parts[0]
    if not drive_namespace.is_dir():
        raise ValueError(
            f"Google Drive is not mounted at the configured namespace: {drive_namespace}"
        )


def _prepare_cache_directory(
    root: Path,
    plan: InferencePlan,
    *,
    allow_local_cache_for_testing: bool,
) -> Path:
    _validate_cache_root_mount(
        root,
        allow_local_cache_for_testing=allow_local_cache_for_testing,
    )
    cache_directory = root / plan.fingerprint_hash / plan.plan_hash
    cache_directory.mkdir(parents=True, exist_ok=True)
    if cache_directory.is_symlink() or not cache_directory.is_dir():
        raise ValueError("cache plan directory must be a non-symlink directory")
    return cache_directory


def _runtime_environment(backend: Any) -> dict[str, Any]:
    environment = local_creation_environment()
    reported = backend.runtime_environment()
    if not isinstance(reported, dict):
        raise TypeError("backend.runtime_environment() must return an object")
    environment.update(reported)
    if environment.get("cuda_driver") is None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            environment["cuda_driver"] = result.stdout.splitlines()[0].strip()
    canonical_json(environment)
    return environment


def _recover_completed_run(
    resume_directory: str | Path | None,
    config: DFIConfig,
    *,
    configuration_sha256: str,
    input_sha256: str,
) -> Path | None:
    if resume_directory is None:
        return None
    run_directory = Path(resume_directory).resolve(strict=False)
    run_json = run_directory / "run.json"
    if not run_json.is_file() or run_json.is_symlink():
        return None
    try:
        receipt = json.loads(run_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("completed run receipt is malformed") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != RUN_SCHEMA_VERSION
        or receipt.get("status") != "complete"
    ):
        raise ValueError("existing run.json is not a compatible complete receipt")
    if not _same_canonical_json(receipt.get("configuration"), config.to_receipt_dict()):
        raise ValueError("completed run configuration differs from the requested config")
    if receipt.get("configuration_source", {}).get("sha256") != configuration_sha256:
        raise ValueError("completed run configuration hash differs from the current file")
    if receipt.get("input", {}).get("sha256") != input_sha256:
        raise ValueError("completed run input hash differs from the current dataset")
    results = run_directory / "results.parquet"
    result_receipt = receipt.get("results")
    if (
        not isinstance(result_receipt, dict)
        or result_receipt.get("path") != "results.parquet"
        or not results.is_file()
        or results.is_symlink()
        or sha256_file(results) != result_receipt.get("sha256")
    ):
        raise ValueError("completed run result does not match its receipt")
    rows = read_parquet_rows(results)
    if len(rows) != result_receipt.get("row_count"):
        raise ValueError("completed run result row count does not match its receipt")
    partial_json = run_directory / "run.partial.json"
    if partial_json.is_symlink():
        raise ValueError("refusing to clean a symlinked partial receipt")
    partial_json.unlink(missing_ok=True)
    partial_directory = run_directory / ".partial"
    if partial_directory.exists():
        if partial_directory.is_symlink() or not partial_directory.is_dir():
            raise ValueError("refusing to clean an unsafe completed-run .partial artifact")
        shutil.rmtree(partial_directory)
    for staging in run_directory.parent.glob(f".{run_directory.name}.finalizing-*"):
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError("refusing to clean an unsafe finalization artifact")
        shutil.rmtree(staging)
    _fsync_directory(run_directory)
    return run_directory


def _inspect_cache_for_run(
    manifest_path: Path,
    partial_directory: Path,
    plan: InferencePlan,
) -> tuple[CacheInspection, dict[str, Any] | None]:
    if not (manifest_path.exists() or manifest_path.is_symlink()):
        return CacheInspection((), (), frozenset()), None
    try:
        inspection = inspect_cache_manifest(
            manifest_path,
            expected_plan_hash=plan.plan_hash,
            expected_fingerprint=plan.fingerprint,
            expected_work_ids=plan.expected_work_ids,
            expected_partitions=plan.partitions,
        )
    except (OSError, ValueError) as exc:
        if manifest_path.is_symlink():
            raise ValueError("refusing to repair a symlinked cache manifest") from exc
        metadata = manifest_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("refusing to repair a non-file cache manifest") from exc
        digest = sha256_file(manifest_path)
        quarantine = partial_directory / f"rejected-cache-manifest-{digest}.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".rejected-cache-manifest-",
            suffix=".partial",
            dir=partial_directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(manifest_path, temporary)
            if sha256_file(temporary) != digest:
                raise ValueError("rejected cache manifest copy changed bytes")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, quarantine)
            _fsync_directory(partial_directory)
        finally:
            temporary.unlink(missing_ok=True)
        if sha256_file(manifest_path) != digest:
            raise ValueError("cache manifest changed during rejection") from exc
        manifest_path.unlink()
        _fsync_directory(manifest_path.parent)
        return (
            CacheInspection((), (), frozenset()),
            {
                "sha256": digest,
                "size_bytes": metadata.st_size,
                "reason": str(exc),
            },
        )
    return inspection, None


def _finalize_resumable_bundle(
    run_directory: Path,
    rows: Iterable[dict[str, Any]],
    *,
    receipt: dict[str, Any],
    plan: InferencePlan,
    run_started_monotonic: float,
    prior_write_seconds: float,
    prior_active_wall_seconds: float,
) -> tuple[Path, Path]:
    parent = run_directory.parent
    staging = parent / f".{run_directory.name}.finalizing-{uuid.uuid4()}"
    try:
        final_write_started = time.perf_counter()
        write_run_bundle(
            staging,
            rows,
            receipt=receipt,
            expected_work_ids=plan.expected_work_ids,
            expected_logical_ids=plan.expected_logical_ids,
        )
        final_write_seconds = time.perf_counter() - final_write_started
        finalized_receipt = json.loads((staging / "run.json").read_text(encoding="utf-8"))
        finalized_receipt["performance"]["result_write_seconds"] = (
            prior_write_seconds + final_write_seconds
        )
        wall_seconds = prior_active_wall_seconds + time.perf_counter() - run_started_monotonic
        finalized_receipt["performance"]["active_wall_seconds"] = wall_seconds
        created_at = datetime.fromisoformat(finalized_receipt["created_at"].replace("Z", "+00:00"))
        completed_at = datetime.now(UTC)
        finalized_receipt["completed_at"] = completed_at.isoformat().replace("+00:00", "Z")
        finalized_receipt["performance"]["elapsed_calendar_seconds"] = (
            completed_at - created_at
        ).total_seconds()
        computed_requests = finalized_receipt["requests"]["computed_requests"]
        finalized_receipt["performance"]["computed_requests_per_second"] = (
            computed_requests / wall_seconds if wall_seconds else 0.0
        )
        atomic_write_json(staging / "run.json", finalized_receipt)
        readback = read_parquet_rows(staging / "results.parquet")
        if len(readback) != len(plan.logical_requests):
            raise ValueError("staged final results changed logical output cardinality")
        os.replace(staging / "results.parquet", run_directory / "results.parquet")
        _fsync_directory(run_directory)
        os.replace(staging / "run.json", run_directory / "run.json")
        _fsync_directory(run_directory)
        (run_directory / "run.partial.json").unlink(missing_ok=True)
        partial_directory = run_directory / ".partial"
        if partial_directory.exists():
            if partial_directory.is_symlink() or not partial_directory.is_dir():
                raise ValueError("refusing to clean an unsafe .partial artifact")
            shutil.rmtree(partial_directory)
        _fsync_directory(run_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return run_directory / "results.parquet", run_directory / "run.json"


def run_experiment(
    config: DFIConfig,
    *,
    config_path: str | Path,
    resume_directory: str | Path | None = None,
    backend: Any | None = None,
    backend_factory: Callable[..., Any] | None = None,
    run_parity: bool = False,
    allow_local_cache_for_testing: bool = False,
) -> Path:
    """Run or resume the complete revision-pinned LLaDA scoring pipeline."""

    run_started = time.perf_counter()
    selected_config_path = Path(config_path).resolve(strict=True)
    source_provenance = {
        "git": _repository_state(selected_config_path),
        "package_source_sha256": _package_source_hash(),
    }
    configuration_sha256 = sha256_file(selected_config_path)
    parsed_config = load_config(selected_config_path)
    if sha256_file(selected_config_path) != configuration_sha256:
        raise ValueError("configuration changed while it was being parsed")
    if not _same_canonical_json(parsed_config.to_receipt_dict(), config.to_receipt_dict()):
        raise ValueError("config object does not match config_path")
    input_sha256 = sha256_file(config.dataset.path)
    recovered = _recover_completed_run(
        resume_directory,
        config,
        configuration_sha256=configuration_sha256,
        input_sha256=input_sha256,
    )
    if recovered is not None:
        return recovered
    records, examples = load_jsonl(config.dataset.path)
    if sha256_file(config.dataset.path) != input_sha256:
        raise ValueError("dataset changed while it was being validated")
    _validate_cache_root_mount(
        config.cache.root,
        allow_local_cache_for_testing=allow_local_cache_for_testing,
    )
    if backend is None:
        factory = LLaDABackend.load if backend_factory is None else backend_factory
        backend = factory(
            repository=config.model.repository,
            revision=config.model.revision,
            tokenizer_revision=config.model.tokenizer_revision,
            remote_code_revision=config.model.remote_code_revision,
            device="cuda",
        )
    plan = build_inference_plan(config, examples, backend)
    cache_directory = _prepare_cache_directory(
        config.cache.root,
        plan,
        allow_local_cache_for_testing=allow_local_cache_for_testing,
    )
    run_directory, partial_directory, state = _open_run_directory(
        config,
        selected_config_path,
        plan,
        input_row_count=len(records),
        configuration_sha256=configuration_sha256,
        input_sha256=input_sha256,
        source_provenance=source_provenance,
        resume_directory=resume_directory,
    )

    creation_environment = _runtime_environment(backend)
    if not isinstance(run_parity, bool):
        raise TypeError("run_parity must be boolean")
    parity_state = state["acceptance"].get("scalar_batch_parity")
    if run_parity and (
        not isinstance(parity_state, dict) or parity_state.get("status") != "passed"
    ):
        parity_logical, parity_requests = _parity_selection(plan)
        backend.reset_peak_vram()
        parity = run_scalar_batch_parity(
            backend,
            parity_requests,
            max_batch_tokens=config.runtime.max_batch_tokens,
            max_batch_rows=config.runtime.max_batch_rows,
            rtol=1e-3,
            atol=1e-3,
        )
        aggregate_report = _aggregate_parity_report(
            parity_logical,
            parity.scalar_results,
            parity.batched_results,
        )
        parity.report["selection"] = {
            "arm": "prior",
            "mask_index": 0,
            "logical_request_count": len(parity_logical),
            "unique_request_count": len(parity_requests),
            "example_ids": [request.example_id for request in parity_logical],
            "family_ids": sorted({request.family_id for request in parity_logical}),
            "family_count": len({request.family_id for request in parity_logical}),
            "logical_ids": [request.logical_id for request in parity_logical],
        }
        parity.report["aggregate_checks"] = aggregate_report
        if not aggregate_report["passed"]:
            parity.report["passed"] = False
            parity.report["failure_count"] += 1
            parity.report["failures"].append(
                "aggregate claim ordering or matched-pair decisions changed"
            )
            canonical_json(parity.report)
            raise ParityMismatchError(parity.report)
        canonical_json(parity.report)
        state["acceptance"]["scalar_batch_parity"] = {
            "status": "passed",
            "plan_hash": plan.plan_hash,
            "model": {
                "repository": config.model.repository,
                "revision": config.model.revision,
                "tokenizer_revision": config.model.tokenizer_revision,
                "remote_code_revision": config.model.remote_code_revision,
            },
            "report": parity.report,
        }
        state["accounting"]["peak_vram_bytes"] = max(
            state["accounting"]["peak_vram_bytes"],
            int(backend.peak_vram_bytes()),
        )
        _write_partial_state(run_directory, state)
    physical_rows: dict[str, dict[str, Any]] = {}
    local_rows, local_shards, local_validation_failures = _load_local_partitions(
        run_directory,
        partial_directory,
        state,
        plan,
    )
    _merge_physical_rows(physical_rows, local_rows.values(), source="local resume state")

    manifest_path = cache_directory / "cache.json"
    inspection, rejected_manifest = _inspect_cache_for_run(
        manifest_path,
        partial_directory,
        plan,
    )
    hydrated_directory = partial_directory / "hydrated"
    hydrated_paths = hydrate_cache_shards(inspection.valid_shards, hydrated_directory)
    downloaded_bytes = 0
    for shard, hydrated_path in zip(inspection.valid_shards, hydrated_paths, strict=True):
        downloaded_bytes += hydrated_path.stat(follow_symlinks=False).st_size
        rows = read_validated_cache_rows(shard, path=hydrated_path)
        _merge_physical_rows(physical_rows, rows, source=f"Drive partition {shard.partition_index}")

    missing_partitions: list[int] = []
    available_ids = set(physical_rows)
    for partition_index, expected_ids in plan.partitions.items():
        covered = available_ids & expected_ids
        if covered == expected_ids:
            continue
        elif covered:
            raise ValueError(f"partition {partition_index} is only partially covered")
        else:
            missing_partitions.append(partition_index)

    valid_drive_partitions = {shard.partition_index for shard in inspection.valid_shards}
    resumed_uploads = [
        shard
        for partition_index, shard in sorted(local_shards.items())
        if partition_index not in valid_drive_partitions
    ]
    requests_by_id = {request.work_id: request for request in plan.physical_requests}
    accounting = state["accounting"]
    telemetry = _telemetry_from_accounting(accounting)
    partition_write_seconds = float(accounting["result_write_seconds"])
    prior_active_wall_seconds = float(accounting["active_wall_seconds"])
    uploader_metrics = CacheUploadMetrics(0, 0, 0, 0, 0, 0.0, 0.0)

    def persist_accounting() -> None:
        _store_accounting(
            accounting,
            telemetry,
            result_write_seconds=partition_write_seconds,
            peak_vram_bytes=int(backend.peak_vram_bytes()),
            active_wall_seconds=(prior_active_wall_seconds + time.perf_counter() - run_started),
        )
        _write_partial_state(run_directory, state)

    should_upload = bool(resumed_uploads or missing_partitions)
    if missing_partitions:
        backend.reset_peak_vram()

    if should_upload:
        uploader = BoundedCacheUploader(
            cache_directory,
            plan_hash=plan.plan_hash,
            fingerprint=plan.fingerprint,
            expected_work_ids=plan.expected_work_ids,
            expected_partitions=plan.partitions,
            max_queue_size=2,
        )
        try:
            with uploader:
                for shard in resumed_uploads:
                    uploader.submit(
                        shard,
                        repair_invalid_partition=True,
                    )
                for partition_index in missing_partitions:
                    expected_ids = plan.partitions[partition_index]
                    partition_requests = tuple(
                        requests_by_id[work_id]
                        for work_id in plan.execution_order
                        if work_id in expected_ids
                    )
                    inference_started = time.perf_counter()
                    try:
                        results, telemetry = execute_inference_requests(
                            backend,
                            partition_requests,
                            max_batch_tokens=config.runtime.max_batch_tokens,
                            max_batch_rows=config.runtime.max_batch_rows,
                            telemetry=telemetry,
                        )
                    except BaseException:
                        telemetry.inference_seconds += time.perf_counter() - inference_started
                        persist_accounting()
                        raise
                    cache_rows = [result.as_cache_row() for result in results]
                    write_started = time.perf_counter()
                    sealed = seal_cache_rows(
                        partial_directory,
                        cache_rows,
                        partition_index=partition_index,
                        expected_partition_work_ids=set(expected_ids),
                        creation_environment=creation_environment,
                    )
                    partition_write_seconds += time.perf_counter() - write_started
                    entries = [
                        entry
                        for entry in state["completed_partitions"]
                        if entry["partition_index"] != partition_index
                    ]
                    entries.append(_partial_entry(sealed))
                    state["completed_partitions"] = sorted(
                        entries, key=lambda entry: entry["partition_index"]
                    )
                    persist_accounting()
                    _merge_physical_rows(physical_rows, cache_rows, source="fresh model inference")
                    uploader.submit(sealed, repair_invalid_partition=True)
        finally:
            uploader_metrics = uploader.metrics
            _add_upload_accounting(accounting, uploader_metrics)
            persist_accounting()

    final_cache = inspect_cache_manifest(
        manifest_path,
        expected_plan_hash=plan.plan_hash,
        expected_fingerprint=plan.fingerprint,
        expected_work_ids=plan.expected_work_ids,
        expected_partitions=plan.partitions,
    )
    if final_cache.failures or final_cache.work_ids != frozenset(plan.expected_work_ids):
        raise ValueError("final Drive cache generation does not exactly cover the inference plan")
    manifest_hash = sha256_file(manifest_path)
    persist_accounting()

    logical_rows = _logical_result_rows(plan, physical_rows)
    interpretation_reason = (
        "checkpoint-knowledge and dataset admission gates have not been established for this run"
    )
    evaluation = evaluate_rows(
        logical_rows,
        interpretation_allowed=False,
        interpretation_reason=interpretation_reason,
    )
    acceptance_receipt = state["acceptance"]

    cache_hit_ids = set(inspection.work_ids)
    local_reuse_ids = set(local_rows) - cache_hit_ids
    peak_vram = int(accounting["peak_vram_bytes"])
    batching_receipt = _telemetry_receipt(telemetry)
    if sha256_file(selected_config_path) != configuration_sha256:
        raise ValueError("configuration changed during the run")
    if sha256_file(config.dataset.path) != input_sha256:
        raise ValueError("dataset changed during the run")
    if not _same_canonical_json(_repository_state(selected_config_path), source_provenance["git"]):
        raise ValueError("Git source state changed during the run")
    if _package_source_hash() != source_provenance["package_source_sha256"]:
        raise ValueError("installed package source changed during the run")
    parity_forward_counts = (
        acceptance_receipt.get("scalar_batch_parity", {})
        .get("report", {})
        .get("forward_counts", {})
    )
    acceptance_forwards = sum(parity_forward_counts.values())
    receipt = {
        "run_uuid": state["run_uuid"],
        "created_at": state["created_at"],
        "git": source_provenance["git"],
        "package_source_sha256": source_provenance["package_source_sha256"],
        "configuration": config.to_receipt_dict(),
        "configuration_source": {
            "path": str(selected_config_path),
            "sha256": configuration_sha256,
        },
        "input": state["input"],
        "model": {
            "repository": config.model.repository,
            "revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "remote_code_revision": config.model.remote_code_revision,
        },
        "environment": creation_environment,
        "protocol": {
            "scoring": config.scoring.protocol,
            "prompt": config.scoring.prompt_protocol,
            "seed": config.seed,
            "arms": list(config.scoring.arms),
            "mask_policy": config.scoring.mask_policy,
            "n_masks": config.scoring.n_masks,
            "temperature": config.scoring.temperature,
            "dtype": config.runtime.dtype,
            "inference_backend": INFERENCE_BACKEND,
        },
        "plan": {
            "schema_version": PLAN_SCHEMA_VERSION,
            "hash": plan.plan_hash,
            "fingerprint_hash": plan.fingerprint_hash,
            "partition_requests": config.runtime.partition_requests,
            "partition_count": len(plan.partitions),
        },
        "requests": {
            "logical_requests": len(plan.logical_requests),
            "unique_requests": len(plan.physical_requests),
            "completed_requests": len(physical_rows),
            "computed_requests": telemetry.completed_requests,
            "forwards": telemetry.forwards + acceptance_forwards,
            "inference_forwards": telemetry.forwards,
            "acceptance_forwards": acceptance_forwards,
        },
        "batching": batching_receipt,
        "performance": {
            "active_wall_seconds": 0.0,
            "elapsed_calendar_seconds": 0.0,
            "computed_requests_per_second": 0.0,
            "result_write_seconds": partition_write_seconds,
            "peak_vram_bytes": peak_vram,
        },
        "cache": {
            "backend": config.cache.backend,
            "root": str(config.cache.root),
            "plan_directory": str(cache_directory),
            "manifest_sha256": manifest_hash,
            "hits": len(cache_hit_ids),
            "misses": len(plan.expected_work_ids - cache_hit_ids),
            "downloaded_shards": len(hydrated_paths),
            "downloaded_bytes": downloaded_bytes,
            "uploaded_shards": accounting["cache_uploaded_shards"],
            "uploaded_bytes": accounting["cache_uploaded_bytes"],
            "validation_failures": len(inspection.failures) + int(rejected_manifest is not None),
            "rejected_manifest": rejected_manifest,
            "uploader_failed_shards": accounting["cache_failed_shards"],
            "uploader_skipped_shards": accounting["cache_skipped_shards"],
            "uploader_queue_stall_seconds": accounting["cache_queue_stall_seconds"],
            "uploader_seconds": accounting["cache_upload_seconds"],
            "reused_creation_environments": {
                "drive": _creation_environments(inspection.valid_shards),
                "local_resume": _creation_environments(local_shards.values()),
            },
            "final_creation_environments": _creation_environments(final_cache.valid_shards),
        },
        "resume": {
            "requested": resume_directory is not None,
            "resume_count": state["resume_count"],
            "reused_local_requests": len(local_reuse_ids),
            "local_validation_failures": local_validation_failures,
        },
        "evaluation": evaluation,
        "acceptance": acceptance_receipt,
        "interpretation_allowed": False,
        "interpretation_reason": interpretation_reason,
    }
    _finalize_resumable_bundle(
        run_directory,
        logical_rows,
        receipt=receipt,
        plan=plan,
        run_started_monotonic=run_started,
        prior_write_seconds=partition_write_seconds,
        prior_active_wall_seconds=prior_active_wall_seconds,
    )
    return run_directory
