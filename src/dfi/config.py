"""Strict, self-contained TOML configuration for one DFI run."""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

ModelRepository: TypeAlias = Literal["GSAI-ML/LLaDA-8B-Base"]
ScoringProtocol: TypeAlias = Literal["analytic-v1"]
ScoringArm: TypeAlias = Literal["prior", "supplied_evidence"]
MaskPolicy: TypeAlias = Literal["fixed-v1"]
PromptProtocol: TypeAlias = Literal["claim-prefix-v1"]
RuntimeDtype: TypeAlias = Literal["bfloat16"]
CacheBackend: TypeAlias = Literal["google_drive"]

IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class DatasetConfig:
    path: Path


@dataclass(frozen=True)
class ModelConfig:
    repository: ModelRepository
    revision: str
    tokenizer_revision: str
    remote_code_revision: str


@dataclass(frozen=True)
class ScoringConfig:
    protocol: ScoringProtocol
    arms: tuple[ScoringArm, ...]
    mask_policy: MaskPolicy
    n_masks: int
    temperature: float
    prompt_protocol: PromptProtocol


@dataclass(frozen=True)
class RuntimeConfig:
    dtype: RuntimeDtype
    max_batch_tokens: int
    max_batch_rows: int
    partition_requests: int


@dataclass(frozen=True)
class CacheConfig:
    backend: CacheBackend
    root: Path


@dataclass(frozen=True)
class DFIConfig:
    config_version: Literal[1]
    seed: int
    output_root: Path
    dataset: DatasetConfig
    model: ModelConfig
    scoring: ScoringConfig
    runtime: RuntimeConfig
    cache: CacheConfig

    def to_receipt_dict(self) -> dict[str, Any]:
        """Return the complete resolved configuration using JSON-safe values."""

        return {
            "config_version": self.config_version,
            "seed": self.seed,
            "output_root": str(self.output_root),
            "dataset": {"path": str(self.dataset.path)},
            "model": {
                "repository": self.model.repository,
                "revision": self.model.revision,
                "tokenizer_revision": self.model.tokenizer_revision,
                "remote_code_revision": self.model.remote_code_revision,
            },
            "scoring": {
                "protocol": self.scoring.protocol,
                "arms": list(self.scoring.arms),
                "mask_policy": self.scoring.mask_policy,
                "n_masks": self.scoring.n_masks,
                "temperature": self.scoring.temperature,
                "prompt_protocol": self.scoring.prompt_protocol,
            },
            "runtime": {
                "dtype": self.runtime.dtype,
                "max_batch_tokens": self.runtime.max_batch_tokens,
                "max_batch_rows": self.runtime.max_batch_rows,
                "partition_requests": self.runtime.partition_requests,
            },
            "cache": {
                "backend": self.cache.backend,
                "root": str(self.cache.root),
            },
        }


def _require_exact_keys(value: Any, required: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        raise ValueError(f"{label} has invalid keys: missing={missing}, unknown={unknown}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_literal(value: Any, expected: str, label: str) -> str:
    selected = _require_string(value, label)
    if selected != expected:
        raise ValueError(f"{label} must be exactly {expected!r}")
    return selected


def _require_integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _require_revision(value: Any, label: str) -> str:
    revision = _require_string(value, label)
    if IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"{label} must be a literal 40-character lowercase hexadecimal commit")
    return revision


def _resolve_path(value: Any, *, config_directory: Path, label: str) -> Path:
    raw = _require_string(value, label)
    selected = Path(raw)
    if not selected.is_absolute():
        selected = config_directory / selected
    return selected.resolve(strict=False)


def load_config(path: str | Path) -> DFIConfig:
    """Parse and fully validate one version-1 DFI TOML configuration."""

    selected = Path(path)
    if not selected.is_file():
        raise ValueError(f"configuration file does not exist: {selected}")
    selected = selected.resolve(strict=True)
    try:
        with selected.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"configuration is not valid TOML: {exc}") from exc

    top = _require_exact_keys(
        raw,
        frozenset(
            {
                "config_version",
                "seed",
                "output_root",
                "dataset",
                "model",
                "scoring",
                "runtime",
                "cache",
            }
        ),
        "configuration",
    )
    dataset = _require_exact_keys(top["dataset"], frozenset({"path"}), "dataset")
    model = _require_exact_keys(
        top["model"],
        frozenset({"repository", "revision", "tokenizer_revision", "remote_code_revision"}),
        "model",
    )
    scoring = _require_exact_keys(
        top["scoring"],
        frozenset(
            {
                "protocol",
                "arms",
                "mask_policy",
                "n_masks",
                "temperature",
                "prompt_protocol",
            }
        ),
        "scoring",
    )
    runtime = _require_exact_keys(
        top["runtime"],
        frozenset({"dtype", "max_batch_tokens", "max_batch_rows", "partition_requests"}),
        "runtime",
    )
    cache = _require_exact_keys(top["cache"], frozenset({"backend", "root"}), "cache")

    config_version = _require_integer(top["config_version"], "config_version", minimum=1)
    if config_version != 1:
        raise ValueError("config_version must be exactly 1")
    seed = _require_integer(top["seed"], "seed", minimum=0)
    config_directory = selected.parent
    output_root = _resolve_path(
        top["output_root"], config_directory=config_directory, label="output_root"
    )
    dataset_path = _resolve_path(
        dataset["path"], config_directory=config_directory, label="dataset.path"
    )
    if not dataset_path.is_file():
        raise ValueError(f"dataset.path does not name an existing file: {dataset_path}")

    repository = cast(
        ModelRepository,
        _require_literal(model["repository"], "GSAI-ML/LLaDA-8B-Base", "model.repository"),
    )
    revision = _require_revision(model["revision"], "model.revision")
    tokenizer_revision = _require_revision(model["tokenizer_revision"], "model.tokenizer_revision")
    remote_code_revision = _require_revision(
        model["remote_code_revision"], "model.remote_code_revision"
    )

    protocol = cast(
        ScoringProtocol,
        _require_literal(scoring["protocol"], "analytic-v1", "scoring.protocol"),
    )
    arms_value = scoring["arms"]
    if not isinstance(arms_value, list) or arms_value not in (
        ["prior"],
        ["prior", "supplied_evidence"],
    ):
        raise ValueError("scoring.arms must be exactly ['prior'] or ['prior', 'supplied_evidence']")
    arms = cast(tuple[ScoringArm, ...], tuple(arms_value))
    mask_policy = cast(
        MaskPolicy,
        _require_literal(scoring["mask_policy"], "fixed-v1", "scoring.mask_policy"),
    )
    n_masks = _require_integer(scoring["n_masks"], "scoring.n_masks", minimum=1)
    temperature = scoring["temperature"]
    if not isinstance(temperature, float) or not math.isfinite(temperature) or temperature != 1.0:
        raise ValueError("scoring.temperature must be represented as the float 1.0")
    prompt_protocol = cast(
        PromptProtocol,
        _require_literal(scoring["prompt_protocol"], "claim-prefix-v1", "scoring.prompt_protocol"),
    )

    dtype = cast(
        RuntimeDtype,
        _require_literal(runtime["dtype"], "bfloat16", "runtime.dtype"),
    )
    max_batch_tokens = _require_integer(
        runtime["max_batch_tokens"], "runtime.max_batch_tokens", minimum=1
    )
    max_batch_rows = _require_integer(
        runtime["max_batch_rows"], "runtime.max_batch_rows", minimum=1
    )
    partition_requests = _require_integer(
        runtime["partition_requests"], "runtime.partition_requests", minimum=1
    )

    backend = cast(
        CacheBackend,
        _require_literal(cache["backend"], "google_drive", "cache.backend"),
    )
    cache_root_raw = _require_string(cache["root"], "cache.root")
    cache_root = Path(cache_root_raw)
    if not cache_root.is_absolute():
        raise ValueError("cache.root must be an explicit absolute path")
    cache_root = cache_root.resolve(strict=False)

    return DFIConfig(
        config_version=1,
        seed=seed,
        output_root=output_root,
        dataset=DatasetConfig(dataset_path),
        model=ModelConfig(
            repository=repository,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            remote_code_revision=remote_code_revision,
        ),
        scoring=ScoringConfig(
            protocol=protocol,
            arms=arms,
            mask_policy=mask_policy,
            n_masks=n_masks,
            temperature=temperature,
            prompt_protocol=prompt_protocol,
        ),
        runtime=RuntimeConfig(
            dtype=dtype,
            max_batch_tokens=max_batch_tokens,
            max_batch_rows=max_batch_rows,
            partition_requests=partition_requests,
        ),
        cache=CacheConfig(backend=backend, root=cache_root),
    )


__all__ = [
    "CacheConfig",
    "DFIConfig",
    "DatasetConfig",
    "ModelConfig",
    "RuntimeConfig",
    "ScoringConfig",
    "load_config",
]
