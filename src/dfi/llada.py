"""Revision-pinned LLaDA request construction and full-logit inference.

This module deliberately imports neither Torch nor Transformers at import time.
The offline analytic package therefore remains usable without the GPU runtime.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from dfi.masking import (
    MaskChoice,
    align_word_pieces,
    apply_mask_geometry,
    canonical_json,
    stable_work_id,
)

LLADA_REPOSITORY = "GSAI-ML/LLaDA-8B-Base"
LLADA_MASK_TOKEN = "<|mdm_mask|>"
LLADA_MASK_TOKEN_ID = 126336
LLADA_PROTOCOL_MAX_LENGTH = 512
PROMPT_PROTOCOL = "claim-prefix-v1"
SCORING_PROTOCOL = "analytic-v1"
INFERENCE_BACKEND = "llada-full-logits-v1"
DTYPE_NAME = "bfloat16"
TEMPERATURE = 1.0

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_WORK_ID_RE = re.compile(r"[0-9a-f]{64}")
_TIED_WEIGHTS_PATCH_LOCK = threading.RLock()


@dataclass(frozen=True)
class LLaDAModelSpec:
    """The immutable Hugging Face identity of the only v0.1 model backend."""

    repository: str
    revision: str
    tokenizer_revision: str
    remote_code_revision: str

    def __post_init__(self) -> None:
        if self.repository != LLADA_REPOSITORY:
            raise ValueError(f"v0.1 supports only repository {LLADA_REPOSITORY!r}")
        for field in ("revision", "tokenizer_revision", "remote_code_revision"):
            value = getattr(self, field)
            if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
                raise ValueError(f"{field} must be a lowercase 40-character commit SHA")


@dataclass(frozen=True)
class ClaimEncoding:
    """Claim-only tokenizer output used to preserve character-offset geometry."""

    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    word_pieces: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class WorkRequest:
    """One unpadded physical LLaDA request for one claim/arm/mask."""

    work_id: str
    example_id: str
    arm: str
    mask_index: int
    mask_rate: float
    word_indices: tuple[int, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    masked_positions: tuple[int, ...]
    target_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if _WORK_ID_RE.fullmatch(self.work_id) is None:
            raise ValueError("work_id must be a lowercase SHA-256 string")
        if not self.example_id or not self.arm:
            raise ValueError("example_id and arm must be non-empty")
        if self.mask_index < 0:
            raise ValueError("mask_index must be non-negative")
        if not math.isfinite(self.mask_rate) or not 0.0 < self.mask_rate <= 1.0:
            raise ValueError("mask_rate must be finite and in (0, 1]")
        if not self.input_ids or len(self.input_ids) != len(self.attention_mask):
            raise ValueError("input_ids and attention_mask must have equal non-zero length")
        if any(value != 1 for value in self.attention_mask):
            raise ValueError("planned requests must be unpadded with an all-one attention mask")
        if not self.masked_positions or len(self.masked_positions) != len(self.target_ids):
            raise ValueError("masked_positions and target_ids must have equal non-zero length")
        if self.masked_positions != tuple(sorted(set(self.masked_positions))):
            raise ValueError("masked_positions must be sorted and unique")
        if any(
            position < 0 or position >= len(self.input_ids) for position in self.masked_positions
        ):
            raise ValueError("masked_positions contain an out-of-range index")
        if any(
            self.input_ids[position] != LLADA_MASK_TOKEN_ID for position in self.masked_positions
        ):
            raise ValueError("every masked position must contain the LLaDA mask token ID")

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class AnalyticResult:
    """Compact sufficient statistics for one physical masked request."""

    work_id: str
    masked_positions: tuple[int, ...]
    target_ids: tuple[int, ...]
    top_ids: tuple[int, ...]
    top_alternative_ids: tuple[int, ...]
    ce: tuple[float, ...]
    entropy: tuple[float, ...]
    collision_probability: tuple[float, ...]
    collision_entropy: tuple[float, ...]
    delta: tuple[float, ...]
    swap_llr: tuple[float, ...]
    expected_drift: tuple[float, ...]
    expected_dispersion: tuple[float, ...]
    top_mismatch: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.masked_positions),
            len(self.target_ids),
            len(self.top_ids),
            len(self.top_alternative_ids),
            len(self.ce),
            len(self.entropy),
            len(self.collision_probability),
            len(self.collision_entropy),
            len(self.delta),
            len(self.swap_llr),
            len(self.expected_drift),
            len(self.expected_dispersion),
            len(self.top_mismatch),
        }
        if lengths != {len(self.target_ids)} or not self.target_ids:
            raise ValueError("all analytic result vectors must have equal non-zero length")

    @property
    def n_masked_pieces(self) -> int:
        return len(self.target_ids)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return math.fsum(values) / len(values)

    def as_cache_row(self) -> dict[str, Any]:
        """Return the exact physical columns consumed by the cache contract."""

        return {
            "work_id": self.work_id,
            "masked_positions": list(self.masked_positions),
            "n_masked_pieces": self.n_masked_pieces,
            "target_ids": list(self.target_ids),
            "top_ids": list(self.top_ids),
            "top_alternative_ids": list(self.top_alternative_ids),
            "ce_by_piece": list(self.ce),
            "entropy_by_piece": list(self.entropy),
            "collision_probability_by_piece": list(self.collision_probability),
            "collision_entropy_by_piece": list(self.collision_entropy),
            "delta_by_piece": list(self.delta),
            "swap_llr_by_piece": list(self.swap_llr),
            "expected_drift_by_piece": list(self.expected_drift),
            "expected_dispersion_by_piece": list(self.expected_dispersion),
            "top_mismatch_by_piece": list(self.top_mismatch),
            "ce_mean": self._mean(self.ce),
            "entropy_mean": self._mean(self.entropy),
            "collision_entropy_mean": self._mean(self.collision_entropy),
            "delta_mean": self._mean(self.delta),
            "swap_llr_mean": self._mean(self.swap_llr),
            "expected_drift_mean": self._mean(self.expected_drift),
            "expected_dispersion_mean": self._mean(self.expected_dispersion),
            "top_mismatch_rate": self._mean(self.top_mismatch),
        }


class ParityMismatchError(AssertionError):
    """Fail-closed scalar/global parity error with a machine-readable report."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        failures = report.get("failures", [])
        preview = "; ".join(str(value) for value in failures[:3])
        if len(failures) > 3:
            preview += f"; and {len(failures) - 3} more"
        super().__init__(f"scalar/global LLaDA parity failed: {preview}")


@dataclass(frozen=True)
class ScalarBatchParity:
    """One parity report plus the exact results used for aggregate checks."""

    report: dict[str, Any]
    scalar_results: tuple[AnalyticResult, ...]
    batched_results: tuple[AnalyticResult, ...]


def prompt_prefix(evidence: str | None) -> str:
    """Render the frozen prefix while keeping claim tokenization separate."""

    if evidence:
        return f"Evidence: {evidence.strip()}\nClaim: "
    return "Claim: "


def plan_length_buckets(
    requests: Sequence[WorkRequest],
    *,
    max_batch_tokens: int,
    max_batch_rows: int,
    bucket_width: int = 64,
) -> tuple[tuple[WorkRequest, ...], ...]:
    """Deterministically bucket and greedily batch unpadded requests.

    The token budget is the physical padded size: rows times maximum length.
    Sorting by semantic work ID makes the physical plan independent of input order.
    """

    for name, value in {
        "max_batch_tokens": max_batch_tokens,
        "max_batch_rows": max_batch_rows,
        "bucket_width": bucket_width,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    seen: set[str] = set()
    sortable: list[tuple[int, int, str, WorkRequest]] = []
    for request in requests:
        if request.work_id in seen:
            raise ValueError(f"duplicate work_id in inference plan: {request.work_id}")
        seen.add(request.work_id)
        length = request.sequence_length
        if length > max_batch_tokens:
            raise ValueError(
                f"request {request.work_id} has {length} tokens, above max_batch_tokens="
                f"{max_batch_tokens}"
            )
        bucket = (length - 1) // bucket_width
        sortable.append((bucket, length, request.work_id, request))
    sortable.sort(key=lambda item: item[:3])

    batches: list[tuple[WorkRequest, ...]] = []
    current: list[WorkRequest] = []
    current_bucket: int | None = None
    current_max_length = 0
    for bucket, length, _, request in sortable:
        proposed_rows = len(current) + 1
        proposed_max = max(current_max_length, length)
        must_flush = bool(current) and (
            bucket != current_bucket
            or proposed_rows > max_batch_rows
            or proposed_rows * proposed_max > max_batch_tokens
        )
        if must_flush:
            batches.append(tuple(current))
            current = []
            current_max_length = 0
        if not current:
            current_bucket = bucket
        current.append(request)
        current_max_length = max(current_max_length, length)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


@contextmanager
def _tied_weights_compat(torch_module: Any) -> Any:
    """Temporarily provide the attribute expected by some Transformers releases."""

    with _TIED_WEIGHTS_PATCH_LOCK:
        module_type = torch_module.nn.Module
        original = module_type.__getattr__

        def patched(module: Any, name: str) -> Any:
            if name == "all_tied_weights_keys":
                return {}
            return original(module, name)

        module_type.__getattr__ = patched
        try:
            yield
        finally:
            module_type.__getattr__ = original


def _runtime_modules() -> tuple[Any, Any, Any]:
    """Import the optional GPU runtime only when a real backend is requested."""

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in a runtime without extras
        raise RuntimeError(
            "LLaDA inference requires the separately installed Torch/Transformers GPU runtime"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} tokenizer output must be a mapping")
    return value


class LLaDABackend:
    """Full-logit scalar reference and globally padded LLaDA backend."""

    def __init__(
        self,
        *,
        spec: LLaDAModelSpec,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        device: str,
        pad_token_id: int,
        vocab_size: int,
    ) -> None:
        self.spec = spec
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch_module
        self.device = device
        self.pad_token_id = int(pad_token_id)
        self.vocab_size = int(vocab_size)
        self._claim_cache: dict[str, ClaimEncoding] = {}
        self._prefix_cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def load(
        cls,
        *,
        repository: str,
        revision: str,
        tokenizer_revision: str,
        remote_code_revision: str,
        device: str = "cuda",
    ) -> LLaDABackend:
        """Load one exact BF16 CUDA snapshot; never resolve a moving revision."""

        spec = LLaDAModelSpec(
            repository=repository,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            remote_code_revision=remote_code_revision,
        )
        if not isinstance(device, str) or not re.fullmatch(r"cuda(?::\d+)?", device):
            raise ValueError("the v0.1 LLaDA backend requires an explicit CUDA device")

        torch, auto_model, auto_tokenizer = _runtime_modules()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the v0.1 LLaDA backend")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support BF16")

        tokenizer = auto_tokenizer.from_pretrained(
            spec.repository,
            revision=spec.tokenizer_revision,
            trust_remote_code=True,
            use_fast=True,
        )
        with _tied_weights_compat(torch):
            model = auto_model.from_pretrained(
                spec.repository,
                revision=spec.revision,
                code_revision=spec.remote_code_revision,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        model = model.to(device).eval()
        pad_token_id, vocab_size = cls._validate_loaded_assets(tokenizer, model)
        return cls(
            spec=spec,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            device=device,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
        )

    @staticmethod
    def _validate_loaded_assets(tokenizer: Any, model: Any) -> tuple[int, int]:
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("loaded LLaDA model has no config")
        config_mask = getattr(config, "mask_token_id", None)
        if config_mask != LLADA_MASK_TOKEN_ID:
            raise ValueError(
                f"model mask_token_id must be {LLADA_MASK_TOKEN_ID}, got {config_mask!r}"
            )
        converted_mask = tokenizer.convert_tokens_to_ids(LLADA_MASK_TOKEN)
        if converted_mask != LLADA_MASK_TOKEN_ID:
            raise ValueError(
                f"tokenizer must map {LLADA_MASK_TOKEN!r} to {LLADA_MASK_TOKEN_ID}, "
                f"got {converted_mask!r}"
            )
        tokenizer_mask = getattr(tokenizer, "mask_token_id", None)
        if tokenizer_mask is not None and tokenizer_mask != LLADA_MASK_TOKEN_ID:
            raise ValueError("tokenizer mask_token_id disagrees with the LLaDA mask token")
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            converted_token = tokenizer.convert_ids_to_tokens(LLADA_MASK_TOKEN_ID)
            if converted_token != LLADA_MASK_TOKEN:
                raise ValueError("tokenizer mask-token reverse conversion is inconsistent")

        tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
        config_pad = getattr(config, "pad_token_id", None)
        if tokenizer_pad is None or config_pad is None or int(tokenizer_pad) != int(config_pad):
            raise ValueError("tokenizer and model must expose the same non-null pad_token_id")

        vocab_size = getattr(config, "vocab_size", None)
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 2:
            raise ValueError("model config must expose an integer vocab_size >= 2")
        if vocab_size <= LLADA_MASK_TOKEN_ID:
            raise ValueError("LLaDA mask token is outside the model vocabulary")
        if vocab_size >= 2**24:
            raise ValueError("vocabulary IDs cannot be packed exactly into FP32 compact output")

        model_max = getattr(config, "max_sequence_length", None)
        if isinstance(model_max, bool) or not isinstance(model_max, int):
            raise ValueError("model config must expose integer max_sequence_length")
        if model_max < LLADA_PROTOCOL_MAX_LENGTH:
            raise ValueError(
                f"model max_sequence_length={model_max} is below the protocol cap "
                f"{LLADA_PROTOCOL_MAX_LENGTH}"
            )
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        if (
            isinstance(tokenizer_max, int)
            and not isinstance(tokenizer_max, bool)
            and tokenizer_max < LLADA_PROTOCOL_MAX_LENGTH
        ):
            raise ValueError("tokenizer model_max_length is below the protocol cap")
        return int(tokenizer_pad), vocab_size

    def is_oom_error(self, error: BaseException) -> bool:
        """Identify only CUDA allocator failures eligible for one runner retry."""

        oom_type = getattr(self._torch.cuda, "OutOfMemoryError", None)
        if isinstance(oom_type, type) and isinstance(error, oom_type):
            return True
        return isinstance(error, RuntimeError) and "CUDA out of memory" in str(error)

    def peak_vram_bytes(self) -> int:
        """Return peak allocated device memory without resetting allocator state."""

        return int(self._torch.cuda.max_memory_allocated(self.device))

    def reset_peak_vram(self) -> None:
        """Reset peak-memory accounting once at a runner-controlled boundary."""

        self._torch.cuda.reset_peak_memory_stats(self.device)

    def runtime_environment(self) -> dict[str, Any]:
        """Return JSON-safe runtime attribution for the final run receipt."""

        torch = self._torch
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            transformers_version: str | None = metadata.version("transformers")
        except metadata.PackageNotFoundError:
            transformers_version = None
        try:
            device_name: str | None = str(torch.cuda.get_device_name(self.device))
        except (AttributeError, RuntimeError):
            device_name = None
        try:
            capability_value = torch.cuda.get_device_capability(self.device)
            device_capability: list[int] | None = [
                int(capability_value[0]),
                int(capability_value[1]),
            ]
        except (AttributeError, RuntimeError, TypeError, IndexError):
            device_capability = None
        try:
            parameter = next(iter(self.model.parameters()))
            parameter_dtype: str | None = str(parameter.dtype).removeprefix("torch.")
        except (AttributeError, StopIteration, TypeError):
            parameter_dtype = None
        return {
            "torch": str(getattr(torch, "__version__", "unknown")),
            "transformers": transformers_version,
            "cuda_runtime": None if cuda_version is None else str(cuda_version),
            "cuda_driver": None,
            "device": self.device,
            "device_name": device_name,
            "device_capability": device_capability,
            "dtype": DTYPE_NAME,
            "model_parameter_dtype": parameter_dtype,
        }

    def encode_claim(self, claim: str) -> ClaimEncoding:
        """Tokenize the claim alone so offsets remain claim-relative."""

        if not isinstance(claim, str) or not claim:
            raise ValueError("claim must be a non-empty string")
        cached = self._claim_cache.get(claim)
        if cached is not None:
            return cached
        encoded = _require_mapping(
            self.tokenizer(
                claim,
                add_special_tokens=False,
                return_offsets_mapping=True,
            ),
            label="claim",
        )
        if "input_ids" not in encoded or "offset_mapping" not in encoded:
            raise ValueError("fast tokenizer must return input_ids and offset_mapping")
        token_ids = tuple(int(value) for value in encoded["input_ids"])
        offsets = tuple((int(value[0]), int(value[1])) for value in encoded["offset_mapping"])
        if not token_ids or len(token_ids) != len(offsets):
            raise ValueError("claim token IDs and offsets must have equal non-zero length")
        word_pieces = align_word_pieces(claim, offsets)
        result = ClaimEncoding(token_ids, offsets, word_pieces)
        self._claim_cache[claim] = result
        return result

    def encode_prefix(self, evidence: str | None) -> tuple[int, ...]:
        prefix = prompt_prefix(evidence)
        cached = self._prefix_cache.get(prefix)
        if cached is not None:
            return cached
        encoded = _require_mapping(
            self.tokenizer(prefix, add_special_tokens=False),
            label="prefix",
        )
        if "input_ids" not in encoded:
            raise ValueError("tokenizer prefix output must contain input_ids")
        token_ids = tuple(int(value) for value in encoded["input_ids"])
        if not token_ids:
            raise ValueError("prompt prefix tokenization must not be empty")
        self._prefix_cache[prefix] = token_ids
        return token_ids

    def prepare_request(
        self,
        *,
        example_id: str,
        claim: str,
        evidence: str | None,
        arm: str,
        choice: MaskChoice,
        mask_policy: str = "fixed-v1",
    ) -> WorkRequest:
        """Build exact multi-piece geometry and its padding-independent work ID."""

        claim_encoding = self.encode_claim(claim)
        prefix_ids = self.encode_prefix(evidence)
        if len(prefix_ids) + len(claim_encoding.token_ids) > LLADA_PROTOCOL_MAX_LENGTH:
            raise ValueError(
                f"sequence exceeds the analytic-v1 cap of {LLADA_PROTOCOL_MAX_LENGTH} tokens"
            )
        geometry = apply_mask_geometry(
            claim_encoding.token_ids,
            claim_encoding.word_pieces,
            choice,
            mask_token_id=LLADA_MASK_TOKEN_ID,
        )
        input_ids = prefix_ids + geometry.masked_input_ids
        attention_mask = (1,) * len(input_ids)
        masked_positions = tuple(len(prefix_ids) + value for value in geometry.piece_indices)
        work_id = stable_work_id(
            scoring_protocol=SCORING_PROTOCOL,
            model_repository=self.spec.repository,
            model_revision=self.spec.revision,
            tokenizer_revision=self.spec.tokenizer_revision,
            remote_code_revision=self.spec.remote_code_revision,
            arm=arm,
            prompt_protocol=PROMPT_PROTOCOL,
            mask_policy=mask_policy,
            mask_rate=choice.mask_rate,
            input_ids=input_ids,
            attention_mask=attention_mask,
            masked_positions=masked_positions,
            target_ids=geometry.target_ids,
            temperature=TEMPERATURE,
            dtype=DTYPE_NAME,
            inference_backend=INFERENCE_BACKEND,
        )
        return WorkRequest(
            work_id=work_id,
            example_id=example_id,
            arm=arm,
            mask_index=choice.mask_index,
            mask_rate=choice.mask_rate,
            word_indices=choice.word_indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            masked_positions=masked_positions,
            target_ids=geometry.target_ids,
        )

    def _validate_request_for_model(self, request: WorkRequest) -> None:
        if request.sequence_length > LLADA_PROTOCOL_MAX_LENGTH:
            raise ValueError("request exceeds the analytic-v1 sequence cap")
        if any(target < 0 or target >= self.vocab_size for target in request.target_ids):
            raise ValueError("request target ID is outside the model vocabulary")

    def _stats_from_selected_logits(
        self,
        selected_logits: Any,
        targets: Any,
    ) -> list[list[float]]:
        """Reduce full selected-position logits on-device, then transfer once."""

        torch = self._torch
        selected = selected_logits.float()
        if (
            len(selected.shape) != 2
            or selected.shape[0] < 1
            or selected.shape[1] != self.vocab_size
        ):
            raise ValueError("selected logits must have shape [positions, model vocabulary]")
        log_probabilities = torch.log_softmax(selected, dim=-1)
        row_indices = torch.arange(selected.shape[0], dtype=torch.long, device=self.device)
        target_log_probabilities = log_probabilities[row_indices, targets]
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        collision = (probabilities * probabilities).sum(dim=-1)
        collision_entropy = -torch.log(collision)
        top_ids = torch.argmax(log_probabilities, dim=-1)

        alternative_log_probabilities = log_probabilities.clone()
        alternative_log_probabilities[row_indices, targets] = float("-inf")
        # torch.argmax returns the first maximum, freezing the lowest-ID tie break.
        top_alternative_ids = torch.argmax(alternative_log_probabilities, dim=-1)
        top_alternative_log_probabilities = log_probabilities[row_indices, top_alternative_ids]

        ce = -target_log_probabilities
        delta = ce - entropy
        swap_llr = top_alternative_log_probabilities - target_log_probabilities
        expected_drift = 1.0 - probabilities[row_indices, targets]
        expected_dispersion = 1.0 - collision
        top_mismatch = (top_ids != targets).float()

        # Vocabulary IDs for this backend are below 2**24, hence exact in FP32.
        packed = torch.stack(
            (
                targets.float(),
                top_ids.float(),
                top_alternative_ids.float(),
                ce,
                entropy,
                collision,
                collision_entropy,
                delta,
                swap_llr,
                expected_drift,
                expected_dispersion,
                top_mismatch,
            ),
            dim=-1,
        )
        return packed.detach().cpu().tolist()

    @staticmethod
    def _result_from_rows(
        request: WorkRequest,
        rows: Sequence[Sequence[float]],
    ) -> AnalyticResult:
        if len(rows) != len(request.target_ids) or any(len(row) != 12 for row in rows):
            raise ValueError("compact model output has invalid cardinality")

        def ints(column: int) -> tuple[int, ...]:
            return tuple(int(row[column]) for row in rows)

        def floats(column: int) -> tuple[float, ...]:
            return tuple(float(row[column]) for row in rows)

        packed_targets = ints(0)
        if packed_targets != request.target_ids:
            raise ValueError("compact model output target IDs disagree with the work request")
        ce = floats(3)
        entropy = floats(4)
        collision_probability = floats(5)
        raw_collision_entropy = floats(6)
        raw_delta = floats(7)
        raw_expected_drift = floats(9)
        raw_expected_dispersion = floats(10)
        collision_entropy = tuple(-math.log(value) for value in collision_probability)
        delta = tuple(
            ce_value - entropy_value for ce_value, entropy_value in zip(ce, entropy, strict=True)
        )
        expected_drift = tuple(1.0 - math.exp(-value) for value in ce)
        expected_dispersion = tuple(1.0 - value for value in collision_probability)
        for label, raw_values, canonical_values in (
            ("collision entropy", raw_collision_entropy, collision_entropy),
            ("delta", raw_delta, delta),
            ("expected drift", raw_expected_drift, expected_drift),
            ("expected dispersion", raw_expected_dispersion, expected_dispersion),
        ):
            if any(
                not math.isclose(raw, canonical, rel_tol=1e-5, abs_tol=1e-6)
                for raw, canonical in zip(raw_values, canonical_values, strict=True)
            ):
                raise ValueError(f"GPU compact output has incoherent {label}")
        return AnalyticResult(
            work_id=request.work_id,
            masked_positions=request.masked_positions,
            target_ids=packed_targets,
            top_ids=ints(1),
            top_alternative_ids=ints(2),
            ce=ce,
            entropy=entropy,
            collision_probability=collision_probability,
            collision_entropy=collision_entropy,
            delta=delta,
            swap_llr=floats(8),
            expected_drift=expected_drift,
            expected_dispersion=expected_dispersion,
            top_mismatch=floats(11),
        )

    def infer_scalar(self, request: WorkRequest) -> AnalyticResult:
        """Reference path: one unpadded row and full sequence-by-vocabulary logits."""

        self._validate_request_for_model(request)
        torch = self._torch
        input_ids = torch.tensor(
            [list(request.input_ids)],
            dtype=torch.long,
            device=self.device,
        )
        attention = torch.tensor(
            [list(request.attention_mask)],
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.tensor(
            request.masked_positions,
            dtype=torch.long,
            device=self.device,
        )
        targets = torch.tensor(request.target_ids, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits
            selected = logits[0].index_select(0, positions)
            rows = self._stats_from_selected_logits(selected, targets)
        return self._result_from_rows(request, rows)

    def _infer_padded_batch(
        self,
        requests: Sequence[WorkRequest],
    ) -> tuple[AnalyticResult, ...]:
        if not requests:
            return ()
        for request in requests:
            self._validate_request_for_model(request)

        torch = self._torch
        max_length = max(request.sequence_length for request in requests)
        input_ids = torch.full(
            (len(requests), max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention = torch.zeros(
            (len(requests), max_length),
            dtype=torch.long,
            device=self.device,
        )
        for row_index, request in enumerate(requests):
            length = request.sequence_length
            input_ids[row_index, :length] = torch.tensor(
                request.input_ids,
                dtype=torch.long,
                device=self.device,
            )
            attention[row_index, :length] = torch.tensor(
                request.attention_mask,
                dtype=torch.long,
                device=self.device,
            )

        selected_rows: list[Any] = []
        target_rows: list[Any] = []
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits
            if len(logits.shape) != 3 or logits.shape[:2] != input_ids.shape:
                raise ValueError(
                    "model logits must have shape [batch, padded sequence, vocabulary]"
                )
            if logits.shape[-1] != self.vocab_size:
                raise ValueError("model logit vocabulary does not match model config")
            for row_index, request in enumerate(requests):
                positions = torch.tensor(
                    request.masked_positions,
                    dtype=torch.long,
                    device=self.device,
                )
                selected_rows.append(logits[row_index].index_select(0, positions))
                target_rows.append(
                    torch.tensor(request.target_ids, dtype=torch.long, device=self.device)
                )
            packed_rows = self._stats_from_selected_logits(
                torch.cat(selected_rows, dim=0),
                torch.cat(target_rows, dim=0),
            )

        results: list[AnalyticResult] = []
        offset = 0
        for request in requests:
            next_offset = offset + len(request.target_ids)
            results.append(self._result_from_rows(request, packed_rows[offset:next_offset]))
            offset = next_offset
        if offset != len(packed_rows):
            raise ValueError("global batch produced unexpected output cardinality")
        return tuple(results)

    def infer_batch(
        self,
        requests: Sequence[WorkRequest],
    ) -> tuple[AnalyticResult, ...]:
        """Run one right-padded, single-forward physical batch in caller order."""

        work_ids = [request.work_id for request in requests]
        if len(work_ids) != len(set(work_ids)):
            raise ValueError("infer_batch requires unique work IDs")
        return self._infer_padded_batch(requests)

    def infer_global(
        self,
        requests: Sequence[WorkRequest],
        *,
        max_batch_tokens: int,
        max_batch_rows: int,
        bucket_width: int = 64,
    ) -> tuple[AnalyticResult, ...]:
        """Run deterministic length-bucketed batches and restore caller order."""

        batches = plan_length_buckets(
            requests,
            max_batch_tokens=max_batch_tokens,
            max_batch_rows=max_batch_rows,
            bucket_width=bucket_width,
        )
        by_work_id: dict[str, AnalyticResult] = {}
        for batch in batches:
            for result in self.infer_batch(batch):
                if result.work_id in by_work_id:
                    raise ValueError(f"duplicate model output for work_id {result.work_id}")
                by_work_id[result.work_id] = result
        if len(by_work_id) != len(requests):
            raise ValueError("global inference output cardinality does not match the work plan")
        return tuple(by_work_id[request.work_id] for request in requests)


_PARITY_NUMERIC_METRICS = (
    "ce",
    "entropy",
    "collision_probability",
    "collision_entropy",
    "delta",
    "swap_llr",
    "expected_drift",
    "expected_dispersion",
    "top_mismatch",
)


def run_scalar_batch_parity(
    backend: LLaDABackend,
    requests: Sequence[WorkRequest],
    max_batch_tokens: int,
    max_batch_rows: int,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> ScalarBatchParity:
    """Run the bounded manual parity gate and retain both result sets.

    Any exact-identity or numeric mismatch raises ParityMismatchError. Its
    report attribute contains the same JSON-safe report with passed=false and
    explicit failure details.
    """

    tolerance_values: dict[str, float] = {}
    for name, value in {"rtol": rtol, "atol": atol}.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        tolerance_values[name] = numeric
    if not requests:
        raise ValueError("parity comparison requires at least one work request")

    batches = plan_length_buckets(
        requests,
        max_batch_tokens=max_batch_tokens,
        max_batch_rows=max_batch_rows,
    )
    expected_work_ids = [request.work_id for request in requests]
    scalar_results = [backend.infer_scalar(request) for request in requests]
    batched_results = list(
        backend.infer_global(
            requests,
            max_batch_tokens=max_batch_tokens,
            max_batch_rows=max_batch_rows,
        )
    )
    scalar_work_ids = [result.work_id for result in scalar_results]
    batched_work_ids = [result.work_id for result in batched_results]

    failures: list[str] = []
    exact = {
        "output_cardinality": (
            len(scalar_results) == len(requests) and len(batched_results) == len(requests)
        ),
        "work_ids": (
            scalar_work_ids == expected_work_ids and batched_work_ids == expected_work_ids
        ),
        "masked_positions": True,
        "target_ids": True,
        "top_ids": True,
        "top_alternative_ids": True,
    }
    if not exact["output_cardinality"]:
        failures.append(
            "output cardinality mismatch: "
            f"expected={len(requests)} scalar={len(scalar_results)} "
            f"batched={len(batched_results)}"
        )
    if scalar_work_ids != expected_work_ids:
        failures.append("scalar work IDs do not exactly match request order")
    if batched_work_ids != expected_work_ids:
        failures.append("batched work IDs do not exactly match request order")

    max_absolute_error: dict[str, float | None] = {
        metric: 0.0 for metric in _PARITY_NUMERIC_METRICS
    }
    comparable_count = min(len(requests), len(scalar_results), len(batched_results))
    for request_index in range(comparable_count):
        request = requests[request_index]
        scalar = scalar_results[request_index]
        batched = batched_results[request_index]
        exact_fields = {
            "masked_positions": (
                scalar.masked_positions == request.masked_positions
                and batched.masked_positions == request.masked_positions
            ),
            "target_ids": (
                scalar.target_ids == request.target_ids and batched.target_ids == request.target_ids
            ),
            "top_ids": scalar.top_ids == batched.top_ids,
            "top_alternative_ids": (scalar.top_alternative_ids == batched.top_alternative_ids),
        }
        for field, matches in exact_fields.items():
            if not matches:
                exact[field] = False
                failures.append(
                    f"request {request.work_id}: exact {field} mismatch at output index "
                    f"{request_index}"
                )

        for metric in _PARITY_NUMERIC_METRICS:
            scalar_values = getattr(scalar, metric)
            batched_values = getattr(batched, metric)
            if len(scalar_values) != len(batched_values):
                max_absolute_error[metric] = None
                failures.append(f"request {request.work_id}: {metric} vector cardinality mismatch")
                continue
            for piece_index, (scalar_value, batched_value) in enumerate(
                zip(scalar_values, batched_values, strict=True)
            ):
                left = float(scalar_value)
                right = float(batched_value)
                if not math.isfinite(left) or not math.isfinite(right):
                    max_absolute_error[metric] = None
                    failures.append(
                        f"request {request.work_id}: {metric}[{piece_index}] is non-finite"
                    )
                    continue
                absolute_error = abs(left - right)
                previous = max_absolute_error[metric]
                if previous is not None:
                    max_absolute_error[metric] = max(previous, absolute_error)
                if not math.isclose(
                    left,
                    right,
                    rel_tol=tolerance_values["rtol"],
                    abs_tol=tolerance_values["atol"],
                ):
                    failures.append(
                        f"request {request.work_id}: {metric}[{piece_index}] "
                        f"scalar={left} batched={right} exceeds tolerance"
                    )

    exact_passed = all(exact.values())
    numeric_passed = not any(
        "exceeds tolerance" in failure
        or "non-finite" in failure
        or "vector cardinality mismatch" in failure
        for failure in failures
    )
    report: dict[str, Any] = {
        "schema_version": "llada-scalar-batch-parity-v1",
        "passed": exact_passed and numeric_passed and not failures,
        "request_count": len(requests),
        "tolerances": tolerance_values,
        "forward_counts": {
            "scalar": len(requests),
            "batched": len(batches),
        },
        "batch_plan": {
            "batch_count": len(batches),
            "rows_per_batch": [len(batch) for batch in batches],
            "padded_tokens_per_batch": [
                len(batch) * max(request.sequence_length for request in batch) for batch in batches
            ],
        },
        "expected_work_ids": expected_work_ids,
        "scalar_work_ids": scalar_work_ids,
        "batched_work_ids": batched_work_ids,
        "exact_checks": exact,
        "numeric_checks": {
            "metrics": list(_PARITY_NUMERIC_METRICS),
            "max_absolute_error": max_absolute_error,
        },
        "failure_count": len(failures),
        "failures": failures,
    }
    canonical_json(report)
    if not report["passed"]:
        raise ParityMismatchError(report)
    return ScalarBatchParity(
        report=report,
        scalar_results=tuple(scalar_results),
        batched_results=tuple(batched_results),
    )


def compare_scalar_and_batched(
    backend: LLaDABackend,
    requests: Sequence[WorkRequest],
    max_batch_tokens: int,
    max_batch_rows: int,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """Run the bounded manual parity gate and return its canonical report."""

    return run_scalar_batch_parity(
        backend,
        requests,
        max_batch_tokens=max_batch_tokens,
        max_batch_rows=max_batch_rows,
        rtol=rtol,
        atol=atol,
    ).report
