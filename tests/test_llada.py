from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import dfi.llada as llada
from dfi.masking import MaskChoice, canonical_json
from dfi.scoring import score_logits

REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
CODE_REVISION = "c" * 40


class FakeTensor:
    def __init__(self, values: Any, *, dtype: Any | None = None) -> None:
        if dtype is None:
            self.values = np.asarray(values)
        else:
            self.values = np.asarray(values, dtype=dtype)
        self.device = "cuda"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @staticmethod
    def _index(value: Any) -> Any:
        if isinstance(value, FakeTensor):
            return value.values.astype(np.int64)
        if isinstance(value, tuple):
            return tuple(FakeTensor._index(item) for item in value)
        return value

    @staticmethod
    def _values(value: Any) -> Any:
        return value.values if isinstance(value, FakeTensor) else value

    def __getitem__(self, key: Any) -> FakeTensor:
        return FakeTensor(self.values[self._index(key)])

    def __setitem__(self, key: Any, value: Any) -> None:
        self.values[self._index(key)] = self._values(value)

    def __len__(self) -> int:
        return len(self.values)

    def float(self) -> FakeTensor:
        return FakeTensor(self.values, dtype=np.float32)

    def clone(self) -> FakeTensor:
        return FakeTensor(self.values.copy())

    def exp(self) -> FakeTensor:
        return FakeTensor(np.exp(self.values))

    def sum(self, dim: int) -> FakeTensor:
        return FakeTensor(self.values.sum(axis=dim))

    def index_select(self, dim: int, indices: FakeTensor) -> FakeTensor:
        return FakeTensor(np.take(self.values, indices.values.astype(np.int64), axis=dim))

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[Any]:
        return self.values.tolist()

    def __neg__(self) -> FakeTensor:
        return FakeTensor(-self.values)

    def __mul__(self, other: Any) -> FakeTensor:
        return FakeTensor(self.values * self._values(other))

    def __rmul__(self, other: Any) -> FakeTensor:
        return self * other

    def __add__(self, other: Any) -> FakeTensor:
        return FakeTensor(self.values + self._values(other))

    def __radd__(self, other: Any) -> FakeTensor:
        return self + other

    def __sub__(self, other: Any) -> FakeTensor:
        return FakeTensor(self.values - self._values(other))

    def __rsub__(self, other: Any) -> FakeTensor:
        return FakeTensor(self._values(other) - self.values)

    def __ne__(self, other: Any) -> FakeTensor:
        return FakeTensor(self.values != self._values(other))


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, *, available: bool = True, bf16: bool = True) -> None:
        self.available = available
        self.bf16 = bf16
        self.reset_devices: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    @staticmethod
    def max_memory_allocated(device: str) -> int:
        assert device.startswith("cuda")
        return 123456

    def reset_peak_memory_stats(self, device: str) -> None:
        self.reset_devices.append(device)

    @staticmethod
    def get_device_name(device: str) -> str:
        assert device.startswith("cuda")
        return "Fake A100"

    @staticmethod
    def get_device_capability(device: str) -> tuple[int, int]:
        assert device.startswith("cuda")
        return (8, 0)


class FakeModule:
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


class FakeTorch:
    __version__ = "fake-torch"
    long = np.int64
    bfloat16 = object()
    nn = SimpleNamespace(Module=FakeModule)
    version = SimpleNamespace(cuda="fake-cuda")

    def __init__(self, *, available: bool = True, bf16: bool = True) -> None:
        self.cuda = FakeCuda(available=available, bf16=bf16)

    @staticmethod
    def tensor(values: Any, *, dtype: Any, device: str) -> FakeTensor:
        del device
        return FakeTensor(values, dtype=dtype)

    @staticmethod
    def full(shape: tuple[int, int], fill: int, *, dtype: Any, device: str) -> FakeTensor:
        del device
        return FakeTensor(np.full(shape, fill, dtype=dtype))

    @staticmethod
    def zeros(shape: tuple[int, int], *, dtype: Any, device: str) -> FakeTensor:
        del device
        return FakeTensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def arange(length: int, *, dtype: Any, device: str) -> FakeTensor:
        del device
        return FakeTensor(np.arange(length, dtype=dtype))

    @staticmethod
    def log_softmax(tensor: FakeTensor, dim: int) -> FakeTensor:
        values = tensor.values.astype(np.float32)
        shifted = values - np.max(values, axis=dim, keepdims=True)
        output = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
        return FakeTensor(output.astype(np.float32))

    @staticmethod
    def log(tensor: FakeTensor) -> FakeTensor:
        return FakeTensor(np.log(tensor.values))

    @staticmethod
    def argmax(tensor: FakeTensor, dim: int) -> FakeTensor:
        return FakeTensor(np.argmax(tensor.values, axis=dim), dtype=np.int64)

    @staticmethod
    def stack(tensors: tuple[FakeTensor, ...], dim: int) -> FakeTensor:
        return FakeTensor(np.stack([tensor.values for tensor in tensors], axis=dim))

    @staticmethod
    def cat(tensors: list[FakeTensor], dim: int) -> FakeTensor:
        return FakeTensor(np.concatenate([tensor.values for tensor in tensors], axis=dim))

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


class FakeTokenizer:
    pad_token_id = 5
    mask_token_id = None
    model_max_length = 4096

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.claims = {
            "Alpha betaword": {
                "input_ids": [1, 2, 3],
                "offset_mapping": [(0, 5), (6, 10), (10, 14)],
            },
            "Gamma": {
                "input_ids": [0],
                "offset_mapping": [(0, 5)],
            },
            "Gamma delta": {
                "input_ids": [0, 1],
                "offset_mapping": [(0, 5), (6, 11)],
            },
        }
        self.prefixes = {
            "Claim: ": {"input_ids": [4]},
            "Evidence: card\nClaim: ": {"input_ids": [4, 0]},
        }

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((text, dict(kwargs)))
        if kwargs.get("return_offsets_mapping"):
            return self.claims[text]
        return self.prefixes[text]

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        if token == llada.LLADA_MASK_TOKEN:
            return llada.LLADA_MASK_TOKEN_ID
        return -1

    @staticmethod
    def convert_ids_to_tokens(token_id: int) -> str:
        if token_id == llada.LLADA_MASK_TOKEN_ID:
            return llada.LLADA_MASK_TOKEN
        return "<unknown>"


class DeterministicModel:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, *, input_ids: FakeTensor, attention_mask: FakeTensor) -> Any:
        ids = input_ids.values.copy()
        attention = attention_mask.values.copy()
        self.calls.append((ids, attention))
        batch, length = ids.shape
        output = np.empty((batch, length, self.vocab_size), dtype=np.float32)
        for row in range(batch):
            context = int(ids[row][attention[row] == 1].sum())
            for position in range(length):
                for token_id in range(self.vocab_size):
                    output[row, position, token_id] = (
                        token_id * 0.17
                        + ((context + position * 3 + token_id * token_id) % 11) * 0.031
                    )
        return SimpleNamespace(logits=FakeTensor(output))

    @staticmethod
    def parameters() -> Any:
        return iter([SimpleNamespace(dtype="torch.bfloat16")])


def make_spec() -> llada.LLaDAModelSpec:
    return llada.LLaDAModelSpec(
        repository=llada.LLADA_REPOSITORY,
        revision=REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        remote_code_revision=CODE_REVISION,
    )


def make_backend() -> tuple[llada.LLaDABackend, FakeTokenizer, DeterministicModel]:
    tokenizer = FakeTokenizer()
    model = DeterministicModel(vocab_size=6)
    backend = llada.LLaDABackend(
        spec=make_spec(),
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
        device="cuda",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=model.vocab_size,
    )
    return backend, tokenizer, model


def make_request(index: int, length: int) -> llada.WorkRequest:
    values = [1] * length
    values[-1] = llada.LLADA_MASK_TOKEN_ID
    return llada.WorkRequest(
        work_id=f"{index:064x}",
        example_id=f"example-{index}",
        arm="prior",
        mask_index=0,
        mask_rate=0.5,
        word_indices=(0,),
        input_ids=tuple(values),
        attention_mask=(1,) * length,
        masked_positions=(length - 1,),
        target_ids=(index % 5,),
    )


def test_module_has_no_eager_gpu_runtime_import() -> None:
    assert "torch" not in llada.__dict__
    assert "transformers" not in llada.__dict__


def test_model_spec_requires_exact_repo_and_immutable_commits() -> None:
    assert make_spec().revision == REVISION
    with pytest.raises(ValueError, match="only repository"):
        llada.LLaDAModelSpec("moving/model", REVISION, TOKENIZER_REVISION, CODE_REVISION)
    with pytest.raises(ValueError, match="revision must be"):
        llada.LLaDAModelSpec(llada.LLADA_REPOSITORY, "main", TOKENIZER_REVISION, CODE_REVISION)
    with pytest.raises(ValueError, match="tokenizer_revision must be"):
        llada.LLaDAModelSpec(llada.LLADA_REPOSITORY, REVISION, "A" * 40, CODE_REVISION)


def test_loader_pins_all_assets_and_restores_scoped_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = FakeTorch()
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = 126081
    loaded_model = SimpleNamespace(
        config=SimpleNamespace(
            mask_token_id=llada.LLADA_MASK_TOKEN_ID,
            pad_token_id=126081,
            vocab_size=126464,
            max_sequence_length=4096,
        )
    )
    loaded_model.to = lambda device: loaded_model
    loaded_model.eval = lambda: loaded_model
    tokenizer_calls: list[tuple[str, dict[str, Any]]] = []
    model_calls: list[tuple[str, dict[str, Any]]] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(repository: str, **kwargs: Any) -> FakeTokenizer:
            tokenizer_calls.append((repository, kwargs))
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(repository: str, **kwargs: Any) -> Any:
            assert FakeModule().all_tied_weights_keys == {}
            model_calls.append((repository, kwargs))
            return loaded_model

    original_getattr = FakeModule.__getattr__
    monkeypatch.setattr(llada, "_runtime_modules", lambda: (torch, AutoModel, AutoTokenizer))
    backend = llada.LLaDABackend.load(
        repository=llada.LLADA_REPOSITORY,
        revision=REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        remote_code_revision=CODE_REVISION,
        device="cuda:0",
    )

    assert backend.model is loaded_model
    assert FakeModule.__getattr__ is original_getattr
    assert tokenizer_calls == [
        (
            llada.LLADA_REPOSITORY,
            {
                "revision": TOKENIZER_REVISION,
                "trust_remote_code": True,
                "use_fast": True,
            },
        )
    ]
    assert model_calls == [
        (
            llada.LLADA_REPOSITORY,
            {
                "revision": REVISION,
                "code_revision": CODE_REVISION,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            },
        )
    ]


def test_loader_restores_patch_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = FakeTorch()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(repository: str, **kwargs: Any) -> FakeTokenizer:
            del repository, kwargs
            return FakeTokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(repository: str, **kwargs: Any) -> Any:
            del repository, kwargs
            assert FakeModule().all_tied_weights_keys == {}
            raise RuntimeError("load failed")

    original_getattr = FakeModule.__getattr__
    monkeypatch.setattr(llada, "_runtime_modules", lambda: (torch, AutoModel, AutoTokenizer))
    with pytest.raises(RuntimeError, match="load failed"):
        llada.LLaDABackend.load(
            repository=llada.LLADA_REPOSITORY,
            revision=REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            remote_code_revision=CODE_REVISION,
        )
    assert FakeModule.__getattr__ is original_getattr


@pytest.mark.parametrize(
    ("available", "bf16", "message"),
    [
        (False, True, "CUDA is required"),
        (True, False, "does not support BF16"),
    ],
)
def test_loader_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    bf16: bool,
    message: str,
) -> None:
    class Never:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError((args, kwargs))

    monkeypatch.setattr(
        llada,
        "_runtime_modules",
        lambda: (FakeTorch(available=available, bf16=bf16), Never, Never),
    )
    with pytest.raises(RuntimeError, match=message):
        llada.LLaDABackend.load(
            repository=llada.LLADA_REPOSITORY,
            revision=REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            remote_code_revision=CODE_REVISION,
        )


def test_loaded_assets_require_exact_mask_pad_and_length() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = 126081
    config = SimpleNamespace(
        mask_token_id=llada.LLADA_MASK_TOKEN_ID,
        pad_token_id=126081,
        vocab_size=126464,
        max_sequence_length=4096,
    )
    model = SimpleNamespace(config=config)
    assert llada.LLaDABackend._validate_loaded_assets(tokenizer, model) == (126081, 126464)

    config.mask_token_id = 1
    with pytest.raises(ValueError, match="mask_token_id"):
        llada.LLaDABackend._validate_loaded_assets(tokenizer, model)
    config.mask_token_id = llada.LLADA_MASK_TOKEN_ID
    config.pad_token_id = 0
    with pytest.raises(ValueError, match="same non-null pad"):
        llada.LLaDABackend._validate_loaded_assets(tokenizer, model)
    config.pad_token_id = 126081
    config.max_sequence_length = 511
    with pytest.raises(ValueError, match="below the protocol cap"):
        llada.LLaDABackend._validate_loaded_assets(tokenizer, model)


def test_prepare_request_uses_separate_prompt_and_multi_piece_geometry() -> None:
    backend, tokenizer, _ = make_backend()
    choice = MaskChoice(mask_index=3, mask_rate=0.5, word_indices=(1,))
    request = backend.prepare_request(
        example_id="alpha",
        claim="Alpha betaword",
        evidence="  card  ",
        arm="evidence",
        choice=choice,
    )

    assert llada.prompt_prefix(None) == "Claim: "
    assert llada.prompt_prefix("  card  ") == "Evidence: card\nClaim: "
    assert tokenizer.calls == [
        (
            "Alpha betaword",
            {"add_special_tokens": False, "return_offsets_mapping": True},
        ),
        ("Evidence: card\nClaim: ", {"add_special_tokens": False}),
    ]
    assert request.input_ids == (4, 0, 1, llada.LLADA_MASK_TOKEN_ID, llada.LLADA_MASK_TOKEN_ID)
    assert request.attention_mask == (1, 1, 1, 1, 1)
    assert request.masked_positions == (3, 4)
    assert request.target_ids == (2, 3)
    assert request.word_indices == (1,)

    repeated = backend.prepare_request(
        example_id="alpha",
        claim="Alpha betaword",
        evidence="  card  ",
        arm="evidence",
        choice=choice,
    )
    assert repeated == request
    assert len(tokenizer.calls) == 2


def test_length_buckets_are_deterministic_and_obey_both_caps() -> None:
    requests = [make_request(1, 3), make_request(2, 4), make_request(3, 5), make_request(4, 7)]
    forward = llada.plan_length_buckets(
        requests,
        max_batch_tokens=10,
        max_batch_rows=2,
        bucket_width=4,
    )
    reverse = llada.plan_length_buckets(
        list(reversed(requests)),
        max_batch_tokens=10,
        max_batch_rows=2,
        bucket_width=4,
    )
    forward_ids = tuple(tuple(item.work_id for item in batch) for batch in forward)
    reverse_ids = tuple(tuple(item.work_id for item in batch) for batch in reverse)
    assert forward_ids == reverse_ids
    assert forward_ids == (
        (requests[0].work_id, requests[1].work_id),
        (requests[2].work_id,),
        (requests[3].work_id,),
    )
    for batch in forward:
        assert len(batch) <= 2
        assert len(batch) * max(item.sequence_length for item in batch) <= 10

    with pytest.raises(ValueError, match="duplicate work_id"):
        llada.plan_length_buckets(
            [requests[0], requests[0]],
            max_batch_tokens=10,
            max_batch_rows=2,
        )
    with pytest.raises(ValueError, match="above max_batch_tokens"):
        llada.plan_length_buckets(
            [make_request(9, 11)],
            max_batch_tokens=10,
            max_batch_rows=2,
        )


def test_scalar_and_global_paths_match_full_logit_reference() -> None:
    backend, _, model = make_backend()
    first = backend.prepare_request(
        example_id="alpha",
        claim="Alpha betaword",
        evidence=None,
        arm="prior",
        choice=MaskChoice(mask_index=0, mask_rate=0.5, word_indices=(1,)),
    )
    second = backend.prepare_request(
        example_id="gamma",
        claim="Gamma",
        evidence=None,
        arm="prior",
        choice=MaskChoice(mask_index=0, mask_rate=0.5, word_indices=(0,)),
    )

    scalar = backend.infer_scalar(first)
    scalar_logits = model.calls[-1][0]
    del scalar_logits
    global_results = backend.infer_global(
        [second, first],
        max_batch_tokens=16,
        max_batch_rows=8,
        bucket_width=64,
    )
    assert tuple(result.work_id for result in global_results) == (second.work_id, first.work_id)
    direct_batch = backend.infer_batch([second, first])
    assert tuple(result.work_id for result in direct_batch) == (second.work_id, first.work_id)
    global_first = global_results[1]
    assert global_first.target_ids == scalar.target_ids
    assert global_first.top_ids == scalar.top_ids
    assert global_first.top_alternative_ids == scalar.top_alternative_ids
    for name in (
        "ce",
        "entropy",
        "collision_probability",
        "collision_entropy",
        "delta",
        "swap_llr",
        "expected_drift",
        "expected_dispersion",
        "top_mismatch",
    ):
        assert getattr(global_first, name) == pytest.approx(getattr(scalar, name), abs=1e-7)

    # The independent NumPy analytic core agrees with the selected full logits.
    ids, attention = model.calls[0]
    output = model(input_ids=FakeTensor(ids), attention_mask=FakeTensor(attention)).logits
    selected = output.values[0, list(first.masked_positions)]
    expected = score_logits(selected, first.target_ids)
    assert scalar.ce == pytest.approx(expected.ce, rel=1e-6, abs=1e-6)
    assert scalar.entropy == pytest.approx(expected.entropy, rel=1e-6, abs=1e-6)
    assert scalar.collision_probability == pytest.approx(
        expected.collision_probability,
        rel=1e-6,
        abs=1e-6,
    )
    assert scalar.top_alternative_ids == tuple(expected.top_alternative_ids)
    assert scalar.swap_llr == pytest.approx(expected.swap_llr, rel=1e-6, abs=1e-6)

    cache_row = scalar.as_cache_row()
    assert cache_row["work_id"] == first.work_id
    assert cache_row["n_masked_pieces"] == 2
    assert cache_row["target_ids"] == [2, 3]
    assert cache_row["ce_mean"] == pytest.approx(sum(scalar.ce) / 2)
    # The exported float32-derived statistics retain exact cache-contract
    # identities after their one device-to-host transfer.
    for index in range(cache_row["n_masked_pieces"]):
        assert cache_row["delta_by_piece"][index] == (
            cache_row["ce_by_piece"][index] - cache_row["entropy_by_piece"][index]
        )
        assert cache_row["expected_dispersion_by_piece"][index] == (
            1.0 - cache_row["collision_probability_by_piece"][index]
        )
    assert not hasattr(backend._torch.cuda, "empty_cache")


def test_runner_telemetry_and_oom_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _, _ = make_backend()
    monkeypatch.setattr(llada.metadata, "version", lambda package: f"fake-{package}")

    assert backend.is_oom_error(backend._torch.cuda.OutOfMemoryError("oom"))
    assert backend.is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not backend.is_oom_error(RuntimeError("different runtime failure"))
    assert backend.peak_vram_bytes() == 123456
    backend.reset_peak_vram()
    assert backend._torch.cuda.reset_devices == ["cuda"]
    assert backend.runtime_environment() == {
        "torch": "fake-torch",
        "transformers": "fake-transformers",
        "cuda_runtime": "fake-cuda",
        "cuda_driver": None,
        "device": "cuda",
        "device_name": "Fake A100",
        "device_capability": [8, 0],
        "dtype": "bfloat16",
        "model_parameter_dtype": "bfloat16",
    }


def parity_requests(backend: llada.LLaDABackend) -> list[llada.WorkRequest]:
    return [
        backend.prepare_request(
            example_id="alpha",
            claim="Alpha betaword",
            evidence=None,
            arm="prior",
            choice=MaskChoice(mask_index=0, mask_rate=0.5, word_indices=(1,)),
        ),
        backend.prepare_request(
            example_id="gamma",
            claim="Gamma",
            evidence=None,
            arm="prior",
            choice=MaskChoice(mask_index=0, mask_rate=0.5, word_indices=(0,)),
        ),
    ]


def test_manual_parity_gate_returns_canonical_pass_report() -> None:
    backend, _, model = make_backend()
    requests = parity_requests(backend)
    report = llada.compare_scalar_and_batched(
        backend,
        requests,
        max_batch_tokens=16,
        max_batch_rows=8,
    )

    assert report["passed"] is True
    assert report["failure_count"] == 0
    assert report["forward_counts"] == {"scalar": 2, "batched": 1}
    assert report["batch_plan"] == {
        "batch_count": 1,
        "rows_per_batch": [2],
        "padded_tokens_per_batch": [8],
    }
    assert all(report["exact_checks"].values())
    assert all(value == 0.0 for value in report["numeric_checks"]["max_absolute_error"].values())
    assert json.loads(canonical_json(report)) == report
    assert len(model.calls) == 3


def test_manual_parity_gate_fails_closed_on_exact_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _, _ = make_backend()
    requests = parity_requests(backend)
    real_global = backend.infer_global

    def mismatched_global(*args: Any, **kwargs: Any) -> tuple[llada.AnalyticResult, ...]:
        results = list(real_global(*args, **kwargs))
        first = results[0]
        changed = (
            (first.top_alternative_ids[0] + 1) % backend.vocab_size,
            *first.top_alternative_ids[1:],
        )
        results[0] = replace(first, top_alternative_ids=changed)
        return tuple(results)

    monkeypatch.setattr(backend, "infer_global", mismatched_global)
    with pytest.raises(llada.ParityMismatchError, match="exact top_alternative_ids") as caught:
        llada.compare_scalar_and_batched(
            backend,
            requests,
            max_batch_tokens=16,
            max_batch_rows=8,
        )
    report = caught.value.report
    assert report["passed"] is False
    assert report["exact_checks"]["top_alternative_ids"] is False
    assert report["failure_count"] == 1
    canonical_json(report)


def test_manual_parity_gate_fails_closed_on_numeric_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _, _ = make_backend()
    requests = parity_requests(backend)
    real_global = backend.infer_global

    def mismatched_global(*args: Any, **kwargs: Any) -> tuple[llada.AnalyticResult, ...]:
        results = list(real_global(*args, **kwargs))
        first = results[0]
        results[0] = replace(first, ce=(first.ce[0] + 0.25, *first.ce[1:]))
        return tuple(results)

    monkeypatch.setattr(backend, "infer_global", mismatched_global)
    with pytest.raises(llada.ParityMismatchError, match=r"ce\[0\].*exceeds tolerance") as caught:
        llada.compare_scalar_and_batched(
            backend,
            requests,
            max_batch_tokens=16,
            max_batch_rows=8,
            rtol=1e-6,
            atol=1e-6,
        )
    report = caught.value.report
    assert report["passed"] is False
    assert report["exact_checks"] == {
        "output_cardinality": True,
        "work_ids": True,
        "masked_positions": True,
        "target_ids": True,
        "top_ids": True,
        "top_alternative_ids": True,
    }
    assert report["numeric_checks"]["max_absolute_error"]["ce"] == pytest.approx(0.25)
    assert report["failure_count"] == 1
    canonical_json(report)
