from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfi.config import load_config
from dfi.data import load_jsonl
from dfi.llada import LLADA_MASK_TOKEN, LLADA_MASK_TOKEN_ID, LLaDABackend, LLaDAModelSpec
from dfi.masking import deterministic_masks
from dfi.pipeline import build_inference_plan

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "llada-tokenizer-mask0.json"


def test_pinned_real_tokenizer_reproduces_frozen_synthetic_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    revision = fixture["tokenizer_revision"]
    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--GSAI-ML--LLaDA-8B-Base"
        / "snapshots"
        / revision
    )
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip("the exact pinned LLaDA tokenizer is not present in the local HF cache")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        fixture["model_repository"],
        revision=revision,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    assert tokenizer.convert_tokens_to_ids(LLADA_MASK_TOKEN) == LLADA_MASK_TOKEN_ID
    assert tokenizer.convert_ids_to_tokens(LLADA_MASK_TOKEN_ID) == LLADA_MASK_TOKEN

    backend = LLaDABackend(
        spec=LLaDAModelSpec(
            repository=fixture["model_repository"],
            revision=revision,
            tokenizer_revision=revision,
            remote_code_revision=revision,
        ),
        tokenizer=tokenizer,
        model=None,
        torch_module=None,
        device="cuda",
        pad_token_id=int(tokenizer.pad_token_id),
        vocab_size=126464,
    )
    assert list(backend.encode_prefix(None)) == fixture["prefix_ids"]
    _, examples = load_jsonl(ROOT / fixture["source_dataset"])
    by_example = {example.example_id: example for example in examples}

    for expected in fixture["requests"]:
        example = by_example[expected["example_id"]]
        encoding = backend.encode_claim(example.claim)
        assert list(encoding.token_ids) == expected["claim_ids"]
        assert [list(span) for span in encoding.offsets] == expected["offsets"]
        assert [list(pieces) for pieces in encoding.word_pieces] == expected["word_pieces"]
        choice = deterministic_masks(
            example.claim,
            example_id=example.example_id,
            arm=fixture["arm"],
            seed=fixture["seed"],
            n_masks=1,
        )[0]
        assert choice.mask_index == fixture["mask_index"]
        assert choice.mask_rate == expected["mask_rate"]
        assert list(choice.word_indices) == expected["word_indices"]
        request = backend.prepare_request(
            example_id=example.example_id,
            claim=example.claim,
            evidence=None,
            arm=fixture["arm"],
            choice=choice,
            mask_policy="fixed-v1",
        )
        assert list(request.masked_positions) == expected["masked_positions"]
        assert list(request.target_ids) == expected["target_ids"]
        assert list(request.input_ids) == expected["masked_input_ids"]
        assert request.work_id == expected["work_id"]

    throughput = load_config(ROOT / "configs" / "a100-throughput.toml")
    throughput_plan = build_inference_plan(throughput, examples, backend)
    assert len(throughput_plan.logical_requests) == 4096
    assert len(throughput_plan.physical_requests) == 4096
    assert (
        throughput_plan.plan_hash
        == "71e4f64392cf73721838c39eaf8f0987608056e67ca292963165d65ab9e0bc40"
    )
