from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dfi.cli import offline_check
from dfi.data import load_jsonl
from dfi.masking import (
    MaskChoice,
    align_word_pieces,
    apply_mask_geometry,
    deterministic_masks,
    stable_work_id,
)
from dfi.scoring import reduce_positions, score_marginals

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "anchors.jsonl"


def _fixture_records() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _replace_claim(record: dict[str, Any], claim: str) -> None:
    record["claim"] = claim
    record["atomic_facts"][0]["text"] = claim
    record["atomic_facts"][0]["claim_span"] = {"start": 0, "end": len(claim)}


def _length_changing_pair() -> list[dict[str, Any]]:
    supported, refuted = copy.deepcopy(_fixture_records()[:2])
    supported_claim = "Alder maps to extraordinarilylongword."
    refuted_claim = "Alder maps to X."
    _replace_claim(supported, supported_claim)
    _replace_claim(refuted, refuted_claim)

    supported_start = supported_claim.index("extraordinarilylongword")
    supported_end = supported_start + len("extraordinarilylongword")
    refuted_start = refuted_claim.index("X")
    refuted_end = refuted_start + 1
    supported["contrast"]["original_span"] = {"start": refuted_start, "end": refuted_end}
    supported["contrast"]["replacement_span"] = {
        "start": supported_start,
        "end": supported_end,
    }
    refuted["contrast"]["original_span"] = {
        "start": supported_start,
        "end": supported_end,
    }
    refuted["contrast"]["replacement_span"] = {"start": refuted_start, "end": refuted_end}
    return [supported, refuted]


def _work_identity(*, temperature: float, scoring_protocol: str = "analytic-v1") -> str:
    return stable_work_id(
        scoring_protocol=scoring_protocol,
        model_repository="synthetic/model",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        remote_code_revision="remote-code-revision",
        arm="prior",
        prompt_protocol="claim-prefix-v1",
        mask_policy="fixed-v1",
        mask_rate=0.5,
        input_ids=[99, 7],
        attention_mask=[1, 1],
        masked_positions=[0],
        target_ids=[3],
        temperature=temperature,
        dtype="float64",
        inference_backend="saved-marginals-v1",
    )


def test_exact_marginal_formulas() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float64)
    stats = score_marginals(probabilities, [0, 2])

    assert stats.ce.tolist() == pytest.approx([-math.log(0.7), -math.log(0.3)])
    assert stats.entropy[0] == pytest.approx(-sum(p * math.log(p) for p in probabilities[0]))
    assert stats.collision_probability.tolist() == pytest.approx([0.54, 0.46])
    assert stats.delta.tolist() == pytest.approx(stats.ce - stats.entropy)
    assert stats.swap_llr.tolist() == pytest.approx([math.log(0.2 / 0.7), math.log(0.6 / 0.3)])
    assert stats.top_alternative_ids.tolist() == [1, 1]
    assert stats.expected_drift.tolist() == pytest.approx([0.3, 0.7])
    assert stats.expected_dispersion.tolist() == pytest.approx([0.46, 0.54])
    assert stats.top_mismatch.tolist() == [0.0, 1.0]

    reduced = reduce_positions(stats)
    assert reduced["n_masked_pieces"] == 2


def test_multi_piece_alignment_masks_every_piece() -> None:
    text = "Alder maps to 17."
    offsets = [(0, 5), (6, 10), (11, 13), (14, 15), (15, 16), (16, 17)]
    pieces = align_word_pieces(text, offsets)
    geometry = apply_mask_geometry(
        [1, 2, 3, 4, 5, 6],
        pieces,
        MaskChoice(mask_index=0, mask_rate=0.5, word_indices=(3,)),
        mask_token_id=99,
    )
    assert geometry.piece_indices == (3, 4)
    assert geometry.target_ids == (4, 5)
    assert geometry.masked_input_ids == (1, 2, 3, 99, 99, 6)


def test_fixed_masks_do_not_depend_on_input_order() -> None:
    claims = {
        "a": "One synthetic claim has several content words.",
        "b": "Another synthetic claim also has content words.",
    }
    forward = {
        key: deterministic_masks(value, example_id=key, arm="prior", seed=7, n_masks=8)
        for key, value in claims.items()
    }
    reverse = {
        key: deterministic_masks(claims[key], example_id=key, arm="prior", seed=7, n_masks=8)
        for key in reversed(claims)
    }
    assert forward == reverse


def test_fixed_mask_policy_matches_frozen_sequence() -> None:
    choices = deterministic_masks(
        "One synthetic claim has several content words.",
        example_id="mask-policy-probe",
        arm="prior",
        seed=7,
        n_masks=4,
    )
    assert [(choice.mask_rate, choice.word_indices) for choice in choices] == [
        (0.3572114630336358, (1, 3, 5)),
        (0.4346862665701139, (2, 3, 5)),
        (0.456596030196209, (3, 4, 5)),
        (0.5006438363360246, (0, 3, 4, 5)),
    ]


def test_canonical_schema_and_synthetic_fixture() -> None:
    schema_bytes = (ROOT / "data" / "dfi-v2.schema.json").read_bytes()
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "a454af060830c338e3f96b6ff2a6b0643b5e4f6cd472e9701254242cdfac8b18"
    )
    records, examples = load_jsonl(ROOT / "data" / "fixtures" / "anchors.jsonl")
    assert len(records) == 8
    assert len({example.family_id for example in examples}) == 4
    assert all(example.knowledge_status is None for example in examples)


def test_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    source = ROOT / "data" / "fixtures" / "anchors.jsonl"
    first = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    first["unknown_field"] = True
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Additional properties"):
        load_jsonl(invalid)


def test_fixture_contrast_spans_cross_reference_the_paired_claim() -> None:
    records, _ = load_jsonl(FIXTURE)
    by_id = {record["example_id"]: record for record in records}

    for record in records:
        contrast = record["contrast"]
        paired = by_id[contrast["paired_example_id"]]
        paired_contrast = paired["contrast"]
        assert contrast["original_span"] == paired_contrast["replacement_span"]
        assert contrast["replacement_span"] == paired_contrast["original_span"]


def test_original_span_is_validated_against_paired_claim(tmp_path: Path) -> None:
    records = _length_changing_pair()
    load_jsonl(_write_jsonl(tmp_path / "valid.jsonl", records))

    invalid = copy.deepcopy(records)
    invalid[0]["contrast"]["original_span"] = invalid[0]["contrast"]["replacement_span"]
    with pytest.raises(ValueError, match="original_span is invalid for paired example"):
        load_jsonl(_write_jsonl(tmp_path / "invalid.jsonl", invalid))


def test_pairing_must_be_reciprocal_and_one_to_one(tmp_path: Path) -> None:
    one_sided = copy.deepcopy(_fixture_records()[:2])
    one_sided[1]["contrast"] = None
    with pytest.raises(ValueError, match=r"pairing .* is not reciprocal"):
        load_jsonl(_write_jsonl(tmp_path / "one-sided.jsonl", one_sided))

    one_to_many = copy.deepcopy(_fixture_records()[:2])
    extra = copy.deepcopy(one_to_many[0])
    extra["example_id"] = "alder-supported-copy"
    extra["atomic_facts"][0]["atomic_id"] = "alder-supported-copy-atom"
    one_to_many.append(extra)
    with pytest.raises(ValueError, match="referenced by multiple examples"):
        load_jsonl(_write_jsonl(tmp_path / "one-to-many.jsonl", one_to_many))


def test_mask_geometry_rejects_invalid_or_ambiguous_pieces() -> None:
    with pytest.raises(ValueError, match="piece index is outside token_ids"):
        apply_mask_geometry(
            [10, 20],
            [(-1,)],
            MaskChoice(mask_index=0, mask_rate=0.5, word_indices=(0,)),
            mask_token_id=99,
        )
    with pytest.raises(ValueError, match="ambiguously belongs to selected words"):
        apply_mask_geometry(
            [10],
            [(0,), (0,)],
            MaskChoice(mask_index=0, mask_rate=0.5, word_indices=(0, 1)),
            mask_token_id=99,
        )
    with pytest.raises(ValueError, match="ambiguously overlaps maskable words"):
        align_word_pieces("New York", [(0, 8)])
    assert align_word_pieces("word.", [(0, 5)]) == ((0,), (0,))


def test_analytic_v1_requires_exact_unit_temperature() -> None:
    assert len(_work_identity(temperature=1.0)) == 64
    with pytest.raises(ValueError, match=r"analytic-v1 requires temperature exactly 1\.0"):
        _work_identity(temperature=0.999)
    assert len(_work_identity(temperature=0.5, scoring_protocol="future-protocol")) == 64


def test_complete_offline_check() -> None:
    summary = offline_check()
    assert summary == {
        "records": 8,
        "families": 4,
        "mask_rows": 8,
        "work_ids": 8,
        "fixture_kind": "synthetic-correctness-smoke",
        "interpretation_allowed": False,
    }
