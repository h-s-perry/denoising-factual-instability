"""Canonical DFI v2 JSONL validation and the narrow scoring view."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


@dataclass(frozen=True)
class ScoringExample:
    """Internal scoring view; this is not a second public data schema."""

    example_id: str
    family_id: str
    split: str
    claim: str
    label: str
    domain: str
    fact_type: str
    prior_familiarity: str
    evidence_condition: str
    evidence_texts: tuple[str, ...]
    paired_example_id: str | None
    knowledge_status: str | None
    knowledge_passed: bool | None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ScoringExample:
        audit = record["knowledge_audit"]
        contrast = record.get("contrast")
        return cls(
            example_id=record["example_id"],
            family_id=record["family_id"],
            split=record["split"],
            claim=record["claim"],
            label=record["label"],
            domain=record["domain"],
            fact_type=record["fact_type"],
            prior_familiarity=record["prior_familiarity"],
            evidence_condition=record["evidence_condition"],
            evidence_texts=tuple(item["text"] for item in record["evidence_sets"]),
            paired_example_id=(
                contrast.get("paired_example_id") if isinstance(contrast, dict) else None
            ),
            knowledge_status=(audit.get("status") if isinstance(audit, dict) else None),
            knowledge_passed=(audit.get("passed") if isinstance(audit, dict) else None),
        )

    def evidence_for_arm(self, arm: str) -> str:
        if arm == "prior":
            return ""
        if arm == "supplied_evidence":
            if not self.evidence_texts:
                raise ValueError(f"{self.example_id}: supplied_evidence arm has no evidence")
            return "\n\n".join(self.evidence_texts)
        raise ValueError(f"unsupported evidence arm: {arm!r}")


def default_schema_path() -> Path:
    """Locate the unchanged canonical schema in a source checkout."""

    source_path = Path(__file__).resolve().parents[2] / "data" / "dfi-v2.schema.json"
    if source_path.is_file():
        return source_path
    raise FileNotFoundError("source-tree schema path is unavailable in this installation")


def _load_schema(path: Path | None) -> dict[str, Any]:
    try:
        if path is not None:
            raw = path.read_text(encoding="utf-8")
        else:
            try:
                raw = default_schema_path().read_text(encoding="utf-8")
            except FileNotFoundError:
                raw = (
                    resources.files("dfi")
                    .joinpath("dfi-v2.schema.json")
                    .read_text(encoding="utf-8")
                )
        value = json.loads(raw)
    except FileNotFoundError as exc:
        raise ValueError(f"schema file not found: {path}") from exc
    Draft202012Validator.check_schema(value)
    return value


def _span_is_valid(span: Any, text: str) -> bool:
    return (
        isinstance(span, dict)
        and isinstance(span.get("start"), int)
        and isinstance(span.get("end"), int)
        and 0 <= span["start"] < span["end"] <= len(text)
    )


def _semantic_row_errors(record: dict[str, Any], line_number: int) -> list[str]:
    prefix = f"line {line_number}"
    errors: list[str] = []
    claim = record["claim"]

    for index, atom in enumerate(record["atomic_facts"]):
        if not _span_is_valid(atom["claim_span"], claim):
            errors.append(f"{prefix}: atomic_facts[{index}].claim_span is invalid")

    if record["evidence_condition"] == "none" and record["evidence_sets"]:
        errors.append(f"{prefix}: evidence_condition=none must have no evidence")
    if record["evidence_condition"] != "none" and not record["evidence_sets"]:
        errors.append(f"{prefix}: non-none evidence condition requires evidence")
    evidence_ids: set[str] = set()
    for index, item in enumerate(record["evidence_sets"]):
        if item["evidence_id"] in evidence_ids:
            errors.append(f"{prefix}: duplicate evidence_id {item['evidence_id']!r}")
        evidence_ids.add(item["evidence_id"])
        for span_index, span in enumerate(item["rationale_spans"]):
            if not _span_is_valid(span, item["text"]):
                errors.append(
                    f"{prefix}: evidence_sets[{index}].rationale_spans[{span_index}] is invalid"
                )
        if record["evidence_condition"].startswith("gold_") and not item["rationale_spans"]:
            errors.append(f"{prefix}: gold evidence requires a rationale span")

    annotators = record["annotations"]["annotator_ids"]
    if len(set(annotators)) < 2:
        errors.append(f"{prefix}: at least two distinct annotators are required")
    if record["split"] == "test" and record["annotations"]["adjudication_status"] != "verified":
        errors.append(f"{prefix}: test examples must be verified")

    contrast = record.get("contrast")
    if contrast is not None and not _span_is_valid(contrast["replacement_span"], claim):
        errors.append(f"{prefix}: contrast.replacement_span is invalid")

    audit = record["knowledge_audit"]
    if isinstance(audit, dict):
        recognized = audit["status"] in {"recall_known", "recognition_known"}
        if audit["recognition_passed"] != recognized:
            errors.append(f"{prefix}: recognition_passed is inconsistent with status")
        if audit["passed"] != (audit["status"] == "recall_known"):
            errors.append(f"{prefix}: knowledge_audit.passed is inconsistent with status")
        if abs(audit["forward_top1_rate"] - audit["recognition_top1_rate"]) > 1e-12:
            errors.append(f"{prefix}: forward_top1_rate must equal recognition_top1_rate")
    return errors


def _cross_row_errors(records: Iterable[dict[str, Any]]) -> list[str]:
    materialized = list(records)
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    family_splits: dict[str, set[str]] = defaultdict(set)
    incoming_pairs: dict[str, list[str]] = defaultdict(list)

    for record in materialized:
        example_id = record["example_id"]
        if example_id in by_id:
            errors.append(f"duplicate example_id: {example_id}")
        by_id[example_id] = record
        family_splits[record["family_id"]].add(record["split"])

    for family_id, splits in sorted(family_splits.items()):
        if len(splits) > 1:
            errors.append(f"family {family_id!r} spans splits {sorted(splits)}")

    for record in materialized:
        contrast = record.get("contrast")
        if not isinstance(contrast, dict):
            continue
        paired_id = contrast["paired_example_id"]
        incoming_pairs[paired_id].append(record["example_id"])
        paired = by_id.get(paired_id)
        if paired is None:
            errors.append(f"{record['example_id']}: paired example {paired_id!r} is missing")
            continue
        if paired_id == record["example_id"]:
            errors.append(f"{record['example_id']}: an example cannot be paired with itself")
            continue
        if paired["family_id"] != record["family_id"]:
            errors.append(f"{record['example_id']}: paired example belongs to another family")
        if not _span_is_valid(contrast["original_span"], paired["claim"]):
            errors.append(
                f"{record['example_id']}: contrast.original_span is invalid for paired "
                f"example {paired_id!r}"
            )
        paired_contrast = paired.get("contrast")
        if (
            not isinstance(paired_contrast, dict)
            or paired_contrast.get("paired_example_id") != record["example_id"]
        ):
            errors.append(f"{record['example_id']}: pairing with {paired_id!r} is not reciprocal")

    for paired_id, source_ids in sorted(incoming_pairs.items()):
        if len(source_ids) > 1:
            errors.append(
                f"paired example {paired_id!r} is referenced by multiple examples "
                f"{sorted(source_ids)}"
            )
    return errors


def load_jsonl(
    path: str | Path,
    *,
    schema_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[ScoringExample]]:
    """Load schema-valid v2 records and enforce documented cross-row invariants."""

    selected = Path(path)
    if not selected.is_file():
        raise ValueError(f"dataset file not found: {selected}")
    schema = _load_schema(Path(schema_path) if schema_path is not None else None)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with selected.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            schema_errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
            for error in schema_errors:
                location = ".".join(str(part) for part in error.absolute_path) or "<row>"
                errors.append(f"line {line_number} {location}: {error.message}")
            if schema_errors or not isinstance(value, dict):
                continue
            errors.extend(_semantic_row_errors(value, line_number))
            records.append(value)

    errors.extend(_cross_row_errors(records))
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        remainder = len(errors) - min(len(errors), 20)
        suffix = f"\n- ... and {remainder} more" if remainder else ""
        raise ValueError(
            f"DFI v2 validation failed with {len(errors)} error(s):\n{preview}{suffix}"
        )
    if not records:
        raise ValueError("dataset contains no records")
    return records, [ScoringExample.from_record(record) for record in records]
