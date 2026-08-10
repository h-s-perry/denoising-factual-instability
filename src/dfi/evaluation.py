"""Claim-, contrast-, and family-aware reductions for sealed DFI rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

RISK_COLUMNS = (
    "ce_mean",
    "delta_mean",
    "swap_llr_mean",
    "expected_drift_mean",
    "expected_dispersion_mean",
    "top_mismatch_rate",
)
SUMMARY_COLUMNS = (
    "ce_mean",
    "entropy_mean",
    "collision_entropy_mean",
    "delta_mean",
    "swap_llr_mean",
    "expected_drift_mean",
    "expected_dispersion_mean",
    "top_mismatch_rate",
)


def reduce_claims(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average per-mask statistics into one row per `(example_id, arm)`."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["example_id"]), str(row["arm"]))].append(row)
    if not groups:
        raise ValueError("cannot reduce an empty result table")

    reduced: list[dict[str, Any]] = []
    metadata = ("family_id", "split", "label", "paired_example_id")
    for (example_id, arm), members in sorted(groups.items()):
        for field in metadata:
            values = {member.get(field) for member in members}
            if len(values) != 1:
                raise ValueError(f"{example_id}/{arm}: inconsistent {field} across masks")
        mask_indices = [int(member["mask_index"]) for member in members]
        if len(set(mask_indices)) != len(mask_indices):
            raise ValueError(f"{example_id}/{arm}: duplicate mask_index")
        row: dict[str, Any] = {
            "example_id": example_id,
            "arm": arm,
            "family_id": members[0]["family_id"],
            "split": members[0]["split"],
            "label": members[0]["label"],
            "paired_example_id": members[0].get("paired_example_id"),
            "n_masks": len(members),
        }
        for column in SUMMARY_COLUMNS:
            values = np.asarray([member[column] for member in members], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{example_id}/{arm}: non-finite {column}")
            row[column] = float(values.mean())
        reduced.append(row)
    return reduced


def binary_auroc(scores: Iterable[float], labels: Iterable[str]) -> float | None:
    """Compute tie-aware AUROC over supported/refuted claims only."""

    positives: list[float] = []
    negatives: list[float] = []
    for score, label in zip(scores, labels, strict=True):
        if label == "refuted":
            positives.append(float(score))
        elif label == "supported":
            negatives.append(float(score))
    if not positives or not negatives:
        return None
    positive = np.asarray(positives, dtype=np.float64)
    negative = np.asarray(negatives, dtype=np.float64)
    differences = positive[:, None] - negative[None, :]
    return float((np.sum(differences > 0.0) + 0.5 * np.sum(differences == 0.0)) / differences.size)


def _contrast_metrics(claims: list[dict[str, Any]], score_column: str) -> dict[str, Any]:
    by_id = {row["example_id"]: row for row in claims}
    if len(by_id) != len(claims):
        raise ValueError("duplicate claim reductions")
    consumed: set[frozenset[str]] = set()
    gaps: list[float] = []

    for row in claims:
        paired_id = row.get("paired_example_id")
        if not paired_id:
            continue
        paired = by_id.get(paired_id)
        if paired is None:
            raise ValueError(f"{row['example_id']}: paired reduction is missing")
        key = frozenset({row["example_id"], paired_id})
        if len(key) != 2 or key in consumed:
            continue
        consumed.add(key)
        labels = {row["label"], paired["label"]}
        if labels != {"supported", "refuted"}:
            continue
        refuted = row if row["label"] == "refuted" else paired
        supported = paired if row["label"] == "refuted" else row
        gaps.append(float(refuted[score_column]) - float(supported[score_column]))

    if not gaps:
        return {"n_declared_contrasts": 0, "wins": 0, "win_rate": None, "mean_gap": None}
    values = np.asarray(gaps, dtype=np.float64)
    wins = int(np.sum(values > 0.0))
    return {
        "n_declared_contrasts": len(values),
        "wins": wins,
        "win_rate": float(wins / len(values)),
        "mean_gap": float(values.mean()),
    }


def _family_metrics(claims: list[dict[str, Any]], score_column: str) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in claims:
        families[row["family_id"]].append(row)
    gaps: list[float] = []
    for _family_id, members in sorted(families.items()):
        supported = [float(row[score_column]) for row in members if row["label"] == "supported"]
        refuted = [float(row[score_column]) for row in members if row["label"] == "refuted"]
        if not supported or not refuted:
            continue
        gaps.append(float(np.mean(refuted) - np.mean(supported)))
    if not gaps:
        return {"n_binary_families": 0, "wins": 0, "win_rate": None, "mean_gap": None}
    values = np.asarray(gaps, dtype=np.float64)
    wins = int(np.sum(values > 0.0))
    return {
        "n_binary_families": len(values),
        "wins": wins,
        "win_rate": float(wins / len(values)),
        "mean_gap": float(values.mean()),
    }


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    interpretation_allowed: bool = False,
    interpretation_reason: str = "data and protocol admission gates have not passed",
) -> dict[str, Any]:
    """Compute aggregate metrics without treating masks as independent examples."""

    materialized = list(rows)
    claims = reduce_claims(materialized)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_arm[claim["arm"]].append(claim)

    arm_metrics: dict[str, Any] = {}
    for arm, arm_claims in sorted(by_arm.items()):
        score_metrics: dict[str, Any] = {}
        for column in RISK_COLUMNS:
            score_metrics[column] = {
                "claim_auroc": binary_auroc(
                    (row[column] for row in arm_claims),
                    (row["label"] for row in arm_claims),
                ),
                "declared_contrasts": _contrast_metrics(arm_claims, column),
                "families": _family_metrics(arm_claims, column),
            }
        arm_metrics[arm] = {
            "n_claims": len(arm_claims),
            "n_families": len({row["family_id"] for row in arm_claims}),
            "label_counts": {
                label: sum(row["label"] == label for row in arm_claims)
                for label in ("supported", "refuted", "insufficient", "ambiguous")
            },
            "scores": score_metrics,
        }

    return {
        "schema_version": "evaluation-v1",
        "n_mask_rows": len(materialized),
        "n_claim_rows": len(claims),
        "arms": arm_metrics,
        "interpretation_allowed": bool(interpretation_allowed),
        "interpretation_reason": interpretation_reason,
        "warning": "DFI measures model/evidence compatibility, not truth.",
    }
