"""Pure analytic DFI formulas over masked-token marginals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class MarginalStats:
    """Positionwise sufficient statistics for one masked request."""

    target_ids: IntArray
    top_ids: IntArray
    top_alternative_ids: IntArray
    ce: FloatArray
    entropy: FloatArray
    collision_probability: FloatArray
    collision_entropy: FloatArray
    delta: FloatArray
    swap_llr: FloatArray
    expected_drift: FloatArray
    expected_dispersion: FloatArray
    top_mismatch: FloatArray


def probabilities_from_logits(logits: npt.ArrayLike) -> FloatArray:
    """Return stable float64 probabilities for `[positions, vocabulary]` logits."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or values.shape[1] < 2:
        raise ValueError("logits must have shape [positions, vocabulary>=2]")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    shifted = values - values.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _validated_probabilities(probabilities: npt.ArrayLike) -> FloatArray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [positions, vocabulary>=2]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("each marginal distribution must sum to one")
    return values


def score_marginals(
    probabilities: npt.ArrayLike,
    target_ids: npt.ArrayLike,
) -> MarginalStats:
    """Compute the exact analytic `T=1` statistics used by DFI.

    `swap_llr` excludes the submitted token when selecting the strongest
    alternative. All logarithms are natural logarithms.
    """

    probs = _validated_probabilities(probabilities)
    targets = np.asarray(target_ids, dtype=np.int64)
    if targets.ndim != 1 or len(targets) != len(probs):
        raise ValueError("target_ids must have one entry per masked position")
    if np.any(targets < 0) or np.any(targets >= probs.shape[1]):
        raise ValueError("target_ids contain an out-of-vocabulary index")

    rows = np.arange(len(probs))
    tiny = np.finfo(np.float64).tiny
    log_probs = np.log(np.clip(probs, tiny, None))
    target_log_probs = log_probs[rows, targets]
    ce = -target_log_probs
    entropy = -(np.where(probs > 0.0, probs * log_probs, 0.0)).sum(axis=1)
    collision_probability = np.square(probs).sum(axis=1)
    collision_entropy = -np.log(np.clip(collision_probability, tiny, None))

    alternative_log_probs = log_probs.copy()
    alternative_log_probs[rows, targets] = -np.inf
    top_alternative_ids = alternative_log_probs.argmax(axis=1).astype(np.int64)
    top_alternative_log_probs = alternative_log_probs[rows, top_alternative_ids]
    top_ids = probs.argmax(axis=1).astype(np.int64)

    return MarginalStats(
        target_ids=targets,
        top_ids=top_ids,
        top_alternative_ids=top_alternative_ids,
        ce=ce,
        entropy=entropy,
        collision_probability=collision_probability,
        collision_entropy=collision_entropy,
        delta=ce - entropy,
        swap_llr=top_alternative_log_probs - target_log_probs,
        expected_drift=1.0 - probs[rows, targets],
        expected_dispersion=1.0 - collision_probability,
        top_mismatch=(top_ids != targets).astype(np.float64),
    )


def score_logits(logits: npt.ArrayLike, target_ids: npt.ArrayLike) -> MarginalStats:
    """Convenience wrapper for full selected-position logits."""

    return score_marginals(probabilities_from_logits(logits), target_ids)


def reduce_positions(
    stats: MarginalStats,
) -> dict[str, float | int]:
    """Reduce positionwise statistics into one compact per-mask row."""

    return {
        "n_masked_pieces": len(stats.ce),
        "ce_mean": float(np.mean(stats.ce)),
        "entropy_mean": float(np.mean(stats.entropy)),
        "collision_entropy_mean": float(np.mean(stats.collision_entropy)),
        "delta_mean": float(np.mean(stats.delta)),
        "swap_llr_mean": float(np.mean(stats.swap_llr)),
        "expected_drift_mean": float(np.mean(stats.expected_drift)),
        "expected_dispersion_mean": float(np.mean(stats.expected_dispersion)),
        "top_mismatch_rate": float(np.mean(stats.top_mismatch)),
    }


def stats_as_lists(stats: MarginalStats) -> dict[str, list[int] | list[float]]:
    """Return the positionwise sufficient statistics in Parquet-friendly form."""

    return {
        "target_ids": stats.target_ids.tolist(),
        "top_ids": stats.top_ids.tolist(),
        "top_alternative_ids": stats.top_alternative_ids.tolist(),
        "ce_by_piece": stats.ce.tolist(),
        "entropy_by_piece": stats.entropy.tolist(),
        "collision_probability_by_piece": stats.collision_probability.tolist(),
        "collision_entropy_by_piece": stats.collision_entropy.tolist(),
        "delta_by_piece": stats.delta.tolist(),
        "swap_llr_by_piece": stats.swap_llr.tolist(),
        "expected_drift_by_piece": stats.expected_drift.tolist(),
        "expected_dispersion_by_piece": stats.expected_dispersion.tolist(),
        "top_mismatch_by_piece": stats.top_mismatch.tolist(),
    }
