"""Deterministic word-to-piece masking and stable work identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)?|[^\w\s]")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "was",
        "were",
        "is",
        "are",
        "in",
        "on",
        "of",
        "for",
        "to",
        "at",
        "by",
        "with",
        "from",
        "as",
        "that",
        "this",
        "he",
        "she",
        "it",
        "they",
    }
)


@dataclass(frozen=True)
class WordSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class MaskChoice:
    mask_index: int
    mask_rate: float
    word_indices: tuple[int, ...]


@dataclass(frozen=True)
class MaskGeometry:
    piece_indices: tuple[int, ...]
    target_ids: tuple[int, ...]
    masked_input_ids: tuple[int, ...]


def word_token_spans(text: str) -> tuple[WordSpan, ...]:
    """Return the notebook-compatible word/punctuation character spans."""

    return tuple(
        WordSpan(match.group(), match.start(), match.end()) for match in WORD_RE.finditer(text)
    )


def normalize_word(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def align_word_pieces(
    text: str,
    offsets: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Map every word span to all overlapping tokenizer pieces.

    Offsets with `(0, 0)` are treated as special tokens. Every alphanumeric
    word must map to at least one piece; silently dropping a multi-piece word
    would change the scientific protocol.
    """

    normalized_offsets: list[tuple[int, int]] = []
    for index, offset in enumerate(offsets):
        if len(offset) != 2:
            raise ValueError(f"offset {index} must contain exactly two integers")
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start or end > len(text):
            raise ValueError(f"offset {index} is outside the claim")
        normalized_offsets.append((start, end))

    spans = word_token_spans(text)
    aligned: list[tuple[int, ...]] = []
    for word in spans:
        pieces = tuple(
            index
            for index, (start, end) in enumerate(normalized_offsets)
            if end > start and end > word.start and start < word.end
        )
        if normalize_word(word.text) and not pieces:
            raise ValueError(f"word {word.text!r} at [{word.start}, {word.end}) has no token piece")
        aligned.append(pieces)

    piece_owners: dict[int, int] = {}
    for word_index, pieces in enumerate(aligned):
        if not normalize_word(spans[word_index].text):
            continue
        for piece in pieces:
            previous = piece_owners.setdefault(piece, word_index)
            if previous != word_index:
                raise ValueError(
                    f"token piece {piece} ambiguously overlaps maskable words "
                    f"{previous} and {word_index}"
                )
    return tuple(aligned)


def _digest(*parts: Any) -> bytes:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def deterministic_masks(
    text: str,
    *,
    example_id: str,
    arm: str,
    seed: int,
    n_masks: int,
) -> tuple[MaskChoice, ...]:
    """Generate fixed-v1 masks without process or batching-order randomness.

    The policy mirrors the canonical notebook's content-word uniform band:
    each mask rate lies in `[0.25, 0.60]`, and content words are selected by a
    SHA-256 rank. Hash-derived ranking is stable across input reordering and
    does not rely on Python's randomized `hash()`.
    """

    if n_masks < 1:
        raise ValueError("n_masks must be positive")
    spans = word_token_spans(text)
    maskable = [index for index, span in enumerate(spans) if normalize_word(span.text)]
    content = [index for index in maskable if normalize_word(spans[index].text) not in STOPWORDS]
    eligible = content or maskable
    if not eligible:
        raise ValueError("claim contains no maskable words")

    choices: list[MaskChoice] = []
    for mask_index in range(n_masks):
        base = _digest("fixed-v1", seed, example_id, arm, mask_index)
        unit = int.from_bytes(base[:8], "big") / float(2**64 - 1)
        mask_rate = 0.25 + 0.35 * unit
        count = max(1, min(len(eligible), round(mask_rate * len(eligible))))
        ranked = sorted(
            eligible,
            key=lambda word_index: (_digest(base.hex(), word_index), word_index),
        )
        choices.append(
            MaskChoice(
                mask_index=mask_index,
                mask_rate=mask_rate,
                word_indices=tuple(sorted(ranked[:count])),
            )
        )
    return tuple(choices)


def apply_mask_geometry(
    token_ids: Sequence[int],
    word_pieces: Sequence[Sequence[int]],
    choice: MaskChoice,
    *,
    mask_token_id: int,
) -> MaskGeometry:
    """Mask every piece of every selected word and retain the exact targets."""

    piece_owners: dict[int, int] = {}
    for word_index in choice.word_indices:
        if word_index < 0 or word_index >= len(word_pieces):
            raise ValueError(f"mask word index {word_index} is out of range")
        selected = tuple(int(piece) for piece in word_pieces[word_index])
        if not selected:
            raise ValueError(f"mask word index {word_index} has no token pieces")
        if len(set(selected)) != len(selected):
            raise ValueError(f"mask word index {word_index} repeats a token piece")
        for piece in selected:
            if piece < 0 or piece >= len(token_ids):
                raise ValueError("piece index is outside token_ids")
            previous = piece_owners.setdefault(piece, word_index)
            if previous != word_index:
                raise ValueError(
                    f"token piece {piece} ambiguously belongs to selected words "
                    f"{previous} and {word_index}"
                )
    piece_indices = tuple(sorted(piece_owners))
    if not piece_indices:
        raise ValueError("mask produced no token pieces")

    masked = [int(token_id) for token_id in token_ids]
    targets = tuple(masked[index] for index in piece_indices)
    for index in piece_indices:
        masked[index] = int(mask_token_id)
    return MaskGeometry(piece_indices, targets, tuple(masked))


def canonical_json(value: Any) -> bytes:
    """Canonical JSON encoding used for all content identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_work_id(
    *,
    scoring_protocol: str,
    model_repository: str,
    model_revision: str,
    tokenizer_revision: str,
    remote_code_revision: str,
    arm: str,
    prompt_protocol: str,
    mask_policy: str,
    mask_rate: float,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    masked_positions: Sequence[int],
    target_ids: Sequence[int],
    temperature: float,
    dtype: str,
    inference_backend: str,
) -> str:
    """Hash every score-affecting field into a stable semantic work ID."""

    input_values = [int(value) for value in input_ids]
    attention_values = [int(value) for value in attention_mask]
    position_values = [int(value) for value in masked_positions]
    target_values = [int(value) for value in target_ids]
    required_strings = {
        "scoring_protocol": scoring_protocol,
        "model_repository": model_repository,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "remote_code_revision": remote_code_revision,
        "arm": arm,
        "prompt_protocol": prompt_protocol,
        "mask_policy": mask_policy,
        "dtype": dtype,
        "inference_backend": inference_backend,
    }
    if any(not value for value in required_strings.values()):
        raise ValueError("work identity string fields must be non-empty")
    if not input_values or len(input_values) != len(attention_values):
        raise ValueError("input_ids and attention_mask must have equal non-zero length")
    if any(value not in {0, 1} for value in attention_values):
        raise ValueError("attention_mask values must be zero or one")
    if not position_values or len(position_values) != len(target_values):
        raise ValueError("masked_positions and target_ids must have equal non-zero length")
    if len(set(position_values)) != len(position_values):
        raise ValueError("masked_positions must be unique")
    if any(position < 0 or position >= len(input_values) for position in position_values):
        raise ValueError("masked_positions contain an out-of-range index")
    if not 0.0 < float(mask_rate) <= 1.0:
        raise ValueError("mask_rate must be in (0, 1]")
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if scoring_protocol == "analytic-v1" and temperature_value != 1.0:
        raise ValueError("analytic-v1 requires temperature exactly 1.0")

    payload = {
        "arm": arm,
        "attention_mask": attention_values,
        "dtype": dtype,
        "inference_backend": inference_backend,
        "input_ids": input_values,
        "mask_policy": mask_policy,
        "mask_rate": float(mask_rate),
        "masked_positions": position_values,
        "model_repository": model_repository,
        "model_revision": model_revision,
        "prompt_protocol": prompt_protocol,
        "remote_code_revision": remote_code_revision,
        "scoring_protocol": scoring_protocol,
        "target_ids": target_values,
        "temperature": temperature_value,
        "tokenizer_revision": tokenizer_revision,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()
