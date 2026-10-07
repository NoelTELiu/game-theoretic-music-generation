from __future__ import annotations

from typing import Sequence

import torch


HOLD_TOKEN_RAW = -2

DURATION_FEATURE_NAMES = [
    "upper_current_hold_length",
    "lower_current_hold_length",
    "upper_steps_since_last_onset",
    "lower_steps_since_last_onset",
    "upper_prev_is_hold",
    "lower_prev_is_hold",
]

DURATION_FEATURE_DIM = len(DURATION_FEATURE_NAMES)


def build_duration_feature_tensor_from_context(
    context: torch.Tensor,
    hold_token_id: int,
    normalizer: float | int | None = None,
) -> torch.Tensor:
    """Compute duration-aware state features from a two-voice context window.

    Parameters
    ----------
    context:
        Token-id context tensor with shape ``[T, 2]`` or ``[B, T, 2]``.
        Voice 0 is upper/soprano and voice 1 is lower/bass.
    hold_token_id:
        Integer token id corresponding to the raw HOLD token ``-2`` in the
        checkpoint vocabulary.
    normalizer:
        Value used to scale count features. If omitted, the context length is
        used. Counts are therefore capped to the visible context window.

    Returns
    -------
    torch.Tensor
        If input was ``[T, 2]``, returns ``[6]``. If input was ``[B, T, 2]``,
        returns ``[B, 6]``. Feature order is defined by
        ``DURATION_FEATURE_NAMES``.

    Notes
    -----
    This intentionally uses only the current context window, not hidden full
    piece history. That keeps training and generation aligned because the model
    state is still a fixed-length context representation.
    """
    if not torch.is_tensor(context):
        context = torch.as_tensor(context, dtype=torch.long)

    if context.ndim == 2:
        single = True
        context_b = context.unsqueeze(0)
    elif context.ndim == 3:
        single = False
        context_b = context
    else:
        raise ValueError(f"Expected context shape [T, 2] or [B, T, 2], got {tuple(context.shape)}")

    if context_b.shape[-1] != 2:
        raise ValueError(f"Expected two voices in the last dimension, got shape {tuple(context_b.shape)}")

    batch_size, context_len, _ = context_b.shape
    if context_len <= 0:
        raise ValueError("Duration features require a non-empty context window.")

    device = context_b.device
    dtype = torch.float32
    denom = float(normalizer if normalizer is not None else context_len)
    denom = max(denom, 1.0)

    hold_mask = context_b.eq(int(hold_token_id))
    reversed_hold = torch.flip(hold_mask, dims=[1])

    # Consecutive HOLD tokens immediately before the prediction time.
    # cumprod stays 1 while the reversed prefix is all HOLD, then becomes 0.
    hold_prefix = torch.cumprod(reversed_hold.to(torch.long), dim=1)
    current_hold_length = hold_prefix.sum(dim=1).to(dtype)

    # Distance, in context steps, to the most recent non-HOLD token.
    reversed_non_hold = ~reversed_hold
    step_numbers = torch.arange(1, context_len + 1, device=device, dtype=torch.long).view(1, context_len, 1)
    large = torch.full((batch_size, context_len, 2), context_len + 1, device=device, dtype=torch.long)
    candidate_steps = torch.where(reversed_non_hold, step_numbers.expand(batch_size, -1, 2), large)
    steps_since_last_onset = candidate_steps.min(dim=1).values
    steps_since_last_onset = torch.where(
        steps_since_last_onset.eq(context_len + 1),
        torch.full_like(steps_since_last_onset, context_len),
        steps_since_last_onset,
    ).to(dtype)

    prev_is_hold = hold_mask[:, -1, :].to(dtype)

    features = torch.stack(
        [
            current_hold_length[:, 0] / denom,
            current_hold_length[:, 1] / denom,
            steps_since_last_onset[:, 0] / denom,
            steps_since_last_onset[:, 1] / denom,
            prev_is_hold[:, 0],
            prev_is_hold[:, 1],
        ],
        dim=-1,
    )

    return features[0] if single else features


def duration_feature_metadata(normalizer: float | int | None = None) -> dict:
    """Return a JSON/checkpoint-friendly metadata dictionary."""
    return {
        "duration_feature_dim": DURATION_FEATURE_DIM,
        "duration_feature_names": list(DURATION_FEATURE_NAMES),
        "duration_feature_normalizer": None if normalizer is None else float(normalizer),
        "duration_feature_source": "fixed_context_window",
        "hold_token_raw": HOLD_TOKEN_RAW,
    }
