from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class NashStepResult:
    """Result of one pure-Nash decision step."""

    pair: Tuple[int, int]
    metrics: Dict[str, float]


def _normalize_actions(actions: Sequence[int] | np.ndarray, vocab_size: int, name: str) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D action array, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError(f"{name} must contain at least one action")
    if np.any(arr < 0) or np.any(arr >= vocab_size):
        bad = arr[(arr < 0) | (arr >= vocab_size)][:10]
        raise ValueError(f"{name} contains action ids outside [0, {vocab_size}): {bad}")
    return np.array(sorted({int(x) for x in arr}), dtype=np.int64)


def _build_utility_matrices(
    soprano_logits: np.ndarray,
    bass_logits: np.ndarray,
    pair_logits: np.ndarray,
    pair_to_id: Dict[Tuple[int, int], int],
    s_actions: np.ndarray,
    b_actions: np.ndarray,
    alpha: float,
    beta_private: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build compatibility and player utilities over the shared A_U x A_L grid."""
    m = int(len(s_actions))
    n = int(len(b_actions))

    compat = np.zeros((m, n), dtype=np.float64)
    for i, s_id in enumerate(s_actions):
        for j, b_id in enumerate(b_actions):
            compat[i, j] = float(pair_logits[pair_to_id[(int(s_id), int(b_id))]])

    upper_private = np.asarray(soprano_logits[s_actions], dtype=np.float64)
    lower_private = np.asarray(bass_logits[b_actions], dtype=np.float64)

    upper_util = compat + float(alpha) * upper_private[:, None]
    lower_util = compat + float(beta_private) * lower_private[None, :]
    return compat, upper_util, lower_util


def find_pure_nash_equilibria(
    upper_util: np.ndarray,
    lower_util: np.ndarray,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Return pure Nash equilibria as row/column indices.

    A pair (i, j) is a pure Nash equilibrium when:
        - upper cannot improve by changing i while lower keeps j fixed
        - lower cannot improve by changing j while upper keeps i fixed
    """
    upper_util = np.asarray(upper_util, dtype=np.float64)
    lower_util = np.asarray(lower_util, dtype=np.float64)

    if upper_util.shape != lower_util.shape:
        raise ValueError(
            f"Utility matrices must have the same shape, got "
            f"{upper_util.shape} and {lower_util.shape}"
        )

    upper_best = upper_util >= (np.max(upper_util, axis=0, keepdims=True) - float(atol))
    lower_best = lower_util >= (np.max(lower_util, axis=1, keepdims=True) - float(atol))
    return np.argwhere(upper_best & lower_best)


def _score_matrix(
    criterion: str,
    compat: np.ndarray,
    upper_util: np.ndarray,
    lower_util: np.ndarray,
) -> np.ndarray:
    if criterion == "welfare":
        return upper_util + lower_util
    if criterion == "compatibility":
        return compat
    if criterion == "upper":
        return upper_util
    if criterion == "lower":
        return lower_util
    raise ValueError(
        "criterion must be one of 'welfare', 'compatibility', 'upper', or 'lower', "
        f"got {criterion!r}"
    )


def _choose_from_indices(
    indices: np.ndarray,
    scores: np.ndarray,
) -> Tuple[int, int]:
    if len(indices) == 0:
        raise ValueError("indices must contain at least one candidate")

    candidate_scores = np.asarray([scores[int(i), int(j)] for i, j in indices], dtype=np.float64)
    chosen_pos = int(np.argmax(candidate_scores))
    chosen_i, chosen_j = indices[chosen_pos]
    return int(chosen_i), int(chosen_j)


def _argmax_pair(scores: np.ndarray) -> Tuple[int, int]:
    flat_index = int(np.argmax(np.asarray(scores, dtype=np.float64).reshape(-1)))
    i, j = np.unravel_index(flat_index, scores.shape)
    return int(i), int(j)


def solve_nash_step(
    soprano_logits: np.ndarray,
    bass_logits: np.ndarray,
    pair_logits: np.ndarray,
    pair_to_id: Dict[Tuple[int, int], int],
    vocab_size: int,
    alpha: float,
    beta_private: float,
    top_k: Optional[int] = 12,
    s_actions: Optional[Sequence[int] | np.ndarray] = None,
    b_actions: Optional[Sequence[int] | np.ndarray] = None,
    tie_break: str = "welfare",
    fallback: str = "welfare",
    atol: float = 1e-9,
) -> NashStepResult:
    """
    Solve one pure-Nash two-player normal-form game on A_U x A_L.

    Utilities:
        U_upper = C(a_U, a_L) + alpha        * V_U(a_U)
        U_lower = C(a_U, a_L) + beta_private * V_L(a_L)

    If one or more pure Nash equilibria exist, `tie_break` chooses among them.
    If none exist, `fallback` chooses a deterministic non-equilibrium pair.
    Report `nash_fallback_used` and regret metrics so this method is treated as
    an ablation, not as guaranteed equilibrium generation.
    """
    soprano_logits = np.asarray(soprano_logits, dtype=np.float64)
    bass_logits = np.asarray(bass_logits, dtype=np.float64)
    pair_logits = np.asarray(pair_logits, dtype=np.float64)

    if s_actions is None or b_actions is None:
        if top_k is not None and 0 < int(top_k) < int(vocab_size):
            k = int(top_k)
            s_actions_arr = np.argpartition(soprano_logits, -k)[-k:]
            b_actions_arr = np.argpartition(bass_logits, -k)[-k:]
        else:
            s_actions_arr = np.arange(vocab_size)
            b_actions_arr = np.arange(vocab_size)
    else:
        s_actions_arr = s_actions
        b_actions_arr = b_actions

    s_actions_arr = _normalize_actions(s_actions_arr, vocab_size=vocab_size, name="s_actions")
    b_actions_arr = _normalize_actions(b_actions_arr, vocab_size=vocab_size, name="b_actions")

    compat, upper_util, lower_util = _build_utility_matrices(
        soprano_logits=soprano_logits,
        bass_logits=bass_logits,
        pair_logits=pair_logits,
        pair_to_id=pair_to_id,
        s_actions=s_actions_arr,
        b_actions=b_actions_arr,
        alpha=alpha,
        beta_private=beta_private,
    )

    equilibria = find_pure_nash_equilibria(
        upper_util=upper_util,
        lower_util=lower_util,
        atol=atol,
    )

    pure_found = len(equilibria) > 0
    if pure_found:
        scores = _score_matrix(tie_break, compat, upper_util, lower_util)
        chosen_i, chosen_j = _choose_from_indices(equilibria, scores)
        fallback_used = False
    else:
        scores = _score_matrix(fallback, compat, upper_util, lower_util)
        chosen_i, chosen_j = _argmax_pair(scores)
        fallback_used = True

    pair = (int(s_actions_arr[chosen_i]), int(b_actions_arr[chosen_j]))

    chosen_upper_payoff = float(upper_util[chosen_i, chosen_j])
    chosen_lower_payoff = float(lower_util[chosen_i, chosen_j])
    chosen_compatibility = float(compat[chosen_i, chosen_j])

    regret_upper = float(np.max(upper_util[:, chosen_j]) - chosen_upper_payoff)
    regret_lower = float(np.max(lower_util[chosen_i, :]) - chosen_lower_payoff)
    welfare = float(chosen_upper_payoff + chosen_lower_payoff)

    metrics = {
        "nash_pure_found": float(1.0 if pure_found else 0.0),
        "nash_fallback_used": float(1.0 if fallback_used else 0.0),
        "nash_num_equilibria": float(len(equilibria)),
        "nash_is_unique": float(1.0 if len(equilibria) == 1 else 0.0),
        "chosen_pair_compatibility": chosen_compatibility,
        "payoff_upper": chosen_upper_payoff,
        "payoff_lower": chosen_lower_payoff,
        "payoff_welfare": welfare,
        "payoff_balance_abs": float(abs(chosen_upper_payoff - chosen_lower_payoff)),
        "regret_upper": regret_upper,
        "regret_lower": regret_lower,
        "regret_total": float(regret_upper + regret_lower),
        "num_upper_candidates": float(len(s_actions_arr)),
        "num_lower_candidates": float(len(b_actions_arr)),
        "num_pair_candidates": float(len(s_actions_arr) * len(b_actions_arr)),
    }
    return NashStepResult(pair=pair, metrics=metrics)
