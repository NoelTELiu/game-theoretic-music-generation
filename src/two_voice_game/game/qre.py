from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence

import numpy as np


@dataclass
class QREStepResult:
    pair: Tuple[int, int]
    metrics: Dict[str, float]


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temperature = max(float(temperature), 1e-8)
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - np.max(z)
    exp_z = np.exp(z)
    total = exp_z.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(z, dtype=np.float64) / len(z)
    return exp_z / total


def entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


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


def solve_qre_step(
    soprano_logits: np.ndarray,
    bass_logits: np.ndarray,
    pair_logits: np.ndarray,
    pair_to_id: Dict[Tuple[int, int], int],
    vocab_size: int,
    alpha: float,
    beta_private: float,
    qre_beta: float,
    iterations: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = 12,
    s_actions: Optional[Sequence[int] | np.ndarray] = None,
    b_actions: Optional[Sequence[int] | np.ndarray] = None,
) -> QREStepResult:
    """
    Solve one QRE-style two-player normal-form game.

    Utilities:
        U_upper = C(a_U, a_L) + alpha        * V_U(a_U)
        U_lower = C(a_U, a_L) + beta_private * V_L(a_L)

    Candidate-set policy:
        - If s_actions and b_actions are provided, QRE uses exactly those
          upper/lower action sets. This is the preferred path for fair
          comparison with no_game_joint_grid.
        - Otherwise, the function falls back to top-k private-head candidates.

    alpha and beta_private are generation-time hyperparameters.
    """
    soprano_logits = np.asarray(soprano_logits, dtype=np.float64)
    bass_logits = np.asarray(bass_logits, dtype=np.float64)
    pair_logits = np.asarray(pair_logits, dtype=np.float64)

    if s_actions is None or b_actions is None:
        if top_k is not None and 0 < top_k < vocab_size:
            s_actions_arr = np.argpartition(soprano_logits, -top_k)[-top_k:]
            b_actions_arr = np.argpartition(bass_logits, -top_k)[-top_k:]
        else:
            s_actions_arr = np.arange(vocab_size)
            b_actions_arr = np.arange(vocab_size)
    else:
        s_actions_arr = s_actions
        b_actions_arr = b_actions

    s_actions_arr = _normalize_actions(s_actions_arr, vocab_size=vocab_size, name="s_actions")
    b_actions_arr = _normalize_actions(b_actions_arr, vocab_size=vocab_size, name="b_actions")

    Uu = np.zeros((len(s_actions_arr), len(b_actions_arr)), dtype=np.float64)
    Ul = np.zeros_like(Uu)

    for i, s_id in enumerate(s_actions_arr):
        for j, b_id in enumerate(b_actions_arr):
            pair_id = pair_to_id[(int(s_id), int(b_id))]
            compat = float(pair_logits[pair_id])
            Uu[i, j] = compat + alpha * float(soprano_logits[s_id])
            Ul[i, j] = compat + beta_private * float(bass_logits[b_id])

    p_u = np.ones(len(s_actions_arr), dtype=np.float64) / len(s_actions_arr)
    p_l = np.ones(len(b_actions_arr), dtype=np.float64) / len(b_actions_arr)

    for _ in range(max(int(iterations), 0)):
        p_u = softmax_np(qre_beta * (Uu @ p_l), temperature=temperature)
        p_l = softmax_np(qre_beta * (p_u @ Ul), temperature=temperature)

    chosen_i = int(rng.choice(np.arange(len(s_actions_arr)), p=p_u))
    chosen_j = int(rng.choice(np.arange(len(b_actions_arr)), p=p_l))
    pair = (int(s_actions_arr[chosen_i]), int(b_actions_arr[chosen_j]))

    chosen_u = Uu[chosen_i, chosen_j]
    chosen_l = Ul[chosen_i, chosen_j]
    metrics = {
        "regret_upper": float(np.max(Uu[:, chosen_j]) - chosen_u),
        "regret_lower": float(np.max(Ul[chosen_i, :]) - chosen_l),
        "entropy_upper": entropy(p_u),
        "entropy_lower": entropy(p_l),
        "payoff_upper": float(chosen_u),
        "payoff_lower": float(chosen_l),
        "num_upper_candidates": float(len(s_actions_arr)),
        "num_lower_candidates": float(len(b_actions_arr)),
        "num_pair_candidates": float(len(s_actions_arr) * len(b_actions_arr)),
    }
    return QREStepResult(pair=pair, metrics=metrics)
