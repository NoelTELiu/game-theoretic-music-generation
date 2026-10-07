from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .qre import entropy, softmax_np


@dataclass
class StackelbergStepResult:
    """Result of one Stackelberg-style leader/follower decision step."""

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
    """
    Build compatibility and player utility matrices over A_U x A_L.

    Returns:
        compat: [num_upper, num_lower]
        U_upper: [num_upper, num_lower]
        U_lower: [num_upper, num_lower]
    """
    m = int(len(s_actions))
    n = int(len(b_actions))

    compat = np.zeros((m, n), dtype=np.float64)
    for i, s_id in enumerate(s_actions):
        for j, b_id in enumerate(b_actions):
            compat[i, j] = float(pair_logits[pair_to_id[(int(s_id), int(b_id))]])

    upper_private = np.asarray(soprano_logits[s_actions], dtype=np.float64)
    lower_private = np.asarray(bass_logits[b_actions], dtype=np.float64)

    U_upper = compat + float(alpha) * upper_private[:, None]
    U_lower = compat + float(beta_private) * lower_private[None, :]
    return compat, U_upper, U_lower


def solve_stackelberg_step(
    soprano_logits: np.ndarray,
    bass_logits: np.ndarray,
    pair_logits: np.ndarray,
    pair_to_id: Dict[Tuple[int, int], int],
    vocab_size: int,
    alpha: float,
    beta_private: float,
    leader: str,
    mode: str,
    rng: np.random.Generator,
    stackelberg_beta: float = 1.0,
    temperature: float = 1.0,
    top_k: Optional[int] = 12,
    s_actions: Optional[Sequence[int] | np.ndarray] = None,
    b_actions: Optional[Sequence[int] | np.ndarray] = None,
) -> StackelbergStepResult:
    """
    Solve one Stackelberg-style leader/follower game on A_U x A_L.

    Utilities:
        U_upper = C(a_U, a_L) + alpha        * V_U(a_U)
        U_lower = C(a_U, a_L) + beta_private * V_L(a_L)

    Supported leaders:
        leader="lower": bass/lower voice commits first; upper best-responds.
        leader="upper": soprano/upper voice commits first; lower best-responds.

    Supported modes:
        mode="deterministic": follower uses argmax response; leader uses argmax
            anticipated payoff. This is the first recommended experiment.
        mode="soft": follower uses a softmax response for each possible leader
            action; leader samples from a softmax over expected leader payoffs.

    Candidate-set policy:
        If s_actions and b_actions are provided, this function uses exactly those
        upper/lower candidate sets. Otherwise it falls back to private-head top-k.
    """
    soprano_logits = np.asarray(soprano_logits, dtype=np.float64)
    bass_logits = np.asarray(bass_logits, dtype=np.float64)
    pair_logits = np.asarray(pair_logits, dtype=np.float64)

    leader = str(leader).lower().strip()
    mode = str(mode).lower().strip()
    if leader not in {"lower", "upper"}:
        raise ValueError(f"leader must be 'lower' or 'upper', got {leader!r}")
    if mode not in {"deterministic", "soft"}:
        raise ValueError(f"mode must be 'deterministic' or 'soft', got {mode!r}")

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

    compat, U_upper, U_lower = _build_utility_matrices(
        soprano_logits=soprano_logits,
        bass_logits=bass_logits,
        pair_logits=pair_logits,
        pair_to_id=pair_to_id,
        s_actions=s_actions_arr,
        b_actions=b_actions_arr,
        alpha=alpha,
        beta_private=beta_private,
    )

    m, n = U_upper.shape
    leader_policy_entropy = 0.0
    follower_response_entropy = 0.0
    leader_expected_payoff = 0.0

    if leader == "lower":
        # Leader action is lower index j; follower action is upper index i.
        if mode == "deterministic":
            best_upper_for_lower = np.argmax(U_upper, axis=0)  # [n]
            anticipated_lower_payoff = np.array(
                [U_lower[int(best_upper_for_lower[j]), j] for j in range(n)],
                dtype=np.float64,
            )
            chosen_j = int(np.argmax(anticipated_lower_payoff))
            chosen_i = int(best_upper_for_lower[chosen_j])
            leader_expected_payoff = float(anticipated_lower_payoff[chosen_j])
        else:
            follower_probs_by_lower = np.zeros((m, n), dtype=np.float64)
            anticipated_lower_payoff = np.zeros(n, dtype=np.float64)
            for j in range(n):
                follower_probs_by_lower[:, j] = softmax_np(
                    float(stackelberg_beta) * U_upper[:, j],
                    temperature=temperature,
                )
                anticipated_lower_payoff[j] = float(np.dot(follower_probs_by_lower[:, j], U_lower[:, j]))

            leader_probs = softmax_np(
                float(stackelberg_beta) * anticipated_lower_payoff,
                temperature=temperature,
            )
            chosen_j = int(rng.choice(np.arange(n), p=leader_probs))
            chosen_i = int(rng.choice(np.arange(m), p=follower_probs_by_lower[:, chosen_j]))
            leader_policy_entropy = entropy(leader_probs)
            follower_response_entropy = entropy(follower_probs_by_lower[:, chosen_j])
            leader_expected_payoff = float(anticipated_lower_payoff[chosen_j])

    else:
        # Leader action is upper index i; follower action is lower index j.
        if mode == "deterministic":
            best_lower_for_upper = np.argmax(U_lower, axis=1)  # [m]
            anticipated_upper_payoff = np.array(
                [U_upper[i, int(best_lower_for_upper[i])] for i in range(m)],
                dtype=np.float64,
            )
            chosen_i = int(np.argmax(anticipated_upper_payoff))
            chosen_j = int(best_lower_for_upper[chosen_i])
            leader_expected_payoff = float(anticipated_upper_payoff[chosen_i])
        else:
            follower_probs_by_upper = np.zeros((m, n), dtype=np.float64)
            anticipated_upper_payoff = np.zeros(m, dtype=np.float64)
            for i in range(m):
                follower_probs_by_upper[i, :] = softmax_np(
                    float(stackelberg_beta) * U_lower[i, :],
                    temperature=temperature,
                )
                anticipated_upper_payoff[i] = float(np.dot(follower_probs_by_upper[i, :], U_upper[i, :]))

            leader_probs = softmax_np(
                float(stackelberg_beta) * anticipated_upper_payoff,
                temperature=temperature,
            )
            chosen_i = int(rng.choice(np.arange(m), p=leader_probs))
            chosen_j = int(rng.choice(np.arange(n), p=follower_probs_by_upper[chosen_i, :]))
            leader_policy_entropy = entropy(leader_probs)
            follower_response_entropy = entropy(follower_probs_by_upper[chosen_i, :])
            leader_expected_payoff = float(anticipated_upper_payoff[chosen_i])

    pair = (int(s_actions_arr[chosen_i]), int(b_actions_arr[chosen_j]))

    chosen_upper_payoff = float(U_upper[chosen_i, chosen_j])
    chosen_lower_payoff = float(U_lower[chosen_i, chosen_j])
    chosen_compatibility = float(compat[chosen_i, chosen_j])

    # Unilateral regret is useful as a diagnostic. In deterministic Stackelberg,
    # the follower regret should be zero for the selected leader action.
    regret_upper = float(np.max(U_upper[:, chosen_j]) - chosen_upper_payoff)
    regret_lower = float(np.max(U_lower[chosen_i, :]) - chosen_lower_payoff)

    if leader == "lower":
        leader_payoff = chosen_lower_payoff
        follower_payoff = chosen_upper_payoff
        follower_regret = regret_upper
        leader_regret_given_follower = regret_lower
    else:
        leader_payoff = chosen_upper_payoff
        follower_payoff = chosen_lower_payoff
        follower_regret = regret_lower
        leader_regret_given_follower = regret_upper

    metrics = {
        "stackelberg_leader_is_lower": float(1.0 if leader == "lower" else 0.0),
        "stackelberg_mode_is_soft": float(1.0 if mode == "soft" else 0.0),
        "stackelberg_beta": float(stackelberg_beta),
        "chosen_pair_compatibility": chosen_compatibility,
        "payoff_upper": chosen_upper_payoff,
        "payoff_lower": chosen_lower_payoff,
        "leader_payoff": float(leader_payoff),
        "follower_payoff": float(follower_payoff),
        "leader_expected_payoff": float(leader_expected_payoff),
        "regret_upper": regret_upper,
        "regret_lower": regret_lower,
        "follower_regret": float(follower_regret),
        "leader_regret_given_follower": float(leader_regret_given_follower),
        "leader_policy_entropy": float(leader_policy_entropy),
        "follower_response_entropy": float(follower_response_entropy),
        "num_upper_candidates": float(m),
        "num_lower_candidates": float(n),
        "num_pair_candidates": float(m * n),
    }
    return StackelbergStepResult(pair=pair, metrics=metrics)
