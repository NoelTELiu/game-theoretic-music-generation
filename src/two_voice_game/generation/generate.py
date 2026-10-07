from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

import numpy as np
import torch

from two_voice_game.game import (
    entropy,
    softmax_np,
    solve_nash_step,
    solve_qre_step,
    solve_stackelberg_step,
)
from two_voice_game.models import MultiHeadMLP
from two_voice_game.features.duration import (
    DURATION_FEATURE_DIM,
    DURATION_FEATURE_NAMES,
    HOLD_TOKEN_RAW,
    build_duration_feature_tensor_from_context,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LoadedGenerator:
    model: MultiHeadMLP
    config: dict
    token_to_id: Dict[int, int]
    id_to_token: Dict[int, int]
    pair_to_id: Dict[Tuple[int, int], int]
    id_to_pair: Dict[int, Tuple[int, int]]
    context_len: int
    duration_aware_state: bool = False
    duration_feature_dim: int = 0
    duration_feature_names: List[str] | None = None
    duration_feature_normalizer: float | None = None
    hold_token_id: int | None = None


def _normalize_int_dict(d: Dict[Any, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def _normalize_pair_to_id(d: Dict[Any, Any]) -> Dict[Tuple[int, int], int]:
    out = {}
    for k, v in d.items():
        if not isinstance(k, tuple):
            raise TypeError(f"Unsupported pair_to_id key type: {type(k)}")
        out[(int(k[0]), int(k[1]))] = int(v)
    return out


def _normalize_id_to_pair(d: Dict[Any, Any]) -> Dict[int, Tuple[int, int]]:
    return {int(k): (int(v[0]), int(v[1])) for k, v in d.items()}


def _get_model_config_from_checkpoint(ckpt: dict) -> dict:
    """
    Support both checkpoint formats.

    New format:
        ckpt["config"]["model"] = {
            "context_len": ...,
            "emb_dim": ...,
            "hidden_dim": ...,
            "dropout": ...
        }

    Old format:
        ckpt["config"] = {
            "context_len": ...,
            "emb_dim": ...,
            "hidden_dim": ...,
            "dropout": ...
        }
    """
    cfg = ckpt.get("config", {})

    if "model" in cfg:
        return cfg["model"]

    required_keys = ["context_len", "emb_dim", "hidden_dim", "dropout"]
    missing = [k for k in required_keys if k not in cfg]

    if missing:
        raise KeyError(
            "Checkpoint config does not contain a 'model' section and is "
            f"missing fallback keys: {missing}. "
            "Please retrain the model or provide a compatible checkpoint."
        )

    return cfg


def load_generator(checkpoint_path: str | Path, device: str = DEVICE) -> LoadedGenerator:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    token_to_id = _normalize_int_dict(ckpt["token_to_id"])
    id_to_token = _normalize_int_dict(ckpt["id_to_token"])
    pair_to_id = _normalize_pair_to_id(ckpt["pair_to_id"])
    id_to_pair = _normalize_id_to_pair(ckpt["id_to_pair"])

    model_cfg = _get_model_config_from_checkpoint(ckpt)

    context_len = int(model_cfg["context_len"])
    duration_aware_state = bool(model_cfg.get("duration_aware_state", False))
    duration_feature_dim = int(model_cfg.get("duration_feature_dim", DURATION_FEATURE_DIM if duration_aware_state else 0))
    duration_feature_names = list(model_cfg.get("duration_feature_names", DURATION_FEATURE_NAMES if duration_aware_state else []))
    duration_feature_normalizer_value = model_cfg.get("duration_feature_normalizer", context_len if duration_aware_state else None)
    duration_feature_normalizer = (
        float(duration_feature_normalizer_value)
        if duration_feature_normalizer_value is not None
        else None
    )
    hold_token_id = token_to_id.get(HOLD_TOKEN_RAW)

    if duration_aware_state:
        if duration_feature_dim != DURATION_FEATURE_DIM:
            raise ValueError(
                "This generator supports exactly the standard duration feature set "
                f"with dim={DURATION_FEATURE_DIM}, but checkpoint config has "
                f"duration_feature_dim={duration_feature_dim}."
            )
        if hold_token_id is None:
            raise KeyError(
                f"Checkpoint uses duration-aware state, but raw HOLD token {HOLD_TOKEN_RAW} "
                "is not present in token_to_id."
            )

    model = MultiHeadMLP(
        vocab_size=len(token_to_id),
        pair_vocab_size=len(pair_to_id),
        context_len=context_len,
        emb_dim=int(model_cfg["emb_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        duration_feature_dim=duration_feature_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return LoadedGenerator(
        model=model,
        config=cfg,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pair_to_id=pair_to_id,
        id_to_pair=id_to_pair,
        context_len=context_len,
        duration_aware_state=duration_aware_state,
        duration_feature_dim=duration_feature_dim,
        duration_feature_names=duration_feature_names,
        duration_feature_normalizer=duration_feature_normalizer,
        hold_token_id=hold_token_id,
    )


def raw_to_id_sequence(raw_tokens: np.ndarray, token_to_id: Dict[int, int]) -> np.ndarray:
    raw_tokens = np.asarray(raw_tokens, dtype=np.int64)

    if raw_tokens.ndim != 2 or raw_tokens.shape[1] != 2:
        raise ValueError(f"Expected raw_tokens shape [T, 2], got {raw_tokens.shape}")

    ids = np.zeros_like(raw_tokens, dtype=np.int64)

    for t in range(raw_tokens.shape[0]):
        for v in range(2):
            raw = int(raw_tokens[t, v])

            if raw not in token_to_id:
                raise KeyError(f"Raw token {raw} is not in checkpoint vocabulary.")

            ids[t, v] = int(token_to_id[raw])

    return ids


def id_to_raw_sequence(id_tokens: np.ndarray, id_to_token: Dict[int, int]) -> np.ndarray:
    id_tokens = np.asarray(id_tokens, dtype=np.int64)
    raw = np.zeros_like(id_tokens, dtype=np.int64)

    for t in range(id_tokens.shape[0]):
        for v in range(2):
            raw[t, v] = int(id_to_token[int(id_tokens[t, v])])

    return raw


def select_heldout_seeds(
    data_path: str | Path,
    split: str,
    context_len: int,
    num_seeds: int = 20,
    seed_indices: Optional[List[int]] = None,
    rng_seed: int = 42,
) -> List[Tuple[int, np.ndarray]]:
    data = np.load(Path(data_path), allow_pickle=True)
    key = f"{split}_tokens"

    if key not in data:
        raise KeyError(
            f"Could not find '{key}' in {data_path}. "
            f"Available keys: {list(data.keys())}"
        )

    pieces = data[key]
    valid_indices = [i for i, p in enumerate(pieces) if len(np.asarray(p)) > context_len]

    if seed_indices is None:
        rng = random.Random(rng_seed)
        chosen = valid_indices[:]
        rng.shuffle(chosen)
        chosen = chosen[:num_seeds]
    else:
        chosen = seed_indices
        for idx in chosen:
            if idx not in valid_indices:
                raise ValueError(
                    f"Piece index {idx} is invalid or too short "
                    f"for context_len={context_len}."
                )

    return [
        (idx, np.asarray(pieces[idx], dtype=np.int64)[:context_len, :2].copy())
        for idx in chosen
    ]


def _model_logits(
    gen: LoadedGenerator,
    generated_ids: np.ndarray,
    device: str = DEVICE,
):
    context = generated_ids[-gen.context_len:]

    if context.shape != (gen.context_len, 2):
        raise ValueError(
            f"Context must have shape [{gen.context_len}, 2], got {context.shape}"
        )

    pos_value = generated_ids.shape[0] % 16

    context_t = torch.tensor(
        context,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    pos_t = torch.tensor(
        [pos_value],
        dtype=torch.long,
        device=device,
    )

    duration_t = None
    if gen.duration_aware_state:
        if gen.hold_token_id is None:
            raise ValueError("duration-aware checkpoint is missing hold_token_id metadata.")
        duration_t = build_duration_feature_tensor_from_context(
            context=context_t,
            hold_token_id=int(gen.hold_token_id),
            normalizer=gen.duration_feature_normalizer or gen.context_len,
        )
        if duration_t.ndim == 1:
            duration_t = duration_t.unsqueeze(0)
        duration_t = duration_t.to(device=device, dtype=torch.float32)
        if duration_t.shape[-1] != gen.duration_feature_dim:
            raise ValueError(
                "Computed duration feature dimension does not match checkpoint: "
                f"{duration_t.shape[-1]} != {gen.duration_feature_dim}"
            )

    with torch.no_grad():
        s_logits, b_logits, pair_logits = gen.model(context_t, pos_t, duration_t)

    return (
        s_logits[0].detach().cpu().numpy(),
        b_logits[0].detach().cpu().numpy(),
        pair_logits[0].detach().cpu().numpy(),
    )


def _sample_index_from_scores(
    scores: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> int:
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    if top_k is not None and 0 < top_k < n:
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        probs = softmax_np(scores[top_idx], temperature=temperature)
        return int(rng.choice(top_idx, p=probs))

    probs = softmax_np(scores, temperature=temperature)
    return int(rng.choice(np.arange(n), p=probs))



def get_voice_candidate_actions(
    soprano_logits: np.ndarray,
    bass_logits: np.ndarray,
    top_k: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the shared voice-level candidate action sets used by the fair
    no-game joint grid baseline and QRE game generation.

    A_U = top-k candidates from the upper/soprano private head.
    A_L = top-k candidates from the lower/bass private head.

    The candidate pair grid is A_U x A_L.
    """
    soprano_logits = np.asarray(soprano_logits, dtype=np.float64)
    bass_logits = np.asarray(bass_logits, dtype=np.float64)
    vocab_size = len(soprano_logits)

    if len(bass_logits) != vocab_size:
        raise ValueError(
            f"Soprano and bass logits must have the same length, got "
            f"{len(soprano_logits)} and {len(bass_logits)}"
        )

    if top_k is not None and 0 < int(top_k) < vocab_size:
        k = int(top_k)
        s_actions = np.argpartition(soprano_logits, -k)[-k:]
        b_actions = np.argpartition(bass_logits, -k)[-k:]
    else:
        s_actions = np.arange(vocab_size)
        b_actions = np.arange(vocab_size)

    s_actions = np.array(sorted(map(int, s_actions)), dtype=np.int64)
    b_actions = np.array(sorted(map(int, b_actions)), dtype=np.int64)
    return s_actions, b_actions


def _pair_grid_scores(
    pair_logits: np.ndarray,
    pair_to_id: Dict[Tuple[int, int], int],
    s_actions: Sequence[int] | np.ndarray,
    b_actions: Sequence[int] | np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Return pair-head scores over the shared candidate grid A_U x A_L."""
    scores: List[float] = []
    pairs: List[Tuple[int, int]] = []

    for s_id in s_actions:
        for b_id in b_actions:
            pair = (int(s_id), int(b_id))
            pair_id = pair_to_id[pair]
            scores.append(float(pair_logits[pair_id]))
            pairs.append(pair)

    return np.asarray(scores, dtype=np.float64), pairs


def _sample_pair_from_candidate_grid(
    pair_logits: np.ndarray,
    pair_to_id: Dict[Tuple[int, int], int],
    s_actions: Sequence[int] | np.ndarray,
    b_actions: Sequence[int] | np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> Tuple[Tuple[int, int], Dict[str, float]]:
    """
    Sample a pair from the shared candidate grid using only pair-head
    compatibility scores C(s, a_U, a_L).

    This is the fair no-game joint decision rule:
        same A_U, same A_L as QRE, but no player-specific utilities and no solver.
    """
    scores, pairs = _pair_grid_scores(
        pair_logits=pair_logits,
        pair_to_id=pair_to_id,
        s_actions=s_actions,
        b_actions=b_actions,
    )
    probs = softmax_np(scores, temperature=temperature)
    chosen_idx = int(rng.choice(np.arange(len(pairs)), p=probs))
    chosen_pair = pairs[chosen_idx]

    metrics = {
        "pair_grid_entropy": entropy(probs),
        "chosen_pair_compatibility": float(scores[chosen_idx]),
        "num_upper_candidates": float(len(s_actions)),
        "num_lower_candidates": float(len(b_actions)),
        "num_pair_candidates": float(len(pairs)),
    }
    return chosen_pair, metrics


def generate_independent(
    gen: LoadedGenerator,
    seed_raw: np.ndarray,
    num_steps: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 16,
    rng_seed: int = 0,
    device: str = DEVICE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Independent-agents baseline.

    The upper and lower voices are generated independently from their private
    heads:

        soprano_logits -> next soprano token
        bass_logits    -> next bass token

    This method does not use:
        - pair_logits
        - payoff matrix
        - game solver

    Research meaning:
        no strategic interaction and no joint pair selection.
    """
    rng = np.random.default_rng(rng_seed)
    generated_ids = raw_to_id_sequence(seed_raw, gen.token_to_id)

    soprano_entropies = []
    bass_entropies = []

    for _ in range(num_steps):
        s_logits, b_logits, _ = _model_logits(gen, generated_ids, device=device)

        s_probs = softmax_np(s_logits, temperature=temperature)
        b_probs = softmax_np(b_logits, temperature=temperature)

        soprano_entropies.append(entropy(s_probs))
        bass_entropies.append(entropy(b_probs))

        s_id = _sample_index_from_scores(
            s_logits,
            rng,
            temperature=temperature,
            top_k=top_k,
        )

        b_id = _sample_index_from_scores(
            b_logits,
            rng,
            temperature=temperature,
            top_k=top_k,
        )

        generated_ids = np.vstack(
            [
                generated_ids,
                np.array([s_id, b_id], dtype=np.int64),
            ]
        )

    return id_to_raw_sequence(generated_ids, gen.id_to_token), {
        "avg_soprano_entropy": float(np.mean(soprano_entropies))
        if soprano_entropies
        else 0.0,
        "avg_bass_entropy": float(np.mean(bass_entropies))
        if bass_entropies
        else 0.0,
    }


def generate_no_game_joint_grid(
    gen: LoadedGenerator,
    seed_raw: np.ndarray,
    num_steps: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 16,
    rng_seed: int = 0,
    device: str = DEVICE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fair no-game joint baseline over a shared voice-level candidate grid.

    Candidate set:
        A_U = top-k upper/soprano candidates from the private upper head
        A_L = top-k lower/bass candidates from the private lower head
        candidate pairs = A_U x A_L

    Decision rule:
        sample a pair from A_U x A_L using only the learned pair compatibility
        score C(s, a_U, a_L), represented by pair_logits.

    This baseline models upper-lower interaction but does not construct
    player-specific utilities and does not solve a game. It is the recommended
    main baseline for comparing against qre_game because qre_game uses the same
    A_U x A_L candidate grid.
    """
    rng = np.random.default_rng(rng_seed)
    generated_ids = raw_to_id_sequence(seed_raw, gen.token_to_id)

    step_infos: List[Dict[str, float]] = []

    for _ in range(num_steps):
        s_logits, b_logits, pair_logits = _model_logits(gen, generated_ids, device=device)

        s_actions, b_actions = get_voice_candidate_actions(
            soprano_logits=s_logits,
            bass_logits=b_logits,
            top_k=top_k,
        )

        next_pair, metrics = _sample_pair_from_candidate_grid(
            pair_logits=pair_logits,
            pair_to_id=gen.pair_to_id,
            s_actions=s_actions,
            b_actions=b_actions,
            rng=rng,
            temperature=temperature,
        )
        step_infos.append(metrics)

        generated_ids = np.vstack(
            [
                generated_ids,
                np.array(next_pair, dtype=np.int64),
            ]
        )

    metrics_out: Dict[str, float] = {}
    if step_infos:
        for key in step_infos[0].keys():
            metrics_out[f"avg_{key}"] = float(np.mean([x[key] for x in step_infos]))

    return id_to_raw_sequence(generated_ids, gen.id_to_token), metrics_out


def generate_no_game_joint_global(
    gen: LoadedGenerator,
    seed_raw: np.ndarray,
    num_steps: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 16,
    rng_seed: int = 0,
    device: str = DEVICE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Strong/global no-game joint baseline.

    This is the original pair-head baseline:
        state -> global pair_logits top-k -> next (soprano, bass)

    It models upper-lower pair compatibility through the joint pair head, but it
    does not use the same A_U x A_L grid as QRE. Therefore, treat it as a strong
    supplementary baseline rather than the fairest main comparison to qre_game.
    """
    rng = np.random.default_rng(rng_seed)
    generated_ids = raw_to_id_sequence(seed_raw, gen.token_to_id)

    pair_entropies = []

    for _ in range(num_steps):
        _, _, pair_logits = _model_logits(gen, generated_ids, device=device)

        pair_probs = softmax_np(pair_logits, temperature=temperature)
        pair_entropies.append(entropy(pair_probs))

        pair_id = _sample_index_from_scores(
            pair_logits,
            rng,
            temperature=temperature,
            top_k=top_k,
        )

        next_pair = gen.id_to_pair[pair_id]

        generated_ids = np.vstack(
            [
                generated_ids,
                np.array(next_pair, dtype=np.int64),
            ]
        )

    return id_to_raw_sequence(generated_ids, gen.id_to_token), {
        "avg_pair_entropy": float(np.mean(pair_entropies))
        if pair_entropies
        else 0.0,
        "avg_num_pair_candidates": float(top_k if top_k is not None else len(gen.id_to_pair)),
    }


# Main no-game joint alias.
# For the main experiment, no_game_joint now means the fair grid baseline.
generate_no_game_joint = generate_no_game_joint_grid


# Backward-compatible alias.
# Old code may still import generate_no_game.
generate_no_game = generate_no_game_joint



def _sample_qre_pair_with_optional_compat_filter(
    soprano_logits: np.ndarray,
    bass_logits: np.ndarray,
    pair_logits: np.ndarray,
    pair_to_id: Dict[Tuple[int, int], int],
    s_actions: Sequence[int] | np.ndarray,
    b_actions: Sequence[int] | np.ndarray,
    rng: np.random.Generator,
    alpha: float = 1.0,
    beta_private: float = 1.0,
    qre_beta: float = 1.0,
    iterations: int = 20,
    temperature: float = 1.0,
    compat_top_pairs: Optional[int] = None,
) -> Tuple[Tuple[int, int], Dict[str, float]]:
    """
    Build a QRE-style policy on the shared A_U x A_L candidate grid, then
    optionally restrict final pair sampling to the top-N pairs by the shared
    compatibility score C(s, a_U, a_L).

    This keeps the strategic upper/lower soft-response behavior, but prevents
    the final sample from coming from very low-compatibility pairs.
    """
    s_actions = np.asarray(s_actions, dtype=np.int64)
    b_actions = np.asarray(b_actions, dtype=np.int64)

    if s_actions.size == 0 or b_actions.size == 0:
        raise ValueError("QRE candidate action sets must be non-empty.")

    m = int(s_actions.size)
    n = int(b_actions.size)

    compat = np.zeros((m, n), dtype=np.float64)
    for i, s_id in enumerate(s_actions):
        for j, b_id in enumerate(b_actions):
            compat[i, j] = float(pair_logits[pair_to_id[(int(s_id), int(b_id))]])

    upper_private = np.asarray(soprano_logits[s_actions], dtype=np.float64)
    lower_private = np.asarray(bass_logits[b_actions], dtype=np.float64)

    upper_util = compat + float(alpha) * upper_private[:, None]
    lower_util = compat + float(beta_private) * lower_private[None, :]

    pi_u = np.full(m, 1.0 / m, dtype=np.float64)
    pi_l = np.full(n, 1.0 / n, dtype=np.float64)

    num_iter = max(int(iterations), 1)
    for _ in range(num_iter):
        expected_upper = upper_util @ pi_l
        expected_lower = pi_u @ lower_util

        # qre_beta controls rationality. temperature is kept as a generation
        # softness parameter for consistency with the existing generator.
        pi_u = softmax_np(float(qre_beta) * expected_upper, temperature=temperature)
        pi_l = softmax_np(float(qre_beta) * expected_lower, temperature=temperature)

    joint_probs = np.outer(pi_u, pi_l)
    flat_joint = np.asarray(joint_probs.reshape(-1), dtype=np.float64)
    n_pairs = int(flat_joint.size)

    filter_enabled = compat_top_pairs is not None and int(compat_top_pairs) > 0
    selected_top_k = n_pairs
    fallback_uniform_on_filter = False

    if filter_enabled:
        selected_top_k = min(int(compat_top_pairs), n_pairs)
        flat_compat = compat.reshape(-1)

        # argpartition is enough because we only need the top-k set, not sorted order.
        top_indices = np.argpartition(flat_compat, -selected_top_k)[-selected_top_k:]

        masked = np.zeros_like(flat_joint, dtype=np.float64)
        masked[top_indices] = flat_joint[top_indices]

        total = float(masked.sum())
        if total <= 0.0 or not np.isfinite(total):
            # Extremely unlikely, but keeps sampling safe if all probability mass
            # underflows outside the retained compatibility set.
            masked[top_indices] = 1.0 / selected_top_k
            fallback_uniform_on_filter = True
        else:
            masked /= total

        sample_probs = masked
    else:
        total = float(flat_joint.sum())
        if total <= 0.0 or not np.isfinite(total):
            sample_probs = np.full(n_pairs, 1.0 / n_pairs, dtype=np.float64)
        else:
            sample_probs = flat_joint / total

    chosen_flat = int(rng.choice(np.arange(n_pairs), p=sample_probs))
    chosen_i, chosen_j = np.unravel_index(chosen_flat, (m, n))
    chosen_pair = (int(s_actions[chosen_i]), int(b_actions[chosen_j]))

    chosen_compat = float(compat[chosen_i, chosen_j])
    chosen_joint_prob_before_filter = float(flat_joint[chosen_flat])
    chosen_joint_prob_after_filter = float(sample_probs[chosen_flat])

    if filter_enabled:
        flat_compat = compat.reshape(-1)
        compat_threshold = float(np.min(flat_compat[top_indices]))
        retained_mass = float(flat_joint[top_indices].sum())
    else:
        compat_threshold = float(np.min(compat))
        retained_mass = 1.0

    metrics = {
        "qre_upper_entropy": entropy(pi_u),
        "qre_lower_entropy": entropy(pi_l),
        "qre_joint_entropy_before_filter": entropy(flat_joint / max(float(flat_joint.sum()), 1e-12)),
        "qre_joint_entropy_after_filter": entropy(sample_probs),
        "chosen_pair_compatibility": chosen_compat,
        "chosen_pair_joint_prob_before_filter": chosen_joint_prob_before_filter,
        "chosen_pair_joint_prob_after_filter": chosen_joint_prob_after_filter,
        "compat_filter_enabled": float(1.0 if filter_enabled else 0.0),
        "compat_top_pairs": float(selected_top_k),
        "compat_filter_retained_mass": retained_mass,
        "compat_filter_threshold": compat_threshold,
        "compat_filter_fallback_uniform": float(1.0 if fallback_uniform_on_filter else 0.0),
        "num_upper_candidates": float(m),
        "num_lower_candidates": float(n),
        "num_pair_candidates": float(n_pairs),
    }

    return chosen_pair, metrics

def generate_qre_game(
    gen: LoadedGenerator,
    seed_raw: np.ndarray,
    num_steps: int,
    alpha: float = 1.0,
    beta_private: float = 1.0,
    qre_beta: float = 1.0,
    iterations: int = 20,
    temperature: float = 1.0,
    top_k: Optional[int] = 12,
    compat_top_pairs: Optional[int] = None,
    rng_seed: int = 0,
    device: str = DEVICE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    QRE-style game generation.

    At each time step:

        1. Use the model to compute:
            V_U(a_U)        from soprano_logits
            V_L(a_L)        from bass_logits
            C(a_U, a_L)     from pair_logits

        2. Build two player-specific utilities:
            U_upper = C(a_U, a_L) + alpha        * V_U(a_U)
            U_lower = C(a_U, a_L) + beta_private * V_L(a_L)

        3. Use a QRE-style iterative logit response solver.

        4. If compat_top_pairs is set, restrict final sampling to the top-N
           pairs by shared compatibility C and renormalize the QRE joint policy
           over that filtered set.

    Research meaning:
        interaction with game solver. The optional compatibility filter is a
        controlled variant for testing whether QRE's weak harmonic metrics come
        from sampling too many low-compatibility pairs.
    """
    rng = np.random.default_rng(rng_seed)
    generated_ids = raw_to_id_sequence(seed_raw, gen.token_to_id)

    step_infos = []

    for _ in range(num_steps):
        s_logits, b_logits, pair_logits = _model_logits(
            gen,
            generated_ids,
            device=device,
        )

        s_actions, b_actions = get_voice_candidate_actions(
            soprano_logits=s_logits,
            bass_logits=b_logits,
            top_k=top_k,
        )

        if compat_top_pairs is None:
            result = solve_qre_step(
                soprano_logits=s_logits,
                bass_logits=b_logits,
                pair_logits=pair_logits,
                pair_to_id=gen.pair_to_id,
                vocab_size=len(gen.token_to_id),
                alpha=alpha,
                beta_private=beta_private,
                qre_beta=qre_beta,
                iterations=iterations,
                rng=rng,
                temperature=temperature,
                top_k=top_k,
                s_actions=s_actions,
                b_actions=b_actions,
            )

            next_pair = result.pair
            step_metrics = result.metrics
        else:
            next_pair, step_metrics = _sample_qre_pair_with_optional_compat_filter(
                soprano_logits=s_logits,
                bass_logits=b_logits,
                pair_logits=pair_logits,
                pair_to_id=gen.pair_to_id,
                s_actions=s_actions,
                b_actions=b_actions,
                rng=rng,
                alpha=alpha,
                beta_private=beta_private,
                qre_beta=qre_beta,
                iterations=iterations,
                temperature=temperature,
                compat_top_pairs=compat_top_pairs,
            )

        step_infos.append(step_metrics)

        generated_ids = np.vstack(
            [
                generated_ids,
                np.array(next_pair, dtype=np.int64),
            ]
        )

    raw = id_to_raw_sequence(generated_ids, gen.id_to_token)

    metrics: Dict[str, float] = {}

    if step_infos:
        for key in step_infos[0].keys():
            metrics[f"avg_{key}"] = float(np.mean([x[key] for x in step_infos]))

    return raw, metrics


def generate_nash_game(
    gen: LoadedGenerator,
    seed_raw: np.ndarray,
    num_steps: int,
    alpha: float = 1.0,
    beta_private: float = 1.0,
    top_k: Optional[int] = 12,
    tie_break: str = "welfare",
    fallback: str = "welfare",
    atol: float = 1e-9,
    rng_seed: int = 0,
    device: str = DEVICE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Pure-Nash game generation.

    Candidate set:
        A_U = top-k upper/soprano candidates from the private upper head
        A_L = top-k lower/bass candidates from the private lower head
        candidate pairs = A_U x A_L

    Utilities:
        U_upper = C(a_U, a_L) + alpha        * V_U(a_U)
        U_lower = C(a_U, a_L) + beta_private * V_L(a_L)

    Decision rule:
        choose a pure Nash equilibrium if one exists. If multiple equilibria
        exist, use `tie_break`. If no pure equilibrium exists, use `fallback`.

    Research meaning:
        simultaneous equilibrium ablation. It lets us test whether a stricter
        "neither voice wants to unilaterally change" decision rule is musically
        useful, and reports fallback/regret so failure cases are visible.
    """
    # Kept for API symmetry with stochastic generators; Nash itself is deterministic.
    _ = np.random.default_rng(rng_seed)
    generated_ids = raw_to_id_sequence(seed_raw, gen.token_to_id)

    step_infos: List[Dict[str, float]] = []

    for _step in range(num_steps):
        s_logits, b_logits, pair_logits = _model_logits(
            gen,
            generated_ids,
            device=device,
        )

        s_actions, b_actions = get_voice_candidate_actions(
            soprano_logits=s_logits,
            bass_logits=b_logits,
            top_k=top_k,
        )

        result = solve_nash_step(
            soprano_logits=s_logits,
            bass_logits=b_logits,
            pair_logits=pair_logits,
            pair_to_id=gen.pair_to_id,
            vocab_size=len(gen.token_to_id),
            alpha=alpha,
            beta_private=beta_private,
            top_k=top_k,
            s_actions=s_actions,
            b_actions=b_actions,
            tie_break=tie_break,
            fallback=fallback,
            atol=atol,
        )

        step_infos.append(result.metrics)

        generated_ids = np.vstack(
            [
                generated_ids,
                np.array(result.pair, dtype=np.int64),
            ]
        )

    raw = id_to_raw_sequence(generated_ids, gen.id_to_token)

    metrics: Dict[str, float] = {}
    if step_infos:
        for key in step_infos[0].keys():
            metrics[f"avg_{key}"] = float(np.mean([x[key] for x in step_infos]))

    return raw, metrics


def generate_stackelberg_game(
    gen: LoadedGenerator,
    seed_raw: np.ndarray,
    num_steps: int,
    alpha: float = 1.0,
    beta_private: float = 1.0,
    leader: str = "lower",
    mode: str = "deterministic",
    stackelberg_beta: float = 1.0,
    temperature: float = 1.0,
    top_k: Optional[int] = 12,
    rng_seed: int = 0,
    device: str = DEVICE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Stackelberg-style leader/follower game generation.

    Candidate set:
        A_U = top-k upper/soprano candidates from the private upper head
        A_L = top-k lower/bass candidates from the private lower head
        candidate pairs = A_U x A_L

    Utilities:
        U_upper = C(a_U, a_L) + alpha        * V_U(a_U)
        U_lower = C(a_U, a_L) + beta_private * V_L(a_L)

    Supported variants:
        leader="lower": bass/lower voice commits first, soprano best-responds.
        leader="upper": soprano/upper voice commits first, bass best-responds.

        mode="deterministic": follower uses argmax response and leader uses
            argmax anticipated payoff. This is the recommended first test.
        mode="soft": follower and leader use softmax policies controlled by
            stackelberg_beta and temperature.

    Research meaning:
        role-structured game solver. The lower-leading variant tests whether
        bass/harmonic grounding provides a better interaction model than the
        simultaneous QRE-style solver.
    """
    rng = np.random.default_rng(rng_seed)
    generated_ids = raw_to_id_sequence(seed_raw, gen.token_to_id)

    step_infos: List[Dict[str, float]] = []

    for _ in range(num_steps):
        s_logits, b_logits, pair_logits = _model_logits(
            gen,
            generated_ids,
            device=device,
        )

        s_actions, b_actions = get_voice_candidate_actions(
            soprano_logits=s_logits,
            bass_logits=b_logits,
            top_k=top_k,
        )

        result = solve_stackelberg_step(
            soprano_logits=s_logits,
            bass_logits=b_logits,
            pair_logits=pair_logits,
            pair_to_id=gen.pair_to_id,
            vocab_size=len(gen.token_to_id),
            alpha=alpha,
            beta_private=beta_private,
            leader=leader,
            mode=mode,
            rng=rng,
            stackelberg_beta=stackelberg_beta,
            temperature=temperature,
            top_k=top_k,
            s_actions=s_actions,
            b_actions=b_actions,
        )

        step_infos.append(result.metrics)

        generated_ids = np.vstack(
            [
                generated_ids,
                np.array(result.pair, dtype=np.int64),
            ]
        )

    raw = id_to_raw_sequence(generated_ids, gen.id_to_token)

    metrics: Dict[str, float] = {}
    if step_infos:
        for key in step_infos[0].keys():
            metrics[f"avg_{key}"] = float(np.mean([x[key] for x in step_infos]))

    return raw, metrics


def save_generation_outputs(
    raw_tokens: np.ndarray,
    output_prefix: str | Path,
    metrics: Optional[Dict[str, float]] = None,
    write_musicxml_file: bool = True,
    tempo_bpm: int = 90,
) -> None:
    """
    Save generation outputs.

    Files:
        .npy         raw token array
        .txt         human-readable raw token text
        .metrics.txt scalar metrics
        .musicxml    optional MusicXML export
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_prefix.with_suffix(".npy"), raw_tokens)
    np.savetxt(output_prefix.with_suffix(".txt"), raw_tokens, fmt="%d")

    if metrics is not None:
        lines = [f"{k}: {v:.6f}" for k, v in sorted(metrics.items())]
        output_prefix.with_suffix(".metrics.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    if write_musicxml_file:
        from two_voice_game.generation.musicxml_export import write_musicxml

        write_musicxml(
            raw_tokens=raw_tokens,
            output_path=output_prefix.with_suffix(".musicxml"),
            tempo_bpm=tempo_bpm,
        )
