from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from two_voice_game.data import ChoraleNextStepDataset, build_pair_vocab, build_vocab
from two_voice_game.features.duration import (
    DURATION_FEATURE_DIM,
    DURATION_FEATURE_NAMES,
    HOLD_TOKEN_RAW,
    build_duration_feature_tensor_from_context,
)
from two_voice_game.models import MultiHeadMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ["data", "output", "model", "training"]:
        if key not in cfg:
            raise ValueError(f"Missing required config section: {key}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_ctx16.yaml")
    parser.add_argument(
        "--duration_aware_state",
        dest="duration_aware_state",
        action="store_true",
        default=None,
        help="Override config and train with duration-aware state features.",
    )
    parser.add_argument(
        "--no_duration_aware_state",
        dest="duration_aware_state",
        action="store_false",
        help="Override config and train with the original context + pos state only.",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return np.load(path, allow_pickle=True)


def prepare_model_config(cfg: dict, cli_duration_aware_state: bool | None) -> dict:
    """Normalize model config and record duration-aware metadata in checkpoints."""
    cfg = copy.deepcopy(cfg)
    model_cfg = cfg.setdefault("model", {})

    duration_aware_state = bool(model_cfg.get("duration_aware_state", False))
    if cli_duration_aware_state is not None:
        duration_aware_state = bool(cli_duration_aware_state)

    context_len = int(model_cfg["context_len"])
    duration_feature_normalizer = float(model_cfg.get("duration_feature_normalizer", context_len))

    model_cfg["duration_aware_state"] = duration_aware_state
    model_cfg["duration_feature_dim"] = DURATION_FEATURE_DIM if duration_aware_state else 0
    model_cfg["duration_feature_names"] = list(DURATION_FEATURE_NAMES) if duration_aware_state else []
    model_cfg["duration_feature_normalizer"] = duration_feature_normalizer if duration_aware_state else None
    model_cfg["duration_feature_source"] = "fixed_context_window" if duration_aware_state else None
    model_cfg["hold_token_raw"] = HOLD_TOKEN_RAW if duration_aware_state else None

    return cfg


def _duration_features_for_batch(
    context: torch.Tensor,
    hold_token_id: int | None,
    duration_aware_state: bool,
    duration_feature_normalizer: float | int | None,
) -> torch.Tensor | None:
    if not duration_aware_state:
        return None
    if hold_token_id is None:
        raise ValueError("duration_aware_state=True requires HOLD=-2 to exist in token_to_id.")
    return build_duration_feature_tensor_from_context(
        context=context,
        hold_token_id=int(hold_token_id),
        normalizer=duration_feature_normalizer,
    )


def run_epoch(
    model,
    loader,
    optimizer=None,
    *,
    hold_token_id: int | None = None,
    duration_aware_state: bool = False,
    duration_feature_normalizer: float | int | None = None,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_s_correct = 0
    total_b_correct = 0
    total_pair_correct = 0
    total_count = 0

    for batch in loader:
        context = batch["context"].to(DEVICE)
        y_s = batch["y_s"].to(DEVICE)
        y_b = batch["y_b"].to(DEVICE)
        y_pair = batch["y_pair"].to(DEVICE)
        pos = batch["pos"].to(DEVICE)
        duration_features = _duration_features_for_batch(
            context=context,
            hold_token_id=hold_token_id,
            duration_aware_state=duration_aware_state,
            duration_feature_normalizer=duration_feature_normalizer,
        )

        if is_train:
            optimizer.zero_grad()

        soprano_logits, bass_logits, pair_logits = model(context, pos, duration_features)
        loss_s = ce(soprano_logits, y_s)
        loss_b = ce(bass_logits, y_b)
        loss_pair = ce(pair_logits, y_pair)
        loss = loss_s + loss_b + loss_pair

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = context.shape[0]
        total_loss += loss.item() * batch_size
        total_s_correct += (soprano_logits.argmax(dim=-1) == y_s).sum().item()
        total_b_correct += (bass_logits.argmax(dim=-1) == y_b).sum().item()
        total_pair_correct += (pair_logits.argmax(dim=-1) == y_pair).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "soprano_acc": total_s_correct / total_count,
        "bass_acc": total_b_correct / total_count,
        "pair_acc": total_pair_correct / total_count,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    valid_metrics: dict,
    config: dict,
    token_to_id: Dict[int, int],
    id_to_token: Dict[int, int],
    pair_to_id: Dict[Tuple[int, int], int],
    id_to_pair: Dict[int, Tuple[int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_metrics": valid_metrics,
        "config": config,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "pair_to_id": pair_to_id,
        "id_to_pair": id_to_pair,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    cfg = prepare_model_config(load_config(args.config), args.duration_aware_state)

    data_path = Path(cfg["data"]["path"])
    save_dir = Path(cfg["output"]["save_dir"])
    context_len = int(cfg["model"]["context_len"])
    emb_dim = int(cfg["model"]["emb_dim"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    dropout = float(cfg["model"]["dropout"])
    duration_aware_state = bool(cfg["model"].get("duration_aware_state", False))
    duration_feature_dim = int(cfg["model"].get("duration_feature_dim", 0))
    duration_feature_normalizer = cfg["model"].get("duration_feature_normalizer", None)
    batch_size = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    learning_rate = float(cfg["training"]["learning_rate"])
    seed = int(cfg["training"]["seed"])

    set_seed(seed)

    print(f"Using config: {args.config}")
    print(f"Using device: {DEVICE}")
    print(f"Data path: {data_path}")
    print(f"Save dir: {save_dir}")
    print(f"Context length: {context_len}")
    print(f"Duration-aware state: {duration_aware_state}")
    if duration_aware_state:
        print(f"Duration feature dim: {duration_feature_dim}")
        print(f"Duration feature normalizer: {duration_feature_normalizer}")
        print(f"Duration features: {', '.join(DURATION_FEATURE_NAMES)}")

    data = load_npz(data_path)
    train_pieces = data["train_tokens"]
    valid_pieces = data["valid_tokens"]

    token_to_id, id_to_token = build_vocab(train_pieces)
    pair_to_id, id_to_pair = build_pair_vocab(token_to_id)
    hold_token_id = token_to_id.get(HOLD_TOKEN_RAW)

    if duration_aware_state and hold_token_id is None:
        raise KeyError(
            f"duration_aware_state=True, but raw HOLD token {HOLD_TOKEN_RAW} "
            "is not in the training vocabulary."
        )

    print(f"Train pieces: {len(train_pieces)}")
    print(f"Valid pieces: {len(valid_pieces)}")
    print(f"Token vocab size: {len(token_to_id)}")
    print(f"Pair vocab size: {len(pair_to_id)}")
    if duration_aware_state:
        print(f"HOLD raw token {HOLD_TOKEN_RAW} -> token id {hold_token_id}")

    train_ds = ChoraleNextStepDataset(train_pieces, token_to_id, pair_to_id, context_len)
    valid_ds = ChoraleNextStepDataset(valid_pieces, token_to_id, pair_to_id, context_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MultiHeadMLP(
        vocab_size=len(token_to_id),
        pair_vocab_size=len(pair_to_id),
        context_len=context_len,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        duration_feature_dim=duration_feature_dim,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_loss = float("inf")
    best_epoch = -1
    print("\nStart training...\n")

    epoch_kwargs = {
        "hold_token_id": hold_token_id,
        "duration_aware_state": duration_aware_state,
        "duration_feature_normalizer": duration_feature_normalizer,
    }

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, **epoch_kwargs)
        valid_metrics = run_epoch(model, valid_loader, optimizer=None, **epoch_kwargs)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f}, "
            f"S={train_metrics['soprano_acc']:.3f}, "
            f"B={train_metrics['bass_acc']:.3f}, "
            f"P={train_metrics['pair_acc']:.3f} | "
            f"valid loss={valid_metrics['loss']:.4f}, "
            f"S={valid_metrics['soprano_acc']:.3f}, "
            f"B={valid_metrics['bass_acc']:.3f}, "
            f"P={valid_metrics['pair_acc']:.3f}"
        )
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                save_dir / "best.pt",
                model,
                optimizer,
                epoch,
                valid_metrics,
                cfg,
                token_to_id,
                id_to_token,
                pair_to_id,
                id_to_pair,
            )
            print(f"  Saved best checkpoint to {save_dir / 'best.pt'}")

    save_checkpoint(
        save_dir / "last.pt",
        model,
        optimizer,
        epochs,
        valid_metrics,
        cfg,
        token_to_id,
        id_to_token,
        pair_to_id,
        id_to_pair,
    )
    print(f"\nSaved last checkpoint to {save_dir / 'last.pt'}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best valid loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
