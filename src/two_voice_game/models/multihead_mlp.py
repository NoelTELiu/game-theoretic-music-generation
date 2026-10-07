from __future__ import annotations

import torch
from torch import nn


class MultiHeadMLP(nn.Module):
    """
    Multi-head MLP utility/scoring model.

    Inputs:
        context:           [batch, context_len, 2]
        pos:               [batch], position in bar, usually t % 16
        duration_features: optional [batch, duration_feature_dim]

    Outputs:
        soprano_logits: [batch, vocab_size]
        bass_logits:    [batch, vocab_size]
        pair_logits:    [batch, pair_vocab_size]

    Backward compatibility:
        duration_feature_dim defaults to 0, so old checkpoints trained with
        context + pos only keep the same architecture and state_dict shape.
    """

    def __init__(
        self,
        vocab_size: int,
        pair_vocab_size: int,
        context_len: int,
        emb_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        num_positions: int = 16,
        duration_feature_dim: int = 0,
    ):
        super().__init__()
        self.context_len = int(context_len)
        self.duration_feature_dim = int(duration_feature_dim)
        if self.duration_feature_dim < 0:
            raise ValueError("duration_feature_dim must be >= 0")

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(num_positions, emb_dim)

        input_dim = self.context_len * 2 * emb_dim + emb_dim + self.duration_feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.soprano_head = nn.Linear(hidden_dim, vocab_size)
        self.bass_head = nn.Linear(hidden_dim, vocab_size)
        self.pair_head = nn.Linear(hidden_dim, pair_vocab_size)

    def forward(
        self,
        context: torch.Tensor,
        pos: torch.Tensor,
        duration_features: torch.Tensor | None = None,
    ):
        batch_size = context.shape[0]
        x = self.token_emb(context).reshape(batch_size, -1)
        p = self.pos_emb(pos)
        inputs = [x, p]

        if self.duration_feature_dim > 0:
            if duration_features is None:
                raise ValueError(
                    "This MultiHeadMLP was created with duration_feature_dim="
                    f"{self.duration_feature_dim}, but duration_features=None was passed."
                )
            if duration_features.ndim != 2:
                raise ValueError(
                    "duration_features must have shape [batch, duration_feature_dim], "
                    f"got {tuple(duration_features.shape)}"
                )
            if duration_features.shape[0] != batch_size:
                raise ValueError(
                    "duration_features batch size does not match context batch size: "
                    f"{duration_features.shape[0]} != {batch_size}"
                )
            if duration_features.shape[1] != self.duration_feature_dim:
                raise ValueError(
                    "duration_features last dimension does not match model duration_feature_dim: "
                    f"{duration_features.shape[1]} != {self.duration_feature_dim}"
                )
            inputs.append(duration_features.to(device=context.device, dtype=x.dtype))

        h = self.encoder(torch.cat(inputs, dim=-1))
        return self.soprano_head(h), self.bass_head(h), self.pair_head(h)
