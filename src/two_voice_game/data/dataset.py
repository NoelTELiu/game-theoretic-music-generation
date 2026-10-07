from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class ChoraleNextStepDataset(Dataset):
    """
    Next-step prediction dataset.

    For each piece and time t:
        input  = piece[t - context_len : t], shape [context_len, 2]
        target = piece[t]

    Context windows never cross piece boundaries.
    """

    def __init__(
        self,
        pieces,
        token_to_id: Dict[int, int],
        pair_to_id: Dict[Tuple[int, int], int],
        context_len: int = 16,
    ):
        self.samples = []
        self.context_len = context_len
        self.token_to_id = token_to_id
        self.pair_to_id = pair_to_id

        for piece_idx, piece in enumerate(pieces):
            arr = np.asarray(piece, dtype=np.int64)
            if len(arr) <= context_len:
                continue

            token_id_arr = np.zeros_like(arr, dtype=np.int64)
            for t in range(len(arr)):
                raw_s = int(arr[t, 0])
                raw_b = int(arr[t, 1])
                if raw_s not in token_to_id:
                    raise KeyError(f"Soprano token {raw_s} in piece {piece_idx} is not in training vocabulary.")
                if raw_b not in token_to_id:
                    raise KeyError(f"Bass token {raw_b} in piece {piece_idx} is not in training vocabulary.")
                token_id_arr[t, 0] = token_to_id[raw_s]
                token_id_arr[t, 1] = token_to_id[raw_b]

            for t in range(context_len, len(arr)):
                context = token_id_arr[t - context_len:t]
                y_s = int(token_id_arr[t, 0])
                y_b = int(token_id_arr[t, 1])
                y_pair = pair_to_id[(y_s, y_b)]
                pos_in_bar = t % 16
                self.samples.append((context, y_s, y_b, y_pair, pos_in_bar))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, y_s, y_b, y_pair, pos_in_bar = self.samples[idx]
        return {
            "context": torch.tensor(context, dtype=torch.long),
            "y_s": torch.tensor(y_s, dtype=torch.long),
            "y_b": torch.tensor(y_b, dtype=torch.long),
            "y_pair": torch.tensor(y_pair, dtype=torch.long),
            "pos": torch.tensor(pos_in_bar, dtype=torch.long),
        }
