from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def build_vocab(train_pieces) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build token vocabulary from the training split only."""
    tokens = set()
    for piece in train_pieces:
        arr = np.asarray(piece)
        for x in arr.flatten():
            tokens.add(int(x))

    token_list = sorted(tokens)
    token_to_id = {tok: i for i, tok in enumerate(token_list)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return token_to_id, id_to_token


def build_pair_vocab(token_to_id: Dict[int, int]):
    """
    Build pair vocabulary as the Cartesian product of all token ids.

    Pair class = (soprano_token_id, bass_token_id)
    """
    token_ids = sorted(token_to_id.values())
    pair_list = [(s_id, b_id) for s_id in token_ids for b_id in token_ids]
    pair_to_id = {pair: i for i, pair in enumerate(pair_list)}
    id_to_pair = {i: pair for pair, i in pair_to_id.items()}
    return pair_to_id, id_to_pair
