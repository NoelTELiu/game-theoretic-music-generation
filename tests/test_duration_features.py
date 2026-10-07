import torch

from two_voice_game.features.duration import build_duration_feature_tensor_from_context


def test_duration_features_from_visible_context():
    # HOLD token id = 0. Upper voice ends with two HOLDs; lower ends with one.
    context = torch.tensor([
        [1, 2],
        [0, 2],
        [0, 0],
    ])
    features = build_duration_feature_tensor_from_context(context, hold_token_id=0, normalizer=3)
    expected = torch.tensor([2/3, 1/3, 1.0, 2/3, 1.0, 1.0], dtype=torch.float32)
    assert torch.allclose(features, expected, atol=1e-6)
