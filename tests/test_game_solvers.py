import numpy as np

from two_voice_game.game.nash import find_pure_nash_equilibria, solve_nash_step
from two_voice_game.game.qre import softmax_np, solve_qre_step
from two_voice_game.game.stackelberg import solve_stackelberg_step


def _pair_vocab(vocab_size=2):
    return {(i, j): i * vocab_size + j for i in range(vocab_size) for j in range(vocab_size)}


def test_softmax_is_probability_distribution():
    p = softmax_np(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p > 0)


def test_find_pure_nash_coordination_game():
    upper = np.array([[2.0, 0.0], [0.0, 1.0]])
    lower = upper.copy()
    eq = {tuple(x) for x in find_pure_nash_equilibria(upper, lower)}
    assert eq == {(0, 0), (1, 1)}


def test_solvers_return_valid_candidate_pairs():
    vocab_size = 2
    pair_to_id = _pair_vocab(vocab_size)
    soprano = np.array([1.0, 0.2])
    bass = np.array([0.1, 1.2])
    pair = np.array([0.4, 1.0, 0.2, 0.5])
    actions = np.array([0, 1])
    rng = np.random.default_rng(7)

    qre = solve_qre_step(
        soprano, bass, pair, pair_to_id, vocab_size,
        alpha=0.5, beta_private=0.5, qre_beta=0.5, iterations=5,
        rng=rng, s_actions=actions, b_actions=actions,
    )
    stk = solve_stackelberg_step(
        soprano, bass, pair, pair_to_id, vocab_size,
        alpha=0.5, beta_private=0.5, leader="lower", mode="deterministic",
        rng=rng, s_actions=actions, b_actions=actions,
    )
    nash = solve_nash_step(
        soprano, bass, pair, pair_to_id, vocab_size,
        alpha=0.5, beta_private=0.5, s_actions=actions, b_actions=actions,
    )

    valid = {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert qre.pair in valid
    assert stk.pair in valid
    assert nash.pair in valid
