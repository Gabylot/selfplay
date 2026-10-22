"""Root mate-seeding tests for the Python and Rust MCTS engines.

Mate seeding gives every root child that delivers checkmate one terminal
pre-visit after root expansion.  It must not replace the normal visit policy,
move choice, or temperature sampling.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import chess
import numpy as np
import pytest

from encoding import NUM_ACTIONS, move_to_policy_index
from mcts import MCTS
from rust_mcts import RustMCTS


# State-independent priors make mirrored positions produce identical trees.
_PRIORS = (np.arange(NUM_ACTIONS, dtype=np.float32) % 97) + 1.0
_PRIORS /= _PRIORS.sum()


class FixedNet:
    """Deterministic network: fixed policy, neutral value."""

    def predict(self, state):
        return _PRIORS.copy(), 0.0

    def predictBatch(self, states):
        n = len(states)
        return np.tile(_PRIORS, (n, 1)), np.zeros(n, dtype=np.float32)


ENGINES = [MCTS, RustMCTS]

WHITE_MATE = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"
BLACK_MATE = "4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1"
WHITE_MULTI = "6k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1"
BLACK_MULTI = "r3r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
QUIET = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_engine(cls, mate_seeding, num_simulations=64):
    return cls(
        network=FixedNet(),
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        batch_size=1,
        mate_seeding=mate_seeding,
    )


def run_search(cls, fen, mate_seeding, num_simulations=64):
    board = chess.Board(fen)
    engine = make_engine(cls, mate_seeding, num_simulations)
    root = engine.get_root(board)
    visit_policy, best_move, _ = engine.search(root)
    return engine, board, root, visit_policy, best_move


def root_n(engine, root):
    if isinstance(engine, RustMCTS):
        return engine._N(root.rs_id)
    return root.N


def child_n(engine, root, action):
    child = root.children[action]
    if isinstance(engine, RustMCTS):
        return engine._N(child.rs_id)
    return child.N


def mate_indices(board):
    out = []
    for move in board.legal_moves:
        probe = board.copy()
        probe.push(move)
        if probe.is_checkmate():
            out.append(move_to_policy_index(move, board))
    return out


@pytest.mark.parametrize("engine_cls", ENGINES)
@pytest.mark.parametrize("fen,mate_uci", [
    (WHITE_MATE, "e1e8"),
    (BLACK_MATE, "e8e1"),
])
def test_white_and_black_mate_is_seeded(engine_cls, fen, mate_uci):
    engine, board, root, visit_policy, _ = run_search(
        engine_cls, fen, mate_seeding=True)

    action = move_to_policy_index(chess.Move.from_uci(mate_uci), board)
    assert child_n(engine, root, action) >= 1
    assert visit_policy[action] > 0.0
    assert root_n(engine, root) == 64


@pytest.mark.parametrize("engine_cls", ENGINES)
@pytest.mark.parametrize("fen", [WHITE_MULTI, BLACK_MULTI])
def test_multiple_mates_each_get_a_seed(engine_cls, fen):
    engine, board, root, visit_policy, _ = run_search(
        engine_cls, fen, mate_seeding=True)

    actions = mate_indices(board)
    assert len(actions) >= 2
    for action in actions:
        assert child_n(engine, root, action) >= 1
        assert visit_policy[action] > 0.0
    assert root_n(engine, root) == 64


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_mirrored_mates_have_identical_visit_policies(engine_cls):
    _, _, _, white_policy, _ = run_search(engine_cls, WHITE_MATE, True)
    _, _, _, black_policy, _ = run_search(engine_cls, BLACK_MATE, True)
    np.testing.assert_array_equal(white_policy, black_policy)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_non_mate_position_is_unchanged(engine_cls):
    engine_off, _, root_off, policy_off, _ = run_search(
        engine_cls, QUIET, mate_seeding=False)
    engine_on, _, root_on, policy_on, _ = run_search(
        engine_cls, QUIET, mate_seeding=True)

    np.testing.assert_array_equal(policy_off, policy_on)
    assert root_n(engine_off, root_off) == root_n(engine_on, root_on)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_temperature_selection_uses_visit_distribution(engine_cls):
    engine, board, root, visit_policy, _ = run_search(
        engine_cls, WHITE_MULTI, mate_seeding=True)

    sampled_policy, move = engine.select_move_with_temperature(root, 1.0)
    np.testing.assert_allclose(sampled_policy, visit_policy, atol=1e-6)
    assert move is not None
    # The old override returned a one-hot mate policy here; seeding must not.
    assert not np.isclose(sampled_policy.max(), 1.0)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_normal_visit_policy_is_still_used(engine_cls):
    engine, board, root, visit_policy, best_move = run_search(
        engine_cls, WHITE_MULTI, mate_seeding=True)

    actions = mate_indices(board)
    assert len(actions) >= 2
    assert np.isclose(visit_policy.sum(), 1.0)
    # Seeding does not force a single mate move to one-hot.
    assert not np.isclose(visit_policy[actions[0]], 1.0)
    assert best_move is not None
