"""Fresh-simulation floor for tree reuse (Python and Rust MCTS).

A recycled root can already hold the full visit budget.  The floor guarantees
that a reused root still gets ``min_fresh_sims`` new simulations, so a seeded
mate pre-visit can be followed by real search instead of ending the search.
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

from chess_metrics import fresh_sim_stats
from encoding import NUM_ACTIONS, move_to_policy_index
from mcts import MCTS
from rust_mcts import RustMCTS


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
QUIET = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_engine(cls, num_simulations=200, min_fresh_sims=0,
                mate_seeding=False):
    return cls(
        network=FixedNet(),
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        batch_size=1,
        mate_seeding=mate_seeding,
        min_fresh_sims=min_fresh_sims,
    )


def root_n(engine, root):
    if isinstance(engine, RustMCTS):
        return engine._N(root.rs_id)
    return root.N


def child_n(engine, root, action):
    child = root.children[action]
    if isinstance(engine, RustMCTS):
        return engine._N(child.rs_id)
    return child.N


def expand(engine, root):
    if isinstance(engine, RustMCTS):
        engine._expand_node(root.rs_id)
    else:
        engine._expand_node(root)


def add_visits(engine, node, value, count):
    if isinstance(engine, RustMCTS):
        for _ in range(count):
            engine._lib.mcts_rs_backprop(node.rs_id, float(value))
    else:
        for _ in range(count):
            engine._backpropagate(node, float(value))


def build_reused_root(cls, fen, mate_uci, other_uci):
    """Build a root with N=199: 6 mate visits + 193 visits on one other move."""
    board = chess.Board(fen)
    engine = make_engine(cls, num_simulations=200, min_fresh_sims=0)
    root = engine.get_root(board)
    expand(engine, root)

    mate_action = move_to_policy_index(chess.Move.from_uci(mate_uci), board)
    other_action = move_to_policy_index(chess.Move.from_uci(other_uci), board)
    assert mate_action in root.children
    assert other_action in root.children
    assert mate_action != other_action

    add_visits(engine, root.children[mate_action], -1.0, 6)
    add_visits(engine, root.children[other_action], 0.0, 193)

    assert root_n(engine, root) == 199
    assert child_n(engine, root, mate_action) == 6
    assert child_n(engine, root, other_action) == 193
    return engine, root, board, mate_action, other_action


@pytest.mark.parametrize("engine_cls", ENGINES)
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("fen,mate_uci,other_uci", [
    (WHITE_MATE, "e1e8", "e1e2"),
    (BLACK_MATE, "e8e1", "e8e7"),
])
def test_reused_root_floor_gives_mate_the_fresh_visits(
        engine_cls, batch_size, fen, mate_uci, other_uci):
    engine, root, board, mate_action, other_action = build_reused_root(
        engine_cls, fen, mate_uci, other_uci)
    before_share = 6 / 199

    engine.batch_size = batch_size
    engine.mate_seeding = True
    engine.min_fresh_sims = 64
    visit_policy, _best_move, stats = engine.search(root)

    assert stats['inherited_root_n'] == 199
    assert stats['reused_root'] is True
    assert stats['budget_new_sims'] == 0
    assert stats['floor_fired'] is True
    assert stats['num_simulations'] == 64

    # Some of the 64 fresh simulations explore other unvisited children
    # first (large sqrt(root.N) exploration term), but the mate must receive
    # most of the new visits and a visibly larger policy share.
    mate_new_visits = child_n(engine, root, mate_action) - 7
    assert mate_new_visits > 32
    assert child_n(engine, root, other_action) == 193
    assert root_n(engine, root) == 200 + 64

    after_share = visit_policy[mate_action]
    assert after_share > before_share + 0.15
    assert after_share > 0.20
    assert visit_policy.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_reused_root_below_budget_is_unchanged(engine_cls):
    def build_and_run(min_fresh_sims):
        engine = make_engine(engine_cls, num_simulations=50,
                             min_fresh_sims=min_fresh_sims)
        board = chess.Board(QUIET)
        root = engine.get_root(board)
        engine.search(root)
        assert root_n(engine, root) == 50

        engine.num_simulations = 200
        policy, _best, stats = engine.search(root)
        return engine, root, policy, stats

    eng0, root0, policy0, stats0 = build_and_run(0)
    eng64, root64, policy64, stats64 = build_and_run(64)

    assert stats0['reused_root'] is True
    assert stats0['budget_new_sims'] == 150
    assert stats0['floor_fired'] is False
    assert stats0['num_simulations'] == 150
    assert stats64['budget_new_sims'] == 150
    assert stats64['floor_fired'] is False
    assert stats64['num_simulations'] == 150
    assert root_n(eng0, root0) == 200
    assert root_n(eng64, root64) == 200
    np.testing.assert_array_equal(policy0, policy64)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_zero_floor_is_legacy_on_fresh_seeded_root(engine_cls):
    def run(min_fresh_sims):
        engine = make_engine(engine_cls, num_simulations=64,
                             min_fresh_sims=min_fresh_sims,
                             mate_seeding=True)
        board = chess.Board(WHITE_MATE)
        root = engine.get_root(board)
        policy, best, stats = engine.search(root)
        return engine, root, policy, best, stats

    eng0, root0, policy0, best0, stats0 = run(0)
    eng64, root64, policy64, best64, stats64 = run(64)

    # Fresh roots are unchanged: seed visit + 63 normal sims = 64.
    assert stats0['inherited_root_n'] == 0
    assert stats0['reused_root'] is False
    assert stats0['floor_fired'] is False
    assert stats0['num_simulations'] == 63
    assert root_n(eng0, root0) == 64
    assert stats64['num_simulations'] == 63
    assert root_n(eng64, root64) == 64
    assert best0.uci() == best64.uci() == "e1e8"
    np.testing.assert_array_equal(policy0, policy64)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_zero_floor_is_legacy_on_reused_root(engine_cls):
    engine, root, _board, mate_action, other_action = build_reused_root(
        engine_cls, WHITE_MATE, "e1e8", "e1e2")

    engine.mate_seeding = True
    engine.min_fresh_sims = 0
    policy, _best_move, stats = engine.search(root)

    assert stats['inherited_root_n'] == 199
    assert stats['reused_root'] is True
    assert stats['budget_new_sims'] == 0
    assert stats['floor_fired'] is False
    assert stats['num_simulations'] == 0
    assert child_n(engine, root, mate_action) == 7
    assert child_n(engine, root, other_action) == 193
    assert root_n(engine, root) == 200
    assert policy[mate_action] == pytest.approx(7 / 200)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_mirrored_reused_roots_give_identical_policies(engine_cls):
    engine_w, root_w, _bw, _mw, _ow = build_reused_root(
        engine_cls, WHITE_MATE, "e1e8", "e1e2")
    engine_b, root_b, _bb, _mb, _ob = build_reused_root(
        engine_cls, BLACK_MATE, "e8e1", "e8e7")

    engine_w.mate_seeding = True
    engine_b.mate_seeding = True
    engine_w.min_fresh_sims = 64
    engine_b.min_fresh_sims = 64
    policy_w, _best_w, stats_w = engine_w.search(root_w)
    policy_b, _best_b, stats_b = engine_b.search(root_b)

    assert stats_w['num_simulations'] == stats_b['num_simulations'] == 64
    np.testing.assert_array_equal(policy_w, policy_b)


def test_fresh_sim_stats_rolling_window():
    stats = fresh_sim_stats([
        {'fresh_sims': [0, 10, 25, 100],
         'floor_fired': [True, False, False, True]},
        {'fresh_sims': [64], 'floor_fired': [True]},
    ])
    assert stats['fresh_sims_mean'] == pytest.approx(39.8)
    assert stats['fresh_sims_p10'] == pytest.approx(4.0)
    assert stats['fresh_floor_fired_share'] == pytest.approx(0.6)
    assert stats['fresh_sims_below25_share'] == pytest.approx(0.4)
    assert stats['fresh_sims_zero_share'] == pytest.approx(0.2)
