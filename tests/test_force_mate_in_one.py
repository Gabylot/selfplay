"""mate-in-one override tests for the Python and Rust MCTS engines.

With ``force_mate_in_one`` enabled, a root position that has an immediate
checkmate must return that move with a one-hot visit policy and as the
selected move at every temperature -- regardless of visit counts or tree
reuse.  With several mating moves, the first one in canonical policy-index
order wins.
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


class SaturatingNet(FixedNet):
    """Value head stuck near +1.0, like the trained net on quiet positions.

    This is the premise of the observed bug: with a saturating value head,
    the mate child loses the visit vote against quiet moves.
    """

    def predict(self, state):
        return _PRIORS.copy(), 0.99

    def predictBatch(self, states):
        n = len(states)
        return np.tile(_PRIORS, (n, 1)), np.full(n, 0.99, dtype=np.float32)


class CountedNet(SaturatingNet):
    """Saturating network that counts every forward pass."""

    def __init__(self):
        self.calls = 0

    def predict(self, state):
        self.calls += 1
        return super().predict(state)

    def predictBatch(self, states):
        self.calls += 1
        return super().predictBatch(states)


ENGINES = [MCTS, RustMCTS]

WHITE_MATE = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"
BLACK_MATE = "4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1"
WHITE_MULTI = "6k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1"
BLACK_MULTI = "r3r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
QUIET = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_engine(cls, force_mate_in_one, num_simulations=200, net_cls=FixedNet):
    return cls(
        network=net_cls(),
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        batch_size=1,
        mate_seeding=True,
        force_mate_in_one=force_mate_in_one,
    )


def run_search(cls, fen, force_mate_in_one, num_simulations=200,
               net_cls=FixedNet):
    board = chess.Board(fen)
    engine = make_engine(cls, force_mate_in_one, num_simulations, net_cls)
    root = engine.get_root(board)
    visit_policy, best_move, _ = engine.search(root)
    return engine, board, root, visit_policy, best_move


def mate_indices(board):
    out = []
    for move in board.legal_moves:
        probe = board.copy()
        probe.push(move)
        if probe.is_checkmate():
            out.append(move_to_policy_index(move, board))
    return sorted(out)


def assert_forced(engine, board, root, visit_policy, best_move):
    """Full contract: search, then selection at every temperature."""
    actions = mate_indices(board)
    expected = actions[0]

    np.testing.assert_allclose(visit_policy.sum(), 1.0)
    assert np.count_nonzero(visit_policy) == 1
    assert visit_policy[expected] == 1.0

    for move in board.legal_moves:
        if move_to_policy_index(move, board) == expected:
            forced_uci = move.uci()
            break
    # The shim Move has no __eq__; compare canonical UCI strings.
    assert best_move.uci() == forced_uci

    for temperature in (1.0, 0.5, 0.1, 0.0):
        sel_policy, sel_move = engine.select_move_with_temperature(
            root, temperature)
        np.testing.assert_allclose(sel_policy.sum(), 1.0)
        assert np.count_nonzero(sel_policy) == 1
        assert sel_policy[expected] == 1.0
        assert sel_move.uci() == forced_uci


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_white_mate_is_forced(engine_cls):
    engine, board, root, visit_policy, best_move = run_search(
        engine_cls, WHITE_MATE, force_mate_in_one=True)
    assert_forced(engine, board, root, visit_policy, best_move)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_black_mate_is_forced(engine_cls):
    engine, board, root, visit_policy, best_move = run_search(
        engine_cls, BLACK_MATE, force_mate_in_one=True)
    assert_forced(engine, board, root, visit_policy, best_move)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_mirrored_mates_share_one_hot_index(engine_cls):
    _, _, _, wp, _ = run_search(engine_cls, WHITE_MATE, True)
    _, _, _, bp, _ = run_search(engine_cls, BLACK_MATE, True)
    white_idx = int(np.flatnonzero(wp)[0])
    black_idx = int(np.flatnonzero(bp)[0])
    assert white_idx == black_idx
    assert white_idx in mate_indices(chess.Board(WHITE_MATE))


def root_n(engine, root):
    if isinstance(engine, RustMCTS):
        return engine._N(root.rs_id)
    return root.N


def children_n(engine, root):
    if isinstance(engine, RustMCTS):
        return {a: engine._N(c.rs_id) for a, c in root.children.items()}
    return {a: c.N for a, c in root.children.items()}


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_forced_shortcircuit_runs_zero_sims(engine_cls):
    """A forced root must return instantly: no simulations, no net call,
    and the root stays unexpanded."""
    for fen, uci in ((WHITE_MATE, "e1e8"), (BLACK_MATE, "e8e1")):
        board = chess.Board(fen)
        engine = make_engine(engine_cls, force_mate_in_one=True,
                             num_simulations=200, net_cls=CountedNet)
        root = engine.get_root(board)
        visit_policy, best_move, stats = engine.search(root)
        assert engine.network.calls == 0
        assert stats['num_simulations'] == 0
        assert stats['forced_mate'] is True
        assert stats['avg_depth'] == 0.0
        assert root_n(engine, root) == 0
        assert not root.is_expanded
        assert_forced(engine, board, root, visit_policy, best_move)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_forced_repeated_search_is_idempotent(engine_cls):
    """Re-searching the same forced root: still instant, same move."""
    engine = make_engine(engine_cls, force_mate_in_one=True,
                         num_simulations=200, net_cls=CountedNet)
    board = chess.Board(WHITE_MATE)
    root = engine.get_root(board)
    for _ in range(3):
        vp, mv, st = engine.search(root)
        assert st['num_simulations'] == 0
        assert st['forced_mate'] is True
    assert engine.network.calls == 0
    assert_forced(engine, board, root, vp, mv)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_forced_on_reused_expanded_root(engine_cls):
    """Reused root with ~199 visits on quiet children: still forced, with
    no additional simulations or network calls."""
    engine = make_engine(engine_cls, force_mate_in_one=False,
                         num_simulations=199, net_cls=CountedNet)
    board = chess.Board(WHITE_MATE)
    root = engine.get_root(board)
    visit_policy, best_move, stats = engine.search(root)
    n_before = root_n(engine, root)
    calls_before = engine.network.calls
    # Expanded tree distributed ~199 visits over the (non-mating) children.
    assert n_before >= 199

    engine.force_mate_in_one = True
    visit_policy, best_move, stats = engine.search(root)
    assert stats['forced_mate'] is True
    assert stats['num_simulations'] == 0
    assert root_n(engine, root) == n_before      # visits unchanged
    assert engine.network.calls == calls_before  # no net call
    assert_forced(engine, board, root, visit_policy, best_move)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_multiple_mates_use_lowest_policy_index(engine_cls):
    for fen in (WHITE_MULTI, BLACK_MULTI):
        engine, board, root, visit_policy, best_move = run_search(
            engine_cls, fen, force_mate_in_one=True)
        actions = mate_indices(board)
        assert len(actions) >= 2
        expected = actions[0]
        assert int(np.flatnonzero(visit_policy)[0]) == expected
        assert visit_policy[expected] == 1.0
        _, sel_move = engine.select_move_with_temperature(root, 1.0)
        assert move_to_policy_index(sel_move, board) == expected


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_no_mate_position_is_unchanged(engine_cls):
    engine_off, _, root_off, policy_off, move_off = run_search(
        engine_cls, QUIET, force_mate_in_one=False)
    engine_on, _, root_on, policy_on, move_on = run_search(
        engine_cls, QUIET, force_mate_in_one=True)

    np.testing.assert_array_equal(policy_off, policy_on)
    assert move_off == move_on
    # Normal visit distributions stay spread out over many moves.
    assert np.count_nonzero(policy_on) > 5
    assert not np.isclose(policy_on.max(), 1.0)
    sampled, _ = engine_on.select_move_with_temperature(root_on, 1.0)
    np.testing.assert_allclose(sampled, policy_on, atol=1e-6)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_flag_off_leaves_search_distribution(engine_cls):
    """Without the flag nothing is one-hot, even at mating positions."""
    engine, board, root, visit_policy, _ = run_search(
        engine_cls, WHITE_MATE, force_mate_in_one=False, num_simulations=200)
    assert visit_policy.sum() == pytest.approx(1.0)
    assert not np.isclose(visit_policy.max(), 1.0)
    sel_policy, _ = engine.select_move_with_temperature(root, 1.0)
    assert not np.isclose(sel_policy.max(), 1.0)
