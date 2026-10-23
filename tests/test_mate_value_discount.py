"""Length-discounted checkmate value tests (``selfplay.mate_value_discount``).

In a game that ends in a real checkmate, every position's value target is
``outcome * plies-to-move * gamma ** d`` where ``d`` is the plies from that
position to the mating move (the mating move's own position has d=1).
Draws, cap lengths and adjudicated games are unchanged.  gamma = 1.0
reproduces the flat +/-1 targets exactly.
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
from selfplay import play_one_game

# King-and-queen mate in two, one side each: Kb6 cuts off the escape rank,
# the only black reply is Kb8, then Qh8# -- all moves verified legal through
# the repo's shim board, and the final position is a genuine checkmate.
WHITE_MATE_FEN = "k7/8/8/1K6/7Q/8/8/8 w - - 0 1"
WHITE_SCRIPT = ["b5b6", "a8b8", "h4h8"]

BLACK_MATE_FEN = "K7/8/8/1k6/7q/8/8/8 b - - 0 1"
BLACK_SCRIPT = ["b5b6", "a8b8", "h4h8"]

GAMMA = 0.5  # exaggerated so every depth is numerically distinct


class _FakeRoot:
    def __init__(self, board):
        self.board = board
        self.children = {}


class ScriptedEngine:
    """Plays a fixed move list; search and selection return the same move."""

    def __init__(self, script):
        self.script = list(script)
        self.pending = None
        self.num_simulations = 1

    def get_root(self, board):
        return _FakeRoot(board.copy())

    def _next_move(self, root):
        if self.pending is None:
            self.pending = chess.Move.from_uci(self.script.pop(0))
        return self.pending

    def search(self, root, *args, **kwargs):
        move = self._next_move(root)
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[move_to_policy_index(move, root.board)] = 1.0
        return policy, move, {'avg_depth': 1.0, 'num_simulations': 1}

    def get_root_child_stats(self, root):
        return []

    def recycle_tree(self, root, move):
        return None

    def select_move_with_temperature(self, root, temperature):
        move = self._next_move(root)
        self.pending = None
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[move_to_policy_index(move, root.board)] = 1.0
        return policy, move


def play(script, fen=None, gamma=GAMMA, max_game_length=6):
    engine = ScriptedEngine(script)
    return play_one_game(engine, max_game_length=max_game_length, fen=fen,
                         mate_value_discount=gamma)


def values(game_data):
    return [float(v) for _s, _p, v, _m in game_data]


def discounts(game_data):
    n = len(game_data)
    return [n - i for i in range(n)]


def test_white_mate_targets_are_discounted():
    game_data, info = play(WHITE_SCRIPT, fen=WHITE_MATE_FEN)
    assert info['termination'] == 'checkmate'
    assert info['result'] == 1.0

    vs = values(game_data)
    assert len(vs) == 3
    # Position plies-to-mate: 3, 2, 1; white/black to move: +, -, +.
    assert vs[0] == pytest.approx(GAMMA ** 3)
    assert vs[1] == pytest.approx(-(GAMMA ** 2))
    assert vs[2] == pytest.approx(GAMMA ** 1)


def test_black_mate_targets_are_discounted():
    game_data, info = play(BLACK_SCRIPT, fen=BLACK_MATE_FEN)
    assert info['termination'] == 'checkmate'
    assert info['result'] == -1.0

    vs = values(game_data)
    assert len(vs) == 3
    # Same colour pattern: black-to-move -> (-1 * -1) = +, white -> -.
    assert vs[0] == pytest.approx(GAMMA ** 3)
    assert vs[1] == pytest.approx(-(GAMMA ** 2))
    assert vs[2] == pytest.approx(GAMMA ** 1)


def test_draw_and_cap_games_are_unchanged():
    script = ["e2e4", "e7e5"]
    for gamma in (1.0, 0.5):
        game_data, info = play(script, fen=None, gamma=gamma,
                               max_game_length=2)
        assert info['termination'] != 'checkmate'
        # A capped game by material balance scores 0 for everyone.
        assert all(v == 0.0 for v in values(game_data))


def test_gamma_one_reproduces_old_flat_targets():
    # Flat targets: winner-to-move +1, opponent-to-move -1; both scripted
    # games have the alternating (win side, opponent, win side) pattern.
    for script, fen in ((WHITE_SCRIPT, WHITE_MATE_FEN),
                        (BLACK_SCRIPT, BLACK_MATE_FEN)):
        game_data, info = play(script, fen=fen, gamma=1.0)
        assert info['termination'] == 'checkmate'
        vs = values(game_data)
        assert len(vs) == 3
        assert vs == [1.0, -1.0, 1.0]


def test_winner_and_loser_targets_are_exact_negatives():
    """Mirrored games train identical player-relative targets; the loser's
    to-move positions mirror the winner's, sign-flipped by the parity of
    who is on the move."""
    white_data, info_w = play(WHITE_SCRIPT, fen=WHITE_MATE_FEN)
    black_data, info_b = play(BLACK_SCRIPT, fen=BLACK_MATE_FEN)
    assert info_w['result'] == 1.0
    assert info_b['result'] == -1.0

    wv = values(white_data)
    bv = values(black_data)
    assert len(wv) == len(bv) == 3
    # Mirror symmetry: both games expect (+g^3, -g^2, +g).
    for x, y in zip(wv, bv):
        assert x == y
        assert x != 0.0

    expected = [GAMMA ** 3, -(GAMMA ** 2), GAMMA ** 1]
    assert wv == pytest.approx(expected, rel=1e-6)
    assert bv == pytest.approx(expected, rel=1e-6)


def test_discount_distance_matches_plies_to_mate():
    game_data, info = play(BLACK_SCRIPT, fen=BLACK_MATE_FEN)
    assert discounts(game_data) == [3, 2, 1]
    # The mating move's own position is exactly d=1 away from the mate.
    vs = values(game_data)
    assert vs[2] == pytest.approx(GAMMA ** 1)


# ── forced mate-in-one short-circuit (play_one_game integration) ────────────

_FORCED_STATS = {
    'avg_depth': 0.0, 'max_depth': 0, 'num_simulations': 0,
    'inherited_root_n': 0, 'budget_new_sims': 0, 'target_new_sims': 0,
    'min_fresh_sims': 64, 'reused_root': False, 'floor_fired': False,
    'forced_mate': True,
}


class ForcedNoSearchEngine(ScriptedEngine):
    """Mimics the short-circuited forced-mate engine: search returns the
    one-hot policy, the scripted move and forced, zero-simulation stats."""

    def search(self, root, *args, **kwargs):
        move = self._next_move(root)
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[move_to_policy_index(move, root.board)] = 1.0
        return policy, move, dict(_FORCED_STATS)


@pytest.mark.parametrize("engine_cls", [ScriptedEngine, ForcedNoSearchEngine])
def test_targets_identical_for_forced_and_regular_engine(engine_cls):
    """Stored policy / value targets are the same whether the search ran
    (old behaviour) or the forced mate short-circuited with zero sims."""
    def run(E):
        engine = E(list(WHITE_SCRIPT))
        return play_one_game(engine, max_game_length=6, fen=WHITE_MATE_FEN,
                             mate_value_discount=GAMMA)

    data_old, info_old = run(ScriptedEngine)
    data_new, info_new = run(ForcedNoSearchEngine)

    for (s1, p1, v1, m1), (s2, p2, v2, m2) in zip(data_old, data_new):
        assert v1 == v2                      # value targets: exact
        np.testing.assert_array_equal(p1, p2)   # policy targets: identical
        np.testing.assert_array_equal(m1, m2)   # legal masks: identical
    assert info_old['fresh_sims'] == [1, 1, 1]
    assert info_new['fresh_sims'] == []       # forced moves excluded
    assert info_new['forced_moves'] == 3
    assert info_old['forced_moves'] == 0
    # The discount is untouched: same gamma^d ladder in both runs.
    assert values(data_new) == values(data_old)
    assert tuple(discounts(data_new)) == (3, 2, 1)


def test_forced_move_gui_snapshot_is_valid():
    """The on_move snapshot stays valid for a forced move: move + forced
    flag, empty candidates, zero depths/sims -- nothing crashes."""
    snaps = []
    engine = ForcedNoSearchEngine(list(WHITE_SCRIPT))
    game_data, info = play_one_game(
        engine, max_game_length=6, fen=WHITE_MATE_FEN,
        on_move=lambda fen, uci, mc, md: snaps.append(md))
    assert len(snaps) == 3
    for md in snaps:
        assert md['forced'] is True
        assert md['candidates'] == []
        assert md['sims'] == 0
        assert md['depth'] == 0.0
        assert md['selected_move']    # a move is present
    assert info['termination'] == 'checkmate'


def test_forced_mate_stats_window_metric():
    """Chess/forced_mate_* window aggregate: share over searched+forced."""
    from chess_metrics import forced_mate_stats
    games = [
        {'forced_moves': 2, 'fresh_sims': [10, 20, 30]},   # 3 searched + 2 forced
        {'forced_moves': 0, 'fresh_sims': [5]},
        {'forced_moves': 1, 'fresh_sims': []},
    ]
    out = forced_mate_stats(games)
    assert out['forced_mate_moves_mean'] == pytest.approx(1.0)   # (2+0+1)/3
    # 3 searched + 1 searched + 0 searched = 4 searched, 3 forced => 3/7.
    assert out['forced_mate_share'] == pytest.approx(3.0 / 7.0)
    assert forced_mate_stats([]) == {}
