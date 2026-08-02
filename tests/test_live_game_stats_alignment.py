"""Regression test: MCTS stats must stay index-aligned with moves.

Bug: ``LiveGameState.update()`` silently dropped ``None`` mcts_stats entries
(e.g. alpha-beta moves in reference games have no MCTS statistics).  This made
``mcts_stats_per_move`` shorter than ``moves``, so the GUI frontend stopped
showing MCTS stats for all subsequent moves once its index ran past the end of
the shorter list.

Fix: always append to ``_mcts_stats`` (even ``None``) so the lists stay
index-aligned.  The frontend already renders ``None`` entries gracefully.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.live_game import LiveGameState


def _mk_stats(move_uci):
    return {
        'selected_move': move_uci,
        'candidates': [
            {'move': move_uci, 'N': 42, 'W': 1.5, 'Q': 0.5, 'P': 0.25},
        ],
    }


def test_stats_stay_aligned_with_mixed_none_and_dict():
    """MCTS stats list length must always equal moves length."""
    lg = LiveGameState()  # no socketio → _emit() is a no-op

    lg.start_game(1, 0)

    lg.update('fen1', 'e2e4', 1, mcts_stats=_mk_stats('e2e4'))
    # Alpha-beta move — no MCTS stats (this is the None case from reference games)
    lg.update('fen2', 'e7e5', 2, mcts_stats=None)
    lg.update('fen3', 'g1f3', 3, mcts_stats=_mk_stats('g1f3'))
    lg.update('fen4', 'b8c6', 4, mcts_stats=None)

    st = lg.get_state()

    assert len(st['moves']) == 4
    assert len(st['mcts_stats_per_move']) == 4, \
        "mcts_stats_per_move must be same length as moves"
    assert st['mcts_stats_per_move'][0] is not None
    assert st['mcts_stats_per_move'][1] is None
    assert st['mcts_stats_per_move'][2] is not None
    assert st['mcts_stats_per_move'][3] is None
    print("  PASS: test_stats_stay_aligned_with_mixed_none_and_dict")


def test_stats_aligned_in_saved_game_history():
    """Completed (saved) games must also have aligned stats lists."""
    lg = LiveGameState()

    lg.start_game(7, 0)
    lg.update('fen1', 'd2d4', 1, mcts_stats=_mk_stats('d2d4'))
    lg.update('fen2', 'd7d5', 2, mcts_stats=None)
    lg.game_over('1-0', 'checkmate')

    hist = lg.get_game_history()
    assert len(hist) == 1
    g = hist[0]
    assert len(g['moves']) == 2
    assert len(g['mcts_stats_per_move']) == 2, \
        "saved game mcts_stats_per_move must be same length as moves"
    assert g['mcts_stats_per_move'][0] is not None
    assert g['mcts_stats_per_move'][1] is None
    print("  PASS: test_stats_aligned_in_saved_game_history")


def test_stats_aligned_all_none():
    """Even when every move has no stats, alignment must hold."""
    lg = LiveGameState()

    lg.start_game(2, 0)
    lg.update('fen1', 'e2e4', 1, mcts_stats=None)
    lg.update('fen2', 'e7e5', 2, mcts_stats=None)
    lg.update('fen3', 'g1f3', 3, mcts_stats=None)

    st = lg.get_state()
    assert len(st['moves']) == 3
    assert len(st['mcts_stats_per_move']) == 3
    assert all(s is None for s in st['mcts_stats_per_move'])
    print("  PASS: test_stats_aligned_all_none")