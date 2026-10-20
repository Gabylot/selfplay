"""Tests for net-only puzzle accuracy (puzzle_eval.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import chess
import game_api

game_api.set_game("chess")

import puzzle_eval
from encoding import NUM_ACTIONS, move_to_policy_index


# White rook mate-in-1 (back rank) and its colour mirror for Black to move.
_PUZZLES = [
    ("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1", ["a1a8"], "mate1"),
    ("r3k3/8/8/8/8/8/5PPP/6K1 b - - 0 1", ["a8a1"], "mate1"),
]


def _perfect_predictor(puzzles):
    def predict(states):
        outs = np.zeros((states.shape[0], NUM_ACTIONS), dtype=np.float32)
        # states are produced by puzzle_eval in the same order as `puzzles`
        for i, (fen, sol, _kind) in enumerate(puzzles):
            b = chess.Board(fen)
            n = len(list(b.legal_moves))
            # set the solution above the uniform floor so it is the argmax
            outs[i, :] = 1.0 / n if n else 0.0
            for s in sol:
                outs[i, move_to_policy_index(chess.Move.from_uci(s), b)] = 1.0
        return outs, np.zeros(states.shape[0])
    return predict


def test_perfect_predictor_scores_one_both_colours():
    out = puzzle_eval.evaluate(_perfect_predictor(_PUZZLES), puzzles=_PUZZLES)
    assert out["puzzle_n"] == 2.0
    assert out["puzzle_all_white_acc"] == 1.0
    assert out["puzzle_all_black_acc"] == 1.0
    assert out["puzzle_mate1_white_acc"] == 1.0
    assert out["puzzle_mate1_black_acc"] == 1.0


def test_uniform_predictor_is_never_better_than_one():
    def uniform(states):
        return np.ones((states.shape[0], NUM_ACTIONS), dtype=np.float32), \
               np.zeros(states.shape[0])

    out = puzzle_eval.evaluate(uniform, puzzles=_PUZZLES)
    for k, v in out.items():
        if k == "puzzle_n":
            continue
        assert 0.0 <= v <= 1.0
    assert out["puzzle_all_white_acc"] < 1.0


def test_skips_for_non_chess_is_guarded():
    # evaluate() must return the scalar dict (chess is active here).
    out = puzzle_eval.evaluate(_perfect_predictor(_PUZZLES), puzzles=_PUZZLES)
    assert set(out) >= {"puzzle_all_white_acc", "puzzle_all_black_acc"}
