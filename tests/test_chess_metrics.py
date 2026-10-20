"""Tests for the main-process rolling chess metrics (chess_metrics.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import chess_metrics as cm


# ── white share / termination mix / length ──────────────────────────────────

def _g(result, termination, length):
    return {"result": result, "termination": termination, "length": length}


def test_white_share_splits_by_termination_family():
    games = [
        _g(1.0, "checkmate", 30),
        _g(-1.0, "checkmate", 30),
        _g(1.0, "checkmate", 30),
        _g(1.0, "material_white", 200),
        _g(-1.0, "material_black", 200),
        _g(0.0, "stalemate", 80),      # draw -> excluded
    ]
    ws = cm.white_share(games)
    assert ws["white_share_overall"] == pytest.approx(3 / 5)   # 3 white / 5 decisive
    assert ws["white_share_checkmate"] == pytest.approx(2 / 3)
    assert ws["white_share_material"] == pytest.approx(1 / 2)


def test_white_share_none_when_no_decisive_games():
    ws = cm.white_share([_g(0.0, "stalemate", 50)])
    assert ws["white_share_overall"] is None
    assert ws["white_share_checkmate"] is None


def test_termination_mix_sums_to_one():
    games = [_g(1.0, "checkmate", 10), _g(-1.0, "checkmate", 10),
             _g(0.0, "max_length", 10)]
    mix = cm.termination_mix(games)
    assert abs(sum(mix.values()) - 1.0) < 1e-12
    assert mix["termination_share_checkmate"] == pytest.approx(2 / 3)


def test_length_percentiles():
    games = [_g(0.0, "draw", L) for L in (10, 20, 30, 40, 50)]
    ls = cm.length_stats(games)
    assert ls["length_median"] == pytest.approx(30.0)
    assert ls["length_p10"] == pytest.approx(14.0)
    assert ls["length_p90"] == pytest.approx(46.0)


def test_empty_inputs():
    assert cm.white_share([])["white_share_overall"] is None
    assert cm.termination_mix([]) == {}
    assert cm.length_stats([]) == {}


# ── policy KL / entropy ─────────────────────────────────────────────────────

def test_entropy_of_uniform_is_log_n():
    prior = np.ones((1, 4))
    mask = np.ones((1, 4))
    out = cm.policy_kl_entropy(np.zeros((1, 4)), prior, mask)
    assert out["masked_policy_entropy"] == pytest.approx(np.log(4.0))


def test_kl_zero_for_identical_distributions():
    t = np.array([[0.5, 0.3, 0.2]])
    m = np.ones((1, 3))
    out = cm.policy_kl_entropy(t, t.copy(), m)
    assert out["root_kl_visit_prior"] == pytest.approx(0.0, abs=1e-12)


def test_kl_skips_one_hot_targets():
    onehot = np.array([[1.0, 0.0, 0.0]])
    m = np.ones((1, 3))
    out = cm.policy_kl_entropy(onehot, np.ones((1, 3)) / 3, m)
    assert "root_kl_visit_prior" not in out
    assert "masked_policy_entropy" in out


def test_mask_restricts_prior():
    prior = np.array([[0.6, 0.2, 0.2]])
    mask = np.array([[1.0, 1.0, 0.0]])          # exclude the 3rd action
    out = cm.policy_kl_entropy(np.zeros((1, 3)), prior, mask)
    # masked + renormalised: [0.75, 0.25] -> entropy
    expected = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    assert out["masked_policy_entropy"] == pytest.approx(expected)


# ── value error by phase ────────────────────────────────────────────────────

def test_value_error_by_phase():
    def predict(chunk):
        # constant prediction 0.0 -> error equals |target|
        return np.zeros(chunk.shape[0])

    recs = [
        [(20, np.zeros((1, 1), np.float16), 1.0),
         (60, np.zeros((1, 1), np.float16), -1.0)],
        [(20, np.zeros((1, 1), np.float16), 0.0)],
    ]
    out = cm.value_error_by_phase(recs, predict)
    assert out["value_error_ply20"] == pytest.approx(0.5)
    assert out["value_error_ply60"] == pytest.approx(1.0)
    assert "value_error_ply100" not in out


# ── RollingChessStats bookkeeping ───────────────────────────────────────────

def test_rolling_stats_captures_phase_positions_in_order():
    roll = cm.RollingChessStats(window=5, log_every=2, phase_plies=(1, 3))
    data = [(np.full((1, 1), i, np.float16), None, float(i), None)
            for i in range(5)]
    roll.add_game(_g(1.0, "checkmate", 5), data)
    assert not roll.due()          # only 1 game
    roll.add_game(_g(0.0, "draw", 5), data)
    assert roll.due()              # 2 games -> due
    # phase records captured (ply, state, target)
    rec = roll._phase[0]
    assert [p for p, _, _ in rec] == [1, 3]
    # phase ply k stores game_data[k-1]; value there == k-1
    assert [float(t) for _, _, t in rec] == [0.0, 2.0]
    roll.reset_counter()
    assert not roll.due()


def test_compute_runs_with_callables():
    roll = cm.RollingChessStats(window=10, log_every=1, phase_plies=(1,))
    roll.add_game(_g(1.0, "checkmate", 1),
                  [(np.zeros((1, 1), np.float16), None, 1.0, None)])
    A = 3
    res = roll.compute(
        predict_batch=lambda s: (np.full((s.shape[0], A), 1.0 / A),
                                 np.zeros(s.shape[0])),
        sample_batch=lambda n: (np.zeros((n, 1, 1), np.float32),
                                np.full((n, A), 1.0 / A),
                                np.zeros(n), np.ones((n, A))))
    assert res["white_share_overall"] == 1.0
    assert "masked_policy_entropy" in res
    assert res["value_error_ply1"] == pytest.approx(1.0)
