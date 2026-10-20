"""TensorBoardLogger.log_chess writes Chess/* scalars on the chosen axis."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

torch = pytest.importorskip("torch")

from tb_logger import TensorBoardLogger


def _read_tags(log_dir, sub):
    from tensorboard.backend.event_processing.event_file_loader import (
        EventFileLoader,
    )
    tags = set()
    for root, _dirs, files in os.walk(os.path.join(log_dir, sub)):
        for f in files:
            if not f.startswith("events.out.tfevents."):
                continue
            for ev in EventFileLoader(os.path.join(root, f)).Load():
                for v in ev.summary.value:
                    tags.add(v.tag)
    return tags


def test_log_chess_step_axis():
    with tempfile.TemporaryDirectory() as d:
        tb = TensorBoardLogger(d, enabled=True)
        tb.log_chess({"white_share_overall": 0.55,
                      "root_kl_visit_prior": 0.1,
                      "skip_me": None}, 42, axis="step")
        tb.close()
        tags = _read_tags(d, "step")
    assert "Chess/white_share_overall" in tags
    assert "Chess/root_kl_visit_prior" in tags
    assert "Chess/skip_me" not in tags


def test_log_chess_game_axis():
    with tempfile.TemporaryDirectory() as d:
        tb = TensorBoardLogger(d, enabled=True)
        tb.log_chess({"length_median": 90.0}, 7, axis="game")
        tb.close()
        tags = _read_tags(d, "game")
    assert "Chess/length_median" in tags
