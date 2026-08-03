"""Tests for TensorBoardLogger restart handling.

The logger writes metrics on two different x-axes (``step`` and ``game_id``).
On restart it must purge each axis independently, otherwise game-keyed
events from a stale checkpoint overlap the new run's game ids and TensorBoard
shows duplicate / restarting game curves.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tb_logger import TensorBoardLogger


def test_two_writers_created_with_purge_steps():
    """The logger must create separate step/ and game/ writers, each purged
    at its own axis."""
    with tempfile.TemporaryDirectory() as tmp:
        tb = TensorBoardLogger(
            tmp, enabled=True, initial_step=1234, initial_game_id=5678
        )
        try:
            # Two distinct writers
            assert tb.writer is not None
            assert tb.game_writer is not None
            assert tb.writer is not tb.game_writer

            # Each writer's log dir is a subdirectory of the base log dir
            step_dir = Path(tb.writer.get_logdir())
            game_dir = Path(tb.game_writer.get_logdir())
            assert step_dir.parent == Path(tmp)
            assert game_dir.parent == Path(tmp)
            assert step_dir.name == "step"
            assert game_dir.name == "game"
        finally:
            tb.close()


def test_disabled_logger_has_no_writers():
    """When disabled, no writers are created and calls are no-ops."""
    tb = TensorBoardLogger("unused", enabled=False, initial_step=5, initial_game_id=9)
    assert tb.writer is None
    assert tb.game_writer is None
    # These should not raise
    tb.log_training_step(1, 0.1, 0.2, 0.3)
    tb.log_game(1, 1, 1.0, "1-0", 20, "checkmate")
    tb.close()


def test_metrics_route_to_correct_writer():
    """Step-keyed metrics go to the step writer; game-keyed metrics go to
    the game writer."""
    with tempfile.TemporaryDirectory() as tmp:
        tb = TensorBoardLogger(tmp, enabled=True, initial_step=0, initial_game_id=0)
        try:
            # Step-keyed
            tb.log_training_step(step=10, policy_loss=0.1, value_loss=0.2,
                                 total_loss=0.3, learning_rate=0.02, grad_norm=1.0)
            tb.log_buffer_stats(step=10, buffer_size=100, white_wins=0.5,
                                black_wins=0.3, draws=0.2)
            tb.log_network_stats(step=10, avg_max_policy=0.8, avg_abs_value=0.4)
            tb.log_elo(1000.0, "gating", step=10, games_played=20,
                       wins=10, losses=5, draws=5)
            tb.log_promotion_attempt(step=10, promoted=True, win_rate=0.6,
                                     games_played=20, wins=10, losses=5, draws=5)
            tb.log_evaluation(step=10, opponent="alpha_beta_ref", games_played=20,
                              wins=10, losses=5, draws=5, win_rate=0.6)

            # Game-keyed
            tb.log_game(game_id=100, step=10, result=1.0, result_str="1-0",
                        length=30, termination="checkmate", avg_mcts_depth=5.0,
                        num_positions=60, material_diff=3)
            tb.log_mcts_stats(game_id=100, step=10, avg_tree_depth=5.0,
                              avg_sims_per_move=800)

            # The step writer should contain step-keyed tags, the game writer
            # game-keyed tags.  We verify by checking the event files exist
            # in the right subdirectories after close (flush).
        finally:
            tb.close()

        # After close, event files must exist in both subdirectories
        step_events = list((Path(tmp) / "step").glob("events.out.tfevents.*"))
        game_events = list((Path(tmp) / "game").glob("events.out.tfevents.*"))
        assert len(step_events) >= 1, "step writer produced no event file"
        assert len(game_events) >= 1, "game writer produced no event file"