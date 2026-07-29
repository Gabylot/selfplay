"""TensorBoard logging for AlphaZero chess engine.

Provides a ``TensorBoardLogger`` class that wraps
``torch.utils.tensorboard.SummaryWriter`` and mirrors the API of
``StatsLogger`` (SQLite) so both can be called side-by-side from the
training loop.

Logs as many metrics as possible:
  - Training losses (policy, value, total, grad norm) + learning rate
  - Per-game stats (result, length, MCTS depth, material diff, termination)
  - Rolling averages (win rate, avg game length over last N games)
  - Histograms (game length, material diff, loss distributions)
  - MCTS stats (avg tree depth, sims per move)
  - Network confidence (avg max policy, avg abs value)
  - Replay buffer composition (size, outcome distribution)
  - Elo ratings (gating, reference)
  - Promotion attempts (win rate, promoted, wins/losses/draws, elo)
  - Evaluation results (win rate, wins/losses/draws per opponent)
  - Termination reason distribution
  - Full config dump + run summary (text)
"""

import json
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Dict, Any

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard logger that mirrors StatsLogger's API.

    All methods are no-ops when ``enabled`` is False, so callers can
    always invoke them without guarding with ``if tb.enabled``.
    """

    def __init__(self, log_dir: str, enabled: bool = True,
                 rolling_window: int = 100, initial_step: int = 0):
        """Initialise the TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard event files.
            enabled: If False, all logging calls become no-ops.
            rolling_window: Number of recent games to track for
                rolling-average metrics.
            initial_step: Step to resume from.  When > 0, the
                ``SummaryWriter`` is told to purge (discard) any events
                at or beyond this step from *previous* event files in
                the same log directory.  This is essential for correct
                training continuation: without it, TensorBoard sees
                overlapping step ranges from the old and new event files
                and the plots appear to restart or jump.
        """
        self.enabled = enabled
        self.log_dir = log_dir
        purge = initial_step if initial_step > 0 else None
        self.writer = SummaryWriter(log_dir, purge_step=purge) if enabled else None
        self.rolling_window = rolling_window

        # Rolling buffers for computed metrics
        self._game_results = deque(maxlen=rolling_window)
        self._game_lengths = deque(maxlen=rolling_window)
        self._material_diffs = deque(maxlen=rolling_window)
        self._termination_reasons: Dict[str, int] = defaultdict(int)
        self._policy_losses = deque(maxlen=rolling_window)
        self._value_losses = deque(maxlen=rolling_window)
        self._total_losses = deque(maxlen=rolling_window)
        self._grad_norms = deque(maxlen=rolling_window)

        # Track last logged game_id for rolling stats
        self._last_game_id = 0

    # ── Internal helper ──────────────────────────────────────────────

    def _log(self, fn, *args, **kwargs):
        """Call a SummaryWriter method only if enabled."""
        if self.writer is not None:
            fn(*args, **kwargs)

    # ── Config / text ────────────────────────────────────────────────

    def log_config(self, step: int, config_dict: dict):
        """Log the full configuration as text."""
        self._log(self.writer.add_text, "Config/full",
                  json.dumps(config_dict, indent=2), step)

    def log_run_summary(self, step: int, summary: str):
        """Log a human-readable run summary as text."""
        self._log(self.writer.add_text, "Summary", summary, step)

    # ── Training ─────────────────────────────────────────────────────

    def log_training_step(self, step: int, policy_loss: float,
                          value_loss: float, total_loss: float,
                          learning_rate: float = None,
                          grad_norm: float = None):
        """Log a training step's losses, learning rate, and gradient norm."""
        self._log(self.writer.add_scalar, "Loss/policy", policy_loss, step)
        self._log(self.writer.add_scalar, "Loss/value", value_loss, step)
        self._log(self.writer.add_scalar, "Loss/total", total_loss, step)
        self._log(self.writer.add_scalars, "Loss/all",
                  {"policy": policy_loss, "value": value_loss,
                   "total": total_loss}, step)

        if learning_rate is not None:
            self._log(self.writer.add_scalar, "LearningRate", learning_rate, step)

        if grad_norm is not None:
            self._log(self.writer.add_scalar, "Gradient/norm", grad_norm, step)

        # Track for rolling histograms
        self._policy_losses.append(policy_loss)
        self._value_losses.append(value_loss)
        self._total_losses.append(total_loss)
        if grad_norm is not None:
            self._grad_norms.append(grad_norm)

        # Periodic histograms of loss distributions
        if step > 0 and step % 50 == 0:
            if len(self._policy_losses) > 1:
                self._log(self.writer.add_histogram,
                          "Histogram/policy_loss",
                          np.array(self._policy_losses), step)
            if len(self._value_losses) > 1:
                self._log(self.writer.add_histogram,
                          "Histogram/value_loss",
                          np.array(self._value_losses), step)
            if len(self._grad_norms) > 1:
                self._log(self.writer.add_histogram,
                          "Histogram/grad_norm",
                          np.array(self._grad_norms), step)

    # ── Games ────────────────────────────────────────────────────────

    def log_game(self, game_id: int, step: int, result: float,
                 result_str: str, length: int, termination: str,
                 avg_mcts_depth: float = 0, num_positions: int = 0,
                 material_diff: int = 0):
        """Log a completed self-play game with rich metrics."""
        self._last_game_id = game_id

        # Per-game scalars
        self._log(self.writer.add_scalar, "Game/result", result, game_id)
        self._log(self.writer.add_scalar, "Game/length", length, game_id)
        self._log(self.writer.add_scalar, "Game/avg_mcts_depth",
                  avg_mcts_depth, game_id)
        self._log(self.writer.add_scalar, "Game/num_positions",
                  num_positions, game_id)
        self._log(self.writer.add_scalar, "Game/material_diff",
                  material_diff, game_id)

        # Text labels
        self._log(self.writer.add_text, "Game/result_str",
                  f"{result_str} (step {step})", game_id)
        self._log(self.writer.add_text, "Game/termination",
                  f"{termination} (step {step})", game_id)

        # Track for rolling stats
        self._game_results.append(result)
        self._game_lengths.append(length)
        self._material_diffs.append(material_diff)
        self._termination_reasons[termination] += 1

        # Rolling averages (need at least 10 games for stability)
        if len(self._game_results) >= 10:
            wins = sum(1 for r in self._game_results if r > 0)
            losses = sum(1 for r in self._game_results if r < 0)
            draws = sum(1 for r in self._game_results if r == 0)
            total = len(self._game_results)
            rolling_wr = (wins + 0.5 * draws) / total
            self._log(self.writer.add_scalar,
                      "Rolling/win_rate", rolling_wr, game_id)
            self._log(self.writer.add_scalar,
                      "Rolling/avg_game_length",
                      float(np.mean(self._game_lengths)), game_id)
            self._log(self.writer.add_scalars, "Rolling/outcome_counts",
                      {"wins": wins, "losses": losses, "draws": draws},
                      game_id)

        # Periodic histograms
        if game_id > 0 and game_id % 50 == 0:
            if len(self._game_lengths) > 1:
                self._log(self.writer.add_histogram,
                          "Histogram/game_length",
                          np.array(self._game_lengths), game_id)
            if len(self._material_diffs) > 1:
                self._log(self.writer.add_histogram,
                          "Histogram/material_diff",
                          np.array(self._material_diffs), game_id)

        # Termination reason distribution (as scalars)
        self._log(self.writer.add_scalars, "Termination/reasons",
                  dict(self._termination_reasons), game_id)

    # ── MCTS ─────────────────────────────────────────────────────────

    def log_mcts_stats(self, game_id: int, step: int,
                       avg_tree_depth: float, avg_sims_per_move: float):
        """Log MCTS search statistics for a game."""
        self._log(self.writer.add_scalar, "MCTS/avg_tree_depth",
                  avg_tree_depth, game_id)
        self._log(self.writer.add_scalar, "MCTS/avg_sims_per_move",
                  avg_sims_per_move, game_id)

    # ── Network ──────────────────────────────────────────────────────

    def log_network_stats(self, step: int, avg_max_policy: float,
                          avg_abs_value: float):
        """Log network confidence trends."""
        self._log(self.writer.add_scalar, "Network/avg_max_policy",
                  avg_max_policy, step)
        self._log(self.writer.add_scalar, "Network/avg_abs_value",
                  avg_abs_value, step)

    # ── Buffer ───────────────────────────────────────────────────────

    def log_buffer_stats(self, step: int, buffer_size: int,
                         white_wins: float, black_wins: float,
                         draws: float):
        """Log replay buffer composition."""
        self._log(self.writer.add_scalar, "Buffer/size", buffer_size, step)
        self._log(self.writer.add_scalars, "Buffer/outcome_distribution",
                  {"white_wins": white_wins, "black_wins": black_wins,
                   "draws": draws}, step)

    # ── Elo ──────────────────────────────────────────────────────────

    def log_elo(self, elo_rating: float, opponent_type: str,
                step: int = None, games_played: int = 0,
                wins: int = 0, losses: int = 0, draws: int = 0):
        """Log an Elo rating update."""
        s = step if step is not None else 0
        self._log(self.writer.add_scalar, f"Elo/{opponent_type}",
                  elo_rating, s)
        self._log(self.writer.add_scalars, f"Elo/{opponent_type}_detail",
                  {"elo": elo_rating, "games": games_played,
                   "wins": wins, "losses": losses, "draws": draws}, s)

    # ── Promotion ────────────────────────────────────────────────────

    def log_promotion_attempt(self, step: int, promoted: bool,
                              win_rate: float, games_played: int,
                              wins: int, losses: int, draws: int,
                              new_elo: float = None, old_elo: float = None):
        """Log a gating/promotion attempt."""
        self._log(self.writer.add_scalar, "Promotion/win_rate",
                  win_rate, step)
        self._log(self.writer.add_scalar, "Promotion/promoted",
                  int(promoted), step)
        self._log(self.writer.add_scalars, "Promotion/results",
                  {"wins": wins, "losses": losses, "draws": draws}, step)
        if new_elo is not None:
            self._log(self.writer.add_scalar, "Promotion/new_elo",
                      new_elo, step)
        if old_elo is not None:
            self._log(self.writer.add_scalar, "Promotion/old_elo",
                      old_elo, step)

    # ── Evaluation ───────────────────────────────────────────────────

    def log_evaluation(self, step: int, opponent: str, games_played: int,
                       wins: int, losses: int, draws: int,
                       win_rate: float = None):
        """Log evaluation match results."""
        if win_rate is None and games_played > 0:
            win_rate = wins / games_played
        self._log(self.writer.add_scalar,
                  f"Evaluation/{opponent}_win_rate", win_rate, step)
        self._log(self.writer.add_scalars, f"Evaluation/{opponent}",
                  {"wins": wins, "losses": losses, "draws": draws}, step)

    # ── Cleanup ──────────────────────────────────────────────────────

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            # Final summary text
            try:
                summary_lines = [
                    f"Total games logged: {self._last_game_id}",
                    f"Rolling window size: {self.rolling_window}",
                ]
                if self._game_results:
                    wins = sum(1 for r in self._game_results if r > 0)
                    losses = sum(1 for r in self._game_results if r < 0)
                    draws = sum(1 for r in self._game_results if r == 0)
                    total = len(self._game_results)
                    wr = (wins + 0.5 * draws) / total if total > 0 else 0
                    summary_lines.append(
                        f"Recent win rate ({total} games): {wr:.2%}"
                    )
                    summary_lines.append(
                        f"Recent avg game length: "
                        f"{np.mean(self._game_lengths):.1f}"
                    )
                self.writer.add_text(
                    "FinalSummary",
                    "\n".join(summary_lines),
                    self._last_game_id,
                )
            except Exception:
                pass
            self.writer.close()
            self.writer = None
