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

Restart handling
----------------
Metrics are written on two different x-axes:

  * ``step``   — training / buffer / network / elo / eval / promotion metrics
  * ``game_id``— per-game metrics (Game/*, Rolling/*, MCTS/*, Termination/*)

A single ``SummaryWriter`` can only carry one ``purge_step``, so on restart
we use **two** writers in separate subdirectories (``step/`` and ``game/``),
each purged at its own axis.  Without this, game-keyed events from a stale
checkpoint would overlap the new run's game ids and TensorBoard would show
duplicate / restarting game curves.
"""

import json
import numpy as np
import shutil
from collections import deque, defaultdict
from typing import Optional, Dict, Any
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard logger that mirrors StatsLogger's API.

    All methods are no-ops when ``enabled`` is False, so callers can
    always invoke them without guarding with ``if tb.enabled``.
    """

    def __init__(self, log_dir: str, enabled: bool = True,
                 rolling_window: int = 100, initial_step: int = 0,
                 initial_game_id: int = 0):
        """Initialise the TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard event files.  Two
                subdirectories are created inside it: ``step/`` for
                step-keyed metrics and ``game/`` for game-keyed metrics.
            enabled: If False, all logging calls become no-ops.
            rolling_window: Number of recent games to track for
                rolling-average metrics.
            initial_step: Step to resume from.  When > 0, pre-existing
                event files in the ``step/`` subdirectory are consolidated
                into a single history file holding events strictly before
                this step, so the resumed run continues seamlessly and
                TensorBoard shows the whole run from the start.
            initial_game_id: Game id to resume from.  When > 0, the
                ``game/`` subdirectory is consolidated the same way before
                the game-keyed writer starts.
        """
        self.enabled = enabled
        self.log_dir = log_dir
        self.rolling_window = rolling_window

        if enabled:
            step_dir = Path(log_dir) / "step"
            game_dir = Path(log_dir) / "game"
            purge_step = initial_step if initial_step > 0 else None
            purge_game = initial_game_id if initial_game_id > 0 else None
            # Deterministic restart: consolidate any pre-existing event files
            # into a single history file inside the same directory (keeping
            # events strictly before the resume point), so the fresh writer
            # continues seamlessly and TensorBoard shows the full run from the
            # start. The in-place merge is order-independent and prevents
            # duplicate/restarting curves without losing history.
            self._merge_previous(log_dir, step_dir, purge_step, axis="step")
            self._merge_previous(log_dir, game_dir, purge_game, axis="game")
            self.writer = SummaryWriter(str(step_dir), purge_step=purge_step)
            self.game_writer = SummaryWriter(str(game_dir), purge_step=purge_game)
        else:
            self.writer = None
            self.game_writer = None

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

    def _log(self, writer, method, *args, **kwargs):
        """Call a SummaryWriter method only if enabled.

        ``method`` is the method name (e.g. ``"add_scalar"``); it is looked
        up lazily so that a ``None`` writer never triggers an attribute
        access on ``None``.
        """
        if writer is not None:
            getattr(writer, method)(*args, **kwargs)

    @staticmethod
    def _merge_previous(log_dir: str, axis_dir: Path, purge_at, axis: str = "axis"):
        """Consolidate previous event files into one clean history file.

        On a resume (``purge_at`` set), read every pre-existing
        ``events.out.tfevents.*`` under ``step/``/``game/`` — both at the
        top level and inside the per-tag subdirectories that ``add_scalars``
        creates (e.g. ``step/Loss_all_policy/``) — and rewrite them as a
        single merged history file *inside the same directory*, keeping only
        events strictly before the resume point (events at/after it will be
        re-logged by the resumed run).  The fresh ``SummaryWriter`` then
        appends a new file alongside the history, so TensorBoard shows the
        full run from the very first step with no duplicate or restarting
        curves — deterministically, independent of file-name ordering.

        The history is deduplicated per ``(step, tag)`` keeping the newest
        value, so even overlapping files (e.g. two worker PIDs that logged
        the same steps before the checkpoint) collapse to a single clean
        series.  Writer-metadata events (``file_version`` / ``session_log``)
        are dropped; the new run writes its own.

        If merging fails for any reason, the old files are archived aside
        (the previous behaviour) so the resumed run never coexists with
        unmerged leftovers.
        """
        d = Path(axis_dir)
        if purge_at is None or not d.exists():
            return
        # The axis directory itself plus every immediate subdirectory it
        # contains (add_scalars writes one subdirectory per scalar key).
        targets = [d] + sorted(x for x in d.iterdir() if x.is_dir())
        for target in targets:
            TensorBoardLogger._merge_dir(log_dir, target, purge_at, axis)

    @staticmethod
    def _merge_dir(log_dir: str, d: Path, purge_at, axis: str):
        """Merge all event files in a single directory into one history file."""
        files = sorted(f for f in d.glob("events.out.tfevents.*") if f.is_file())
        if not files:
            return
        try:
            from tensorboard.backend.event_processing.event_file_loader import (
                EventFileLoader,
            )
            from tensorboard.summary.writer.record_writer import RecordWriter

            # Read every file, keeping the newest event per (step, tag).
            best = {}  # (step, tag) -> (wall_time, event)
            for f in files:
                try:
                    loader = EventFileLoader(str(f))
                    events = list(loader.Load())
                except Exception as exc:
                    print(f"[TB] merge: skipping unreadable file {f.name}: {exc}")
                    continue
                for ev in events:
                    step = ev.step
                    if step >= purge_at:
                        continue
                    what = ev.WhichOneof("what")
                    if what in ("file_version", "session_log", "source_metadata"):
                        continue
                    wt = ev.wall_time
                    for v in ev.summary.value:
                        key = (step, v.tag)
                        if key not in best or best[key][0] < wt:
                            best[key] = (wt, ev)

            if not best:
                # Nothing below the resume point survives: clear the axis dir.
                for f in files:
                    f.unlink()
                print(f"[TB] cleared {len(files)} previous event file(s) from "
                      f"{d.name}/ (nothing below resume point {purge_at})")
                return

            # Emit each winning event once, in a deterministic order.
            seen = set()
            kept = []
            for (wt, ev) in sorted(best.values(), key=lambda kv: (kv[0], kv[1].step)):
                key = (ev.step, tuple(v.tag for v in ev.summary.value),
                       ev.WhichOneof("what"))
                if key in seen:
                    continue
                seen.add(key)
                kept.append(ev)

            ts = int(min(ev.wall_time for ev in kept)) or 1
            # Never reuse an existing filename.  The timestamp is derived from
            # the earliest event, so a previous resume produces the *same* name
            # and that file is one of ``files`` below.  Reusing it would make
            # the cleanup loop delete the history we just wrote (and silently
            # wipe every restart's TensorBoard history).
            suffix = 0
            while True:
                tag = "" if suffix == 0 else f".{suffix}"
                merged_name = (f"events.out.tfevents.{ts:010d}.Peter."
                               f"merged{tag}.{axis}.0")
                merged_path = d / merged_name
                if not merged_path.exists():
                    break
                suffix += 1
            with open(merged_path, "wb") as fp:
                rw = RecordWriter(fp)
                for ev in kept:
                    rw.write(ev.SerializeToString())
                rw.close()
            for f in files:
                if f == merged_path:
                    continue
                f.unlink()
            print(f"[TB] merged {len(files)} previous event file(s) from "
                  f"{d}/ -> {merged_name} ({len(kept)} events < {purge_at})")
        except Exception as exc:
            print(f"[TB] merge failed ({exc}); archiving previous files instead")
            TensorBoardLogger._archive_dir(log_dir, d)

    @staticmethod
    def _archive_previous(log_dir: str, axis_dir: Path, purge_at):
        """Fallback: move existing event files aside before resuming.

        Used if the in-place merge fails.  Relocates every pre-existing
        ``events.out.tfevents.*`` out of ``step/``/``game/`` — including
        the per-tag ``add_scalars`` subdirectories — to a sibling
        ``<log_dir>_tb_archive/<axis>/`` (and ``.../<axis>/<subdir>/``)
        folder that sits outside the run tree TensorBoard scans.
        """
        d = Path(axis_dir)
        if purge_at is None or not d.exists():
            return
        targets = [d] + sorted(x for x in d.iterdir() if x.is_dir())
        for target in targets:
            TensorBoardLogger._archive_dir(log_dir, target)

    @staticmethod
    def _archive_dir(log_dir: str, d: Path):
        arch_root = Path(log_dir).with_name(Path(log_dir).name + "_tb_archive")
        rel = d.relative_to(Path(log_dir))
        arch = arch_root / rel
        arch.mkdir(parents=True, exist_ok=True)
        moved = 0
        for f in d.glob("events.out.tfevents.*"):
            if f.is_file():
                shutil.move(str(f), str(arch / f.name))
                moved += 1
        if moved:
            print(f"[TB] archived {moved} previous event file(s) from {d} -> {arch}")

    # ── Config / text ────────────────────────────────────────────────

    def log_config(self, step: int, config_dict: dict):
        """Log the full configuration as text."""
        self._log(self.writer, "add_text", "Config/full",
                  json.dumps(config_dict, indent=2), step)

    def log_run_summary(self, step: int, summary: str):
        """Log a human-readable run summary as text."""
        self._log(self.writer, "add_text", "Summary", summary, step)

    # ── Training ─────────────────────────────────────────────────────

    def log_training_step(self, step: int, policy_loss: float,
                          value_loss: float, total_loss: float,
                          learning_rate: float = None,
                          grad_norm: float = None):
        """Log a training step's losses, learning rate, and gradient norm."""
        self._log(self.writer, "add_scalar", "Loss/policy", policy_loss, step)
        self._log(self.writer, "add_scalar", "Loss/value", value_loss, step)
        self._log(self.writer, "add_scalar", "Loss/total", total_loss, step)
        self._log(self.writer, "add_scalars", "Loss/all",
                  {"policy": policy_loss, "value": value_loss,
                   "total": total_loss}, step)

        if learning_rate is not None:
            self._log(self.writer, "add_scalar", "LearningRate", learning_rate, step)

        if grad_norm is not None:
            self._log(self.writer, "add_scalar", "Gradient/norm", grad_norm, step)

        # Track for rolling histograms
        self._policy_losses.append(policy_loss)
        self._value_losses.append(value_loss)
        self._total_losses.append(total_loss)
        if grad_norm is not None:
            self._grad_norms.append(grad_norm)

        # Periodic histograms of loss distributions
        if step > 0 and step % 50 == 0:
            if len(self._policy_losses) > 1:
                self._log(self.writer, "add_histogram",
                          "Histogram/policy_loss",
                          np.array(self._policy_losses), step)
            if len(self._value_losses) > 1:
                self._log(self.writer, "add_histogram",
                          "Histogram/value_loss",
                          np.array(self._value_losses), step)
            if len(self._grad_norms) > 1:
                self._log(self.writer, "add_histogram",
                          "Histogram/grad_norm",
                          np.array(self._grad_norms), step)

    # ── Games ────────────────────────────────────────────────────────

    def log_game(self, game_id: int, step: int, result: float,
                 result_str: str, length: int, termination: str,
                 avg_mcts_depth: float = 0, num_positions: int = 0,
                 material_diff: int = 0):
        """Log a completed self-play game with rich metrics."""
        self._last_game_id = game_id

        # Per-game scalars (game-keyed writer)
        self._log(self.game_writer, "add_scalar", "Game/result", result, game_id)
        self._log(self.game_writer, "add_scalar", "Game/length", length, game_id)
        self._log(self.game_writer, "add_scalar", "Game/avg_mcts_depth",
                  avg_mcts_depth, game_id)
        self._log(self.game_writer, "add_scalar", "Game/num_positions",
                  num_positions, game_id)
        self._log(self.game_writer, "add_scalar", "Game/material_diff",
                  material_diff, game_id)

        # Text labels
        self._log(self.game_writer, "add_text", "Game/result_str",
                  f"{result_str} (step {step})", game_id)
        self._log(self.game_writer, "add_text", "Game/termination",
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
            self._log(self.game_writer, "add_scalar",
                      "Rolling/win_rate", rolling_wr, game_id)
            self._log(self.game_writer, "add_scalar",
                      "Rolling/avg_game_length",
                      float(np.mean(self._game_lengths)), game_id)
            self._log(self.game_writer, "add_scalars", "Rolling/outcome_counts",
                      {"wins": wins, "losses": losses, "draws": draws},
                      game_id)

        # Periodic histograms
        if game_id > 0 and game_id % 50 == 0:
            if len(self._game_lengths) > 1:
                self._log(self.game_writer, "add_histogram",
                          "Histogram/game_length",
                          np.array(self._game_lengths), game_id)
            if len(self._material_diffs) > 1:
                self._log(self.game_writer, "add_histogram",
                          "Histogram/material_diff",
                          np.array(self._material_diffs), game_id)

        # Termination reason distribution (as scalars)
        self._log(self.game_writer, "add_scalars", "Termination/reasons",
                  dict(self._termination_reasons), game_id)

    # ── MCTS ─────────────────────────────────────────────────────────

    def log_mcts_stats(self, game_id: int, step: int,
                       avg_tree_depth: float, avg_sims_per_move: float):
        """Log MCTS search statistics for a game."""
        self._log(self.game_writer, "add_scalar", "MCTS/avg_tree_depth",
                  avg_tree_depth, game_id)
        self._log(self.game_writer, "add_scalar", "MCTS/avg_sims_per_move",
                  avg_sims_per_move, game_id)

    # ── Network ──────────────────────────────────────────────────────

    def log_network_stats(self, step: int, avg_max_policy: float,
                          avg_abs_value: float):
        """Log network confidence trends.

        ``avg_max_policy`` is the mean top *legal-move* probability (masked and
        renormalised over legal moves)."""
        self._log(self.writer, "add_scalar", "Network/avg_max_policy",
                  avg_max_policy, step)
        self._log(self.writer, "add_scalar", "Network/avg_abs_value",
                  avg_abs_value, step)

    # ── Chess quality (main-process rolling metrics) ─────────────────

    def log_chess(self, scalars: dict, x: int, axis: str = "game"):
        """Log a batch of ``Chess/<name>`` scalars.

        ``axis`` selects the x-axis/writer: ``"game"`` uses the game-keyed
        writer (rolling per-game metrics) and ``"step"`` the step-keyed writer
        (checkpoint / evaluation events).  ``None`` values are skipped.
        """
        writer = self.game_writer if axis == "game" else self.writer
        for name, value in scalars.items():
            if value is None:
                continue
            self._log(writer, "add_scalar", f"Chess/{name}", float(value), x)

    # ── Buffer ───────────────────────────────────────────────────────

    def log_buffer_stats(self, step: int, buffer_size: int,
                         white_wins: float, black_wins: float,
                         draws: float):
        """Log replay buffer composition."""
        self._log(self.writer, "add_scalar", "Buffer/size", buffer_size, step)
        self._log(self.writer, "add_scalars", "Buffer/outcome_distribution",
                  {"white_wins": white_wins, "black_wins": black_wins,
                   "draws": draws}, step)

    # ── Elo ──────────────────────────────────────────────────────────

    def log_elo(self, elo_rating: float, opponent_type: str,
                step: int = None, games_played: int = 0,
                wins: int = 0, losses: int = 0, draws: int = 0):
        """Log an Elo rating update."""
        s = step if step is not None else 0
        self._log(self.writer, "add_scalar", f"Elo/{opponent_type}",
                  elo_rating, s)
        self._log(self.writer, "add_scalars", f"Elo/{opponent_type}_detail",
                  {"elo": elo_rating, "games": games_played,
                   "wins": wins, "losses": losses, "draws": draws}, s)

    # ── Promotion ────────────────────────────────────────────────────

    def log_promotion_attempt(self, step: int, promoted: bool,
                              win_rate: float, games_played: int,
                              wins: int, losses: int, draws: int,
                              new_elo: float = None, old_elo: float = None):
        """Log a gating/promotion attempt."""
        self._log(self.writer, "add_scalar", "Promotion/win_rate",
                  win_rate, step)
        self._log(self.writer, "add_scalar", "Promotion/promoted",
                  int(promoted), step)
        self._log(self.writer, "add_scalars", "Promotion/results",
                  {"wins": wins, "losses": losses, "draws": draws}, step)
        if new_elo is not None:
            self._log(self.writer, "add_scalar", "Promotion/new_elo",
                      new_elo, step)
        if old_elo is not None:
            self._log(self.writer, "add_scalar", "Promotion/old_elo",
                      old_elo, step)

    # ── Evaluation ───────────────────────────────────────────────────

    def log_evaluation(self, step: int, opponent: str, games_played: int,
                       wins: int, losses: int, draws: int,
                       win_rate: float = None):
        """Log evaluation match results."""
        if win_rate is None and games_played > 0:
            win_rate = wins / games_played
        self._log(self.writer, "add_scalar",
                  f"Evaluation/{opponent}_win_rate", win_rate, step)
        self._log(self.writer, "add_scalars", f"Evaluation/{opponent}",
                  {"wins": wins, "losses": losses, "draws": draws}, step)

    # ── Cleanup ──────────────────────────────────────────────────────

    def close(self):
        """Close the TensorBoard writers."""
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
        if self.game_writer is not None:
            self.game_writer.close()
            self.game_writer = None