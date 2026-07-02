"""Match state tracker for model-vs-model matches with real-time GUI updates.

SocketIO events emitted:
  'match_state'        — Full match state (scores, progress, current game)
  'match_game_update'  — Move-by-move update of the current game
  'match_complete'     — Match finished with final results
"""

import threading
from typing import List, Optional, Dict, Any
from collections import deque


class MatchState:
    """Thread-safe match state tracker with SocketIO push events."""

    def __init__(self, socketio=None, max_history: int = 50):
        self._lock = threading.Lock()
        self._socketio = socketio

        # Match info
        self._model_a_name: str = "Model A"
        self._model_b_name: str = "Model B"
        self._total_games: int = 0
        self._completed_games: int = 0
        self._wins_a: int = 0
        self._wins_b: int = 0
        self._draws: int = 0
        self._status: str = "idle"  # idle | running | complete

        # Current game state
        self._game_number: int = 0
        self._game_label: str = ""
        self._a_is_white: bool = True
        self._moves: List[str] = []
        self._fens: List[str] = []
        self._mcts_stats: List[dict] = []
        self._result: Optional[str] = None
        self._termination: Optional[str] = None
        self._move_number: int = 0

        # Completed games history
        self._game_history: deque = deque(maxlen=max_history)

    # ── Public API called by match.py ─────────────────────────────────────

    def set_socketio(self, socketio):
        self._socketio = socketio

    def set_match_info(self, model_a: str, model_b: str,
                       total_games: int):
        """Set match metadata before starting."""
        with self._lock:
            self._model_a_name = model_a
            self._model_b_name = model_b
            self._total_games = total_games
            self._status = "idle"
            self._wins_a = 0
            self._wins_b = 0
            self._draws = 0
            self._completed_games = 0
        self._emit_state()

    def start_game(self, game_number: int, game_type: str = "match",
                   game_label: str = "", model_a_name: str = None,
                   model_b_name: str = None, a_is_white: bool = True):
        """Called when a new game starts (live_start event)."""
        with self._lock:
            if self._moves:
                self._save_game()
            self._game_number = game_number
            self._game_label = game_label
            self._a_is_white = a_is_white if a_is_white is not None else True
            self._moves = []
            self._fens = []
            self._mcts_stats = []
            self._result = None
            self._termination = None
            self._move_number = 0
            self._status = "running"
            if model_a_name:
                self._model_a_name = model_a_name
            if model_b_name:
                self._model_b_name = model_b_name
        self._emit_state()

    def update_game(self, board_fen: str, move_uci: str, move_number: int,
                    mcts_stats: dict = None):
        """Called on each move (live_move event)."""
        with self._lock:
            self._moves.append(move_uci)
            self._fens.append(board_fen)
            self._move_number = move_number
            if mcts_stats is not None:
                self._mcts_stats.append(mcts_stats)
        self._emit_game_update()

    def end_game(self, result: str, termination: str):
        """Called when the current game ends (live_end event)."""
        with self._lock:
            self._result = result
            self._termination = termination
        self._emit_game_update()

    def replay_full_game(self, fens, moves, mcts_stats=None):
        """Replay a completed game on the match board."""
        with self._lock:
            self._fens = list(fens)
            self._moves = list(moves)
            self._move_number = len(moves)
            if mcts_stats:
                self._mcts_stats = list(mcts_stats)
            self._save_game()
        self._emit_state()

    def update_scores(self, wins_a, wins_b, draws):
        """Update match scores."""
        with self._lock:
            self._wins_a = wins_a
            self._wins_b = wins_b
            self._draws = draws
            self._completed_games = wins_a + wins_b + draws
        self._emit_state()

    def set_complete(self):
        """Mark the match as complete."""
        with self._lock:
            self._status = "complete"
        self._emit_complete()

    def get_state(self):
        """Get full match state (thread-safe)."""
        with self._lock:
            return self._build_state()

    def get_game_history(self):
        """Get list of completed games."""
        with self._lock:
            return list(self._game_history)

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_state(self):
        """Build full state dict. Must be called with lock held."""
        current_fen = self._fens[-1] if self._fens else (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        return {
            'model_a': self._model_a_name,
            'model_b': self._model_b_name,
            'total_games': self._total_games,
            'completed_games': self._completed_games,
            'wins_a': self._wins_a,
            'wins_b': self._wins_b,
            'draws': self._draws,
            'status': self._status,
            'game_number': self._game_number,
            'game_label': self._game_label,
            'a_is_white': self._a_is_white,
            'moves': list(self._moves),
            'fens': list(self._fens),
            'move_number': self._move_number,
            'current_fen': current_fen,
            'last_move': self._moves[-1] if self._moves else None,
            'result': self._result,
            'termination': self._termination,
        }

    def _save_game(self):
        """Save current game to history. Must be called with lock held."""
        if not self._moves:
            return
        self._game_history.appendleft({
            'game_number': self._game_number,
            'game_label': self._game_label,
            'a_is_white': self._a_is_white,
            'moves': list(self._moves),
            'fens': list(self._fens),
            'result': self._result,
            'termination': self._termination,
            'num_moves': len(self._moves),
        })

    def _emit_state(self):
        """Push full match state to browser."""
        if self._socketio is None:
            return
        with self._lock:
            state = self._build_state()
        self._socketio.emit('match_state', state)

    def _emit_game_update(self):
        """Push current game move update to browser."""
        if self._socketio is None:
            return
        with self._lock:
            state = self._build_state()
        self._socketio.emit('match_game_update', state)

    def _emit_complete(self):
        """Push match complete event."""
        if self._socketio is None:
            return
        with self._lock:
            state = self._build_state()
        self._socketio.emit('match_complete', state)
