"""Live game state for real-time self-play and eval viewing.

Each LiveGameState instance is associated with a worker_id so the GUI
can route push events to the correct grid tile and the expanded viewer.

Push events emitted:
  'worker_tile_update'   — compact state for the grid thumbnail (every move)
  'worker_detail_update' — full state for the expanded viewer (every move,
                           only consumed when that worker's tile is open)
  'eval_live_game_update'— full state for the eval board (eval games only)
"""

import threading
from typing import List, Optional, Dict, Any
from collections import deque


class LiveGameState:
    """Thread-safe live game state. One instance per worker (or eval board)."""

    def __init__(self, socketio=None, max_history: int = 20,
                 worker_id: int = -1, is_eval: bool = False):
        """
        Args:
            socketio:   Flask-SocketIO instance for push events.
            max_history: Max completed games to keep for replay.
            worker_id:  Index of the worker this state belongs to (-1 = unset).
            is_eval:    If True, emits eval-specific events instead of worker events.
        """
        self._lock       = threading.Lock()
        self._socketio   = socketio
        self._worker_id  = worker_id
        self._is_eval    = is_eval

        # Current game state
        self._game_id:    int          = 0
        self._step:       int          = 0
        self._moves:      List[str]    = []
        self._fens:       List[str]    = []
        self._start_fen:  str          = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self._status:     str          = "idle"
        self._result:     Optional[str]= None
        self._termination:Optional[str]= None
        self._move_number:int          = 0
        self._mcts_stats: List[dict]   = []
        self._game_type:  str          = "selfplay"
        self._match_info: Optional[str]= None

        # Completed games history
        self._game_history: deque = deque(maxlen=max_history)

    # ── Public API called by training/eval loops ──────────────────────────

    def start_game(self, game_id: int, step: int,
                   game_type: str = "selfplay", match_info: str = None):
        with self._lock:
            if self._status == "playing" and self._moves:
                self._save_game()
            self._game_id     = game_id
            self._step        = step
            self._moves       = []
            self._fens        = []
            self._mcts_stats  = []
            self._start_fen   = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            self._status      = "playing"
            self._result      = None
            self._termination = None
            self._move_number = 0
            self._game_type   = game_type
            self._match_info  = match_info
        self._emit()

    def update(self, board_fen: str, move_uci: str, move_number: int,
               mcts_stats: dict = None):
        with self._lock:
            self._moves.append(move_uci)
            self._fens.append(board_fen)
            self._move_number = move_number
            if mcts_stats is not None:
                self._mcts_stats.append(mcts_stats)
        self._emit()

    def game_over(self, result: str, termination: str):
        with self._lock:
            self._result      = result
            self._termination = termination
            self._status      = "finished"
            self._save_game()
        self._emit()

    # ── State getters ─────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._full_state()

    def get_tile_state(self) -> Dict[str, Any]:
        """Compact state for grid thumbnail."""
        with self._lock:
            return {
                'worker_id':   self._worker_id,
                'game_id':     self._game_id,
                'status':      self._status,
                'num_moves':   len(self._moves),
                'latest_fen':  self._fens[-1] if self._fens else 'start',
                'result':      self._result,
                'termination': self._termination,
                'game_type':   self._game_type,
                'match_info':  self._match_info,
            }

    def get_game_history(self) -> list:
        with self._lock:
            return list(self._game_history)

    def get_game_by_id(self, game_id: int) -> Optional[dict]:
        with self._lock:
            for game in self._game_history:
                if game['game_id'] == game_id:
                    return game
        return None

    def set_socketio(self, socketio):
        self._socketio = socketio

    def set_worker_id(self, worker_id: int):
        self._worker_id = worker_id

    # ── Internal ──────────────────────────────────────────────────────────

    def _full_state(self) -> Dict[str, Any]:
        """Must be called with lock held."""
        return {
            'worker_id':          self._worker_id,
            'game_id':            self._game_id,
            'step':               self._step,
            'moves':              list(self._moves),
            'fens':               list(self._fens),
            'mcts_stats_per_move':list(self._mcts_stats),
            'start_fen':          self._start_fen,
            'status':             self._status,
            'result':             self._result,
            'termination':        self._termination,
            'move_number':        self._move_number,
            'current_fen':        self._fens[-1] if self._fens else self._start_fen,
            'last_move':          self._moves[-1] if self._moves else None,
            'game_type':          self._game_type,
            'match_info':         self._match_info,
        }

    def _save_game(self):
        """Must be called with lock held."""
        if not self._moves:
            return
        self._game_history.appendleft({
            'game_id':            self._game_id,
            'step':               self._step,
            'moves':              list(self._moves),
            'fens':               list(self._fens),
            'mcts_stats_per_move':list(self._mcts_stats),
            'start_fen':          self._start_fen,
            'result':             self._result,
            'termination':        self._termination,
            'num_moves':          len(self._moves),
            'game_type':          self._game_type,
            'match_info':         self._match_info,
        })

    def _emit(self):
        """Push update to all connected browser clients."""
        if self._socketio is None:
            return

        if self._is_eval:
            # Eval board — single full-state event
            with self._lock:
                state = self._full_state()
            self._socketio.emit('eval_live_game_update', state)
        else:
            # Self-play / eval-in-grid workers
            # 1. Compact tile update (always, for the grid)
            tile = self.get_tile_state()
            self._socketio.emit('worker_tile_update', tile)

            # 2. Full detail update (for the expanded viewer — browser ignores
            #    it if this worker isn't the one currently expanded)
            with self._lock:
                full = self._full_state()
            self._socketio.emit('worker_detail_update', full)

    # ── Backwards-compat alias ────────────────────────────────────────────
    def _emit_update(self):
        self._emit()