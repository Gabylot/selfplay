"""Live game state for real-time self-play viewing.

Provides a thread-safe shared state object that the self-play loop
updates after each move. The GUI server reads this state and emits
WebSocket events to update the browser chessboard.
"""

import threading
from typing import List, Optional, Dict, Any
from collections import deque


class LiveGameState:
    """Thread-safe live game state shared between self-play and GUI.
    
    The self-play loop calls update() after each move.
    The GUI server reads the state and emits it via SocketIO.
    """
    
    def __init__(self, socketio=None, max_history: int = 20):
        """
        Args:
            socketio: Flask-SocketIO instance for emitting events
            max_history: Max completed games to keep for replay
        """
        self._lock = threading.Lock()
        self._socketio = socketio
        
        # Current game state
        self._game_id: int = 0
        self._step: int = 0
        self._moves: List[str] = []          # UCI move strings
        self._fens: List[str] = []           # FEN after each move
        self._start_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self._status: str = "idle"           # idle | playing | finished
        self._result: Optional[str] = None   # "1-0", "0-1", "1/2-1/2", None
        self._termination: Optional[str] = None
        self._move_number: int = 0
        
        # Completed games history (for replay)
        self._game_history: deque = deque(maxlen=max_history)
    
    def start_game(self, game_id: int, step: int):
        """Called when a new self-play game begins."""
        with self._lock:
            # Save previous game if one was in progress
            if self._status == "playing" and self._moves:
                self._save_game()
            
            self._game_id = game_id
            self._step = step
            self._moves = []
            self._fens = []
            self._start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            self._status = "playing"
            self._result = None
            self._termination = None
            self._move_number = 0
        
        self._emit_update()
    
    def update(self, board_fen: str, move_uci: str, move_number: int):
        """Called after each move in the self-play game.
        
        Args:
            board_fen: FEN of the board AFTER the move
            move_uci: UCI string of the move that was just played (e.g. "e2e4")
            move_number: Current half-move count
        """
        with self._lock:
            self._moves.append(move_uci)
            self._fens.append(board_fen)
            self._move_number = move_number
        
        self._emit_update()
    
    def game_over(self, result: str, termination: str):
        """Called when the self-play game ends.
        
        Args:
            result: "1-0", "0-1", or "1/2-1/2"
            termination: "checkmate", "stalemate", "repetition", etc.
        """
        with self._lock:
            self._result = result
            self._termination = termination
            self._status = "finished"
            self._save_game()
        
        self._emit_update()
    
    def _save_game(self):
        """Save current game to history. Must be called with lock held."""
        if not self._moves:
            return
        game = {
            'game_id': self._game_id,
            'step': self._step,
            'moves': list(self._moves),
            'fens': list(self._fens),
            'start_fen': self._start_fen,
            'result': self._result,
            'termination': self._termination,
            'num_moves': len(self._moves),
        }
        self._game_history.appendleft(game)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current live game state (thread-safe)."""
        with self._lock:
            return {
                'game_id': self._game_id,
                'step': self._step,
                'moves': list(self._moves),
                'fens': list(self._fens),
                'start_fen': self._start_fen,
                'status': self._status,
                'result': self._result,
                'termination': self._termination,
                'move_number': self._move_number,
                'current_fen': self._fens[-1] if self._fens else self._start_fen,
                'last_move': self._moves[-1] if self._moves else None,
            }
    
    def get_game_history(self) -> list:
        """Get list of completed games (for replay)."""
        with self._lock:
            return list(self._game_history)
    
    def get_game_by_id(self, game_id: int) -> Optional[dict]:
        """Get a specific completed game by ID."""
        with self._lock:
            for game in self._game_history:
                if game['game_id'] == game_id:
                    return game
        return None
    
    def set_socketio(self, socketio):
        """Set the SocketIO instance (called after app is created)."""
        self._socketio = socketio
    
    def _emit_update(self):
        """Emit a live_game_update event to all connected clients."""
        if self._socketio is None:
            return
        state = self.get_state()
        self._socketio.emit('live_game_update', state)