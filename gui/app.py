"""Flask + SocketIO web dashboard for AlphaZero chess engine.

Provides:
- Live board viewer (self-play games)
- Training dashboard with charts (loss curves, Elo, outcomes)
- Controls for training parameters
- WebSocket-based real-time updates

Runs as a separate thread/process alongside the training loop.
Reads from the same SQLite stats database.
"""

import json
import time
import threading
from pathlib import Path

from flask import Flask, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO

from stats import StatsLogger
from gui.live_game import LiveGameState


app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / "templates"))
app.config['SECRET_KEY'] = 'alphazero-chess-engine'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


@app.route('/chess_pieces/<path:filename>')
def chess_piece(filename):
    """Serve chess piece images from the chess_pieces directory."""
    pieces_dir = str(Path(__file__).parent.parent / "chess_pieces")
    return send_from_directory(pieces_dir, filename)

# Global references (set when starting the server)
_stats: StatsLogger = None
_config = None
_live_game: LiveGameState = None
_eval_live_game: LiveGameState = None


@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/api/summary')
def api_summary():
    """Get summary statistics."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_summary())


@app.route('/api/training_losses')
def api_training_losses():
    """Get training loss history."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_training_losses())


@app.route('/api/game_outcomes')
def api_game_outcomes():
    """Get game outcome history."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_game_outcomes())


@app.route('/api/elo_history')
def api_elo_history():
    """Get Elo rating history."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_elo_history())


@app.route('/api/promotion_history')
def api_promotion_history():
    """Get promotion attempt history."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_promotion_history())


@app.route('/api/evaluation_history')
def api_evaluation_history():
    """Get evaluation match history."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_evaluation_history())


@app.route('/api/network_stats')
def api_network_stats():
    """Get network confidence trend data."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_network_stats_history())


@app.route('/api/buffer_stats')
def api_buffer_stats():
    """Get replay buffer composition."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_buffer_stats_history())


@app.route('/api/mcts_stats')
def api_mcts_stats():
    """Get MCTS statistics."""
    if _stats is None:
        return jsonify({'error': 'No stats connection'}), 503
    return jsonify(_stats.get_mcts_stats_history())


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("[GUI] Client connected")


@socketio.on('request_update')
def handle_request_update():
    """Handle client request for live update."""
    if _stats is not None:
        summary = _stats.get_summary()
        socketio.emit('update', summary)


@socketio.on('request_live_state')
def handle_request_live_state():
    """Handle client request for current live game state."""
    if _live_game is not None:
        state = _live_game.get_state()
        socketio.emit('live_game_update', state)


@socketio.on('request_game_history')
def handle_request_game_history():
    """Handle client request for completed game history."""
    if _live_game is not None:
        history = _live_game.get_game_history()
        socketio.emit('game_history', history)


@socketio.on('request_replay_game')
def handle_request_replay_game(data):
    """Handle client request to replay a specific completed game."""
    if _live_game is not None:
        game_id = data.get('game_id')
        game = _live_game.get_game_by_id(game_id)
        if game is not None:
            emit_data = {
                'game_id': game['game_id'],
                'step': game['step'],
                'moves': game['moves'],
                'fens': game['fens'],
                'start_fen': game['start_fen'],
                'result': game['result'],
                'termination': game['termination'],
                'num_moves': game['num_moves'],
            }
            # Include MCTS stats if available
            if 'mcts_stats_per_move' in game:
                emit_data['mcts_stats_per_move'] = game['mcts_stats_per_move']
            socketio.emit('replay_game', emit_data)


# ===== Eval live game handlers =====

@socketio.on('request_eval_live_state')
def handle_request_eval_live_state():
    """Handle client request for current eval live game state."""
    if _eval_live_game is not None:
        state = _eval_live_game.get_state()
        socketio.emit('eval_live_game_update', state)


@socketio.on('request_eval_game_history')
def handle_request_eval_game_history():
    """Handle client request for completed eval game history."""
    if _eval_live_game is not None:
        history = _eval_live_game.get_game_history()
        socketio.emit('eval_game_history', history)


@socketio.on('request_replay_eval_game')
def handle_request_replay_eval_game(data):
    """Handle client request to replay a specific completed eval game."""
    if _eval_live_game is not None:
        game_id = data.get('game_id')
        game = _eval_live_game.get_game_by_id(game_id)
        if game is not None:
            emit_data = {
                'game_id': game['game_id'],
                'step': game['step'],
                'moves': game['moves'],
                'fens': game['fens'],
                'start_fen': game['start_fen'],
                'result': game['result'],
                'termination': game['termination'],
                'num_moves': game['num_moves'],
                'game_type': game.get('game_type', 'eval'),
                'match_info': game.get('match_info', None),
            }
            if 'mcts_stats_per_move' in game:
                emit_data['mcts_stats_per_move'] = game['mcts_stats_per_move']
            socketio.emit('replay_eval_game', emit_data)


def start_gui_server(stats=None, config=None, live_game=None, eval_live_game=None):
    """Start the Flask + SocketIO GUI server.
    
    Args:
        stats: StatsLogger instance for reading data (or None to read from existing DB)
        config: Config object for server settings
        live_game: LiveGameState instance for live board viewing
        eval_live_game: LiveGameState instance for eval game viewing
    """
    global _stats, _config, _live_game, _eval_live_game
    _stats = stats
    _config = config
    _live_game = live_game
    _eval_live_game = eval_live_game
    
    # Connect the socketio instance to the live game state
    if _live_game is not None:
        _live_game.set_socketio(socketio)
    if _eval_live_game is not None:
        _eval_live_game.set_socketio(socketio)
    
    host = config.gui.host if config else "127.0.0.1"
    port = config.gui.port if config else 5000
    
    # If no stats provided, try to connect to existing DB
    if _stats is None and config:
        db_path = str(Path(config.main.output_dir) / config.main.run_name / config.stats.db_path)
        if Path(db_path).exists():
            _stats = StatsLogger(db_path)
    
    # Start the Flask app
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True, 
                 use_reloader=False, log_output=True)
