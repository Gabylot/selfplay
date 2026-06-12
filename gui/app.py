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

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from stats import StatsLogger


app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / "templates"))
app.config['SECRET_KEY'] = 'alphazero-chess-engine'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global references (set when starting the server)
_stats: StatsLogger = None
_config = None


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


def start_gui_server(stats=None, config=None):
    """Start the Flask + SocketIO GUI server.
    
    Args:
        stats: StatsLogger instance for reading data (or None to read from existing DB)
        config: Config object for server settings
    """
    global _stats, _config
    _stats = stats
    _config = config
    
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