"""Flask + SocketIO GUI server — updated for parallel workers and new push events."""

from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO
from stats import StatsLogger
from gui.live_game import LiveGameState

app = Flask(__name__, template_folder=str(Path(__file__).parent/"templates"))
app.config['SECRET_KEY'] = 'alphazero-chess'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

_stats = None
_config = None
_worker_live_games = []
_eval_live_game = None


@app.route('/chess_pieces/<path:fn>')
def chess_piece(fn):
    return send_from_directory(str(Path(__file__).parent.parent/"chess_pieces"), fn)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/summary')
def api_summary():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_summary())

@app.route('/api/training_losses')
def api_training_losses():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_training_losses())

@app.route('/api/game_outcomes')
def api_game_outcomes():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_game_outcomes())

@app.route('/api/elo_history')
def api_elo_history():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_elo_history())

@app.route('/api/promotion_history')
def api_promotion_history():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_promotion_history())

@app.route('/api/evaluation_history')
def api_evaluation_history():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_evaluation_history())

@app.route('/api/network_stats')
def api_network_stats():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_network_stats_history())

@app.route('/api/buffer_stats')
def api_buffer_stats():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_buffer_stats_history())

@app.route('/api/mcts_stats')
def api_mcts_stats():
    if _stats is None: return jsonify({'error':'no stats'}),503
    return jsonify(_stats.get_mcts_stats_history())

@app.route('/api/num_workers')
def api_num_workers():
    return jsonify({'num_workers': len(_worker_live_games)})


# ── SocketIO ──────────────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    socketio.emit('num_workers', {'num_workers': len(_worker_live_games)})

@socketio.on('request_worker_states')
def on_request_worker_states():
    states = [wlg.get_tile_state() for wlg in _worker_live_games]
    socketio.emit('worker_states', states)

@socketio.on('request_worker_detail')
def on_request_worker_detail(data):
    wid = data.get('worker_id', 0)
    if 0 <= wid < len(_worker_live_games):
        socketio.emit('worker_detail_update', _worker_live_games[wid].get_state())

@socketio.on('request_game_history')
def on_request_game_history(data=None):
    wid = (data or {}).get('worker_id', 0)
    if 0 <= wid < len(_worker_live_games):
        socketio.emit('game_history', {
            'worker_id': wid,
            'history':   _worker_live_games[wid].get_game_history(),
        })

@socketio.on('request_replay_game')
def on_request_replay_game(data):
    wid = data.get('worker_id', 0)
    gid = data.get('game_id')
    if 0 <= wid < len(_worker_live_games):
        game = _worker_live_games[wid].get_game_by_id(gid)
        if game:
            socketio.emit('replay_game', {**game, 'worker_id': wid})

@socketio.on('request_eval_live_state')
def on_request_eval_live_state():
    if _eval_live_game:
        socketio.emit('eval_live_game_update', _eval_live_game.get_state())

@socketio.on('request_eval_game_history')
def on_request_eval_game_history():
    if _eval_live_game:
        socketio.emit('eval_game_history', _eval_live_game.get_game_history())

@socketio.on('request_replay_eval_game')
def on_request_replay_eval_game(data):
    if _eval_live_game:
        game = _eval_live_game.get_game_by_id(data.get('game_id'))
        if game: socketio.emit('replay_eval_game', game)


def start_gui_server(stats=None, config=None, worker_live_games=None, eval_live_game=None):
    global _stats, _config, _worker_live_games, _eval_live_game
    _stats             = stats
    _config            = config
    _worker_live_games = worker_live_games or []
    _eval_live_game    = eval_live_game

    for wlg in _worker_live_games:
        wlg.set_socketio(socketio)
    if _eval_live_game:
        _eval_live_game.set_socketio(socketio)

    host = config.gui.host if config else "127.0.0.1"
    port = config.gui.port if config else 5000

    if _stats is None and config:
        db = str(Path(config.main.output_dir)/config.main.run_name/config.stats.db_path)
        if Path(db).exists():
            _stats = StatsLogger(db)

    socketio.run(app, host=host, port=port,
                 allow_unsafe_werkzeug=True, use_reloader=False, log_output=False)