"""Model-vs-model match orchestrator for AlphaZero chess engine.

Runs a series of games between two independently loaded model checkpoints
through the GPU inference server, streaming live updates to the GUI.

Usage: python main.py match <model_a.pt> <model_b.pt> --games 20 --gui
"""

import io
import time
import threading
from pathlib import Path
import torch

from config import Config
from network import create_model_from_config, load_checkpoint
from selfplay import ParallelSelfPlay
from gui.match_state import MatchState

def _extract_model_name(checkpoint_path):
    return Path(checkpoint_path).stem

def _load_model_weights(config, checkpoint_path):
    net = create_model_from_config(config)
    net.eval()
    ckpt = load_checkpoint(checkpoint_path, net)
    return net, ckpt

def _serialize_weights(network):
    buf = io.BytesIO()
    torch.save(network.state_dict(), buf)
    return buf.getvalue()

def run_match(config, checkpoint_a, checkpoint_b,
              num_games=20, num_workers=4,
              num_simulations=None,
              gui_enabled=False,
              match_state=None):
    """Run a model-vs-model match through the GPU inference server."""
    model_a_name = _extract_model_name(checkpoint_a)
    model_b_name = _extract_model_name(checkpoint_b)

    if num_simulations is not None:
        config.mcts.num_simulations = num_simulations

    print(f"\n{'='*60}")
    print(f"  MODEL MATCH")
    print(f"  {model_a_name}  vs  {model_b_name}")
    print(f"  Games: {num_games}   Workers: {num_workers}")
    print(f"  Simulations: {config.mcts.num_simulations}")
    print(f"{'='*60}\n")

    # Load both models
    print(f"[Match] Loading model A from: {checkpoint_a}")
    net_a, ckpt_a = _load_model_weights(config, checkpoint_a)
    step_a = ckpt_a.get('step', 0)
    print(f"[Match] Loading model B from: {checkpoint_b}")
    net_b, ckpt_b = _load_model_weights(config, checkpoint_b)
    step_b = ckpt_b.get('step', 0)

    # Create match state for GUI
    if match_state is None:
        match_state = MatchState(max_history=num_games)
    match_state.set_match_info(model_a_name, model_b_name, num_games)

    # Start worker pool with GPU inference
    config.inference.use_gpu = True
    psp = ParallelSelfPlay(config, num_workers=num_workers)
    psp.start()

    # Push weights to GPU server (net_a -> weight_q, net_b -> weight_q_b)
    print("[Match] Pushing weights to GPU server...")
    psp.push_eval_weights(net_a, net_b)

    # Build eval tasks
    serialized_a = _serialize_weights(net_a)
    serialized_b = _serialize_weights(net_b)

    eval_tasks = []
    for gi in range(num_games):
        eval_tasks.append({
            'type': 'eval',
            'eval_type': 'network',
            'weights_a': serialized_a,
            'weights_b': serialized_b,
            'a_is_white': (gi % 2 == 0),
            'game_label': f"Game {gi+1}/{num_games}",
        })

    # Launch GUI if requested
    if gui_enabled:
        _start_gui_thread(config, match_state)

    # Dispatch and collect results
    dispatched = psp.dispatch_eval_games(eval_tasks)
    print(f"[Match] Dispatched {dispatched} games to {num_workers} workers")

    collected = 0
    wins_a = wins_b = draws = 0
    live_game_id = 0

    while collected < dispatched:
        result = psp.collect_one(timeout=300.0)
        if result is None:
            print("[Match WARN] Timeout waiting for result")
            break
        if result.get('done'):
            continue

        rt = result.get('type')

        # Live GUI events
        if rt == 'live_start':
            live_game_id += 1
            label = f"Game {live_game_id}/{num_games}"
            match_state.start_game(live_game_id, 'match', label,
                                   model_a_name=model_a_name,
                                   model_b_name=model_b_name,
                                   a_is_white=result.get('a_is_white', True))
            continue

        if rt == 'live_move':
            match_state.update_game(result['fen'], result['move'],
                                    result['move_number'],
                                    mcts_stats=result.get('mcts_stats'))
            continue

        if rt == 'live_end':
            match_state.end_game(result['result'],
                                 result.get('termination', ''))
            continue

        # Full eval result
        if rt != 'eval':
            continue

        collected += 1
        res = result['result']
        a_white = result['a_is_white']

        # Tally
        if res == '1-0':
            if a_white:
                wins_a += 1
                winner = model_a_name
            else:
                wins_b += 1
                winner = model_b_name
        elif res == '0-1':
            if a_white:
                wins_b += 1
                winner = model_b_name
            else:
                wins_a += 1
                winner = model_a_name
        else:
            draws += 1
            winner = 'Draw'

        # Replay full game on match board
        fens = result.get('fens', [])
        moves = result.get('moves', [])
        mstats = result.get('mcts_stats', [])
        match_state.replay_full_game(fens, moves, mstats)

        # Update match scores
        match_state.update_scores(wins_a, wins_b, draws)

        print(f"  [{collected:3d}/{dispatched}] {res:6s}  "
              f"({model_a_name}={wins_a}  {model_b_name}={wins_b}  "
              f"Draws={draws})  Winner: {winner}")

    # Final results
    psp.stop()

    win_rate_a = (wins_a + 0.5 * draws) / collected if collected > 0 else 0.0
    win_rate_b = (wins_b + 0.5 * draws) / collected if collected > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  MATCH COMPLETE")
    print(f"  {model_a_name:30s}  {wins_a:3d}W  {win_rate_a:.1%}")
    print(f"  {model_b_name:30s}  {wins_b:3d}W  {win_rate_b:.1%}")
    print(f"  {'Draws':30s}  {draws:3d}")
    print(f"  Games: {collected}/{num_games}")
    print(f"{'='*60}\n")

    match_state.set_complete()

    return {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'wins_a': wins_a,
        'wins_b': wins_b,
        'draws': draws,
        'num_games': collected,
        'win_rate_a': win_rate_a,
        'win_rate_b': win_rate_b,
        'checkpoint_a': checkpoint_a,
        'checkpoint_b': checkpoint_b,
    }

def _start_gui_thread(config, match_state):
    """Start Flask-SocketIO GUI server in a background thread."""
    from gui.app import start_gui_server
    threading.Thread(
        target=start_gui_server,
        args=(None, config, [], None, match_state),
        daemon=True,
    ).start()
    time.sleep(0.5)
    print(f"[Match] GUI -> http://{config.gui.host}:{config.gui.port}/match")
