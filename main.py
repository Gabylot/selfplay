"""Main entry point for the AlphaZero Chess Engine.

Orchestrates self-play, training, evaluation, and optionally the GUI.

Usage:
    python main.py train          # Run training loop (self-play + training + eval)
    python main.py gui            # Launch GUI server (connects to running training)
    python main.py train --gui    # Run training with GUI
    python main.py evaluate       # Run evaluation matches only
    python main.py sanity         # Run sanity check (tiny config, quick test)
"""

import argparse
import sys
import os
import time
import threading
import signal
import json
from pathlib import Path

import torch
import numpy as np

from config import get_config
from network import AlphaZeroNet, create_model_from_config, save_checkpoint, load_checkpoint
from mcts import MCTS
from selfplay import self_play_game, ReplayBuffer
from training import train_one_step, create_optimizer
from evaluation import Evaluator
from stats import StatsLogger
from gui.live_game import LiveGameState


# Global flag for graceful shutdown
_shutdown = False


def signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[INFO] Shutdown signal received._finishing current operation...")


def run_training(config, gui_enabled: bool = False):
    """Main training loop: self-play → training → evaluation → repeat."""
    global _shutdown
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Create network
    network = create_model_from_config(config)
    network.to(device)
    print(f"[INFO] Network created: {sum(p.numel() for p in network.parameters())} parameters")
    
    # Setup directories
    output_dir = Path(config.main.output_dir) / config.main.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Setup stats
    stats_db_path = output_dir / config.stats.db_path
    stats = StatsLogger(str(stats_db_path))
    print(f"[INFO] Stats database: {stats_db_path}")
    
    # Setup live game state
    live_game = LiveGameState(max_history=20)
    
    # Setup replay buffer
    buffer = ReplayBuffer(max_size=config.buffer.max_size)
    
    # Setup evaluator
    evaluator = Evaluator(config, stats)
    
    # Create optimizer (ensure numeric types from YAML)
    optimizer = create_optimizer(
        network, 
        lr=float(config.training.learning_rate),
        momentum=float(config.training.momentum),
        weight_decay=float(config.training.weight_decay),
    )
    
    # Global counters
    step = 0
    game_id = 0
    best_network = create_model_from_config(config)  # Copy for gating reference
    best_network.load_state_dict(network.state_dict())
    best_network.to(device)
    
    # Log initial config
    stats.log_config(step, config.to_dict())
    
    # Determine self-play network source
    use_latest = (config.selfplay.network_source == "latest")
    
    # Launch GUI if requested
    gui_thread = None
    if gui_enabled:
        from gui.app import start_gui_server
        gui_thread = threading.Thread(target=start_gui_server, args=(stats, config, live_game), daemon=True)
        gui_thread.start()
        print(f"[INFO] GUI server started at http://{config.gui.host}:{config.gui.port}")
    
    # Load checkpoint if exists
    latest_ckpt = checkpoints_dir / "latest.pt"
    if latest_ckpt.exists():
        print(f"[INFO] Loading checkpoint from {latest_ckpt}")
        ckpt = load_checkpoint(str(latest_ckpt), network, optimizer)
        step = ckpt.get('step', 0)
        game_id = ckpt.get('game_id', 0)
        print(f"[INFO] Resumed from step {step}, game {game_id}")
    
    print(f"\n{'='*60}")
    print(f"  AlphaZero Training Started")
    print(f"  Step: {step}, Games: {game_id}")
    print(f"  Self-play network: {'latest' if use_latest else 'gated_best'}")
    print(f"  MCTS simulations: {config.mcts.num_simulations}")
    print(f"{'='*60}\n")
    
    # --- Main training loop ---
    while not _shutdown:
        # --- Self-Play Phase ---
        print(f"[Step {step}] Playing self-play games...")
        
        # Determine which network to use for self-play
        if use_latest:
            current_net = network
        else:
            current_net = best_network
        
        num_selfplay_games = config.training.training_steps_per_iteration
        
        for _ in range(num_selfplay_games):
            if _shutdown:
                break
            
            # Notify live game viewer that a new game is starting
            live_game.start_game(game_id + 1, step)
            
            # Create on_move callback for live board updates
            def _on_move(fen, uci, move_num, _gid=game_id+1, _step=step):
                live_game.update(fen, uci, move_num)
            
            game_data, game_info = self_play_game(current_net, config, on_move=_on_move)
            buffer.add_game(game_data)
            
            game_id += 1
            
            # Notify live game viewer that the game is over
            result_str = game_info['result_str']
            live_game.game_over(result_str, game_info['termination'])
            
            # Log game to stats
            stats.log_game(
                game_id=game_id, step=step,
                result=game_info['result'],
                result_str=game_info['result_str'],
                length=game_info['length'],
                termination=game_info['termination'],
                avg_mcts_depth=game_info['avg_mcts_depth'],
                num_positions=game_info['num_positions'],
            )
            
            # Log MCTS stats
            stats.log_mcts_stats(
                game_id=game_id, step=step,
                avg_tree_depth=game_info['avg_mcts_depth'],
                avg_sims_per_move=config.mcts.num_simulations,
            )
            
            print(f"  Game {game_id}: {game_info['termination']} | "
                  f"Result: {game_info['result_str']} | "
                  f"Length: {game_info['length']} | "
                  f"Buffer: {len(buffer)} positions")
        
        if _shutdown:
            break
        
        # --- Training Phase ---
        print(f"[Step {step}] Training network...")
        
        for _ in range(config.training.training_steps_per_iteration):
            if _shutdown:
                break
            
            loss_dict = train_one_step(
                network, optimizer, buffer, int(config.training.batch_size), device
            )
            step += 1
            
            # Log losses (separate policy/value)
            stats.log_training_step(
                step=step,
                policy_loss=loss_dict['policy_loss'],
                value_loss=loss_dict['value_loss'],
                total_loss=loss_dict['total_loss'],
                learning_rate=config.training.learning_rate,
            )
            
            print(f"  Training step {step}: "
                  f"policy={loss_dict['policy_loss']:.4f} "
                  f"value={loss_dict['value_loss']:.4f} "
                  f"total={loss_dict['total_loss']:.4f}")
        
        # --- Buffer stats ---
        outcome_dist = buffer.get_outcome_distribution()
        stats.log_buffer_stats(
            step=step, buffer_size=len(buffer),
            white_wins=outcome_dist['white_wins'],
            black_wins=outcome_dist['black_wins'],
            draws=outcome_dist['draws'],
        )
        
        # --- Network confidence stats ---
        # Sample a few positions to measure confidence
        if len(buffer) > 0:
            sample_size = min(50, len(buffer))
            indices = np.random.choice(len(buffer), size=sample_size, replace=False)
            states = np.array([buffer.buffer[i][0] for i in indices])
            policies, values = network.predict_batch(states)
            avg_max_policy = float(np.mean(np.max(policies, axis=1)))
            avg_abs_value = float(np.mean(np.abs(values)))
            stats.log_network_stats(step, avg_max_policy, avg_abs_value)
        
        # --- Checkpoint ---
        if step % config.training.checkpoint_interval == 0:
            ckpt_path = checkpoints_dir / f"step_{step}.pt"
            save_checkpoint(network, optimizer, str(ckpt_path), step, 
                          extra={'game_id': game_id})
            save_checkpoint(network, optimizer, str(checkpoints_dir / "latest.pt"), step,
                          extra={'game_id': game_id})
            print(f"  Checkpoint saved: {ckpt_path}")
        
        # --- Evaluation phase (monitoring — does NOT gate self-play) ---
        if game_id % config.evaluation.eval_interval == 0:
            print(f"[Step {step}] Running evaluation (monitoring)...")
            
            # Gating match (monitoring only)
            gating_result = evaluator.run_gating_match(
                network, best_network, step, verbose=True
            )
            
            if gating_result['promoted']:
                best_network.load_state_dict(network.state_dict())
                print(f"  [GATE] Network promoted! New Elo: {evaluator.best_elo:.0f}")
            
            # Reference match vs alpha-beta
            ref_result = evaluator.run_reference_match(network, step, verbose=True)
            print(f"  vs Alpha-Beta: {ref_result['win_rate']:.2%} win rate")
        
        # Summary
        print(f"\n{'='*40}")
        print(f"  Summary: step={step}, games={game_id}, buffer={len(buffer)}")
        print(f"  Stats DB: {stats_db_path}")
        print(f"{'='*40}\n")
    
    # Save final checkpoint
    save_checkpoint(network, optimizer, str(checkpoints_dir / "latest.pt"), step,
                    extra={'game_id': game_id})
    
    stats.close()
    print("[INFO] Training loop finished.")


def run_sanity_check(config):
    """Run a quick sanity check with tiny config."""
    print("\n" + "="*60)
    print("  SANITY CHECK: Tiny Config Pipeline Test")
    print("="*60 + "\n")
    
    # Override config for tiny test
    overrides = {
        'network': {
            'num_residual_blocks': 2,
            'num_filters': 16,
            'num_policy_channels': 8,
            'num_value_channels': 8,
            'value_fc_size': 32,
        },
        'mcts': {'num_simulations': 10},
        'selfplay': {
            'max_game_length': 50,
            'temperature_threshold': 15,
        },
        'training': {
            'batch_size': 8,
            'training_steps_per_iteration': 2,
            'checkpoint_interval': 5,
        },
        'evaluation': {
            'eval_interval': 2,
            'gate_games': 4,
            'ref_opponent_games': 4,
        },
        'buffer': {'max_size': 1000},
    }
    
    local_config = get_config(config_path=None, overrides=overrides)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create network
    network = create_model_from_config(local_config)
    network.to(device)
    param_count = sum(p.numel() for p in network.parameters())
    print(f"Network: {param_count} parameters")
    
    # Test encoding (quick sanity)
    import chess
    from encoding import board_to_tensor, get_legal_move_mask, move_to_policy_index, policy_index_to_move
    board = chess.Board()
    tensor = board_to_tensor(board)
    print(f"Board tensor shape: {tensor.shape} (expected: 18, 8, 8)")
    mask = get_legal_move_mask(board)
    print(f"Legal move mask sum: {mask.sum():.0f} (expected: 20 for starting position)")
    
    # Test MCTS
    print("\n--- Testing MCTS ---")
    mcts = MCTS(network, num_simulations=10, c_puct=1.5)
    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)
    print(f"Best move: {best_move}")
    print(f"Visit policy sum: {visit_policy.sum():.4f}")
    print(f"Avg depth: {stats['avg_depth']:.2f}")
    
    # Test self-play game
    print("\n--- Testing Self-Play Game ---")
    game_data, game_info = self_play_game(network, local_config)
    print(f"Game result: {game_info['result_str']}")
    print(f"Game length: {game_info['length']} moves")
    print(f"Termination: {game_info['termination']}")
    print(f"Positions generated: {game_info['num_positions']}")
    
    # Test replay buffer
    print("\n--- Testing Replay Buffer ---")
    buffer = ReplayBuffer(max_size=1000)
    buffer.add_game(game_data)
    print(f"Buffer size: {len(buffer)}")
    states, policies, values = buffer.sample_batch(8)
    print(f"Sampled batch: states={states.shape}, policies={policies.shape}, values={values.shape}")
    
    # Test training step
    print("\n--- Testing Training Step ---")
    optimizer = create_optimizer(network)
    losses = train_one_step(network, optimizer, buffer, batch_size=8, device=device)
    print(f"Policy loss: {losses['policy_loss']:.4f}")
    print(f"Value loss: {losses['value_loss']:.4f}")
    print(f"Total loss: {losses['total_loss']:.4f}")
    
    # Test stats logging
    print("\n--- Testing Stats Logging ---")
    stats_db = StatsLogger(str(output_dir / "sanity_stats.db"))
    stats_db.log_training_step(1, losses['policy_loss'], losses['value_loss'], losses['total_loss'])
    stats_db.log_game(game_id=1, step=1, result=game_info['result'], 
                      result_str=game_info['result_str'],
                      length=game_info['length'], termination=game_info['termination'])
    
    summary = stats_db.get_summary()
    print(f"Stats summary: total_games={summary['total_games']}, current_step={summary['current_step']}")
    stats_db.close()
    
    # Test alpha-beta
    print("\n--- Testing Alpha-Beta ---")
    from evaluation import alpha_beta_best_move
    move = alpha_beta_best_move(board, depth=2)
    print(f"Alpha-beta best move: {move}")
    
    print("\n" + "="*60)
    print("  SANITY CHECK PASSED - All components working!")
    print("="*60 + "\n")


def run_gui_only(config):
    """Launch GUI server only (connects to running training DB)."""
    from gui.app import start_gui_server
    print(f"[INFO] Starting GUI server at http://{config.gui.host}:{config.gui.port}")
    start_gui_server(stats=None, config=config)
    # If stats is None, GUI reads from existing DB file


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Engine")
    parser.add_argument("mode", choices=["train", "gui", "evaluate", "sanity"],
                       help="Run mode: train, gui, evaluate, sanity")
    parser.add_argument("--gui", action="store_true",
                       help="Enable GUI alongside training")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--sims", type=int, default=None,
                       help="Override MCTS simulation count")
    parser.add_argument("--blocks", type=int, default=None,
                       help="Override number of residual blocks")
    parser.add_argument("--filters", type=int, default=None,
                       help="Override number of filters")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Override run name")
    
    args = parser.parse_args()
    
    # Build overrides from CLI
    overrides = {}
    if args.sims is not None:
        overrides.setdefault('mcts', {})['num_simulations'] = args.sims
    if args.blocks is not None:
        overrides.setdefault('network', {})['num_residual_blocks'] = args.blocks
    if args.filters is not None:
        overrides.setdefault('network', {})['num_filters'] = args.filters
    if args.run_name is not None:
        overrides.setdefault('main', {})['run_name'] = args.run_name
    
    config = get_config(path=args.config, overrides=overrides if overrides else None)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Global output dir for sanity check
    global output_dir
    output_dir = Path(config.main.output_dir) / config.main.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "train":
        run_training(config, gui_enabled=args.gui)
    
    elif args.mode == "gui":
        run_gui_only(config)
    
    elif args.mode == "evaluate":
        # Run evaluation only
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = create_model_from_config(config)
        network.to(device)
        
        # Load checkpoint
        ckpt_path = Path(config.main.output_dir) / config.main.run_name / "checkpoints" / "latest.pt"
        if ckpt_path.exists():
            print(f"[INFO] Loading checkpoint from {ckpt_path}")
            load_checkpoint(str(ckpt_path), network)
        else:
            print("[WARN] No checkpoint found. Using random network — results will be meaningless.")
        
        stats = StatsLogger(str(Path(config.main.output_dir) / config.main.run_name / config.stats.db_path))
        evaluator = Evaluator(config, stats)
        
        print("\n--- Running Reference Evaluation ---")
        ref_result = evaluator.run_reference_match(network, step=0, verbose=True)
        print(f"vs Alpha-Beta: {ref_result['win_rate']:.2%} win rate")
        
        stats.close()
    
    elif args.mode == "sanity":
        run_sanity_check(config)


if __name__ == "__main__":
    main()