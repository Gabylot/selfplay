"""Sanity check: runs a handful of self-play games + training steps end-to-end.

Uses tiny config (2 residual blocks, 16 filters, 10 MCTS sims) to verify
the pipeline works before committing to a longer run.

Usage:
    python sanity_check.py
"""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from config import get_config
    from network import create_model_from_config
    from mcts import MCTS
    from selfplay import self_play_game, ReplayBuffer
    from training import train_one_step, create_optimizer
    from evaluation import alpha_beta_best_move
    from stats import StatsLogger
    from encoding import board_to_tensor, get_legal_move_mask, NUM_PLANES
    import torch
    import chess
    import numpy as np

    print("\n" + "=" * 60)
    print("  SANITY CHECK: Tiny Config Pipeline Test")
    print("=" * 60 + "\n")

    # Tiny config overrides
    overrides = {
        "network": {
            "num_residual_blocks": 2,
            "num_filters": 16,
            "num_policy_channels": 8,
            "num_value_channels": 8,
            "value_fc_size": 32,
        },
        "mcts": {"num_simulations": 10},
        "selfplay": {
            "max_game_length": 50,
            "temperature_threshold": 15,
        },
        "training": {
            "batch_size": 8,
            "training_steps_per_iteration": 2,
            "checkpoint_interval": 5,
        },
        "evaluation": {
            "eval_interval": 2,
            "gate_games": 4,
            "ref_opponent_games": 4,
        },
        "buffer": {"max_size": 1000},
    }

    config = get_config(overrides=overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Test encoding ---
    print("\n--- 1. Testing Board Encoding ---")
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (NUM_PLANES, 8, 8), f"Wrong shape: {tensor.shape}"
    print(f"  Board tensor shape: {tensor.shape} (expected: ({NUM_PLANES}, 8, 8)) OK")

    mask = get_legal_move_mask(board)
    assert mask.sum() == 20, f"Wrong legal moves: {mask.sum()} (expected 20)"
    print(f"  Legal move mask sum: {mask.sum():.0f} (expected: 20) OK")

    # --- 2. Test network ---
    print("\n--- 2. Testing Network ---")
    network = create_model_from_config(config)
    network.to(device)
    param_count = sum(p.numel() for p in network.parameters())
    print(f"  Network parameters: {param_count}")

    # Forward pass
    tensor_batch = tensor[np.newaxis, ...]
    policy_logits, value = network.forward(torch.from_numpy(tensor_batch).float().to(device))
    assert policy_logits.shape == (1, 4672), f"Policy shape wrong: {policy_logits.shape}"
    assert value.shape == (1, 1), f"Value shape wrong: {value.shape}"
    print(f"  Policy shape: {policy_logits.shape} OK")
    print(f"  Value shape: {value.shape}, value: {value.item():.4f} OK")

    # --- 3. Test MCTS ---
    print("\n--- 3. Testing MCTS ---")
    mcts = MCTS(network, num_simulations=10, c_puct=1.5)
    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)
    assert best_move is not None, "MCTS returned no move"
    print(f"  Best move: {best_move}")
    print(f"  Visit policy sum: {visit_policy.sum():.4f}")
    print(f"  Avg depth: {stats['avg_depth']:.2f}")
    print("  MCTS search OK")

    # --- 4. Test self-play game ---
    print("\n--- 4. Testing Self-Play Game ---")
    game_data, game_info = self_play_game(network, config)
    assert len(game_data) > 0, "No positions generated"
    assert game_info['length'] > 0, "Game length is 0"
    print(f"  Game result: {game_info['result_str']}")
    print(f"  Game length: {game_info['length']} moves")
    print(f"  Termination: {game_info['termination']}")
    print(f"  Positions generated: {game_info['num_positions']}")
    print(f"  Avg MCTS depth: {game_info['avg_mcts_depth']:.2f}")
    print("  Self-play game OK")

    # --- 5. Test replay buffer ---
    print("\n--- 5. Testing Replay Buffer ---")
    buffer = ReplayBuffer(max_size=1000)
    buffer.add_game(game_data)
    assert len(buffer) == len(game_data), f"Buffer size mismatch: {len(buffer)} != {len(game_data)}"
    states, policies, values = buffer.sample_batch(8)
    assert states.shape[0] > 0, "Empty batch"
    assert states.shape[1:] == (NUM_PLANES, 8, 8), f"State shape wrong: {states.shape}"
    assert policies.shape[1] == 4672, f"Policy shape wrong: {policies.shape}"
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Sampled batch: states={states.shape}, policies={policies.shape}, values={values.shape}")
    dist = buffer.get_outcome_distribution()
    print(f"  Outcome distribution: {dist}")
    print("  Replay buffer OK")

    # --- 6. Test training step ---
    print("\n--- 6. Testing Training Step ---")
    optimizer = create_optimizer(network)
    losses = train_one_step(network, optimizer, buffer, batch_size=8, device=device)
    assert 'policy_loss' in losses, "Missing policy_loss"
    assert 'value_loss' in losses, "Missing value_loss"
    assert 'total_loss' in losses, "Missing total_loss"
    assert losses['total_loss'] > 0, "Total loss should be positive"
    print(f"  Policy loss: {losses['policy_loss']:.4f}")
    print(f"  Value loss: {losses['value_loss']:.4f}")
    print(f"  Total loss: {losses['total_loss']:.4f}")
    print("  Training step OK")

    # --- 7. Test stats logging ---
    print("\n--- 7. Testing Stats Logging ---")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    stats_db = StatsLogger(db_path)
    stats_db.log_training_step(1, losses['policy_loss'], losses['value_loss'], losses['total_loss'])
    stats_db.log_game(
        game_id=1, step=1, result=game_info['result'],
        result_str=game_info['result_str'],
        length=game_info['length'], termination=game_info['termination'],
    )
    summary = stats_db.get_summary()
    assert summary['total_games'] == 1, f"Wrong game count: {summary['total_games']}"
    assert summary['current_step'] == 1, f"Wrong step: {summary['current_step']}"
    print(f"  Stats summary: total_games={summary['total_games']}, step={summary['current_step']}")
    stats_db.close()
    os.unlink(db_path)
    print("  Stats logging OK")

    # --- 8. Test alpha-beta reference ---
    print("\n--- 8. Testing Alpha-Beta Reference ---")
    move = alpha_beta_best_move(chess.Board(), depth=2)
    assert move is not None, "Alpha-beta returned no move"
    print(f"  Alpha-beta best move from start: {move}")
    print("  Alpha-beta reference OK")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED - Pipeline is functional!")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())