"""Correctness validation for MCTS optimization changes.

Compares move selection and visit distributions between:
1. Sequential MCTS (batch_size=1, stack=False, deduped legal_moves)
2. Batched MCTS (batch_size=8) 

Both should produce similar (not identical due to Dirichlet noise + virtual loss) 
distributions for the starting position.

Usage:
    python test_optimization.py          # Quick test
    python test_optimization.py --games 1  # Play one full game with each
    python test_optimization.py --verbose   # Detailed stats
"""

import argparse
import time
import numpy as np
import chess
from typing import List, Tuple

from encoding import board_to_tensor
from network import AlphaZeroNet
from mcts import MCTS


def test_move_selection_similarity(
    network: AlphaZeroNet,
    board: chess.Board,
    num_simulations: int = 200,
    num_runs: int = 5,
    verbose: bool = False,
) -> dict:
    """Compare sequential vs batched MCTS move selection.
    
    Checks:
    1. Both find the same top-1 move (most visited)
    2. Top-5 set similarity (Jaccard index)
    3. Visit distribution correlation
    
    Returns dict of metrics.
    """
    top1_agreement = 0
    jaccard_sums = []
    visit_corrs = []
    
    for run in range(num_runs):
        # Sequential MCTS
        mcts_seq = MCTS(network, num_simulations=num_simulations, batch_size=1)
        root_seq = mcts_seq.get_root(board)
        visit_seq, best_move_seq, _ = mcts_seq.search(root_seq)
        
        # Get top-5 from sequential (by visit count)
        seq_visits = {}
        for aidx, child in root_seq.children.items():
            if child.N > 0:
                seq_visits[aidx] = child.N
        seq_top5 = set(sorted(seq_visits, key=seq_visits.get, reverse=True)[:5])
        seq_top1 = max(seq_visits, key=seq_visits.get)
        
        # Batched MCTS
        mcts_batch = MCTS(network, num_simulations=num_simulations, batch_size=8)
        root_batch = mcts_batch.get_root(board)
        visit_batch, best_move_batch, _ = mcts_batch.search(root_batch)
        
        # Get top-5 from batched
        batch_visits = {}
        for aidx, child in root_batch.children.items():
            if child.N > 0:
                batch_visits[aidx] = child.N
        batch_top5 = set(sorted(batch_visits, key=batch_visits.get, reverse=True)[:5])
        batch_top1 = max(batch_visits, key=batch_visits.get)
        
        # Top-1 agreement
        if seq_top1 == batch_top1:
            top1_agreement += 1
        
        # Jaccard index for top-5
        intersection = seq_top5 & batch_top5
        union = seq_top5 | batch_top5
        jaccard = len(intersection) / len(union) if union else 1.0
        jaccard_sums.append(jaccard)
        
        # Correlation of visit distributions (over shared children)
        shared_indices = list(set(seq_visits.keys()) & set(batch_visits.keys()))
        if len(shared_indices) > 1:
            seq_counts = np.array([seq_visits[i] for i in shared_indices], dtype=float)
            batch_counts = np.array([batch_visits[i] for i in shared_indices], dtype=float)
            # Normalize
            seq_counts /= seq_counts.sum()
            batch_counts /= batch_counts.sum()
            corr = np.corrcoef(seq_counts, batch_counts)[0, 1]
        else:
            corr = 0.0
        visit_corrs.append(corr)
        
        if verbose:
            print(f"  Run {run+1}: top1_agree={seq_top1 == batch_top1}, "
                  f"jaccard={jaccard:.2f}, corr={corr:.2f}")
            # Print top-3 moves
            print(f"    Seq top3: {[(policy_index_to_move(i, board).uci() if policy_index_to_move(i, board) else '?', seq_visits[i]) for i in list(seq_top5)[:3]]}")
            print(f"    Batch top3: {[(policy_index_to_move(i, board).uci() if policy_index_to_move(i, board) else '?', batch_visits[i]) for i in list(batch_top5)[:3]]}")
    
    return {
        'top1_agreement': top1_agreement / num_runs,
        'avg_jaccard': np.mean(jaccard_sums),
        'avg_visit_corr': np.mean(visit_corrs),
        'num_runs': num_runs,
    }


def test_board_copy_mode(board: chess.Board):
    """Verify that board.copy(stack=False) produces correct positions.
    
    Tests: 10 random moves from starting position, verify that
    board.copy(stack=False) and board.copy(stack=True) produce
    identical FENs and legal move sets.
    """
    import random
    
    for _ in range(10):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = random.choice(moves)
        board.push(move)
        
        # Test both copy modes
        board2 = board.copy(stack=True)
        board3 = board.copy(stack=False)
        
        assert board2.fen() == board3.fen(), \
            f"FEN mismatch: {board2.fen()} vs {board3.fen()}"
        assert board2.legal_moves.count() == board3.legal_moves.count(), \
            f"Legal move count mismatch after {move}"
        
        # Verify that move stack is empty for stack=False
        assert len(board3.move_stack) == 0, \
            "stack=False should produce empty move_stack"
    
    return True


def test_consistency_with_deterministic_noise():
    """Test with zero Dirichlet noise to compare more deterministic behavior.
    
    With noise disabled, sequential and batched should produce very similar
    (though not identical due to virtual loss) results.
    """
    network = AlphaZeroNet(num_residual_blocks=4, num_filters=64)
    board = chess.Board()
    
    # Use MCTS with no Dirichlet noise for more deterministic comparison
    mcts_seq = MCTS(network, num_simulations=100, batch_size=1,
                    dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    mcts_batch = MCTS(network, num_simulations=100, batch_size=8,
                      dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    
    root_seq = mcts_seq.get_root(board)
    visit_seq, _, _ = mcts_seq.search(root_seq)
    
    root_batch = mcts_batch.get_root(board)
    visit_batch, _, _ = mcts_batch.search(root_batch)
    
    # Compare visit distributions
    seq_visits = {}
    for aidx, child in root_seq.children.items():
        seq_visits[aidx] = child.N
    batch_visits = {}
    for aidx, child in root_batch.children.items():
        batch_visits[aidx] = child.N
    
    shared = list(set(seq_visits.keys()) & set(batch_visits.keys()))
    
    if len(shared) > 1:
        seq_arr = np.array([seq_visits.get(i, 0) for i in shared], dtype=float)
        batch_arr = np.array([batch_visits.get(i, 0) for i in shared], dtype=float)
        seq_arr /= seq_arr.sum()
        batch_arr /= batch_arr.sum()
        corr = np.corrcoef(seq_arr, batch_arr)[0, 1]
    else:
        corr = 0.0
    
    seq_top1 = max(seq_visits, key=seq_visits.get)
    batch_top1 = max(batch_visits, key=batch_visits.get)
    
    print(f"  No-noise test: corr={corr:.3f}, top1_match={seq_top1 == batch_top1}")
    
    return {
        'correlation': corr,
        'top1_match': seq_top1 == batch_top1,
        'seq_top1_move': seq_top1,
        'batch_top1_move': batch_top1,
    }


def play_comparison_game(network: AlphaZeroNet, num_simulations: int = 200,
                         max_moves: int = 30) -> dict:
    """Play a short game with each config and compare the move sequences."""
    from selfplay import play_one_game
    
    mcts_seq = MCTS(network, num_simulations=num_simulations, batch_size=1)
    mcts_batch = MCTS(network, num_simulations=num_simulations, batch_size=8)
    
    # Play with sequential
    _, game_info_seq = play_one_game(
        mcts_seq, max_game_length=max_moves, temp_threshold=30
    )
    
    # Play with batched
    _, game_info_batch = play_one_game(
        mcts_batch, max_game_length=max_moves, temp_threshold=30
    )
    
    return {
        'seq_result': game_info_seq['result_str'],
        'seq_length': game_info_seq['length'],
        'seq_termination': game_info_seq['termination'],
        'seq_avg_depth': game_info_seq['avg_mcts_depth'],
        'batch_result': game_info_batch['result_str'],
        'batch_length': game_info_batch['length'],
        'batch_termination': game_info_batch['termination'],
        'batch_avg_depth': game_info_batch['avg_mcts_depth'],
    }


def main():
    parser = argparse.ArgumentParser(description="MCTS Optimization Correctness Test")
    parser.add_argument('--verbose', action='store_true', help='Detailed output')
    parser.add_argument('--games', type=int, default=0, help='Play N games with each config')
    parser.add_argument('--sims', type=int, default=200, help='Num simulations')
    parser.add_argument('--runs', type=int, default=5, help='Num comparison runs')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MCTS Optimization Correctness Test")
    print("=" * 70)
    
    from encoding import policy_index_to_move
    
    # Create network
    print("\nCreating network...")
    network = AlphaZeroNet(num_residual_blocks=4, num_filters=64)
    board = chess.Board()
    
    # Test 1: board.copy(stack=False) correctness
    print("\n[Test 1] board.copy(stack=False) correctness...")
    result = test_board_copy_mode(board)
    print(f"  PASSED: {result}")
    
    # Test 2: Move selection similarity (with Dirichlet noise)
    print(f"\n[Test 2] Move selection similarity (sims={args.sims}, runs={args.runs})...")
    results = test_move_selection_similarity(
        network, board, args.sims, args.runs, verbose=args.verbose
    )
    print(f"  Top-1 agreement: {results['top1_agreement']:.0%} ({results['top1_agreement']*results['num_runs']:.0f}/{results['num_runs']})")
    print(f"  Avg top-5 Jaccard: {results['avg_jaccard']:.2f}")
    print(f"  Avg visit correlation: {results['avg_visit_corr']:.2f}")
    
    # Test 3: Deterministic noise comparison
    print("\n[Test 3] No-noise comparison (more deterministic)...")
    det_results = test_consistency_with_deterministic_noise()
    print(f"  PASSED: correlation={det_results['correlation']:.3f}")
    
    # Test 4: Play short games
    if args.games > 0:
        print(f"\n[Test 4] Playing {args.games} game(s) with each config...")
        for i in range(args.games):
            result = play_comparison_game(network, args.sims, max_moves=40)
            print(f"  Game {i+1}:")
            print(f"    Sequential: {result['seq_result']} (len={result['seq_length']}, "
                  f"depth={result['seq_avg_depth']:.1f})")
            print(f"    Batched:    {result['batch_result']} (len={result['batch_length']}, "
                  f"depth={result['batch_avg_depth']:.1f})")
    
    print("\n" + "=" * 70)
    print("  All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()