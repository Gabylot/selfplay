"""Isolate the source of the White-checkmate bias.

Plays N short self-play games with a FRESH RANDOM network (untrained)
using sequential MCTS (batch_size=1, no virtual loss).  If the White
bias already appears here, the bug is structural (encoding / self-play
mechanics).  If results are balanced, the bias is training-induced.

Usage:
    python test_color_balance.py [num_games] [num_sims] [batch_size]
"""

import sys
import numpy as np
import chess
from collections import Counter

from network import AlphaZeroNet
from mcts import MCTS
from selfplay import play_one_game


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print(f"Random network, {n_games} games, {n_sims} sims, batch_size={batch_size}")

    net = AlphaZeroNet(
        num_residual_blocks=8,
        num_filters=64,
        num_policy_channels=64,
        num_value_channels=1,
        value_fc_size=128,
    )
    net.eval()

    wins = Counter()      # 'white' / 'black' / 'draw'
    terms = Counter()
    term_results = Counter()   # (termination, 'white'/'black'/'draw') -> count
    white_cm = black_cm = 0

    for i in range(n_games):
        mcts = MCTS(
            network=net,
            num_simulations=n_sims,
            c_puct=1.5,
            dirichlet_alpha=0.25,
            dirichlet_epsilon=0.25,
            batch_size=batch_size,
            c_virtual_loss=0.5,
            max_game_length=200,
            adjudicate_material=False,
            piece_values={'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9},
        )
        gd, gi = play_one_game(
            mcts_engine=mcts,
            max_game_length=200,
            adjudicate_material=False,
            piece_values={'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9},
            temp_threshold=20,
            temp_high=1.0,
            temp_low=0.1,
        )
        r = gi['result']
        if r > 0.5:
            outcome = 'white'
        elif r < -0.5:
            outcome = 'black'
        else:
            outcome = 'draw'
        wins[outcome] += 1
        terms[gi['termination']] += 1
        term_results[(gi['termination'], outcome)] += 1
        if gi['termination'] == 'checkmate':
            if r > 0.5:
                white_cm += 1
            elif r < -0.5:
                black_cm += 1

        if (i + 1) % 10 == 0:
            print(f"  after {i+1:3d} games: white={wins['white']:3d} "
                  f"black={wins['black']:3d} draw={wins['draw']:3d}", flush=True)

    print("\n=== RESULTS ===")
    print(f"Games:    {n_games}")
    print(f"White:    {wins['white']}  ({100.0*wins['white']/n_games:.1f}%)")
    print(f"Black:    {wins['black']}  ({100.0*wins['black']/n_games:.1f}%)")
    print(f"Draws:    {wins['draw']}  ({100.0*wins['draw']/n_games:.1f}%)")
    print(f"White cm: {white_cm}")
    print(f"Black cm: {black_cm}")
    print(f"\nTerminations: {dict(terms)}")

    print("\n--- W/B/D split by termination type ---")
    for term in sorted(terms):
        w = term_results[(term, 'white')]
        b = term_results[(term, 'black')]
        d = term_results[(term, 'draw')]
        print(f"  {term:16s} total={terms[term]:3d}  "
              f"white={w:3d}  black={b:3d}  draw={d:3d}")

    tw = wins['white'] + wins['black']
    if tw > 0:
        frac = wins['white'] / tw
        n = tw
        # Standard error of the proportion under H0: p=0.5
        se = np.sqrt(0.25 / n)
        z = (frac - 0.5) / se
        print(f"\nWhite share of decisive games: {frac:.3f}  "
              f"(z={z:.2f} vs H0 p=0.5, n={n})")


if __name__ == "__main__":
    main()