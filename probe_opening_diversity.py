"""Opening-diversity probe (NOT training).

Plays the current checkpoint against itself, sequentially, from:
  (a) the standard start  (baseline, what the run always uses), and
  (b) a set of balanced opening FENs, each also played in its colour-mirror,
and reports White's decisive share in each bucket.

If the White edge collapses toward 50% when games no longer funnel through the
single standard opening, the imbalance is opening-convergence-driven; if it
persists, the net/search itself is colour-asymmetric.

Usage: python probe_opening_diversity.py --games 3 --baseline 12 [--sims 200]
"""
import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, r"f:\python\selfplay")

import numpy as np
import chess

from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS
from selfplay import play_one_game

CKPT = "output/default/checkpoints/latest.pt"

# ~8-12 ply, roughly balanced, standard openings (White to move).
OPENINGS = [
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",  # Ruy Lopez
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",     # QGD
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",        # Sicilian
    "rnbqkb1r/ppp1pppp/5n2/3p4/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 0 4",       # KIA
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",    # Italian
    "rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",     # French
]


def reflect(fen):
    """180 deg colour-mirror as a FEN string (colours flipped, board rotated)."""
    parts = fen.split()
    rows = parts[0].split("/")
    out = []
    for r in reversed(rows):
        r = r[::-1]
        out.append("".join(c.lower() if c.isupper() else c.upper() for c in r))
    parts[0] = "/".join(out)
    parts[1] = "b" if parts[1] == "w" else "w"
    parts[2] = "-"  # point reflection doesn't preserve castling files; drop
    if parts[3] != "-":
        f = ord(parts[3][0]) - ord("a")
        rk = int(parts[3][1])
        parts[3] = chr(ord("a") + (7 - f)) + str(9 - rk)
    return " ".join(parts)


def build_mcts(net, config):
    return MCTS(net,
                num_simulations=config.mcts.num_simulations,
                c_puct=config.mcts.c_puct,
                dirichlet_alpha=config.mcts.dirichlet_alpha,
                dirichlet_epsilon=config.mcts.dirichlet_epsilon,
                batch_size=config.mcts.batch_size,
                c_virtual_loss=config.mcts.c_virtual_loss,
                max_game_length=config.selfplay.max_game_length,
                adjudicate_material=config.selfplay.adjudicate_material,
                piece_values=config.selfplay.piece_values,
                adjudicate_graded=config.selfplay.adjudicate_graded,
                adjudicate_scaling=config.selfplay.adjudicate_scaling)


def run_game(net, config, fen, seed):
    np.random.seed(seed)
    m = build_mcts(net, config)
    _, info = play_one_game(
        mcts_engine=m,
        max_game_length=config.selfplay.max_game_length,
        adjudicate_material=config.selfplay.adjudicate_material,
        piece_values=config.selfplay.piece_values,
        temp_threshold=config.selfplay.temperature_threshold,
        temp_high=config.selfplay.temperature_high,
        temp_low=config.selfplay.temperature_low,
        adjudicate_graded=config.selfplay.adjudicate_graded,
        adjudicate_scaling=config.selfplay.adjudicate_scaling,
        fen=fen)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--games", type=int, default=3, help="games per opening (and its mirror)")
    ap.add_argument("--baseline", type=int, default=12, help="games from the standard start")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=100000)
    args = ap.parse_args()

    config = get_config()
    if args.sims:
        config.mcts.num_simulations = args.sims
    net = create_model_from_config(config)
    load_checkpoint(args.ckpt, net)
    net.eval()
    print(f"checkpoint={args.ckpt}  sims={config.mcts.num_simulations}  "
          f"max_len={config.selfplay.max_game_length}")

    buckets = {"baseline_standard": [], "openings_asis": [], "openings_mirror": []}
    seed = args.seed0

    print(f"\n=== baseline: {args.baseline} games from standard start ===")
    for _ in range(args.baseline):
        t = time.time()
        info = run_game(net, config, None, seed)
        seed += 1
        buckets["baseline_standard"].append(info["result"])
        print(f"  {info['termination']:<20} {info['result_str']:<6} "
              f"len={info['length']:>3} ({time.time()-t:.0f}s)")

    for fen in OPENINGS:
        for tag, f in (("openings_asis", fen), ("openings_mirror", reflect(fen))):
            print(f"\n=== {tag}: {f} ===")
            for _ in range(args.games):
                t = time.time()
                info = run_game(net, config, f, seed)
                seed += 1
                buckets[tag].append(info["result"])
                print(f"  {info['termination']:<20} {info['result_str']:<6} "
                      f"len={info['length']:>3} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 66)
    print(f"{'bucket':<22}{'n':>5}{'W':>5}{'B':>5}{'D':>5}"
          f"{'W/(W+B)':>10}{'z':>8}")
    for name, res in buckets.items():
        w = sum(1 for r in res if r > 0.001)
        b = sum(1 for r in res if r < -0.001)
        d = len(res) - w - b
        n = w + b
        wd = 100 * w / n if n else float("nan")
        z = (w - b) / (n ** 0.5) if n else 0.0
        print(f"{name:<22}{len(res):>5}{w:>5}{b:>5}{d:>5}{wd:>9.1f}%{z:>8.2f}")


if __name__ == "__main__":
    main()
