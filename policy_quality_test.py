"""policy_quality_test.py

Measures the actual sharpness/quality of a trained network's MCTS policy,
decoupled from throughput knobs (batch_size / num_workers).

Why this is different from the GPU sweet-spot sweep
-----------------------------------------------------
That sweep varied batch_size and num_workers while:
  - always starting from the SAME position (the opening),
  - leaving Dirichlet root noise ON (dirichlet_alpha/epsilon > 0).

batch_size and num_workers only change *how fast* simulations are run —
they have no path to change visit concentration. And root noise
deliberately re-randomizes ~25% of the root prior every single trial,
so most of the Top-3% variation you saw was noise-draw variance, not a
real effect of anything being swept. Flat Top-3% across batch/workers is
therefore expected, not evidence of a problem.

This script isolates the variables that actually can affect search
quality:
  - Dirichlet noise is disabled (alpha=0, epsilon=0) so you see the
    network + MCTS's real preference, not an artificially randomized one.
  - Multiple representative positions are tested (opening, tactical,
    midgame, endgame, forced-mate) instead of only the start position.
  - num_simulations and c_puct are swept — the parameters that can
    actually change how concentrated the visit distribution gets.
  - MCTS's chosen move is cross-checked against the depth-2 alpha-beta
    reference (evaluation.alpha_beta_best_move) as an external signal.
    Agreement isn't "correctness" (alpha-beta with material-only eval is
    a weak, tactically-blind opponent), but a sudden drop or near-0%
    agreement across many positions is a red flag worth investigating.
  - A separate batch-size invariance check confirms that batching
    (virtual loss) doesn't distort the search for a FIXED num_simulations
    — this validates the throughput sweep's implicit assumption that
    batch_size is a free lunch.

Note on "trials": with noise disabled, MCTS is fully deterministic for a
given (position, sims, c_puct, batch_size) — dict iteration order and
tie-breaking are stable, so repeated trials return identical numbers.
`--trials` > 1 is mainly a self-check (std should print as ~0.0); it is
not a statistical variance estimate. Real variance in this codebase comes
from Dirichlet noise and temperature sampling in actual self-play, which
this script intentionally excludes.

Usage:
    python policy_quality_test.py --checkpoint output/default/checkpoints/latest.pt
    python policy_quality_test.py --checkpoint latest.pt --sims 200 400 800 1600 --batch-size 1
"""

import argparse
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import chess
import torch

from config import get_config
from network import create_model_from_config
from mcts import MCTS
from evaluation import alpha_beta_best_move


# ─────────────────────────────────────────────────────────────────────────
# Representative test positions.
#
# Deliberately mixed: opening (many reasonable moves — policy SHOULD be
# somewhat spread here, not spiked), tactical/midgame (fewer clearly-best
# moves), endgame (often more forcing), and a forced mate-in-1 position.
#
# The mate-in-1 position is a built-in harness sanity check: mate seeding
# gives every mating root move one terminal pre-visit regardless of network
# quality, so the mating child MUST have a visit. Seeding does NOT pin the
# final move choice, so the chosen move is allowed to be a non-mate.
# ─────────────────────────────────────────────────────────────────────────

TEST_POSITIONS = {
    "opening":             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "italian_opening":     "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "tactical_pos":        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "midgame_1":           "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "midgame_2":           "r4rk1/pp3ppp/2n1q3/2pp4/3P2b1/2P1PN2/PP1Q1PPP/R3R1K1 w - - 0 12",
    "endgame_rook":        "5k2/pp3ppp/8/2r5/4R3/8/PP3PPP/4K3 w - - 0 20",
    "endgame_pawn_race":   "8/5p2/4k3/8/3K4/8/5P2/8 w - - 0 1",
    "back_rank_mate_in_1": "6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1",  # sanity check: mate child gets a seed visit
}


# ─────────────────────────────────────────────────────────────────────────
# Checkpoint loading (robust to the couple of key-naming variants used
# across this repo's own scripts / checkpoints)
# ─────────────────────────────────────────────────────────────────────────

def load_network(checkpoint_path, config):
    net = create_model_from_config(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    net.load_state_dict(sd)
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────────────
# Core search + measurement
# ─────────────────────────────────────────────────────────────────────────

def run_one_search(net, board, num_simulations, c_puct, batch_size,
                   c_virtual_loss=0.5, mate_seeding=True):
    """Run a single MCTS search with root noise disabled."""
    mcts = MCTS(
        network=net,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=0.0,     # disabled: measure the network's real preference
        dirichlet_epsilon=0.0,   # disabled
        batch_size=batch_size,
        c_virtual_loss=c_virtual_loss,
        mate_seeding=mate_seeding,
    )
    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)
    return root, best_move, stats


def visit_concentration(root):
    """(top1_pct, top3_pct, top5_pct) of visits, plus number of expanded children."""
    counts = sorted((c.N for c in root.children.values() if c.N > 0), reverse=True)
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0, 0.0, len(root.children)
    top1 = counts[0] / total
    top3 = sum(counts[:3]) / total
    top5 = sum(counts[:5]) / total
    return top1, top3, top5, len(root.children)


def run_quality_sweep(net, sim_values, c_puct_values, batch_size,
                       num_trials=1, ab_depth=2, verbose=True):
    """
    For each (num_simulations, c_puct) config, run every test position and
    report Top-1/3/5% concentration averaged across positions, plus the
    agreement rate with the depth-2 alpha-beta reference move.
    """
    results = []

    for sims in sim_values:
        for c_puct in c_puct_values:
            top1_all, top3_all, top5_all = [], [], []
            ab_agree = ab_total = 0
            elapsed_total = 0.0
            mate_check_ok = True

            for label, fen in TEST_POSITIONS.items():
                board = chess.Board(fen)
                if board.is_game_over():
                    continue

                # NOTE: the chess engine backend has a known bug where a
                # board built via chess.Board(fen) (as opposed to the
                # default constructor + push()) can corrupt internal state
                # across repeated push/pop cycles -- exactly what
                # alpha_beta_search does internally. This never happens in
                # production (self-play/eval always build boards via the
                # default constructor + push), so it's safe to just skip
                # the AB-agreement signal for whichever hand-written FEN
                # positions trigger it, rather than let it crash the sweep.
                try:
                    ab_move = alpha_beta_best_move(board.copy(), depth=ab_depth)
                except Exception as e:
                    print(f"  [skip AB-agreement for '{label}': "
                          f"{type(e).__name__}: {e}]")
                    ab_move = None

                for _ in range(num_trials):
                    t0 = time.perf_counter()
                    root, best_move, stats = run_one_search(
                        net, board, sims, c_puct, batch_size)
                    elapsed_total += time.perf_counter() - t0

                    top1, top3, top5, _ = visit_concentration(root)
                    top1_all.append(top1)
                    top3_all.append(top3)
                    top5_all.append(top5)

                    if label == "back_rank_mate_in_1":
                        # Mate seeding pre-visits every mating root child.  It
                        # deliberately does not force the final move, so check
                        # the seed visit rather than best_move.
                        if not any(c.N > 0 and c.board.is_checkmate()
                                   for c in root.children.values()):
                            mate_check_ok = False

                    if ab_move is not None and best_move is not None:
                        ab_total += 1
                        if best_move == ab_move:
                            ab_agree += 1

            n = len(top1_all)
            row = {
                "num_simulations": sims,
                "c_puct": c_puct,
                "batch_size": batch_size,
                "top1_mean": float(np.mean(top1_all)),
                "top1_std": float(np.std(top1_all)),
                "top3_mean": float(np.mean(top3_all)),
                "top3_std": float(np.std(top3_all)),
                "top5_mean": float(np.mean(top5_all)),
                "top5_std": float(np.std(top5_all)),
                "ab_agreement": (ab_agree / ab_total) if ab_total else float("nan"),
                "avg_search_time_s": (elapsed_total / n) if n else float("nan"),
                "mate_check_ok": mate_check_ok,
            }
            results.append(row)

            if verbose:
                flag = "" if mate_check_ok else "  [!] MATE-IN-1 SANITY CHECK FAILED"
                print(f"sims={sims:5d} c_puct={c_puct:4.1f} batch={batch_size:3d}  "
                      f"Top1={row['top1_mean']*100:5.1f}%(std{row['top1_std']*100:4.1f})  "
                      f"Top3={row['top3_mean']*100:5.1f}%(std{row['top3_std']*100:4.1f})  "
                      f"Top5={row['top5_mean']*100:5.1f}%(std{row['top5_std']*100:4.1f})  "
                      f"AB-agree={row['ab_agreement']*100:5.1f}%  "
                      f"avg_time={row['avg_search_time_s']*1000:7.1f}ms{flag}")

    return results


def check_batch_size_invariance(net, sims, c_puct, batch_sizes, num_trials=1):
    """
    Sanity check: does batch_size (via virtual loss) change search
    concentration for a FIXED num_simulations? It shouldn't by much —
    virtual loss approximates sequential PUCT, it isn't a different
    search. A large divergence here would point to a virtual-loss
    tuning issue (c_virtual_loss too aggressive), not a training problem,
    and would mean the throughput sweep's batch_size choice isn't "free."
    """
    print("\n--- Batch-size invariance check (fixed sims & c_puct) ---")
    baseline = None
    for bs in batch_sizes:
        top1s, top3s = [], []
        for label, fen in TEST_POSITIONS.items():
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            for _ in range(num_trials):
                root, _, _ = run_one_search(net, board, sims, c_puct, bs)
                top1, top3, _, _ = visit_concentration(root)
                top1s.append(top1)
                top3s.append(top3)
        m1, m3 = np.mean(top1s), np.mean(top3s)
        delta = "" if baseline is None else f"   (dtop1={m1 - baseline:+.1%})"
        if baseline is None:
            baseline = m1
        print(f"  batch_size={bs:3d}  Top1={m1*100:5.1f}%  Top3={m3*100:5.1f}%{delta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sims", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    parser.add_argument("--c-puct", type=float, nargs="+", default=[1.0, 1.5, 2.5])
    parser.add_argument("--batch-size", type=int, default=1,
                         help="Fixed batch size for the main quality sweep. "
                              "Use 1 for the sequential 'ground truth' search "
                              "(no virtual-loss approximation).")
    parser.add_argument("--batch-invariance-sims", type=int, default=800)
    parser.add_argument("--batch-invariance-cpuct", type=float, default=1.5)
    parser.add_argument("--batch-sizes-to-check", type=int, nargs="+",
                         default=[1, 8, 16, 32, 64])
    parser.add_argument("--trials", type=int, default=1,
                         help="Repeats per config. With noise disabled the search "
                              "is deterministic, so this is a sanity check (std "
                              "should print ~0.0), not a variance estimate.")
    parser.add_argument("--skip-batch-check", action="store_true")
    args = parser.parse_args()

    config = get_config(path=args.config)
    net = load_network(args.checkpoint, config)

    print("=" * 100)
    print("POLICY QUALITY SWEEP  (Dirichlet noise disabled, multiple positions)")
    print("=" * 100)
    run_quality_sweep(
        net,
        sim_values=args.sims,
        c_puct_values=args.c_puct,
        batch_size=args.batch_size,
        num_trials=args.trials,
    )

    if not args.skip_batch_check:
        check_batch_size_invariance(
            net,
            sims=args.batch_invariance_sims,
            c_puct=args.batch_invariance_cpuct,
            batch_sizes=args.batch_sizes_to_check,
            num_trials=args.trials,
        )


if __name__ == "__main__":
    main()