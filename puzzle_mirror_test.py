"""Puzzle colour-mirror test on a checkpoint (frame version 2).

Builds tactical puzzles (mate-in-1 and free/hanging-piece), then presents each
BOTH as-is (P) and as its true chess colour mirror (sigma P = flip ranks + swap
colours).  The "correct move" for the mirror is the sigma-image of the original
solution.  Tempo / first-move advantage cannot apply, so any accuracy gap
between White-to-move and Black-to-move presentations is pure net colour bias.

On the colour-neutral frame (rank-flip encoder + mover-relative policy) the net
sees *identical* input tensors for P and sigma P, and the policy index commutes
with sigma, so the accuracy gap must be exactly 0 for any checkpoint -- trained
or not.  A nonzero gap means the encoder / policy-indexing frame is broken, or
the checkpoint predates the current frame (which is refused loudly on load).

Usage:
    python puzzle_mirror_test.py --checkpoint output/default/checkpoints/latest.pt
    python puzzle_mirror_test.py --checkpoint output/run/checkpoints/latest.pt \
        --num-puzzles 2000 --seed 0
"""
import argparse
import os
import random
import sys
import warnings
from collections import Counter

# The embedded interpreter used here does not put the script directory on
# sys.path, so add it explicitly (keeps the script runnable from anywhere).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

import numpy as np
import torch

import chess
import game_api

game_api.set_game("chess")

from config import get_config
from network import create_model_from_config, load_checkpoint
from encoding import get_legal_move_mask, policy_index_to_move
from probe_opening_diversity_gpu import reflect

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def sq_sigma(s):
    """Rank-flip (a1<->a8), files preserved -- matches the frame's encoder."""
    return (s & 7) | ((7 - (s >> 3)) << 3)


def move_sigma(m):
    return chess.Move(sq_sigma(m.from_square), sq_sigma(m.to_square),
                      promotion=m.promotion)


def gives_mate(b, m):
    b.push(m)
    ok = b.is_checkmate()
    b.pop()
    return ok


def hanging_captures(b):
    """All no-recapture captures of an undefended piece worth >=3 at max value."""
    bestg, best = 0, []
    for m in list(b.legal_moves):
        if not b.is_capture(m):
            continue
        vic = b.piece_at(m.to_square)
        vv = VAL[vic.piece_type] if vic is not None else 1
        if vv < 3:
            continue
        opp = not b.turn
        b.push(m)
        defended = b.is_attacked_by(opp, m.to_square)
        b.pop()
        if defended:
            continue
        if vv > bestg:
            bestg, best = vv, [m]
        elif vv == bestg:
            best.append(m)
    return best


def gen(target, seed=0, max_iters=600000):
    rng = random.Random(seed)
    out, it = [], 0
    while len(out) < target and it < max_iters:
        it += 1
        b = chess.Board()
        for ply in range(rng.randint(8, 70)):
            if b.is_game_over():
                break
            lm = list(b.legal_moves)
            if not lm:
                break
            caps = [m for m in lm if b.is_capture(m)]
            m = rng.choice(caps) if (caps and rng.random() < 0.5) else rng.choice(lm)
            b.push(m)
            if ply >= 6:
                sol = [mm for mm in b.legal_moves if gives_mate(b, mm)]
                if sol:
                    out.append((b.fen(), [mm.uci() for mm in sol], "mate1"))
                    break
                hg = hanging_captures(b)
                if hg:
                    out.append((b.fen(), [mm.uci() for mm in hg], "hanging"))
                    break
    return out, it


def encode_all(puzzles):
    """Return (states, metas) with both orientations of every puzzle."""
    sts, metas = [], []
    for fen, sol, kind in puzzles:
        P = chess.Board(fen)
        Q = chess.Board(reflect(fen))
        solq = [move_sigma(chess.Move.from_uci(s)).uci() for s in sol]
        for B, solset in ((P, sol), (Q, solq)):
            sts.append(game_api.encode(B))
            metas.append((B.turn, set(solset), B))
    return np.stack(sts), metas


class Raw:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predictBatch(self, S):
        x = torch.from_numpy(np.asarray(S, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            p, v = self.net(x)
            return torch.softmax(p, -1).cpu().numpy(), v.cpu().numpy().reshape(-1)

    predict_batch = predictBatch


def score(predict_batch, sts, metas, label):
    outs = []
    for i in range(0, len(sts), 512):
        pr = predict_batch(sts[i:i + 512])
        outs.append(np.asarray(pr[0] if isinstance(pr, tuple) else pr))
    P = np.concatenate(outs)
    okW = nW = okB = nB = 0
    for k, (turn, solset, B) in enumerate(metas):
        mask = get_legal_move_mask(B)
        idx = int((P[k] + (1 - mask) * (-1e9)).argmax())
        mv = policy_index_to_move(idx, B)
        good = 1 if (mv is not None and mv.uci() in solset) else 0
        if turn == chess.WHITE:
            nW += 1
            okW += good
        else:
            nB += 1
            okB += good
    aw, ab = okW / max(nW, 1), okB / max(nB, 1)
    gap = aw - ab
    print(f"{label:<5}{nW:>8}{aw:>8.3f}{nB:>8}{ab:>8.3f}{gap:>+10.3f}")
    return gap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a frame-version-2 checkpoint (.pt).")
    ap.add_argument("--num-puzzles", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-9,
                    help="Max |accuracy gap| accepted as a passing frame (default 1e-9).")
    args = ap.parse_args()

    puzzles, iters = gen(args.num_puzzles, seed=args.seed)
    print(f"generated {len(puzzles)} puzzles in {iters} random games")
    print("  kinds:", dict(Counter(k for _, _, k in puzzles)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   checkpoint: {args.checkpoint}")

    cfg = get_config()
    net = create_model_from_config(cfg)
    load_checkpoint(args.checkpoint, net)   # raises FrameVersionError on stale frames
    net.to(device)
    net.eval()
    raw = Raw(net, device)

    worst_gap = 0.0
    for kind in ("ALL", "mate1", "hanging"):
        sub = puzzles if kind == "ALL" else [p for p in puzzles if p[2] == kind]
        if not sub:
            continue
        print(f"\n--- {kind} (n={len(sub)}) ---")
        sts, metas = encode_all(sub)
        print(f"{'net':<5}{'White #':>8}{'W acc':>8}{'Black #':>8}{'B acc':>8}{'gap(W-B)':>10}")
        gap = score(raw.predictBatch, sts, metas, "RAW")
        if kind == "ALL":
            worst_gap = abs(gap)

    if worst_gap <= args.tolerance:
        print(f"\nPASS: colour-mirror gap = {worst_gap:.2e} (<= {args.tolerance:g}) "
              f"-- frame is colour-neutral.")
    else:
        print(f"\nFAIL: colour-mirror gap = {worst_gap:.2e} (> {args.tolerance:g}) "
              f"-- the encoder/policy frame is NOT colour-neutral.")
        sys.exit(1)


if __name__ == "__main__":
    main()
