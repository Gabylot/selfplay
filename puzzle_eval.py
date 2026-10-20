"""Net-only tactical puzzle accuracy, split by side to move.

Used by the training loop at checkpoint time (never per move, never in a
worker).  The puzzle set is generated ONCE per process by random play and then
cached, so the expensive part is paid at most once; each checkpoint only runs
a small batched forward pass.

Two puzzle families are built:
  * ``mate1``   -- positions where a legal move is checkmate (mate-in-1);
  * ``hanging`` -- positions with a free (undefended) capture worth >= 3.

Accuracy is the fraction of puzzles whose top masked policy move is any of the
accepted solution moves, reported separately for White-to-move and
Black-to-move positions (colour balance of tactical vision).
"""

import random

import numpy as np

import chess
import game_api
from encoding import get_legal_move_mask, policy_index_to_move

_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

_PUZZLES = None


def _gives_mate(b, m):
    b.push(m)
    ok = b.is_checkmate()
    b.pop()
    return ok


def _hanging_captures(b):
    bestg, best = 0, []
    for m in list(b.legal_moves):
        if not b.is_capture(m):
            continue
        vic = b.piece_at(m.to_square)
        vv = _VAL[vic.piece_type] if vic is not None else 1
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


def generate(num_mate1=256, num_hanging=256, seed=20240101, max_iters=2000000):
    """Generate a mixed puzzle set by random play (cached callers only)."""
    rng = random.Random(seed)
    mate1, hanging = [], []
    it = 0
    while (len(mate1) < num_mate1 or len(hanging) < num_hanging) and it < max_iters:
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
            if ply < 6:
                continue
            if len(mate1) < num_mate1:
                sol = [mm for mm in b.legal_moves if _gives_mate(b, mm)]
                if sol:
                    mate1.append((b.fen(), [mm.uci() for mm in sol], 'mate1'))
                    break
            if len(hanging) < num_hanging:
                hg = _hanging_captures(b)
                if hg:
                    hanging.append((b.fen(), [mm.uci() for mm in hg], 'hanging'))
                    break
    return mate1 + hanging


def get_puzzles(**kwargs):
    global _PUZZLES
    if _PUZZLES is None:
        _PUZZLES = generate(**kwargs)
    return _PUZZLES


def evaluate(predict_batch, puzzles=None):
    """Return accuracy scalars for the current net.

    ``predict_batch(states f32 (N,P,H,W)) -> (priors (N,A), values)``.
    """
    if not game_api.is_game('chess'):
        return {}
    puzzles = puzzles if puzzles is not None else get_puzzles()
    if not puzzles:
        return {}

    states, metas = [], []
    for fen, sol, kind in puzzles:
        b = chess.Board(fen)
        states.append(game_api.encode(b))
        metas.append((b.turn, set(sol), b, kind))
    states = np.stack(states).astype(np.float32)

    outs = []
    for i in range(0, len(states), 256):
        pr = predict_batch(states[i:i + 256])
        outs.append(np.asarray(pr[0] if isinstance(pr, tuple) else pr))
    P = np.concatenate(outs)

    tally = {}          # (kind, colour) -> [ok, n]

    def _add(kind, colour, good):
        k = (kind, colour)
        a = tally.setdefault(k, [0, 0])
        a[0] += good
        a[1] += 1

    for k, (turn, solset, b, kind) in enumerate(metas):
        mask = get_legal_move_mask(b)
        idx = int((P[k] + (1.0 - mask) * (-1e9)).argmax())
        mv = policy_index_to_move(idx, b)
        good = 1 if (mv is not None and mv.uci() in solset) else 0
        colour = 'white' if turn == chess.WHITE else 'black'
        _add(kind, colour, good)
        _add('all', colour, good)

    out = {'puzzle_n': float(len(puzzles))}
    for (kind, colour), (ok, n) in tally.items():
        out[f'puzzle_{kind}_{colour}_acc'] = ok / max(n, 1)
    return out
