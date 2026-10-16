import warnings, time, sys
warnings.filterwarnings('ignore')
import numpy as np, torch, chess
from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS

FENS  = [
    "6k1/8/8/8/8/4R3/5R2/K7 w - - 0 1",   # K+2R vs K  (mate in 2)
    "7k/8/8/5K2/8/6Q1/8/8 w - - 0 1",     # K+Q vs K  (mate in 1)
]
CKPT  = "output/119/checkpoints/latest.pt"
SIM   = 800

# ── Ground truth: forced-mate search (brute force, python-chess) ──
def can_mate(board, plies):
    """True if side to move can force checkmate within `plies` plies."""
    if board.is_checkmate() or board.is_game_over() or plies <= 0:
        return False
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            return True
        if nb.is_game_over():
            continue
        replies = []
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r); replies.append(rb)
        if all(can_mate(rb, plies - 1) for rb in replies):
            return True
    return False

def mate_moves(board, plies):
    """First moves that force mate within `plies` plies."""
    out = []
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            out.append(m); continue
        if nb.is_game_over():
            continue
        replies = []
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r); replies.append(rb)
        if all(can_mate(rb, plies - 1) for rb in replies):
            out.append(m)
    return out

def ground_truth(fen):
    b = chess.Board(fen)
    print("position:", fen)
    print("side to move:", "White" if b.turn == chess.WHITE else "Black",
          "| legal moves:", len(b.legal_moves))
    for plies in range(1, 10, 2):          # 1,3,5,7,9 plies = mate in 1..5 moves
        if can_mate(b, plies):
            mm = mate_moves(b, plies)
            print(f"  GROUND TRUTH: forced mate in {(plies + 1) // 2} move(s) "
                  f"({plies} plies)  winning first moves: {[m.uci() for m in mm]}")
            return
    print("  GROUND TRUTH: no forced mate found within 5 moves "
          "(could still be longer / not mate)")
    return

config = get_config()
net = create_model_from_config(config)
load_checkpoint(CKPT, net)
net.eval()

def run(fen, tag, eps, bs):
    m = MCTS(net, num_simulations=SIM, c_puct=config.mcts.c_puct,
             dirichlet_alpha=config.mcts.dirichlet_alpha,
             dirichlet_epsilon=eps, batch_size=bs,
             c_virtual_loss=config.mcts.c_virtual_loss,
             max_game_length=config.selfplay.max_game_length,
             adjudicate_material=False)
    b = chess.Board(fen)
    root = m.get_root(b)
    t0 = time.perf_counter()
    vp, bm, _ = m.search(root)
    dt = time.perf_counter() - t0
    top = sorted(root.children.values(), key=lambda c: -c.N)
    print(f"    --- {tag} (eps={eps}, batch={bs}) [{dt:.1f}s] ---")
    print(f"      root Q = {root.Q:+.3f}   best = {bm}")
    for i, c in enumerate(top[:6]):
        print(f"      #{i+1} {c.move.uci():6s} N={c.N:4d}  Q={c.Q:+.3f}")
    sys.stdout.flush()
    return top, root.Q

for fen in FENS:
    print("\n" + "=" * 60)
    ground_truth(fen)
    print("  MCTS at %d sims, latest checkpoint:" % SIM)
    for tag, eps, bs in [("sequential  ", 0.0, 1),
                         ("sequential  ", 0.25, 1),
                         ("batched     ", 0.0, 32),
                         ("batched     ", 0.25, 32)]:
        run(fen, tag, eps, bs)
