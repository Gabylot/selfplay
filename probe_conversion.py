import warnings, time, sys, random
warnings.filterwarnings('ignore')
import numpy as np, torch, chess
from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS

CKPT = "output/119/checkpoints/latest.pt"

config = get_config()
torch.set_num_threads(max(1, torch.get_num_threads()))
net = create_model_from_config(config)
load_checkpoint(CKPT, net)
net.eval()

def make_mcts(sims, eps, max_len):
    return MCTS(net, num_simulations=sims, c_puct=config.mcts.c_puct,
                dirichlet_alpha=config.mcts.dirichlet_alpha,
                dirichlet_epsilon=eps, batch_size=config.mcts.batch_size,
                c_virtual_loss=config.mcts.c_virtual_loss,
                max_game_length=max_len, adjudicate_material=False)

def play_game(fen, sims, eps, max_len=120, seed=0):
    rng = random.Random(seed)
    m = make_mcts(sims, eps, max_len)
    board = chess.Board(fen)
    moves = []
    max_q = 0.0
    min_q = 0.0
    root = None
    t0 = time.perf_counter()
    while (not board.is_game_over()
           and not board.is_repetition(3)
           and len(moves) < max_len):
        if root is None:
            root = m.get_root(board)
        m.search(root)
        max_q = max(max_q, root.Q)
        min_q = min(min_q, root.Q)
        _, selected = m.select_move_with_temperature(root, 0.1)
        if selected is None:
            selected = max(root.children, key=lambda c: c.N).move
        moves.append(selected)
        board.push(selected)
        root = m.recycle_tree(root, selected)
    dt = time.perf_counter() - t0

    if board.is_checkmate():
        term, result = "checkmate", board.result()
    elif board.is_stalemate():
        term, result = "stalemate", "1/2-1/2"
    elif board.is_fivefold_repetition():
        term, result = "fivefold", "1/2-1/2"
    elif board.is_repetition(3):
        term, result = "threefold", "1/2-1/2"
    elif board.is_seventyfive_moves():
        term, result = "75-move", "1/2-1/2"
    elif board.is_insufficient_material():
        term, result = "insufficient", "1/2-1/2"
    else:
        term, result = "max_length", "1/2-1/2"

    conv = term == "checkmate" and result.startswith("1-0")
    return {"fen": fen, "sims": sims, "eps": eps, "term": term, "result": result,
            "converted": conv, "length": len(moves), "max_q": max_q, "min_q": min_q,
            "moves": moves, "dt": dt}

START = [
    ("KQ far  ", "8/8/8/8/2Q5/8/4K3/7k w - - 0 1"),
    ("KQ near ", "7k/8/8/5K2/8/6Q1/8/8 w - - 0 1"),
    ("KR      ", "8/8/8/8/8/3R4/3K4/7k w - - 0 1"),
    ("KRR     ", "7k/8/8/8/8/8/5R2/K5R1 w - - 0 1"),
]

SETTINGS = [(800, 0.0), (2000, 0.0), (2000, 0.25)]

print("=" * 78)
print("CONVERSION FEASIBILITY: full self-play games from sparse endgame starts")
print("latest checkpoint | temp=0.1 (late-game greedy) | tree recycled | max 120 plies")
print("=" * 78)
summary = []
for tag, fen in START:
    for sims, eps in SETTINGS:
        try:
            g = play_game(fen, sims, eps)
        except Exception as e:
            print(f"{tag} sims={sims} eps={eps}: EXCEPTION {e}")
            continue
        mv = " ".join(m.uci() for m in g["moves"][:12])
        if len(g["moves"]) > 12:
            mv += f" ...(+{len(g['moves'])-12})"
        print(f"\n{tag}  sims={sims:4d} eps={eps:<4} "
              f"-> {g['term']:12s} {g['result']:6s} in {g['length']:3d} plies "
              f"[{g['dt']:.0f}s]  Q-range {g['min_q']:+.2f}..{g['max_q']:+.2f}")
        print(f"     {mv}")
        summary.append((tag.strip(), sims, eps, g["term"], g["length"], g["converted"]))
        sys.stdout.flush()

print("\n" + "=" * 78)
print("SUMMARY  (converted = game ended in checkmate)")
print("=" * 78)
for tag, sims, eps, term, length, conv in summary:
    print(f"  {tag:6s} sims={sims:4d} eps={eps:<4} {term:12s} len={length:3d} "
          f"{'CONVERTED' if conv else 'draw/stall'}")
print("done")
