import warnings, time
warnings.filterwarnings('ignore')
import numpy as np, torch, chess
from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS

FEN = "6k1/8/8/8/8/4R3/5R2/K7 w - - 0 1"
CKPT = "output/119/checkpoints/latest.pt"
SIM = 800

config = get_config()
net = create_model_from_config(config)
load_checkpoint(CKPT, net)
net.eval()

def run(eps, bs):
    m = MCTS(net, num_simulations=SIM, c_puct=config.mcts.c_puct,
             dirichlet_alpha=config.mcts.dirichlet_alpha,
             dirichlet_epsilon=eps, batch_size=bs,
             c_virtual_loss=config.mcts.c_virtual_loss,
             max_game_length=config.selfplay.max_game_length,
             adjudicate_material=False)
    b = chess.Board(FEN)
    root = m.get_root(b)
    t0 = time.perf_counter()
    vp, bm, stats = m.search(root)
    dt = time.perf_counter() - t0
    print(f"eps={eps} bs={bs}: best={bm} rootQ={root.Q:+.3f} "
          f"top3={[(c.move.uci(), int(c.N)) for c in sorted(root.children.values(), key=lambda c:-c.N)[:3]]} "
          f"{dt:.1f}s")
    return vp

v_seq0 = run(0.0, 1)
v_seqN = run(0.25, 1)
v_bat0 = run(0.0, 32)
v_batN = run(0.25, 32)

print("\ndeterminism check:")
print("  seq eps0 vs bat eps0 identical:", np.array_equal(v_seq0, v_bat0),
      " max_abs_diff=", np.abs(v_seq0 - v_bat0).max())
print("  seq eps.25 vs bat eps.25 identical:", np.array_equal(v_seqN, v_batN),
      " max_abs_diff=", np.abs(v_seqN - v_batN).max())

# mate-in-distance for the winning line
b = chess.Board(FEN)
print("\nlegal moves:", b.legal_moves.count())
print("black can capture a rook?", any(b.is_capture(m) for m in b.legal_moves))
