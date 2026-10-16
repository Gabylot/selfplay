import warnings, time
warnings.filterwarnings('ignore')
import numpy as np, torch, chess, glob
from collections import OrderedDict
from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS
from encoding import policy_index_to_move
from evaluation import alpha_beta_best_move

# (fen, category)  -- tactical vs quiet, to test within-set variance
FENS = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "OPENING"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "TACTICAL"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "TACTICAL"),
    ("r1bqkbnr/pp1ppppp/2n5/2p5/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4", "TACTICAL"),
    ("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4", "TACTICAL"),
    ("rnbqk2r/pppp1ppp/4pn2/8/2P5/2N2N2/PP1PPPP1/R1BQKB1R w KQkq - 0 4", "TACTICAL"),
    ("3r1rk1/pp3ppp/2pb4/8/8/2P5/PP3PPP/R2R2K1 w - - 3 12", "MIXED"),
    ("rr2kbnr/pp3ppp/2n1p3/8/8/8/PPPPP1PP/R4RK1 w - - 0 1", "MIXED"),
    ("4k3/8/8/8/3K4/8/4P3/8 w - - 0 1", "QUIET"),
    ("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", "QUIET"),
    ("8/8/8/3k4/3P4/2K5/8/8 w - - 0 1", "QUIET"),
    ("rnbqkbnr/ppppppp1/8/8/6p1/8/PPPPPPPP/RNBQKBNR w KQkq - 4 4", "MIXED"),
]

SIM = 800
CK_TREND = [200, 800, 1400, 2000, 2800]          # early→mid→late
CK_AB = 2800                                       # latest for the A/B
GPU = False

config = get_config()
device = torch.device("cpu")

def entropy(p):
    nz = p[p > 0]
    return -np.sum(nz * np.log(nz)) if nz.size else 0.0

def make_mcts(net, c_puct, eps):
    return MCTS(net, num_simulations=SIM, c_puct=c_puct,
                dirichlet_alpha=config.mcts.dirichlet_alpha,
                dirichlet_epsilon=eps,
                batch_size=32, c_virtual_loss=config.mcts.c_virtual_loss,
                max_game_length=config.selfplay.max_game_length,
                adjudicate_material=False)

def load_net(cp):
    net = create_model_from_config(config).to(device)
    load_checkpoint(cp, net); net.eval()
    return net

def argmax_move(vp, board):
    idx = int(np.argmax(vp))
    return policy_index_to_move(idx, board)

# ── Reference moves (alpha-beta, net-independent ground truth) ──
print("=== computing alpha-beta reference moves (depth 4 & 5) ===")
refs = []
for fen, _cat in FENS:
    b = chess.Board(fen)
    r4 = alpha_beta_best_move(b.copy(), 4)
    r5 = alpha_beta_best_move(b.copy(), 5)
    refs.append((r4, r5))
    print(f"  {fen.split()[0]:38s} ab4={r4}  ab5={r5}")

# ── Part A: entropy trend at 800 sims + per-FEN variance ──
print("\n=== PART A: target entropy at %d sims across checkpoints ===" % SIM)
grid = OrderedDict((c, {}) for c in CK_TREND)
for ck in CK_TREND:
    net = load_net(f"output/119/checkpoints/step_{ck}.pt")
    m = make_mcts(net, config.mcts.c_puct, config.mcts.dirichlet_epsilon)
    for j, (fen, cat) in enumerate(FENS):
        b = chess.Board(fen)
        vp, _, _ = m.search(m.get_root(b))
        grid[ck][j] = entropy(vp)
    print(f"  step {ck}: mean={np.mean(list(grid[ck].values())):.3f} "
          f"by-category TACTICAL={np.mean([grid[ck][i] for i,c in enumerate(FENS) if c[1]=='TACTICAL']):.2f} "
          f"QUIET={np.mean([grid[ck][i] for i,c in enumerate(FENS) if c[1]=='QUIET']):.2f}")

print("\n  per-FEN entropy matrix (sims=%d, rows=FEN, cols=checkpoints):" % SIM)
hdr = "  %-38s " % "FEN / cat"
for ck in CK_TREND: hdr += f"{ck:>6}"
print(hdr)
for j, (fen, cat) in enumerate(FENS):
    row = "  %-38s " % fen.split()[0]
    for ck in CK_TREND:
        row += f"{grid[ck][j]:>6.2f}"
    print(row + f"  [{cat}]")

# ── Part B: A/B entropy AND move-quality at 800 sims, latest ckpt ──
print("\n=== PART B: sharpening A/B at 800 sims, checkpoint %d ===" % CK_AB)
CONFIGS = OrderedDict([
    ("baseline  c1.5 e.25", (1.5, 0.25)),
    ("low_puct  c0.8 e.25", (0.8, 0.25)),
    ("low_eps   c1.5 e.10", (1.5, 0.10)),
    ("low_both  c0.8 e.10", (0.8, 0.10)),
])
net = load_net(f"output/119/checkpoints/step_{CK_AB}.pt")
for name, (cp, eps) in CONFIGS.items():
    m = make_mcts(net, cp, eps)
    Hs, agree4, agree5, tot = [], 0, 0, 0
    for i, (fen, _cat) in enumerate(FENS):
        b = chess.Board(fen)
        vp, bm, _ = m.search(m.get_root(b))
        mv = policy_index_to_move(int(np.argmax(vp)), b) or bm
        Hs.append(entropy(vp))
        if mv and refs[i][0] and mv == refs[i][0]: agree4 += 1
        if mv and refs[i][1] and mv == refs[i][1]: agree5 += 1
        tot += 1
    print(f"  {name:20s} H={np.mean(Hs):.3f} eff={np.exp(np.mean(Hs)):.1f}  "
          f"agree_ab4={agree4}/{tot}  agree_ab5={agree5}/{tot}")