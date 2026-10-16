import warnings, time, sys
warnings.filterwarnings('ignore')
import numpy as np, torch, chess
from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS
from encoding import NUM_ACTIONS

CKPT = "output/119/checkpoints/latest.pt"
SIM  = 800

config = get_config()
device = torch.device("cpu")
net = create_model_from_config(config)
load_checkpoint(CKPT, net)
net.eval()

def make_mcts(eps, bs, cls=MCTS, **kw):
    return cls(net, num_simulations=SIM, c_puct=config.mcts.c_puct,
               dirichlet_alpha=config.mcts.dirichlet_alpha,
               dirichlet_epsilon=eps, batch_size=bs,
               c_virtual_loss=config.mcts.c_virtual_loss,
               max_game_length=config.selfplay.max_game_length,
               adjudicate_material=False, **kw)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — broad root-Q probe over clearly-winning sparse endgames
# ─────────────────────────────────────────────────────────────────────────────
# (fen, expected)  expected: WIN -> should be ~+1; DRAW -> ~0
POSITIONS = [
    # KQ vs K
    ("7k/8/8/8/8/8/8/KQ6 w - - 0 1", "WIN"),
    ("8/8/8/3k4/8/8/8/KQ6 w - - 0 1", "WIN"),
    ("8/8/4k3/8/8/3Q4/8/4K3 w - - 0 1", "WIN"),
    ("5k2/8/8/8/8/4Q3/8/4K3 w - - 0 1", "WIN"),
    ("7k/8/8/8/8/6Q1/8/7K w - - 0 1", "WIN"),
    # KR vs K
    ("7k/8/8/8/8/8/8/R6K w - - 0 1", "WIN"),
    ("8/8/4k3/8/8/3R4/8/4K3 w - - 0 1", "WIN"),
    ("6k1/8/8/8/8/2R5/8/4K3 w - - 0 1", "WIN"),
    ("8/8/8/8/3k4/8/8/1RK5 w - - 0 1", "WIN"),
    # KRR vs K
    ("6k1/8/8/8/8/4R3/5R2/K7 w - - 0 1", "WIN"),
    ("7k/8/8/8/8/8/5R2/K5R1 w - - 0 1", "WIN"),
    ("8/8/8/2k5/8/8/3RR3/3K4 w - - 0 1", "WIN"),
    # KBB vs K
    ("6k1/8/8/8/8/8/1B2B3/7K w - - 0 1", "WIN"),
    ("8/8/8/3k4/8/8/2B1B3/2K5 w - - 0 1", "WIN"),
    # KBN vs K  (hardest canonical win)
    ("7k/8/8/8/8/8/5N2/5BK1 w - - 0 1", "WIN"),
    ("8/8/8/3k4/8/8/2N1B3/2K5 w - - 0 1", "WIN"),
    # KQ vs minor
    ("8/8/8/2k5/8/8/3Q2B1/3K4 w - - 0 1", "WIN"),
    ("8/8/8/2k5/8/8/2NQ4/2K5 w - - 0 1", "WIN"),
    # KR vs KN
    ("8/8/8/2k5/8/8/3R1N2/3K4 w - - 0 1", "WIN"),
    # KPP / KP (clearly winning, advanced + supported)
    ("8/8/8/8/8/4P3/4K3/7k w - - 0 1", "WIN"),
    ("8/8/8/8/8/8/2KP4/7k w - - 0 1", "WIN"),
    ("8/8/8/8/8/8/2PKP3/k7 w - - 0 1", "WIN"),
    # Controls — draws
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "DRAW"),
    ("4k3/8/8/8/8/8/5B2/4K3 w - - 0 1", "DRAW"),
    ("4k3/8/8/8/8/8/5N2/4K3 w - - 0 1", "DRAW"),
    ("4k3/8/8/8/8/8/3N1N2/4K3 w - - 0 1", "DRAW"),
    ("4k3/7r/8/8/8/8/3R4/4K3 w - - 0 1", "DRAW"),
]

print("=" * 88)
print("PART 1 — root Q, 800 sims, latest checkpoint, across sparse endgames")
print("=" * 88)
print(f"{'material':<14}{'Q(b1,e0)':>10}{'Q(b32,e0)':>10}{'Q(b32,e.25)':>12}  verdict")
rows = []
for fen, exp in POSITIONS:
    try:
        b = chess.Board(fen)
    except Exception as e:
        print(f"{fen.split()[0]:<14}  INVALID FEN ({e})")
        continue
    m1 = make_mcts(0.0, 1); r1 = m1.get_root(b); m1.search(r1); q1 = r1.Q
    m2 = make_mcts(0.0, 32); r2 = m2.get_root(b); m2.search(r2); q2 = r2.Q
    m3 = make_mcts(0.25, 32); r3 = m3.get_root(b); m3.search(r3); q3 = r3.Q
    mat = fen.split()[0]
    near_win = q2 > 0.5
    verdict = ("known" if exp == "WIN" and near_win else
               "WRONG/0" if exp == "WIN" else
               "correct" if exp == "DRAW" and abs(q2) < 0.5 else "??")
    rows.append((mat, q1, q2, q3, exp, verdict))
    print(f"{mat:<14}{q1:>+10.3f}{q2:>+10.3f}{q3:>+12.3f}  {exp:4s} {verdict}")

n_win = sum(1 for r in rows if r[4] == "WIN")
n_known = sum(1 for r in rows if r[4] == "WIN" and r[3] > 0.5)
print(f"\nwinning positions known to value head: {n_known}/{n_win}  "
      f"(root Q > 0.5 @ batch32, eps0)")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — terminal vs network backprop origins down the winning line
# ─────────────────────────────────────────────────────────────────────────────
class TrackingMCTS(MCTS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.origins = {}           # root-child uci -> {origin: count}
        self.node_origin = {}       # id(node) -> origin tag
        self.total_backprops = 0

    def _get_terminal_value(self, node):
        v = super()._get_terminal_value(node)
        res = node.board.result()
        if res != "*":
            self.node_origin[id(node)] = "real_terminal"
        elif node.board.ply() >= self.max_game_length:
            self.node_origin[id(node)] = "adjudicated"
        else:
            self.node_origin[id(node)] = "soft_draw"
        return v

    def _expand_node_with_data(self, node, policy, value):
        self.node_origin[id(node)] = "network"
        return super()._expand_node_with_data(node, policy, value)

    def _record(self, leaf):
        origin = self.node_origin.get(id(leaf), "unknown")
        rc = self._root_child(leaf)
        self.total_backprops += 1
        if rc is None:
            return
        d = self.origins.setdefault(rc.move.uci(),
                                    {"real_terminal": 0, "network": 0,
                                     "soft_draw": 0, "adjudicated": 0, "unknown": 0})
        d[origin] += 1

    def _backpropagate(self, node, value):
        self._record(node)
        super()._backpropagate(node, value)

    def _backpropagate_with_virtual_loss(self, node, value):
        self._record(node)
        super()._backpropagate_with_virtual_loss(node, value)

    @staticmethod
    def _root_child(node):
        cur = node
        while cur.parent is not None and cur.parent.parent is not None:
            cur = cur.parent
        if cur.parent is None:
            return None
        return cur

def instrument(fen, tag, eps, bs, topn=4):
    m = make_mcts(eps, bs, cls=TrackingMCTS)
    b = chess.Board(fen)
    root = m.get_root(b)
    m.search(root)
    print(f"\n  --- {tag} (eps={eps}, batch={bs})  total backprops={m.total_backprops} "
          f"root Q={root.Q:+.3f}")
    # root Q/best already known; print children table
    kids = sorted(root.children.values(), key=lambda c: -c.N)[:topn]
    print(f"    {'move':6s}{'N':>6s}{'Q':>8s}  {'real#':>6s}{'net#':>6s}{'draw#':>6s}"
          f"  {'%terminal':>9s}")
    for c in kids:
        o = m.origins.get(c.move.uci(), {})
        rt = o.get("real_terminal", 0); nw = o.get("network", 0)
        sd = o.get("soft_draw", 0) + o.get("adjudicated", 0)
        pct = 100.0 * rt / (rt + nw + sd) if (rt + nw + sd) else 0.0
        print(f"    {c.move.uci():6s}{c.N:>6d}{c.Q:>+8.3f}  {rt:>6d}{nw:>6d}{sd:>6d}"
              f"  {pct:>8.1f}%")

print("\n" + "=" * 88)
print("PART 2 — backprop origins (real checkmate leaf vs network leaf), 800 sims")
print("=" * 88)
for tag, eps, bs in [("sequential", 0.0, 1), ("sequential", 0.25, 1),
                     ("batched   ", 0.0, 32), ("batched   ", 0.25, 32)]:
    instrument("6k1/8/8/8/8/4R3/5R2/K7 w - - 0 1", tag, eps, bs)
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — sign convention under the batched / virtual-loss code path
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("PART 3 — terminal sign convention: sequential vs batched (virtual loss)")
print("=" * 88)
class MockNet:
    def predict(self, state):
        return (np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS), 0.0
    def predictBatch(self, states_batch):
        n = states_batch.shape[0]
        return (np.ones((n, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS,
                np.zeros(n, dtype=np.float32))

for bs in (1, 32):
    m = MCTS(MockNet(), num_simulations=200, dirichlet_alpha=0.0,
             dirichlet_epsilon=0.0, batch_size=bs, adjudicate_material=False)
    board = chess.Board("r1bqkbnr/1ppp1ppp/p1n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    root = m.get_root(board)
    m.search(root)
    child = next((c for c in root.children.values()
                  if c.move and c.move.uci() == "f3f7"), None)
    ok = child is not None and abs(child.Q - (-1.0)) < 0.01
    print(f"  batch_size={bs:2d}: f3f7 Q={child.Q:+.3f}  "
          f"{'PASS (mate child Q=-1.0)' if ok else 'FAIL'}")
    assert ok, "sign convention broken under batched/virtual-loss path"
print("  batched sign-convention check passed")
