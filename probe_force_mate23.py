import warnings, time, sys, random
warnings.filterwarnings('ignore')
import numpy as np, torch, chess
from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS

CKPT = "output/119/checkpoints/latest.pt"
config = get_config()
net = create_model_from_config(config)
load_checkpoint(CKPT, net)
net.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Exact forced-mate solver (proof-based, independent of the value head)
# ─────────────────────────────────────────────────────────────────────────────
_memo = {}
def forced_mate(board, plies):
    """Return a move from `board` (side to move) that forces checkmate within
    `plies` plies (1, 3, or 5), or None.  Memoized on (fen, plies)."""
    if board.is_game_over() or plies <= 0:
        return None
    key = (board.fen(), plies)
    if key in _memo:
        return _memo[key]
    if plies == 1:
        for m in board.legal_moves:
            nb = board.copy(stack=False); nb.push(m)
            if nb.is_checkmate():
                _memo[key] = m
                return m
        _memo[key] = None
        return None
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            _memo[key] = m
            return m
        if nb.is_game_over():
            continue                      # stalemate/draw after m -> not mate
        ok = True
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r)
            if rb.is_game_over():
                ok = False; break         # black replied with mate/stalemate
            if forced_mate(rb, plies - 2) is None:
                ok = False; break
        if ok:
            _memo[key] = m
            return m
    _memo[key] = None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MCTS subclass that forces mate in 1 / 2 / 3 at move selection
# ─────────────────────────────────────────────────────────────────────────────
class ForceMateMCTS(MCTS):
    def __init__(self, *a, max_force_plies=5, min_pieces_gate=8, **k):
        super().__init__(*a, **k)
        self.max_force_plies = max_force_plies
        self.min_pieces_gate = min_pieces_gate
        self.forced = 0                    # times override fired
        self.forced_log = []               # (move, plies) entries

    def select_move_with_temperature(self, root, temperature):
        # exact proof override, gated to sparse positions (cheap)
        b = root.board
        n_pieces = len(b.piece_map())
        if n_pieces <= self.min_pieces_gate:
            for plies in range(1, self.max_force_plies + 1, 2):
                fm = forced_mate(b, plies)
                if fm is not None:
                    from mcts import NUM_ACTIONS, move_to_policy_index
                    idx = move_to_policy_index(fm, b)
                    vp = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    vp[idx] = 1.0
                    self.forced += 1
                    self.forced_log.append((fm, plies, b.fen()))
                    return vp, fm
        return super().select_move_with_temperature(root, temperature)

def make_mcts(sims, eps, cls, max_len=120):
    return cls(net, num_simulations=sims, c_puct=config.mcts.c_puct,
               dirichlet_alpha=config.mcts.dirichlet_alpha,
               dirichlet_epsilon=eps, batch_size=config.mcts.batch_size,
               c_virtual_loss=config.mcts.c_virtual_loss,
               max_game_length=max_len, adjudicate_material=False)

def play_game(fen, sims, eps, max_len=120, seed=0):
    rng = random.Random(seed)
    m = make_mcts(sims, eps, ForceMateMCTS, max_len)
    board = chess.Board(fen)
    moves = []
    root = None
    t0 = time.perf_counter()
    while (not board.is_game_over()
           and not board.is_repetition(3)
           and len(moves) < max_len):
        if root is None:
            root = m.get_root(board)
        m.search(root)
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
    return {"term": term, "result": result, "converted": conv,
            "length": len(moves), "moves": moves, "forced": m.forced,
            "forced_log": m.forced_log, "dt": dt}

START = [
    ("KQ far ", "8/8/8/8/2Q5/8/4K3/7k w - - 0 1"),
    ("KQ near", "7k/8/8/5K2/8/6Q1/8/8 w - - 0 1"),
    ("KR     ", "8/8/8/8/8/3R4/3K4/7k w - - 0 1"),
    ("KRR    ", "7k/8/8/8/8/8/5R2/K5R1 w - - 0 1"),
]
SETTINGS = [(800, 0.0), (2000, 0.0), (2000, 0.25)]

print("=" * 78)
print("FORCE MATE 1/2/3 override — replay of probe_conversion games")
print("base outcome  -> override outcome   (same MCTS seed, temp=0.1)")
print("=" * 78)
BASE = {  # from probe_conversion run: (start, sims, eps) -> outcome
    ("KQ far ", 800, 0.0): "threefold",
    ("KQ far ", 2000, 0.0): "max_length",
    ("KQ far ", 2000, 0.25): "max_length",
    ("KQ near", 800, 0.0): "mate",
    ("KQ near", 2000, 0.0): "mate",
    ("KQ near", 2000, 0.25): "mate",
    ("KR     ", 800, 0.0): "mate",
    ("KR     ", 2000, 0.0): "max_length",
    ("KR     ", 2000, 0.25): "max_length",
    ("KRR    ", 800, 0.0): "mate",
    ("KRR    ", 2000, 0.0): "mate",
    ("KRR    ", 2000, 0.25): "mate",
}
for tag, fen in START:
    for sims, eps in SETTINGS:
        g = play_game(fen, sims, eps)
        base = BASE[(tag, sims, eps)]
        fire = ""
        if g["forced"]:
            fm, plies, _ = g["forced_log"][0]
            fire = f"  override fired at move {g['length']-1 if not g['converted'] else 'start'}: {fm} (mate in {(plies+1)//2})"
        print(f"{tag} sims={sims:4d} eps={eps:<4} "
              f"base={base:10s} -> override={g['term']:10s} {g['result']:6s} "
              f"len={g['length']:3d} forced={g['forced']}{fire}  [{g['dt']:.0f}s]")
        if not g["converted"]:
            mv = " ".join(m.uci() for m in g["moves"][:10])
            print(f"      {mv} ...")
        sys.stdout.flush()
print("done")
