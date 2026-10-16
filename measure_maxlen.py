import warnings, time, sys, json, os, argparse, random
warnings.filterwarnings('ignore')
import numpy as np, torch, chess
from config import get_config
from network import create_model_from_config, load_checkpoint
from mcts import MCTS, NUM_ACTIONS, move_to_policy_index
from selfplay import play_one_game

CKPT = "output/119/checkpoints/latest.pt"
config = get_config()
net = create_model_from_config(config)
load_checkpoint(CKPT, net)
net.eval()

# ── exact forced-mate solver (validated in verify_force_mate.py) ──
_memo = {}
def forced_mate(board, plies):
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
            continue
        ok = True
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r)
            if rb.is_game_over():
                ok = False; break
            if forced_mate(rb, plies - 2) is None:
                ok = False; break
        if ok:
            _memo[key] = m
            return m
    _memo[key] = None
    return None

class ForceMateMCTS(MCTS):
    def __init__(self, *a, max_force_plies=5, min_pieces_gate=8, **k):
        super().__init__(*a, **k)
        self.max_force_plies = max_force_plies
        self.min_pieces_gate = min_pieces_gate
        self.forced = 0

    def select_move_with_temperature(self, root, temperature):
        b = root.board
        if len(b.piece_map()) <= self.min_pieces_gate:
            for plies in range(1, self.max_force_plies + 1, 2):
                fm = forced_mate(b, plies)
                if fm is not None:
                    idx = move_to_policy_index(fm, b)
                    vp = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    vp[idx] = 1.0
                    self.forced += 1
                    return vp, fm
        return super().select_move_with_temperature(root, temperature)

def build_mcts(force):
    cls = ForceMateMCTS if force else MCTS
    return cls(net, num_simulations=config.mcts.num_simulations,
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

def run_one(force, seed):
    global _memo
    _memo = {}
    np.random.seed(seed)
    random.seed(seed)
    m = build_mcts(force)
    t0 = time.perf_counter()
    _, info = play_one_game(
        mcts_engine=m,
        max_game_length=config.selfplay.max_game_length,
        adjudicate_material=config.selfplay.adjudicate_material,
        piece_values=config.selfplay.piece_values,
        temp_threshold=config.selfplay.temperature_threshold,
        temp_high=config.selfplay.temperature_high,
        temp_low=config.selfplay.temperature_low,
        adjudicate_graded=config.selfplay.adjudicate_graded,
        adjudicate_scaling=config.selfplay.adjudicate_scaling)
    dt = time.perf_counter() - t0
    forced = getattr(m, "forced", 0)
    return {"condition": "force" if force else "base", "seed": seed,
            "termination": info["termination"], "result_str": info["result_str"],
            "length": info["length"], "forced": forced, "seconds": round(dt, 1)}

OUT = "output/119/measure_maxlen.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["base", "force"], required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    force = args.condition == "force"

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["condition"], r["seed"]))
                except Exception:
                    pass
    tally = {}
    for i in range(args.count):
        seed = args.start + i
        if (args.condition, seed) in done:
            print(f"  seed {seed}: already done, skipping", flush=True)
            continue
        r = run_one(force, seed)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        tally[r["termination"]] = tally.get(r["termination"], 0) + 1
        print(f"  [{args.condition}] seed={seed:3d} {r['termination']:20s} "
              f"{r['result_str']:6s} len={r['length']:3d} forced={r['forced']} "
              f"{r['seconds']:.0f}s   -> {tally}", flush=True)

if __name__ == "__main__":
    main()
