import warnings, time, sys, json, os, argparse, random
warnings.filterwarnings('ignore')
import numpy as np, torch, chess
import multiprocessing as mp
from measure_maxlen import config, net, build_mcts, forced_mate, _memo
from selfplay import play_one_game

def _init_worker():
    torch.set_num_threads(1)

def _run_base(seed):
    global _memo
    _memo = {}
    np.random.seed(seed); random.seed(seed)
    m = build_mcts(force=False)
    fens = []
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
        adjudicate_scaling=config.selfplay.adjudicate_scaling,
        on_move=lambda fen, mv, mn, ms=None: fens.append(fen))
    return {"seed": seed, "termination": info["termination"],
            "result_str": info["result_str"], "length": info["length"],
            "seconds": round(time.perf_counter() - t0, 1),
            "fens": fens}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="output/119/measure_convertibility.jsonl")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line); done.add(r["seed"])
            except Exception:
                pass
    seeds = [s for s in range(args.start, args.start + args.count) if s not in done]
    print(f"base games to run: {len(seeds)} (threads=1, {args.workers} workers)", flush=True)

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_base, seeds, chunksize=1), 1):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            el = time.perf_counter() - t0; eta = el / i * (len(seeds) - i)
            print(f"  seed={r['seed']:3d} {r['termination']:20s} {r['result_str']:6s} "
                  f"len={r['length']:3d}  done {i}/{len(seeds)} eta~{eta/60:.0f}min",
                  flush=True)
    print(f"finished {len(seeds)} games in {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()