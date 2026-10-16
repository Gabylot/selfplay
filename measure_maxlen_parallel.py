import warnings, time, sys, json, os, argparse
warnings.filterwarnings('ignore')
import torch
import multiprocessing as mp
from measure_maxlen import run_one

def _init_worker():
    torch.set_num_threads(1)

def _run_task(task):
    cond, seed = task
    return run_one(force=(cond == "force"), seed=seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="output/119/measure_maxlen_big.jsonl")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["condition"], r["seed"]))
            except Exception:
                pass

    tasks = []
    for seed in range(args.start, args.start + args.count):
        for cond in ("base", "force"):
            if (cond, seed) not in done:
                tasks.append((cond, seed))

    print(f"total seeds [start={args.start}, +{args.count}); "
          f"{len(tasks)} tasks to run with {args.workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), 1):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            el = time.perf_counter() - t0
            rate = el / i
            eta = rate * (len(tasks) - i)
            print(f"  [{r['condition']:5s}] seed={r['seed']:3d} "
                  f"{r['termination']:20s} {r['result_str']:6s} len={r['length']:3d} "
                  f"forced={r['forced']}  done {i}/{len(tasks)}  "
                  f"eta~{eta/60:.0f}min", flush=True)

    print(f"\nfinished {len(tasks)} tasks in {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()