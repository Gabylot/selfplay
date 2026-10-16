import json, collections, math

PATH = "output/119/measure_maxlen1t.jsonl"
rows = {}
for line in open(PATH):
    r = json.loads(line)
    rows.setdefault(r["seed"], {})[r["condition"]] = r

seeds = sorted(rows)
base_c, force_c = [], []
pairs = []
for s in seeds:
    b, f = rows[s].get("base"), rows[s].get("force")
    if b is None or f is None:
        continue
    base_c.append(b); force_c.append(f)
    pairs.append((s, b, f))

def stall(r):  # game that never reached a decisive result
    return r["termination"] in ("max_length", "fifty_moves")

n = len(pairs)
b_stall = sum(stall(b) for _, b, _ in pairs)
f_stall = sum(stall(f) for _, _, f in pairs)
conv = sum(1 for _, b, f in pairs if stall(b) and not stall(f) and f["termination"] == "checkmate")
revert = sum(1 for _, b, f in pairs if not stall(b) and stall(f))
lost = sum(1 for _, b, f in pairs if not stall(b) and not stall(f)
           and b["result_str"] != f["result_str"])

print(f"paired games (threads=1, deterministic): {n}")
print(f"\nbase  : max_length+fifty = {b_stall:2d}/{n} ({100*b_stall/n:.0f}%)"
      f"   | non-decisive total = {sum(1 for _,b,_ in pairs if b['termination']!='checkmate')}/{n}")
print(f"force : max_length+fifty = {f_stall:2d}/{n} ({100*f_stall/n:.0f}%)"
      f"   | non-decisive total = {sum(1 for _,_,f in pairs if f['termination']!='checkmate')}/{n}")
print(f"\nconverted to checkmate : {conv}/{b_stall}   (stall -> checkmate)")
print(f"regressed (checkmate->stall): {revert}")
print(f"result changed on non-stall games: {lost}")

print("\nper-seed pair table:")
print(f"{'seed':>4} {'base':<22}{'force':<24} forced")
for s, b, f in pairs:
    mark = "  <-- converted" if (stall(b) and not stall(f)) else \
           "  <-- REGRESSED" if (not stall(b) and stall(f)) else ""
    print(f"{s:>4} {b['termination']+f' len={b['length']}':<22}"
          f"{f['termination']+f' len={f['length']}':<24} {f['forced']}{mark}")

# McNemar exact test on discordant pairs
discordant = conv + revert
p = 0.5 ** discordant * (2 if discordant else 1) if discordant else 1.0
print(f"\nMcNemar (exact binomial on {discordant} discordant pairs): p ~ {p:.4f}")

# outcomes
res_b = collections.Counter(b["result_str"] for _, b, _ in pairs)
res_f = collections.Counter(f["result_str"] for _, _, f in pairs)
print(f"\noutcomes base : {dict(res_b)}")
print(f"outcomes force: {dict(res_f)}")
