"""Is the White edge real chess or a truncation/adjudication artifact?

Slices game_log by termination and re-derives the White-vs-Black decisive
split, overall and banded by training step and game length, focusing on games
that ended by CHECKMATE (a real chess result) vs material adjudication at the
max-length cutoff.
"""
import argparse
import sqlite3
import sys

sys.path.insert(0, r"f:\python\selfplay")

_ap = argparse.ArgumentParser()
_ap.add_argument("--db", default="output/default/stats.db")
DB = _ap.parse_args().db


def ztest(w, b):
    n = w + b
    if n == 0:
        return float("nan")
    return (w - b) / (n ** 0.5)


con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("SELECT step, result, length, termination, material_diff "
            "FROM game_log")
rows = cur.fetchall()
con.close()

print(f"game_log rows: {len(rows)}")

CHECKMATE = "checkmate"
MATERIAL = ("material_white", "material_black")
TRUNC = ("material_white", "material_black", "max_length")
DRAW = ("insufficient_material", "repetition", "fifty_moves", "stalemate",
        "fivefold_repetition", "seventyfive_moves", "max_length", "draw")


def sign(r):
    if r > 0:
        return "W"
    if r < 0:
        return "B"
    return "D"


def report(title, sel):
    w = b = d = 0
    for st, r, ln, t, md in rows:
        if not sel(st, r, ln, t, md):
            continue
        s = sign(r)
        w += s == "W"
        b += s == "B"
        d += s == "D"
    z = ztest(w, b)
    tot = w + b + d
    wd = w / (w + b) if (w + b) else float("nan")
    print(f"{title:<34}{w:>7}{b:>7}{d:>7}"
          f"{100*w/tot:>8.1f}%{100*b/tot:>8.1f}%"
          f"{100*wd:>9.1f}%{z:>8.2f}")
    return w, b, d


print("\n=== Decisive split by termination definition ===")
print(f"{'segment':<34}{'W':>7}{'B':>7}{'D':>7}{'W%':>9}{'B%':>9}"
      f"{'W/(W+B)':>10}{'z':>8}")
report("ALL games", lambda *a: True)
report("CHECKMATE only (real chess)", lambda st, r, ln, t, md: t == CHECKMATE)
report("MATERIAL adjudication only", lambda st, r, ln, t, md: t in MATERIAL)
report("NON-TRUNCATED (all but material/max_len)",
       lambda st, r, ln, t, md: t not in TRUNC)

print("\n=== material winners: strength of the margin ===")
for label, want in (("material_white", "W"), ("material_black", "B")):
    diffs = [abs(md) for st, r, ln, t, md in rows if t == label]
    if diffs:
        diffs.sort()
        n = len(diffs)
        print(f"  {label:<16} n={n:>6}  |diff| mean={sum(diffs)/n:6.2f}  "
              f"median={diffs[n//2]:5.1f}  p90={diffs[int(n*0.9)]:5.1f}")

print("\n=== CHECKMATE only, by training step ===")
smax = max(st for st, *_ in rows)
band = 3200
print(f"{'step band':<20}{'W':>7}{'B':>7}{'W/(W+B)':>10}{'z':>8}")
for lo in range(0, smax + 1, band):
    hi = lo + band
    report(f"{lo}-{hi-1}",
           lambda st, r, ln, t, md, lo=lo, hi=hi:
           (t == CHECKMATE and lo <= st < hi))

print("\n=== CHECKMATE only, by game length ===")
print(f"{'len band':<20}{'W':>7}{'B':>7}{'W/(W+B)':>10}{'z':>8}")
for lo in range(0, 150, 30):
    hi = lo + 30
    report(f"len {lo}-{hi-1}",
           lambda st, r, ln, t, md, lo=lo, hi=hi:
           (t == CHECKMATE and lo <= ln < hi))

print("\n=== Material adjudication, by training step ===")
print(f"{'step band':<20}{'W':>7}{'B':>7}{'W/(W+B)':>10}{'z':>8}")
for lo in range(0, smax + 1, band):
    hi = lo + band
    report(f"{lo}-{hi-1}",
           lambda st, r, ln, t, md, lo=lo, hi=hi:
           (t in MATERIAL and lo <= st < hi))
