"""Diagnostic: is the white/black win-rate imbalance real, and where does it come from?

Usage:  python diag_color_imbalance.py [--db output/default/stats.db] [--recent N]
"""
import argparse
import math
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="output/default/stats.db")
ap.add_argument("--recent", type=int, default=0,
                help="Restrict analysis to the N most recent selfplay games.")
ap.add_argument("--step-window", type=int, default=0,
                help="Restrict to games from the last N training steps.")
args = ap.parse_args()

con = sqlite3.connect(args.db)
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("tables:", [r[0] for r in cur.fetchall()])
cur.execute("PRAGMA table_info(game_log)")
print("game_log columns:", [r[1] for r in cur.fetchall()])
cur.execute("SELECT COUNT(*), MIN(step), MAX(step) FROM game_log")
n, smin, smax = cur.fetchone()
print(f"game_log rows: {n}  (step {smin}..{smax})")


def report(title, rows):
    """rows: list of (label, white_wins, black_wins, draws)."""
    print(f"\n=== {title} ===")
    print(f"{'segment':<28} {'W':>7} {'B':>7} {'D':>7} "
          f"{'W%':>7} {'B%':>7} {'score%':>7} {'z':>7}")
    for label, w, b, d in rows:
        t = w + b + d
        if t == 0:
            continue
        wp, bp = 100.0 * w / t, 100.0 * b / t
        score = 100.0 * (w + 0.5 * d) / t
        # Two-proportion z-test on decisive games only (H0: W == B).
        dec = w + b
        if dec > 0:
            p0 = 0.5
            z = (w / dec - p0) / math.sqrt(p0 * (1 - p0) / dec)
        else:
            z = float("nan")
        print(f"{label:<28} {w:>7} {b:>7} {d:>7} "
              f"{wp:>6.1f}% {bp:>6.1f}% {score:>6.1f}% {z:>7.1f}")


# ── Overall ──────────────────────────────────────────────────────────────────
# Results can be graded material adjudications (any value in [-1,1]), so group
# by SIGN, not by exact equality to +/-1.0/0.0.
where, params = [], []
if args.recent:
    where.append("id > (SELECT MAX(id) - ? FROM game_log)")
    params.append(args.recent)
if args.step_window:
    where.append("step >= (SELECT MAX(step) - ? FROM game_log)")
    params.append(args.step_window)
wsql = (" WHERE " + " AND ".join(where)) if where else ""


def sign_groups(where_clause, wp):
    """Return (white, black, draw) counts grouping result by sign."""
    sql = ("SELECT CASE WHEN result > 0 THEN 'W' WHEN result < 0 THEN 'B' "
           "ELSE 'D' END AS g, COUNT(*) FROM game_log"
           + where_clause + " GROUP BY g")
    d = dict((r[0], r[1]) for r in cur.execute(sql, wp).fetchall())
    return d.get("W", 0), d.get("B", 0), d.get("D", 0)


def seg_where(op, args_val, base):
    if not args_val:
        return base
    clause = (" AND " if base else " WHERE ") + op + " ?"
    return base + clause


# ── Overall ──────────────────────────────────────────────────────────────────
w, b, d = sign_groups(wsql, params)
report("Overall", [("all games", w, b, d)])

# ── By termination reason ────────────────────────────────────────────────────
cur.execute(f"SELECT DISTINCT termination FROM game_log{wsql} ORDER BY 1",
            params)
terms = [r[0] for r in cur.fetchall()]
rows = []
for t in terms:
    wc = wsql + (" AND " if wsql else " WHERE ") + "termination = ?"
    rows.append((t, *sign_groups(wc, params + [t])))
report("By termination", rows)

# ── By training step (banded) ────────────────────────────────────────────────
cur.execute(f"SELECT MIN(step), MAX(step) FROM game_log{wsql}", params)
s0, s1 = cur.fetchone()
band = max(1, (s1 - s0 + 1) // 10)
rows = []
for lo in range(s0, s1 + 1, band):
    hi = lo + band - 1
    wc = (" WHERE step BETWEEN ? AND ?"
          + ((" AND " + " AND ".join(where)) if where else ""))
    wsql_b, params_b = wc, [lo, hi] + params
    rows.append((f"step {lo}-{hi}", *sign_groups(wsql_b, params_b)))
report("By training step", rows)

# ── By game length band ──────────────────────────────────────────────────────
rows = []
for lo, hi, label in [(0, 30, "len 1-30"), (31, 60, "len 31-60"),
                      (61, 90, "len 61-90"), (91, 120, "len 91-120"),
                      (121, 149, "len 121-149"), (150, 10**9, "len >=150")]:
    wc = (" WHERE length BETWEEN ? AND ?"
          + ((" AND " + " AND ".join(where)) if where else ""))
    rows.append((label, *sign_groups(wc, [lo, hi] + params)))
report("By game length", rows)

# ── Self-play vs adjudication source ─────────────────────────────────────────
rows = []
for t in ["checkmate", "material_white", "material_black", "max_length",
          "repetition", "insufficient_material", "fifty_moves", "stalemate"]:
    wc = wsql + (" AND " if wsql else " WHERE ") + "termination = ?"
    cnt = sign_groups(wc, params + [t])
    if sum(cnt):
        rows.append((t, *cnt))
report("By termination (grouped)", rows)

# ── material_diff conditional on result sign ─────────────────────────────────
cur.execute(
    "SELECT CASE WHEN result > 0 THEN 'white win' WHEN result < 0 "
    "THEN 'black win' ELSE 'draw' END AS g, AVG(material_diff), COUNT(*) "
    "FROM game_log" + wsql + " GROUP BY g", params)
print("\n=== Avg material_diff (white - black) by result sign ===")
for g, avg, c in cur.fetchall():
    print(f"  {g:<12} avg_diff={avg if avg is None else round(avg, 2):>7}  n={c}")

