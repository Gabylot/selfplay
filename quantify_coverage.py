import warnings
warnings.filterwarnings('ignore')
import numpy as np, sqlite3, os

BUFFER = "output/119/checkpoints/replay_buffer.npz"
DB     = "output/119/stats.db"

d = np.load(BUFFER, allow_pickle=False)
states = d['states']; values = d['values']
n = len(states)
print(f"buffer: {n} positions  (max_size buffer, {d['total_games']} games total)")

# ── piece-count histogram from the current-position planes (0..11 of ply 0) ──
pieces = states[:, 0:12].sum(axis=(1, 2, 3))     # total pieces on the board
pawns  = states[:, 0].sum(axis=(1, 2)) + states[:, 6].sum(axis=(1, 2))
nonpwn = pieces - pawns

print("\npiece-count histogram:")
hist = np.bincount(pieces.astype(int), minlength=33)
for pc in range(2, 17):
    cnt = hist[pc] if pc < len(hist) else 0
    print(f"  {pc:2d} pieces: {cnt:6d}  ({100*cnt/n:5.2f}%)")

def frac(leq):
    return 100.0 * np.mean(pieces <= leq)
print(f"\nlow-piece fractions of the buffer:")
for leq in (4, 5, 6, 7, 8, 10):
    print(f"  <= {leq} pieces: {frac(leq):6.2f}%  "
          f"({int(np.sum(pieces <= leq))} positions)")
print(f"  non-pawn piece count <= 2 (only pawns+pawns): "
      f"{100*np.mean(nonpwn <= 2):.2f}%")

# ── value targets within low-piece buckets ──
print("\nvalue-target statistics by piece count:")
for lo, hi in [(2, 4), (5, 6), (7, 8), (9, 32)]:
    sel = (pieces >= lo) & (pieces <= hi)
    if sel.sum() == 0:
        print(f"  {lo}-{hi}: none")
        continue
    v = values[sel]
    print(f"  {lo}-{hi} pieces: n={sel.sum():5d}  mean_v={v.mean():+.3f}  "
          f"mean|v|={np.abs(v).mean():.3f}  %|v|>0.5={100*np.mean(np.abs(v)>0.5):.1f}%")

# ── termination distribution from the game log ──
con = sqlite3.connect(DB)
cur = con.cursor()
rows = cur.execute(
    "SELECT termination, COUNT(*), ROUND(AVG(length),1) FROM game_log GROUP BY termination "
    "ORDER BY COUNT(*) DESC").fetchall()
tot = sum(r[1] for r in rows)
print("\ngame termination distribution (self-play games logged):")
for term, cnt, avg_len in rows:
    print(f"  {term:<24} {cnt:6d}  ({100*cnt/tot:5.1f}%)  avg_len={avg_len}")
r = cur.execute("SELECT result, COUNT(*) FROM game_log GROUP BY result").fetchall()
print("\noutcome distribution:")
for res, cnt in r:
    print(f"  {res:<6} {cnt:6d}  ({100*cnt/tot:5.1f}%)")

# games that reached a low piece count at ANY point are not directly logged;
# but game length gives an idea of simplification pressure
r = cur.execute("SELECT AVG(length), MIN(length), MAX(length) FROM game_log").fetchone()
print(f"\nself-play game length: avg={r[0]:.1f} min={r[1]} max={r[2]} half-moves")
