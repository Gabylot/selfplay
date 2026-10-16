import sqlite3, sys, math

db = sys.argv[1] if len(sys.argv) > 1 else 'output/119/stats.db'
con = sqlite3.connect(db)
cur = con.cursor()

rows = cur.execute(
    "SELECT game_id, step, result, result_str, length, termination "
    "FROM game_log ORDER BY game_id"
).fetchall()
print(f"{db}: {len(rows)} games")

def is_white_win(r):
    return r is not None and r > 0.5
def is_black_win(r):
    return r is not None and r < -0.5

W = sum(1 for _,_,r,_,_,_ in rows if is_white_win(r))
B = sum(1 for _,_,r,_,_,_ in rows if is_black_win(r))
D = len(rows) - W - B
print(f"ALL: white={W} black={B} draw={D}")
dec = W + B
if dec:
    z = (W/dec - 0.5) / math.sqrt(0.25/dec)
    print(f"white share of decisive = {W/dec:.4f}  (z={z:.2f})")

# checkmate breakdown
cm_rows = [r for r in rows if r[5] == 'checkmate']
cw = sum(1 for _,_,r,_,_,_ in cm_rows if is_white_win(r))
cb = sum(1 for _,_,r,_,_,_ in cm_rows if is_black_win(r))
print(f"CHECKMATE: white={cw} black={cb}  (decisive cm={len(cm_rows)})")

# bucket 0: first 500 games
def bucket(lo, hi, label):
    sl = [r for r in rows if lo <= r[0] < hi]
    if not sl:
        print(f"{label}: no games")
        return
    w = sum(1 for _,_,r,_,_,_ in sl if is_white_win(r))
    b = sum(1 for _,_,r,_,_,_ in sl if is_black_win(r))
    cm = [r for r in sl if r[5]=='checkmate']
    cw = sum(1 for _,_,r,_,_,_ in cm if is_white_win(r))
    cb = sum(1 for _,_,r,_,_,_ in cm if is_black_win(r))
    print(f"{label}: n={len(sl)} white={w} black={b} draw={len(sl)-w-b} | cm white={cw} cm black={cb}")

bucket(0, 500, "games  0-499 ")
bucket(500, 1000, "games 500-999 ")
bucket(1000, 2000, "games 1k-2k  ")
bucket(2000, 3500, "games 2k-3.5k")
bucket(3500, len(rows), "games 3.5k+  ")

# bucket by training step instead?  check step range
steps = [r[1] for r in rows]
print("step range:", min(steps), max(steps), "step_0 count:", steps.count(0))