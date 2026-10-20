import sqlite3

con = sqlite3.connect("output/default/stats.db")
c = con.cursor()
rows = list(c.execute("SELECT id,step,result,termination FROM game_log ORDER BY id"))
con.close()

bounds = [0, 2802, 7446, 12446, 17446, 22446, 26634, 28019, 28750, len(rows)]


def zc(w, b):
    n = w + b
    return (w - b) / (n ** 0.5) if n else 0.0


def tally(seg, terms):
    w = b = 0
    for gid, st, res, term in seg:
        if term in terms:
            if res > 0.001:
                w += 1
            elif res < -0.001:
                b += 1
    return w, b


def pct(a, b):
    return f"{100*a/(a+b):4.1f}%" if a + b else "  -  "


print(f"{'seg ids':<14}{'step range':<16}"
      f"{'CHECKMATE W/B (W%)':<28}{'MATERIAL W/B (W%)':<28}")
for i in range(len(bounds) - 1):
    seg = rows[bounds[i]:bounds[i + 1]]
    if not seg:
        continue
    sr0, sr1 = seg[0][1], seg[-1][1]
    cw, cb = tally(seg, ("checkmate",))
    mw, mb = tally(seg, ("material_white", "material_black"))
    print(f"{bounds[i]:>4}-{bounds[i+1]-1:<9}{sr0:>6}-{sr1:<7}  "
          f"{cw:>4}/{cb:<4} {pct(cw,cb)} z={zc(cw,cb):+5.1f}    "
          f"{mw:>4}/{mb:<4} {pct(mw,mb)} z={zc(mw,mb):+5.1f}")
