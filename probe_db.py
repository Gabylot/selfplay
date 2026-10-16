import sqlite3, sys, os

db = sys.argv[1] if len(sys.argv) > 1 else 'output/119/stats.db'
con = sqlite3.connect(db)
cur = con.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('tables:', tables)

for t in tables:
    try:
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n} rows, cols={cols}")
    except Exception as e:
        print(f"  {t}: err {e}")