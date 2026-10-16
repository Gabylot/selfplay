import sqlite3, time, sys
name = 'output/200/stats.db'
c = sqlite3.connect(name); cur = c.cursor()
cur.execute("SELECT timestamp FROM game_log ORDER BY id DESC LIMIT 1")
t1 = cur.fetchone()[0]
time.sleep(20)
cur.execute("SELECT timestamp FROM game_log ORDER BY id DESC LIMIT 1")
t2 = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM game_log WHERE timestamp > ?", (t1,))
cnt = cur.fetchone()[0]
dt = t2 - t1
cur.execute("SELECT AVG(length), AVG(num_positions) FROM game_log WHERE timestamp > ?", (t1,))
avg_len, avg_pos = cur.fetchone()
print(f'games/20s: {cnt}  games/sec={cnt/dt:.3f}')
print(f'avg length={avg_len:.0f} pos={avg_pos:.0f}')
# training throughput
cur.execute("SELECT timestamp FROM training_log ORDER BY id DESC LIMIT 1")
tt2 = cur.fetchone()[0]
cur.execute("SELECT timestamp FROM training_log WHERE id > (SELECT MAX(id)-3 FROM training_log) ORDER BY id DESC LIMIT 1")
tt1 = cur.fetchone()[0] if cur.rowcount or False else tt2
print('training timestamps recent:', tt1, tt2, 'diff', round(tt2-tt1,1))