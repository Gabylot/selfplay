import warnings
warnings.filterwarnings('ignore')
import sqlite3, numpy as np, os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DB = 'output/119/stats.db'
BUFFERS = sorted(glob.glob('output/119/checkpoints/replay_buffer.npz'))
print('buffer snapshots:', BUFFERS)

# ---- 1. Policy loss trajectory ----
con = sqlite3.connect(DB); cur = con.cursor()
rows = cur.execute(
    "SELECT step, policy_loss, value_loss, total_loss FROM training_log ORDER BY step"
).fetchall()
steps = np.array([r[0] for r in rows])
ploss = np.array([r[1] for r in rows])

print(f"--- Policy-loss trend: {len(rows)} steps ---")
print(f"first 100 mean : {ploss[:100].mean():.4f}")
for lo_idx in range(100, len(ploss), 300):
    seg = ploss[lo_idx:lo_idx+300]
    end = steps[min(lo_idx + len(seg) - 1, len(steps) - 1)]
    print(f"  steps {steps[lo_idx]:>6}..{end:>6}: mean={seg.mean():.4f}")

tail = min(300, len(ploss))
sl = ploss[-tail:]
print(f"\nLAST {tail} steps: first={sl[0]:.4f} mean={sl.mean():.4f} last={sl[-1]:.4f} "
      f"min={sl.min():.4f} max={sl.max():.4f}")
slope = np.polyfit(np.arange(len(sl)), sl, 1)[0]
trend = ('DECLINING -> undertrained' if slope < -0.0001
         else ('FLAT' if abs(slope) < 0.0001 else 'RISING'))
print(f"slope over last {tail}: {slope:+.6f} per step  ({trend})")

# ---- 2. Target entropy ----
print("\n=== target-policy entropy (floor for policy loss) ===")
for path in BUFFERS:
    try:
        d = np.load(path, allow_pickle=False)
        pol = d['policies']
        print(f"\n{os.path.basename(path)}: n={len(pol)}")
    except Exception as e:
        print(path, 'ERR', e)
        continue
    s = pol.sum(axis=1)
    print(f"  row-sums: min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f}")
    with np.errstate(divide='ignore'):
        logp = np.log(pol, out=np.full_like(pol, -np.inf), where=pol > 0)
        contrib = np.where(pol > 0, -pol * logp, 0.0)
    ent = contrib.sum(axis=1)
    print(f"  target entropy H: mean={ent.mean():.4f} median={np.median(ent):.4f} "
          f"min={ent.min():.4f} max={ent.max():.4f}")
    print(f"  effective support exp(H): mean={np.exp(ent).mean():.2f} "
          f"median={np.median(np.exp(ent)):.2f}")

# ---- plot ----
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(steps, ploss, lw=0.6, alpha=0.6, label='policy_loss')
if len(ploss) >= 3:
    kern = np.ones(101) / 101
    conv = np.convolve(ploss, kern, 'valid')
    ax.plot(steps[:len(conv)], conv, lw=1.5, color='tab:red', label='MA(101)')
ax.set_xlabel('step'); ax.set_ylabel('policy loss (CE)')
ax.set_title('policy_loss trajectory (undertrained vs floor)')
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
out = 'policy_loss_trend.png'
fig.savefig(out)
print(f"\nwrote {out}")