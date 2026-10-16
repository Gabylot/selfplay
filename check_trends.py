"""Scan the last N steps of training and flag trends that look statistically
or qualitatively "weird" -- rising losses, confidence collapse, win-rate
imbalance beyond chance, spiking draw/unknown-termination rates, gating
stagnation, etc.

Read-only against stats.db. Does not modify any existing files.

Usage:
    python check_trends.py [path/to/stats.db] [--window N] [--recent-games N]

Defaults:
    --window        200   (training steps to look at for loss/confidence trends)
    --recent-games  500   (most recent games to check for outcome/termination anomalies)
"""

import sys
import argparse
import sqlite3
import math
from collections import defaultdict


# ── Helpers ──────────────────────────────────────────────────────────────

def linreg_slope(xs, ys):
    """Simple least-squares slope. Returns 0.0 if insufficient/degenerate data."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def pct_change(old, new):
    if old == 0:
        return float('inf') if new != 0 else 0.0
    return 100.0 * (new - old) / abs(old)


def split_half(rows, value_idx):
    """Split rows in half (by order) and return (avg_first_half, avg_second_half)."""
    n = len(rows)
    if n < 4:
        return None, None
    mid = n // 2
    first = [r[value_idx] for r in rows[:mid]]
    second = [r[value_idx] for r in rows[mid:]]
    return sum(first) / len(first), sum(second) / len(second)


WARN = "  [!] "
OK = "      "


# ── Checks ───────────────────────────────────────────────────────────────

def check_losses(conn, window):
    print("=== Training Loss Trend (last window) ===")
    rows = conn.execute(
        "SELECT step, policy_loss, value_loss, total_loss FROM training_log ORDER BY step DESC LIMIT ?",
        (window,)
    ).fetchall()
    rows.reverse()
    if len(rows) < 4:
        print("  Not enough data.\n")
        return

    steps = [r[0] for r in rows]
    for name, idx in [("policy_loss", 1), ("value_loss", 2), ("total_loss", 3)]:
        vals = [r[idx] for r in rows]
        slope = linreg_slope(steps, vals)
        first_half, second_half = split_half(rows, idx)
        change = pct_change(first_half, second_half)
        flag = slope > 0 and change > 5
        prefix = WARN if flag else OK
        direction = "RISING" if slope > 0 else "falling/flat"
        print(f"{prefix}{name:<14} slope={slope:+.6f}/step  "
              f"first_half_avg={first_half:.5f} -> second_half_avg={second_half:.5f} "
              f"({change:+.1f}%)  [{direction}]")
        if flag:
            print(f"        -> {name} is trending UP over the last {window} steps. "
                  f"Worth checking LR, gradient norms, or a data/label issue.")
    print()


def check_confidence(conn, window):
    print("=== Network Confidence Trend (last window) ===")
    rows = conn.execute(
        "SELECT step, avg_max_policy, avg_abs_value FROM network_stats ORDER BY step DESC LIMIT ?",
        (window,)
    ).fetchall()
    rows.reverse()
    if len(rows) < 4:
        print("  Not enough data.\n")
        return

    steps = [r[0] for r in rows]
    for name, idx in [("avg_max_policy", 1), ("avg_abs_value", 2)]:
        vals = [r[idx] for r in rows]
        slope = linreg_slope(steps, vals)
        first_half, second_half = split_half(rows, idx)
        change = pct_change(first_half, second_half)
        flag = change < -15  # more than 15% relative drop flagged for attention
        prefix = WARN if flag else OK
        direction = "DROPPING" if slope < 0 else "rising/flat"
        print(f"{prefix}{name:<14} slope={slope:+.7f}/step  "
              f"first_half_avg={first_half:.5f} -> second_half_avg={second_half:.5f} "
              f"({change:+.1f}%)  [{direction}]")
        if flag:
            print(f"        -> {name} dropped >15% over the last {window} steps. "
                  f"Cross-check against value_loss/policy_loss trend above -- if loss is "
                  f"flat/improving, this is likely honest recalibration to harder positions, "
                  f"not a regression. If loss is also worsening, investigate further.")
    print()


def check_game_outcomes(conn, recent_games):
    print(f"=== Game Outcome Sanity Check (last {recent_games} games) ===")
    rows = conn.execute(
        "SELECT step, result, length, termination, material_diff FROM game_log ORDER BY id DESC LIMIT ?",
        (recent_games,)
    ).fetchall()
    rows.reverse()
    if not rows:
        print("  No games logged.\n")
        return

    n = len(rows)
    wins = sum(1 for r in rows if r[1] > 0)
    losses = sum(1 for r in rows if r[1] < 0)
    draws = sum(1 for r in rows if r[1] == 0)
    decisive = wins + losses

    print(f"  n={n}  white_wins={wins}  black_wins={losses}  draws={draws} "
          f"(draw_rate={100*draws/n:.1f}%)")

    # Win/loss balance check vs binomial expectation (p=0.5 among decisive games)
    if decisive >= 30:
        expected = decisive * 0.5
        sd = math.sqrt(decisive * 0.5 * 0.5)
        z = (wins - expected) / sd if sd > 0 else 0
        flag = abs(z) > 3  # >3 sigma is a real anomaly
        prefix = WARN if flag else OK
        print(f"{prefix}white/black balance among decisive games: "
              f"z={z:+.2f} (expected ~{expected:.0f} white wins, sd={sd:.1f})")
        if flag:
            print(f"        -> Win-rate imbalance is statistically significant (>3 sigma). "
                  f"Check for a color-dependent bug (e.g. asymmetric color assignment, "
                  f"a1a8-style board-orientation bug, or castling-rights encoding issue).")

    # Termination breakdown + unknown-termination flag
    term_counts = defaultdict(int)
    for r in rows:
        term_counts[r[3]] += 1
    unknown_count = term_counts.get('unknown', 0)
    if unknown_count > 0:
        print(f"{WARN}'unknown' termination appeared {unknown_count}/{n} times "
              f"({100*unknown_count/n:.2f}%) -- this indicates MCTS returned no move "
              f"(see tests/test_game_termination.py). Should be 0 after the encoding fix; "
              f"any nonzero count here means the bug is still reachable.")
    else:
        print(f"{OK}no 'unknown' terminations in this window (good).")

    # Draw-rate trend within the window (first half vs second half)
    if n >= 20:
        mid = n // 2
        first_draws = sum(1 for r in rows[:mid] if r[1] == 0) / mid
        second_draws = sum(1 for r in rows[mid:] if r[1] == 0) / (n - mid)
        change = pct_change(first_draws, second_draws)
        flag = abs(change) > 20
        prefix = WARN if flag else OK
        print(f"{prefix}draw rate trend: {100*first_draws:.1f}% -> {100*second_draws:.1f}% "
              f"({change:+.1f}% relative change)")

    # Avg game length trend + how close to cap
    lengths = [r[2] for r in rows]
    avg_len = sum(lengths) / n
    print(f"{OK}avg game length: {avg_len:.1f}")

    print()


def check_gating(conn):
    print("=== Gating / Promotion Trend ===")
    rows = conn.execute(
        "SELECT step, promoted, win_rate FROM promotion_log ORDER BY step ASC"
    ).fetchall()
    if not rows:
        print("  No promotion attempts logged yet.\n")
        return

    recent = rows[-10:]
    promotions = sum(1 for r in recent if r[1])
    avg_wr = sum(r[2] for r in recent) / len(recent)
    flag = promotions == 0 and len(recent) >= 10
    prefix = WARN if flag else OK
    print(f"{prefix}last {len(recent)} gating attempts: {promotions} promoted, "
          f"avg win_rate={avg_wr:.1%}")
    if flag:
        print(f"        -> No promotions in the last {len(recent)} gating attempts. "
              f"Recall gating is a monitoring signal only (self-play always uses latest "
              f"network per this project's design), so this isn't blocking training, but "
              f"a long promotion drought can indicate the eval win-rate threshold "
              f"({'config.evaluation.gate_win_threshold'}) is miscalibrated or that "
              f"gating variance is too high for the current gate_games count.")
    print()


def check_reference_eval(conn):
    print("=== Reference Opponent (alpha-beta) Win Rate Trend ===")
    rows = conn.execute(
        "SELECT step, win_rate FROM evaluation_log WHERE opponent='alpha_beta_ref' ORDER BY step ASC"
    ).fetchall()
    if len(rows) < 3:
        print("  Not enough evaluation history yet.\n")
        return

    steps = [r[0] for r in rows]
    wrs = [r[1] for r in rows]
    slope = linreg_slope(steps, wrs)
    first_half, second_half = split_half(rows, 1)
    change = pct_change(first_half, second_half) if first_half else 0.0
    flag = slope < 0 and change < -10
    prefix = WARN if flag else OK
    print(f"{prefix}win_rate vs alpha-beta: slope={slope:+.6f}/step  "
          f"{first_half:.1%} -> {second_half:.1%} ({change:+.1f}%)")
    if flag:
        print(f"        -> Win rate against the reference opponent is declining. "
              f"Since self-play always trains on the latest network, this could indicate "
              f"real regression rather than just noisy small-sample eval matches.")
    print()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", default="output/6head/stats.db")
    parser.add_argument("--window", type=int, default=200,
                        help="Training steps to analyze for loss/confidence trends")
    parser.add_argument("--recent-games", type=int, default=500,
                        help="Most recent games to analyze for outcome anomalies")
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(args.db_path)
    except Exception as e:
        print(f"Could not open {args.db_path}: {e}")
        return

    print(f"Analyzing {args.db_path}  (window={args.window} steps, "
          f"recent_games={args.recent_games})\n")

    check_losses(conn, args.window)
    check_confidence(conn, args.window)
    check_game_outcomes(conn, args.recent_games)
    check_gating(conn)
    check_reference_eval(conn)

    print("Done. Lines marked [!] are worth a closer look; unmarked lines look normal.")

    conn.close()


if __name__ == "__main__":
    main()