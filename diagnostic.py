"""diagnose_fastchess_bug_4.py

Script 3 showed 400 push/pop cycles from chess.Board() (default ctor) are
completely safe. But every real failure so far came from chess.Board(fen)
with a non-default FEN. This isolates FEN-parsing specifically: it builds
the SAME resulting position two different ways --

  (a) chess.Board(fen)              -- direct FEN parse
  (b) chess.Board() + push(moves)   -- reached by playing the moves out

-- then runs the identical nested depth-2 push/pop cycle test (from
diagnostic 3) on each. If (a) breaks and (b) doesn't, that nails FEN
parsing as leaving some internal cache under-initialized. It also checks
whether board.copy(stack=True) on a freshly-parsed board is itself the
trigger, independent of push/pop.

Run directly:
    python diagnose_fastchess_bug_4.py
"""

import chess


def nested_cycle_test(board, label):
    """Exact depth-2 nested push/pop pattern from diagnostic_3, parameterized
    on the starting board. Returns True if it completes without error."""
    root_moves = list(board.legal_moves)
    total = 0
    for i, root_mv in enumerate(root_moves):
        board.push(root_mv)
        reply_moves = list(board.legal_moves)
        for j, reply_mv in enumerate(reply_moves):
            total += 1
            try:
                board.push(reply_mv)
                board.pop()
            except Exception as e:
                print(f"  [{label}] FAIL at root #{i} ({root_mv.uci()}), "
                      f"reply #{j} ({reply_mv.uci()}), cycle {total}: "
                      f"{type(e).__name__}: {e}")
                return False
        board.pop()
    print(f"  [{label}] OK -- {total} cycles, no corruption")
    return True


# A simple, unremarkable position reached after two quiet developing
# moves -- nothing tactical, nothing near-edge, no captures involved yet.
TARGET_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
MOVES_TO_REACH_IT = ["e2e4", "e7e5", "g1f3"]

print("=" * 70)
print("Test 1: identical position reached two different ways")
print("=" * 70)

print("\n(a) Direct FEN parse:")
b_fen = chess.Board(TARGET_FEN)
print(f"    FEN: {b_fen.fen()}")
nested_cycle_test(b_fen, "FEN-parsed")

print("\n(b) Built by pushing moves from the default position:")
b_pushed = chess.Board()
for uci in MOVES_TO_REACH_IT:
    b_pushed.push(chess.Move.from_uci(uci))
print(f"    FEN: {b_pushed.fen()}")
assert b_pushed.fen().split(' ')[0] == TARGET_FEN.split(' ')[0], \
    "sanity check: piece placement should match"
nested_cycle_test(b_pushed, "push-built")

# ─────────────────────────────────────────────────────────────────────────
# Test 2: does board.copy(stack=True) on a freshly-parsed board matter,
# independent of push/pop? (policy_quality_test.py / diagnostic_2 always
# called board.copy(stack=True) before alpha_beta_best_move.)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Test 2: does an intervening .copy(stack=True) matter?")
print("=" * 70)

print("\n(a) FEN-parsed, cycle test run DIRECTLY on it (no copy):")
b1 = chess.Board(TARGET_FEN)
nested_cycle_test(b1, "FEN-parsed, no copy")

print("\n(b) FEN-parsed, then .copy(stack=True), cycle test on the COPY:")
b2 = chess.Board(TARGET_FEN)
b2_copy = b2.copy(stack=True)
nested_cycle_test(b2_copy, "FEN-parsed + copy(stack=True)")

print("\n(c) push-built, then .copy(stack=True), cycle test on the COPY:")
b3 = chess.Board()
for uci in MOVES_TO_REACH_IT:
    b3.push(chess.Move.from_uci(uci))
b3_copy = b3.copy(stack=True)
nested_cycle_test(b3_copy, "push-built + copy(stack=True)")

print("\n" + "=" * 70)
print("Interpretation:")
print("  - If ONLY the FEN-parsed variants fail (with or without an extra")
print("    copy), the bug is in FEN parsing under-initializing some")
print("    incremental cache that push()-built boards get for free.")
print("  - If FEN-parsed-without-copy is fine but FEN-parsed+copy fails,")
print("    the bug is specifically in copy(stack=True) on a board with an")
print("    empty move_stack.")
print("  - If push-built+copy also fails, copy(stack=True) itself is the")
print("    common trigger regardless of how the board was constructed.")
print("=" * 70)