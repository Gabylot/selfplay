"""Test edge-file underpromotion encoding/decoding.

Hypothesis: underpromotion captures on the a-file or h-file may produce
an action_idx that, when decoded back via policy_index_to_move, returns
None or a different move — because the resulting to_file falls outside
0..7 for one of the diagonal capture directions, OR because the index
collides with a different legal move's index.

If a move's action_idx decodes to None, that move can still become a
*child* in MCTS (since _expand_node_with_data builds children directly
from `legal_moves`, not by decoding indices) — but if MCTS ever selects
that exact child as the most-visited one, _get_visit_policy /
select_move_with_temperature call policy_index_to_move(best_idx, ...) to
recover the move, get None back, and `best_move` becomes None even
though root.children is non-empty. That produces the "unknown"
termination in play_one_game.

This test checks every legal move in a few edge-file promotion
positions (both colors, both edge files, all capture directions where
applicable) for:
  1. Roundtrip correctness: decode(encode(move)) == move
  2. Index collisions: do two different legal moves share an action_idx?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
from encoding import move_to_policy_index, policy_index_to_move, get_all_policy_indices


def check_position(fen: str, label: str):
    print(f"\n=== {label} ===")
    print(f"FEN: {fen}")
    board = chess.Board(fen)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        print("  (no legal moves)")
        return

    # --- Roundtrip check ---
    seen_indices = {}
    roundtrip_failures = []
    collisions = []

    for move in legal_moves:
        try:
            idx = move_to_policy_index(move, board)
        except ValueError as e:
            roundtrip_failures.append((move, "encode_error", str(e)))
            continue

        # Collision check
        if idx in seen_indices:
            collisions.append((move, seen_indices[idx], idx))
        else:
            seen_indices[idx] = move

        # Roundtrip check
        decoded = policy_index_to_move(idx, board)
        if decoded is None:
            roundtrip_failures.append((move, "decode_none", idx))
        elif decoded != move:
            roundtrip_failures.append((move, "decode_mismatch", idx, decoded))

    # --- Report ---
    promo_moves = [m for m in legal_moves if m.promotion is not None]
    print(f"  Total legal moves: {len(legal_moves)}  (promotions: {len(promo_moves)})")

    for move in promo_moves:
        idx = move_to_policy_index(move, board)
        decoded = policy_index_to_move(idx, board)
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        status = "OK" if decoded == move else f"FAIL (decoded={decoded})"
        print(f"  {move}  from_file={from_file} to_file={to_file}  "
              f"idx={idx}  -> {status}")

    if collisions:
        print("  COLLISIONS FOUND:")
        for move, other, idx in collisions:
            print(f"    idx={idx}: {move} collides with {other}")
    else:
        print("  No index collisions.")

    if roundtrip_failures:
        print("  ROUNDTRIP FAILURES:")
        for f in roundtrip_failures:
            print(f"    {f}")
    else:
        print("  All roundtrips OK.")


if __name__ == "__main__":
    # White pawn on a7, capturing toward b8 with underpromotion.
    # a-file: "forward_left" or "forward_right" may push to_file to -1.
    check_position(
        "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "White pawn a7, capture+promote toward b8 (a-file edge)",
    )

    # White pawn on h7, capturing toward g8 with underpromotion.
    # h-file: the other diagonal direction may push to_file to 8.
    check_position(
        "4k1n1/7P/8/8/8/8/8/4K3 w - - 0 1",
        "White pawn h7, capture+promote toward g8 (h-file edge)",
    )

    # Black pawn on a2, capturing toward b1 with underpromotion.
    check_position(
        "4k3/8/8/8/8/8/p7/1N2K3 b - - 0 1",
        "Black pawn a2, capture+promote toward b1 (a-file edge)",
    )

    # Black pawn on h2, capturing toward g1 with underpromotion.
    check_position(
        "4k3/8/8/8/8/8/7p/4K1N1 b - - 0 1",
        "Black pawn h2, capture+promote toward g1 (h-file edge)",
    )

    # Crowded edge-file position: both a-file and h-file promotions
    # available simultaneously, plus straight-forward promotions, to
    # also exercise the collision check more thoroughly.
    check_position(
        "1n2k1n1/P6P/8/8/8/8/8/4K3 w - - 0 1",
        "White pawns a7 and h7, both with capture+promote options",
    )

    check_position(
        "4k3/8/8/8/8/8/p6p/1N2K1N1 b - - 0 1",
        "Black pawns a2 and h2, both with capture+promote options",
    )

    # Full sanity sweep: get_all_policy_indices should be injective
    # (no two legal moves sharing an index) for every position above,
    # already covered per-position, but also dump the full index set
    # for the most complex position for manual inspection if needed.
    print("\n=== Full index map for crowded White position ===")
    board = chess.Board("1n2k1n1/P6P/8/8/8/8/8/4K3 w - - 0 1")
    index_map = get_all_policy_indices(board)
    for move, idx in sorted(index_map.items(), key=lambda kv: kv[1]):
        print(f"  {move} -> {idx}")