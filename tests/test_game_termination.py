"""Tests to diagnose the "unknown" draw termination bug.

Observation from game #6: Game ended at 51 half-moves (move 26 in chess notation)
with 1/2-1/2 result and "unknown" termination. The last move was a capture.
Move 51 is far below max_game_length (150), so termination should not be "max_length".
It's also not repetition (capture changes the position beyond 3-fold rep).

The code path for this:
1. play_one_game() loop breaks at selfplay.py:200 (both selected_move and best_move are None)
2. Falls through to selfplay.py:272 → termination = "unknown" (move_count < max_game_length)
3. This happens when MCTS returns None for a move

Root cause hypothesis: MCTS root node ends up with 0 children after expansion,
meaning ALL legal moves failed move_to_policy_index() encoding.
"""

import sys
import os
import math
import numpy as np
import chess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoding import (
    move_to_policy_index, policy_index_to_move, board_to_tensor,
    get_legal_move_mask, get_all_policy_indices, NUM_ACTIONS,
    _find_queen_move_plane, _find_knight_move_plane,
    UNDERPROMOTION_OFFSETS, UNDERPROMOTION_DIRS, UNDERPROMOTION_PIECES,
)
from mcts import MCTS, MCTSNode
from config import get_config, Config


# =============================================================
# SECTION 1: Encoding Roundtrip Tests
# =============================================================

def _test_encoding_roundtrip(board, label=""):
    """Test that every legal move can be encoded and decoded correctly.
    Returns list of failing moves, or empty list if all pass.
    """
    failures = []
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return []  # No legal moves (stalemate/checkmate), nothing to test
    
    indices = get_all_policy_indices(board)
    
    for move in legal_moves:
        idx = indices.get(move)
        if idx is None:
            failures.append(("encoding_failed", move, str(move)))
            continue
        
        decoded = policy_index_to_move(idx, board)
        if decoded is None:
            failures.append(("decoding_failed", move, str(move), idx))
        elif decoded != move:
            failures.append(("mismatch", move, str(move), decoded, str(decoded), idx))
    
    return failures


def test_all_starting_moves():
    """Test that all 20 starting moves encode/decode correctly."""
    board = chess.Board()
    failures = _test_encoding_roundtrip(board, "starting position")
    assert len(failures) == 0, f"Starting position encoding failures: {failures}"


def test_all_moves_random_positions():
    """Test encoding on many random positions to catch edge cases."""
    board = chess.Board()
    
    for depth in range(50):
        failures = _test_encoding_roundtrip(board, f"position at depth {depth}")
        assert len(failures) == 0, f"Position at depth {depth}: {failures}"
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = np.random.choice(legal_moves)
        board.push(move)


def test_captures_roundtrip():
    """Specifically test capture moves encoding/decoding.
    This is important because the bug occurred on a capture move."""
    board = chess.Board()
    
    # Play moves that typically lead to captures — use full UCI (4-char)
    openings = [
        "e2e4", "d7d5", "e4d5",  # pawn capture exd5
        "g8f6", "c2c4", "e7e6", "b1c3", "f8b4",  # prepare more complex captures
    ]
    
    for move_uci in openings:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
    
    failures = _test_encoding_roundtrip(board, "captures test")
    assert len(failures) == 0, f"Capture position failures: {failures}"


def test_en_passant_encoding():
    """Test en passant captures specifically."""
    # Set up a proper en passant position
    # White pawn on a2, black pawn on c4, white pushes a2-a4
    board = chess.Board("4k3/8/8/8/2p5/P7/8/4K3 w - - 0 2")
    a4 = chess.Move.from_uci("a2a4")
    if a4 in board.legal_moves:
        board.push(a4)
        # Now black to move, en passant target is a3
        ep = board.ep_square
        if ep is not None:
            for move in board.legal_moves:
                if move.to_square == board.ep_square:
                    # This is an en passant capture
                    idx = move_to_policy_index(move, board)
                    decoded = policy_index_to_move(idx, board)
                    assert decoded == move, f"En passant mismatch: {move} vs {decoded}"
        
        failures = _test_encoding_roundtrip(board, "en passant")
        assert len(failures) == 0, f"En passant failures: {failures}"
    else:
        # a2a4 not legal, try different approach
        board = chess.Board()
        # Play 1. e4 d5 2. e5 f5 3. exf6 (en passant)
        for uci in ["e2e4", "d7d5", "e4e5", "f7f5"]:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)
        # Now white can capture en passant
        failures = _test_encoding_roundtrip(board, "en passant game")
        assert len(failures) == 0, f"En passant game failures: {failures}"


def test_promotion_encoding():
    """Test promotion moves."""
    board = chess.Board("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    failures = _test_encoding_roundtrip(board, "white pawn promotion")
    assert len(failures) == 0, f"Promotion position failures: {failures}"
    
    board = chess.Board("4k3/8/8/8/8/8/1p6/4K3 b - - 0 1")
    failures = _test_encoding_roundtrip(board, "black pawn promotion")
    assert len(failures) == 0, f"Black promotion failures: {failures}"


def test_underpromotion_encoding():
    """Test underpromotion moves specifically - KNOWN BUG area."""
    fens = [
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "White underpromote"),
        ("4k3/8/8/8/8/8/1p6/4K3 b - - 0 1", "Black underpromote"),
        # A position where ONLY underpromotions are legal
        ("r1bqkbnr/PPPPPppp/8/8/8/8/pppppPPP/R1BQKBNR w KQkq - 0 4", "many white promotions"),
        ("r1bqkbnr/pppppPPP/8/8/8/8/PPPPPppp/R1BQKBNR b KQkq - 0 4", "many black promotions"),
    ]
    
    for fen, label in fens:
        board = chess.Board(fen)
        failures = _test_encoding_roundtrip(board, label)
        if failures:
            print(f"FAILURES for {label} ({fen}):")
            for f in failures:
                print(f"  {f}")
        # We'll note but not assert — this is the suspected bug area


def test_castling_encoding():
    """Test castling moves."""
    for fen in [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # All castling available
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",  # Black to move
    ]:
        board = chess.Board(fen)
        failures = _test_encoding_roundtrip(board, f"castling: {fen}")
        assert len(failures) == 0, f"Castling failures for {fen}: {failures}"


def test_endgame_positions():
    """Test encoding on endgame positions (common at move 51)."""
    endgames = [
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/4P3/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/4P3/8/8/8/4K3 b - - 0 1",
        "r1bqkb1r/pppppppp/2n2n2/4P3/2B5/2N5/PPPP1PPP/R1BQK2R w KQkq - 0 1",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
        "r4rk1/pp3ppp/2n1q3/2pp4/3P2b1/2P1PN2/PP1Q1PPP/R3R1K1 w - - 0 12",
        "r1bq1rk1/pp3ppp/2np4/2pNp3/4P3/2PP1N2/PP3PPP/R1BQR1K1 w - - 0 10",
    ]
    
    for fen in endgames:
        board = chess.Board(fen)
        failures = _test_encoding_roundtrip(board, fen)
        assert len(failures) == 0, f"Endgame failures for {fen}: {failures}"


def test_queen_move_plane_all_distances():
    """Test that _find_queen_move_plane handles all 8 directions × 7 distances."""
    for d_idx, (qdr, qdc) in enumerate([
        (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
    ]):
        for dist in range(1, 8):
            dr = qdr * dist
            dc = qdc * dist
            plane = _find_queen_move_plane(dr, dc)
            expected = d_idx * 7 + (dist - 1)
            assert plane == expected, (
                f"Queen move direction {d_idx} ({qdr},{qdc}), "
                f"dist={dist}: got {plane}, expected {expected}"
            )


# =============================================================
# SECTION 2: MCTS Edge Case Tests
# =============================================================

class MockNetwork:
    """Mock network that returns uniform random predictions."""
    def predict(self, state):
        policy = np.random.dirichlet(np.ones(NUM_ACTIONS)) * 0.5
        value = np.random.uniform(-0.1, 0.1)
        return policy, value


def test_mcts_root_no_children():
    """Test MCTS behavior when root has no children after expansion.
    
    This simulates the case where ALL legal moves fail move_to_policy_index().
    """
    mock_net = MockNetwork()
    mcts = MCTS(mock_net, num_simulations=10, c_puct=1.5)
    
    board = chess.Board()
    root = mcts.get_root(board)
    
    # Simulate: expand root but leave children dict empty
    root.is_expanded = True  # Prevent re-expansion
    
    visit_policy, best_move, stats = mcts.search(root)
    
    # Expected behavior: best_move is None, visit_policy is zeros
    assert visit_policy is not None, "visit_policy should not be None"
    assert visit_policy.shape == (NUM_ACTIONS,), f"Shape mismatch: {visit_policy.shape}"
    assert visit_policy.sum() == 0, "visit_policy should be all zeros"
    assert best_move is None, f"best_move should be None, got {best_move}"
    
    # Now test select_move_with_temperature
    vp, selected = mcts.select_move_with_temperature(root, temperature=1.0)
    assert selected is None, "select_move_with_temperature should return None"
    assert vp.sum() == 0, "visit policy from temperature should be all zeros"


def test_mcts_normal_search():
    """Test MCTS normal search works."""
    mock_net = MockNetwork()
    mcts = MCTS(mock_net, num_simulations=10, c_puct=1.5)
    
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)
    
    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)
    
    # Normal case: should have a valid best_move
    assert best_move is not None, (
        f"MCTS returned None best_move for normal position!\n"
        f"Board: {board.fen()}\n"
        f"Legal moves: {list(board.legal_moves)}\n"
        f"Root children: {len(root.children)}"
    )


def test_mcts_after_capture():
    """Test that MCTS returns a valid move after a capture."""
    mock_net = MockNetwork()
    mcts = MCTS(mock_net, num_simulations=20, c_puct=1.5)
    
    board = chess.Board()
    
    # Play a game reaching a position with captures
    moves_sequence = [
        "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6",
        "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "h7h6",
        "g5h4", "b7b6", "c4d5", "e6d5",
        "f1b5", "c7c5", "d4c5", "b6c5", "e8g8", "c8b7",
        "f1e1", "b8d7",
    ]
    
    for uci in moves_sequence:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)
    
    assert len(list(board.legal_moves)) > 0, "No legal moves!"
    
    # Find and play a capture
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            break
    
    # Now test MCTS on position after capture
    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)
    
    assert best_move is not None, (
        f"MCTS returned None after capture!\n"
        f"Board: {board.fen()}\n"
        f"Legal moves: {list(board.legal_moves)}\n"
        f"Root children: {len(root.children)}"
    )


# =============================================================
# SECTION 3: Self-Play Termination Tests
# =============================================================

def test_play_one_game_normal_termination():
    """Test that play_one_game normally terminates with checkmate/stalemate/etc."""
    from selfplay import play_one_game
    
    mock_net = MockNetwork()
    mcts = MCTS(mock_net, num_simulations=10, c_puct=1.5)
    
    game_data, game_info = play_one_game(
        mcts,
        max_game_length=50,
        adjudicate_material=False,
        temp_threshold=15,
        temp_high=1.0,
        temp_low=0.1,
    )
    
    # Should never have "unknown" termination with a functioning MCTS
    assert game_info['termination'] != 'unknown', (
        f"Game terminated with 'unknown'!\n"
        f"Info: {game_info}"
    )


def test_play_one_game_best_move_none():
    """Test: what happens when BOTH search() and select_move_with_temperature()
    return None. This simulates the real bug: root.children = {} → both are None."""
    from selfplay import play_one_game
    
    class AllNoneMCTS:
        """Always returns None for both search and temperature selection."""
        def get_root(self, board):
            node = MCTSNode(board.copy())
            node.is_expanded = True  # Skip expansion
            return node
        
        def search(self, root):
            return np.zeros(NUM_ACTIONS, dtype=np.float32), None, {'avg_depth': 0}
        
        def select_move_with_temperature(self, root, temperature):
            return np.zeros(NUM_ACTIONS, dtype=np.float32), None
    
    mock_net = MockNetwork()
    game_data, game_info = play_one_game(
        AllNoneMCTS(),
        max_game_length=150,
        adjudicate_material=False,
        temp_threshold=30,
        temp_high=1.0,
        temp_low=0.1,
    )
    
    # The game should end immediately (0 moves because both return None first call)
    print(f"AllNone MCTS result: termination={game_info['termination']}, "
          f"length={game_info['length']}, result_str={game_info['result_str']}")
    
    # This is the path that produces "unknown" termination
    if game_info['termination'] == 'unknown':
        print(f"  ★ Confirmed: play_one_game produces 'unknown' when MCTS returns None")
    print(f"  Game info: {game_info}")


# =============================================================
# SECTION 4: Comprehensive Edge Case Tests
# =============================================================

def test_encoding_after_captures():
    """Test that all moves can be encoded after various captures."""
    board = chess.Board()
    np.random.seed(42)
    
    for game_num in range(5):
        board.reset()
        move_count = 0
        captures_made = 0
        
        while not board.is_game_over() and move_count < 100:
            legal = list(board.legal_moves)
            if not legal:
                break
            
            # Check encoding when captures have occurred
            if captures_made > 0:
                failures = _test_encoding_roundtrip(board, f"game_{game_num}_move_{move_count}")
                if failures:
                    print(f"FAILURES at game {game_num}, move {move_count}:")
                    for f in failures:
                        print(f"  {f}")
                    # Don't assert yet — we know underpromotion decoding fails
            
            # Sometimes play a capture
            captures = [m for m in legal if board.is_capture(m)]
            if captures and np.random.random() < 0.3:
                move = np.random.choice(captures)
                captures_made += 1
            else:
                move = np.random.choice(legal)
            
            board.push(move)
            move_count += 1


def test_large_number_of_queen_moves():
    """Test encoding of queen moves, which has the most complex plane logic."""
    board = chess.Board("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1")
    failures = _test_encoding_roundtrip(board, "queen position")
    assert len(failures) == 0, f"Queen position failures: {failures}"
    
    board = chess.Board("4k3/8/8/3Q4/3Q4/8/8/4K3 w - - 0 1")
    failures = _test_encoding_roundtrip(board, "two queens")
    assert len(failures) == 0, f"Two queens failures: {failures}"


def test_very_crowded_positions():
    """Test encoding in crowded positions where many captures have occurred."""
    fens = [
        "r1bqkb1r/pp3ppp/2n1p3/2pp4/3P4/2PBP3/PP3PPP/RN1QKBNR w KQkq - 0 6",
        "r1bq1rk1/2p2ppp/p1p5/1p6/3P4/5N2/PPP2PPP/R1BQR1K1 w - - 0 10",
        "1r1q1rk1/1b1p2pp/pp1bpn2/2p5/2BP4/2N1PN2/PP3PPP/R1BQR1K1 w - - 0 11",
        "4k3/pp3ppp/8/2r5/4R3/8/PP3PPP/4K3 w - - 0 20",
        "4k3/5ppp/8/8/1r6/8/5PPP/4K2R w K - 0 25",
    ]
    
    for fen in fens:
        board = chess.Board(fen)
        failures = _test_encoding_roundtrip(board, fen)
        assert len(failures) == 0, f"Crowded position failures for {fen}: {failures}"


# =============================================================
# SECTION 5: Debug underpromotion decoding bug
# =============================================================

def test_underpromotion_decode_debug():
    """Debug why underpromotion moves fail to decode.
    
    Test findings show decoding fails for moves like:
    - f7f8r (index 3939), f7f8b (index 3936), f7f8n (index 3933)
    - a2a1r (index 654), a2a1b (index 651), a2a1n (index 648)
    """
    print("\n=== Debugging underpromotion encoding/decoding ===")
    
    # Test white pawn promotion f7f8
    board = chess.Board("4k3/8/8/8/8/8/1P6/4K3 w - - 0 1")
    # Actually need pawn on f7, not b2
    board = chess.Board("4k3/5P2/8/8/8/8/8/4K3 w - - 0 1")
    
    for move in board.legal_moves:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            idx = move_to_policy_index(move, board)
            decoded = policy_index_to_move(idx, board)
            
            from_rank = chess.square_rank(move.from_square)
            from_file = chess.square_file(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            to_file = chess.square_file(move.to_square)
            
            print(f"\nMove: {move}")
            print(f"  From: ({from_rank},{from_file}) To: ({to_rank},{to_file})")
            print(f"  Promotion: {chess.piece_symbol(move.promotion)}")
            print(f"  Turn: {'white' if board.turn == chess.WHITE else 'black'}")
            print(f"  Index: {idx}")
            print(f"  Decoded: {decoded}")
            
            # Manually trace policy_index_to_move
            source_idx = idx // 73
            plane = idx % 73
            print(f"  source_idx={source_idx}, plane={plane}")
            
            f_rank = source_idx // 8
            f_file = source_idx % 8
            print(f"  from_rank={f_rank}, from_file={f_file}")
            
            if plane >= 64:
                under_idx = plane - 64
                piece_idx = under_idx // 3
                dir_idx = under_idx % 3
                promo_piece = UNDERPROMOTION_PIECES[piece_idx]
                print(f"  under_idx={under_idx}, piece_idx={piece_idx} ({UNDERPROMOTION_PIECES[piece_idx]}), dir_idx={dir_idx} ({UNDERPROMOTION_DIRS[dir_idx]})")
                
                dir_name = UNDERPROMOTION_DIRS[dir_idx]
                dr_raw, dc_raw = UNDERPROMOTION_OFFSETS[dir_name]
                print(f"  raw offset: ({dr_raw},{dc_raw})")
                
                dr = dr_raw
                if board.turn == chess.BLACK:
                    dr = -dr_raw
                print(f"  adjusted dr: {dr}")
                print(f"  computed to_rank: {f_rank + dr}")
                
            assert decoded is not None, (
                f"BUG CONFIRMED: policy_index_to_move returns None for underpromotion {move}!"
            )


# =============================================================
# SECTION 6: Test with only-underpromotion positions
# =============================================================

def test_position_with_only_underpromotions():
    """Test a position where ALL legal moves are underpromotions.
    This is the direct trigger for the "unknown" termination bug."""
    
    # Create a position where pawns on 7th rank can only underpromote
    # (Queen promotions would be queen-like moves and encode differently)
    # Actually any pawn on 7th rank always has both queen promotions AND underpromotions
    # as legal moves. So ALL moves can never be only underpromotions.
    # 
    # But this test checks: what if MCTS tree doesn't encode them correctly?
    
    board = chess.Board("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    
    # All legal moves for this position
    print("\n=== Position: 4k3/1P6/8/8/8/8/8/4K3 w ===")
    for move in board.legal_moves:
        idx = move_to_policy_index(move, board)
        decoded = policy_index_to_move(idx, board)
        if decoded is None:
            print(f"  ★ FAIL: {move} (idx={idx}) → None")
        elif decoded != move:
            print(f"  ★ MISMATCH: {move} (idx={idx}) → {decoded}")
        else:
            print(f"  ✓ OK: {move} (idx={idx})")
    
    # Now check what MCTS would do with this position
    mock_net = MockNetwork()
    mcts = MCTS(mock_net, num_simulations=10, c_puct=1.5)
    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)
    
    print(f"\n  MCTS best_move: {best_move}")
    print(f"  Root children: {len(root.children)}")
    
    if best_move is None:
        print(f"  ★ MCTS returned None! All children failed policy_index_to_move!")
        
        # Check which children failed
        for child_idx, child in root.children.items():
            decoded = policy_index_to_move(child_idx, board)
            print(f"  Child idx={child_idx}: move={child.move}, decoded={decoded}")
    else:
        print(f"  ✓ MCTS returned valid move: {best_move}")


# =============================================================
# SECTION 7: Run tests
# =============================================================

if __name__ == "__main__":
    import traceback
    
    print("=" * 60)
    print("Running encoding diagnostic tests...")
    print("=" * 60)
    
    tests = [
        ("test_all_starting_moves", test_all_starting_moves),
        ("test_all_moves_random_positions", test_all_moves_random_positions),
        ("test_captures_roundtrip", test_captures_roundtrip),
        ("test_en_passant_encoding", test_en_passant_encoding),
        ("test_promotion_encoding", test_promotion_encoding),
        ("test_castling_encoding", test_castling_encoding),
        ("test_endgame_positions", test_endgame_positions),
        ("test_queen_move_plane_all_distances", test_queen_move_plane_all_distances),
        ("test_mcts_root_no_children", test_mcts_root_no_children),
        ("test_mcts_normal_search", test_mcts_normal_search),
        ("test_mcts_after_capture", test_mcts_after_capture),
        ("test_play_one_game_normal_termination", test_play_one_game_normal_termination),
        ("test_play_one_game_best_move_none", test_play_one_game_best_move_none),
        ("test_underpromotion_decode_debug", test_underpromotion_decode_debug),
        ("test_position_with_only_underpromotions", test_position_with_only_underpromotions),
        ("test_encoding_after_captures", test_encoding_after_captures),
        ("test_large_number_of_queen_moves", test_large_number_of_queen_moves),
        ("test_very_crowded_positions", test_very_crowded_positions),
    ]
    
    passed = 0
    failed = 0
    
    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")