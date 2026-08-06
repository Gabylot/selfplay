"""Board state and move encoding for the AlphaZero chess engine.

Board Representation — matches the AlphaZero paper's M*T + L structure
(Silver et al. 2018, Table S1, Chess column):

    N x N x (M*T + L), where:
      M = 14   per-timestep planes:
                 - 6 planes:  player's (P1) pieces, one per piece type
                 - 6 planes:  opponent's (P2) pieces, one per piece type
                 - 2 planes:  repetition count for that position
                              (plane 12: occurred >= 2 times;
                               plane 13: occurred >= 3 times)
      T = 8    history steps (t, t-1, ..., t-7), zero-filled before game start
      L = 7    constant-valued planes, added ONCE (not repeated per timestep):
                 - player's colour (side to move)                    (1 plane)
                 - P1 castling rights: kingside, queenside           (2 planes)
                 - P2 castling rights: kingside, queenside           (2 planes)
                 - no-progress count (halfmove clock)                (1 plane)
                 - total move count                                  (1 plane)

    Total planes = 14*8 + 7 = 119.

    The board is oriented to the perspective of the CURRENT player (P1):
      - When White is to move, the board is in the absolute frame
        (row 0 = rank 8, col 0 = file a).
      - When Black is to move, the board is rotated 180 degrees
        (row 0 = rank 1, col 0 = file h), so the current player's home
        rank is always at row 0 and P1's pawns always advance toward
        increasing row.
      - Piece planes represent "player's pieces" (P1) vs "opponent's
        pieces" (P2) rather than white/black.
      - Castling planes are player-relative (P1 = current player's
        rights, P2 = opponent's rights).
      - The colour plane holds P1's absolute colour (1.0 if White is to
        move, else 0.0) — this is the paper's "player's colour".

    NOTE: the policy / action output is NOT player-relative.  Per the
    AlphaZero chess paper, the 8x8x73 policy uses ABSOLUTE compass
    directions (N = toward rank 8 for both players).  The network maps a
    player-oriented input to absolute-direction move outputs.  This is
    the opposite of Shogi, where the action space is also player-relative.

    NOTE: this is a BREAKING change to the input representation.  Any
    existing checkpoints and replay-buffer data were trained on the old
    104-plane absolute white/black encoding and are INCOMPATIBLE — they
    must be discarded (start a fresh run with checkpoints/ and
    replay_buffer.npz removed or backed up).

    Note on repetition planes: computed per-timestep by walking
    board.is_repetition(n) upward (n=2,3) and setting the corresponding
    binary plane.  This matches the paper's two per-timestep repetition
    planes (>=2 and >=3 occurrences).

    Note on no-progress-count plane: board.halfmove_clock, normalized by
    NO_PROGRESS_NORM (100 half-moves = the 50-move-rule threshold).

    Note on the move-count plane: normalized by MOVE_COUNT_NORM, which is
    derived from the config's max_game_length (half-moves) divided by 2
    (full moves), so the plane reaches ~1.0 at the game-length cap.

    Note: En passant is NOT encoded as a separate plane — it is implicitly
    determined by the board state (the en-passant square is a property of
    the position, and the network can infer it from the piece positions
    and the last move in the history).  The paper's Table S1 doesn't list
    it either.

Move Encoding (8x8x73 = 4672 action space) — unchanged, matches Table S2:
    Planes 0-55:  Queen-like moves (8 directions × 7 distances)
    Planes 56-63: Knight moves (8 offsets)
    Planes 64-72: Underpromotions (3 pieces × 3 horizontal offsets)
    - Queen promotions are encoded as queen-like moves (forward direction)
"""

import numpy as np
from wrapt import lru_cache
import chess
from typing import Optional

from config import get_config

# Piece type to plane index mapping (within a single 14-plane history group)
# Planes 0-5 = P1 (current player) pieces, 6-11 = P2 (opponent) pieces.
PIECE_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

# ── Per-timestep planes: M = 14 (6 P1 pieces + 6 P2 pieces + 2 repetition) ──
PLANES_PER_HISTORY = 14
NUM_HISTORY_STEPS = 8
NUM_HISTORY_PLANES = PLANES_PER_HISTORY * NUM_HISTORY_STEPS  # 112

# Within a history group:
#   planes 0-5   = P1 pieces (PAWN..KING)
#   planes 6-11  = P2 pieces (PAWN..KING)
#   plane 12     = P1 repetition (position occurred >= 2 times)
#   plane 13     = P2 repetition (position occurred >= 3 times)
PLANE_REPETITION_P1 = 12
PLANE_REPETITION_P2 = 13

# ── L = 7 constant-valued planes, appended ONCE after all history planes ──
PLANE_SIDE_TO_MOVE = NUM_HISTORY_PLANES        # 112  (P1's absolute colour)
PLANE_CASTLING_P1_K = NUM_HISTORY_PLANES + 1   # 113
PLANE_CASTLING_P1_Q = NUM_HISTORY_PLANES + 2   # 114
PLANE_CASTLING_P2_K = NUM_HISTORY_PLANES + 3   # 115
PLANE_CASTLING_P2_Q = NUM_HISTORY_PLANES + 4   # 116
PLANE_NO_PROGRESS_COUNT = NUM_HISTORY_PLANES + 5  # 117
PLANE_MOVE_COUNT = NUM_HISTORY_PLANES + 6      # 118

# Backward-compat aliases (P1 = current player, P2 = opponent).
PLANE_CASTLING_WK = PLANE_CASTLING_P1_K
PLANE_CASTLING_WQ = PLANE_CASTLING_P1_Q
PLANE_CASTLING_BK = PLANE_CASTLING_P2_K
PLANE_CASTLING_BQ = PLANE_CASTLING_P2_Q

NUM_EXTRA_PLANES = 7
NUM_PLANES = NUM_HISTORY_PLANES + NUM_EXTRA_PLANES  # 119

# Normalization divisors for the constant-valued planes.
# Move-count divisor: max_game_length is in half-moves (plies), but
# board.fullmove_number counts full moves, so divide by 2.  The default
# max_game_length = 150 half-moves gives a norm of 75 full moves.
MOVE_COUNT_NORM = get_config().selfplay.max_game_length / 2.0
# No-progress = halfmove clock; 100 half-moves triggers the 50-move rule.
NO_PROGRESS_NORM = 100.0

NUM_ACTIONS = 8 * 8 * 73  # 4672

# Queen move directions: (dr, dc)
# row 0 = rank 8 (top), col 0 = file a (left)
# NOTE: these are ABSOLUTE directions (N = toward rank 8), used for the
# policy output.  The input board is player-oriented, but the action
# space is absolute per the AlphaZero chess paper.
QUEEN_DIRECTIONS = [
    (-1, 0),   # 0: N  (toward rank 8)
    (-1, +1),  # 1: NE
    (0, +1),   # 2: E  (toward file h)
    (+1, +1),  # 3: SE
    (+1, 0),   # 4: S  (toward rank 1)
    (+1, -1),  # 5: SW
    (0, -1),   # 6: W  (toward file a)
    (-1, -1),  # 7: NW
]

# Knight move offsets: (dr, dc)
KNIGHT_OFFSETS = [
    (-2, +1),  # 56
    (-2, -1),  # 57
    (-1, +2),  # 58
    (-1, -2),  # 59
    (+1, +2),  # 60
    (+1, -2),  # 61
    (+2, +1),  # 62
    (+2, -1),  # 63
]

# Underpromotion encoding (from white's perspective: "forward" = toward rank 8)
# Index 64-72 = piece_type_offset * 3 + direction_offset
# piece_type: 0=knight, 1=bishop, 2=rook
# direction: 0=forward, 1=forward-left, 2=forward-right
UNDERPROMOTION_OFFSETS = {
    # (dr, dc) for white pawn moving forward (toward rank 8)
    # chess.square_rank(): rank 0 = rank 1, rank 7 = rank 8
    # Forward for white = increasing rank = +dr
    # Forward for black = decreasing rank = -dr (inverted in policy_index_to_move)
    "forward":      (1, 0),
    "forward_left": (1, -1),
    "forward_right": (1, +1),
}
UNDERPROMOTION_DIRS = ["forward", "forward_left", "forward_right"]
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Promotion plane index = 64 + piece_idx * 3 + dir_idx
def _underpromotion_plane(piece_type, dir_idx):
    """Get the plane index for an underpromotion."""
    piece_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[piece_type]
    return 64 + piece_idx * 3 + dir_idx


def _rotate_square(sq: int) -> int:
    """Rotate a square 180 degrees (player-oriented frame).

    ``(rank, file) -> (7 - rank, 7 - file)``.  Used when Black is to move
    so the current player's home rank is always at array row 0.
    """
    return 63 - sq


def _encode_single_position(board, plane_offset, tensor):
    """Encode a single board position's PIECE + REPETITION planes into
    ``tensor`` at ``plane_offset``.  Side-to-move / castling are written
    once as global L-planes by ``board_to_tensor``.

    The position is oriented to the perspective of the current player
    (``board.turn``): when Black is to move the board is rotated 180
    degrees, and piece planes represent P1 (current player) vs P2
    (opponent) rather than white/black.

    Uses the Rust ``FastBoard.piece_map()`` to iterate only over occupied
    squares, and extracts ``piece_type`` / ``color`` from the raw
    ``FastPiece`` objects – no Python ``Piece`` creation.
    """
    p1_color = board.turn
    rotate = (p1_color == chess.BLACK)

    pm = board._b.piece_map()  # dict[int, FastPiece]  (square → raw piece)

    for sq, p in pm.items():
        pt = p.piece_type        # 1..6 (PAWN..KING)
        color = p.color          # True = WHITE, False = BLACK
        if rotate:
            sq = _rotate_square(sq)
        row = sq >> 3
        col = sq & 7
        if color == p1_color:
            tensor[plane_offset + pt - 1, row, col] = 1.0
        else:
            tensor[plane_offset + 6 + pt - 1, row, col] = 1.0

    # Repetition planes (per-timestep, from this position's own history).
    if board.is_repetition(2):
        tensor[plane_offset + PLANE_REPETITION_P1, :, :] = 1.0
    if board.is_repetition(3):
        tensor[plane_offset + PLANE_REPETITION_P2, :, :] = 1.0


def _encode_global_planes(board: chess.Board, tensor: np.ndarray):
    """Write the L=7 constant-valued planes (once, not per-timestep)."""
    # Player's colour (P1's absolute colour): 1.0 if White to move.
    tensor[PLANE_SIDE_TO_MOVE, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights, player-relative: P1 = current player, P2 = opponent.
    p1 = board.turn
    p2 = not board.turn
    tensor[PLANE_CASTLING_P1_K, :, :] = 1.0 if board.has_kingside_castling_rights(p1) else 0.0
    tensor[PLANE_CASTLING_P1_Q, :, :] = 1.0 if board.has_queenside_castling_rights(p1) else 0.0
    tensor[PLANE_CASTLING_P2_K, :, :] = 1.0 if board.has_kingside_castling_rights(p2) else 0.0
    tensor[PLANE_CASTLING_P2_Q, :, :] = 1.0 if board.has_queenside_castling_rights(p2) else 0.0

    # No-progress count (halfmove clock; 100 half-moves = 50-move rule)
    tensor[PLANE_NO_PROGRESS_COUNT, :, :] = min(board.halfmove_clock / NO_PROGRESS_NORM, 1.0)

    # Total move count
    tensor[PLANE_MOVE_COUNT, :, :] = board.fullmove_number / MOVE_COUNT_NORM


def _board_to_tensor_rust(board: chess.Board) -> np.ndarray:
    """Fast-path tensor encode entirely in Rust (FastBoard.encode_tensor).

    Bit-identical to the pure-Python encoding (verified), ~6x faster.
    """
    b = board._b
    if b is None:
        return board_to_tensor(board, 8)
    if hasattr(b, "encode_tensor"):
        data = b.encode_tensor(NO_PROGRESS_NORM, MOVE_COUNT_NORM)
        return np.frombuffer(data, dtype=np.float32).reshape(NUM_PLANES, 8, 8)
    return board_to_tensor(board, 8)


def board_to_tensor(board: chess.Board,
                    history_length: int = 8) -> np.ndarray:
    """Encode a chess.Board as a (119, 8, 8) float32 numpy array.

    Structure: M*T + L, per the AlphaZero paper (Table S1, Chess column):
      - M=14 planes per history step (6 P1 pieces + 6 P2 pieces + 2
        repetition), repeated for T=8 history steps (112 planes)
      - L=7 constant-valued planes appended once: player's colour, 4
        player-relative castling rights, no-progress count, total move
        count.

    The board is oriented to the current player's perspective: rotated
    180 degrees when Black is to move, with P1/P2 piece planes and
    player-relative castling.

    **Optimization**: Instead of creating 8 separate board copies (one per
    history step), we copy the board once and walk backwards by popping
    moves, encoding each position's piece planes in-place.  This eliminates
    8 ``board.copy(stack=True)`` calls per tensor (a major hot-path saving).

    Args:
        board: Current board position (must have full move history)
        history_length: Number of historical positions to encode (default 8)

    Returns:
        tensor: (119, 8, 8) float32 numpy array
    """
    if history_length == 8 and hasattr(board._b, "encode_tensor"):
        return _board_to_tensor_rust(board)
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Work on a single copy — we'll pop moves to walk back in history.
    b = board.copy(stack=True)
    num_moves = len(b.move_stack)

    # Encode piece planes from most-recent (ply 0) backwards.
    for ply in range(history_length):
        plane_offset = ply * PLANES_PER_HISTORY

        if ply <= num_moves:
            _encode_single_position(b, plane_offset, tensor)
            if ply < num_moves and ply + 1 < history_length:
                b.pop()
        else:
            # Before the game started: fill with an empty board.
            pass

    # Global L-planes, computed once from the *current* position (not
    # affected by the history walk above, which only reads `b`).
    _encode_global_planes(board, tensor)

    return tensor


def board_to_tensor_batch(board: chess.Board) -> np.ndarray:
    """Encode board as batch tensor (1, 119, 8, 8)."""
    return board_to_tensor(board)[np.newaxis, ...]


def square_to_rank_file(square: int):
    """Convert chess square index to (rank, file). rank 0 = rank 1, file 0 = file a."""
    return chess.square_rank(square), chess.square_file(square)


def rank_file_to_square(rank: int, file: int) -> int:
    """Convert (rank, file) to chess square index. rank 0 = rank 1."""
    return chess.square(file, rank)

def move_to_policy_index(move: chess.Move, board: chess.Board) -> int:
    """Convert a chess.Move to a flat policy index (0-4671).

    The policy space is organized as 8*8*73, where for each source square
    (in rank-file order, rank 0 first), there are 73 possible move planes.

    NOTE: the policy uses ABSOLUTE compass directions (N = toward rank 8
    for both players), per the AlphaZero chess paper.  The input board is
    player-oriented, but the action space is absolute.
    """
    # The expensive part here is *not* the index math but the cache KEY: the
    # old @lru_cache was keyed on (move, board), and Board.__hash__ computes a
    # full fen() on EVERY call.  Underpromotion direction depends only on
    # `board.turn`, so cache on (move, turn) and avoid hashing the board.
    return _move_to_policy_index_cached(move, board.turn)


@lru_cache(maxsize=8192)
def _move_to_policy_index_cached(move: chess.Move, turn: bool) -> int:
    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    # dr, dc relative to source (in rank-file coordinates)
    dr = to_rank - from_rank
    dc = to_file - from_file

    # Underpromotion direction is turn-dependent (no board access needed).
    is_promotion = move.promotion is not None

    if is_promotion and move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        # Underpromotion
        # For white: forward = +1 rank, for black: forward = -1 rank
        if turn == chess.WHITE:
            if dr == 1 and dc == 0:
                dir_idx = 0  # forward
            elif dr == 1 and dc == -1:
                dir_idx = 1  # forward-left
            elif dr == 1 and dc == 1:
                dir_idx = 2  # forward-right
            else:
                raise ValueError(f"Invalid underpromotion move: {move}")
        else:
            if dr == -1 and dc == 0:
                dir_idx = 0  # forward
            elif dr == -1 and dc == 1:
                dir_idx = 1  # forward-left (from black's perspective, going toward rank 1)
            elif dr == -1 and dc == -1:
                dir_idx = 2  # forward-right
            else:
                raise ValueError(f"Invalid underpromotion move: {move}")

        plane = _underpromotion_plane(move.promotion, dir_idx)
    else:
        # Queen-like or knight move (including queen promotions)
        # Try to find as queen-like move
        plane = _find_queen_move_plane(dr, dc)
        if plane is None:
            # Try knight move
            plane = _find_knight_move_plane(dr, dc)
        if plane is None:
            raise ValueError(f"Cannot encode move {move} (dr={dr}, dc={dc})")

    # Flat index: (from_rank * 8 + from_file) * 73 + plane
    source_idx = from_rank * 8 + from_file
    return source_idx * 73 + plane


def _find_queen_move_plane(dr, dc):
    """Find the plane index for a queen-like move with given delta."""
    if dr == 0 and dc == 0:
        return None

    # Find direction
    for d_idx, (qdr, qdc) in enumerate(QUEEN_DIRECTIONS):
        # Check if (dr, dc) is in this direction
        if qdr == 0:
            if dr != 0:
                continue
            if qdc > 0 and dc <= 0:
                continue
            if qdc < 0 and dc >= 0:
                continue
            dist = abs(dc)
        elif qdc == 0:
            if dc != 0:
                continue
            if qdr > 0 and dr <= 0:
                continue
            if qdr < 0 and dr >= 0:
                continue
            dist = abs(dr)
        else:
            # Diagonal: dr/dc must match the direction ratio
            if qdr * dc != qdc * dr:
                continue
            if (dr > 0 and qdr < 0) or (dr < 0 and qdr > 0):
                continue
            if (dc > 0 and qdc < 0) or (dc < 0 and qdc > 0):
                continue
            dist = abs(dr) if abs(qdr) == 1 else abs(dr)
            # Verify it's actually on the diagonal with matching magnitude
            if abs(dr) != abs(dc):
                continue

        if 1 <= dist <= 7:
            return d_idx * 7 + (dist - 1)

    return None


def _find_knight_move_plane(dr, dc):
    """Find the plane index for a knight move with given delta."""
    for k_idx, (kdr, kdc) in enumerate(KNIGHT_OFFSETS):
        if dr == kdr and dc == kdc:
            return 56 + k_idx
    return None


def policy_index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a flat policy index (0-4671) to a chess.Move.

    Returns None if the move is not valid for the given board state.
    """
    source_idx = index // 73
    plane = index % 73

    from_rank = source_idx // 8
    from_file = source_idx % 8
    from_square = rank_file_to_square(from_rank, from_file)

    piece = board.piece_at(from_square)
    if piece is None:
        return None

    if plane < 56:
        # Queen-like move
        d_idx = plane // 7
        dist = (plane % 7) + 1
        dr, dc = QUEEN_DIRECTIONS[d_idx]
        to_rank = from_rank + dr * dist
        to_file = from_file + dc * dist
    elif plane < 64:
        # Knight move
        k_idx = plane - 56
        kdr, kdc = KNIGHT_OFFSETS[k_idx]
        to_rank = from_rank + kdr
        to_file = from_file + kdc
    else:
        # Underpromotion
        under_idx = plane - 64
        piece_idx = under_idx // 3
        dir_idx = under_idx % 3
        promo_piece = UNDERPROMOTION_PIECES[piece_idx]

        dir_name = UNDERPROMOTION_DIRS[dir_idx]
        dr, dc = UNDERPROMOTION_OFFSETS[dir_name]

        # Adjust direction based on color
        if board.turn == chess.BLACK:
            dr = -dr
            dc = -dc

        to_rank = from_rank + dr
        to_file = from_file + dc

    # Bounds check
    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        return None

    to_square = rank_file_to_square(to_rank, to_file)

    # Determine promotion
    promotion = None
    if piece.piece_type == chess.PAWN:
        if (board.turn == chess.WHITE and to_rank == 7) or \
           (board.turn == chess.BLACK and to_rank == 0):
            if plane < 56 or plane >= 64:
                # Queen-like move for a pawn to promotion rank = queen promotion
                # Underpromotion planes have their own piece type
                if plane < 56:
                    promotion = chess.QUEEN
                else:
                    under_idx = plane - 64
                    piece_idx = under_idx // 3
                    promotion = UNDERPROMOTION_PIECES[piece_idx]

    move = chess.Move(from_square, to_square, promotion=promotion)

    # Verify the move is legal
    if move in board.legal_moves:
        return move

    # If not legal, try without promotion (for queen-like pawn forward moves)
    if promotion == chess.QUEEN:
        move_no_promo = chess.Move(from_square, to_square, promotion=None)
        # This shouldn't happen for a pawn reaching the last rank, but just in case
        if move_no_promo in board.legal_moves:
            return move_no_promo

    return None


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Get a (4672,) binary mask of legal moves for the current position."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for move in board.legal_moves:
        try:
            idx = move_to_policy_index(move, board)
            mask[idx] = 1.0
        except ValueError:
            # Move encoding failed — shouldn't happen for legal moves
            continue
    return mask


def get_legal_move_mask_from_moves(legal_moves: list, board: chess.Board) -> np.ndarray:
    """Get a (4672,) binary mask from a precomputed list of legal moves.

    This avoids a second iteration over board.legal_moves when the caller
    has already generated the legal moves list.

    Args:
        legal_moves: List of chess.Move objects (already computed)
        board: Board state (needed for move_to_policy_index context)

    Returns:
        mask: (4672,) binary mask
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for move in legal_moves:
        try:
            idx = move_to_policy_index(move, board)
            mask[idx] = 1.0
        except ValueError:
            continue
    return mask


def get_all_policy_indices(board: chess.Board) -> dict:
    """Get mapping from legal move to policy index for the current position."""
    result = {}
    for move in board.legal_moves:
        try:
            idx = move_to_policy_index(move, board)
            result[move] = idx
        except ValueError:
            continue
    return result


def policy_to_move_dict(board: chess.Board, policy: np.ndarray, top_k: int = 5):
    """Convert a raw policy vector to the top-k legal moves with probabilities.

    Args:
        board: Current board state
        policy: (4672,) raw policy logits or probabilities
        top_k: Number of top moves to return

    Returns:
        List of (move, probability) tuples, sorted by probability descending.
    """
    mask = get_legal_move_mask(board)
    masked = policy * mask

    # Renormalize
    total = masked.sum()
    if total > 0:
        masked = masked / total

    # Get top-k indices
    flat_indices = np.argsort(-masked)[:top_k]

    result = []
    for idx in flat_indices:
        if masked[idx] > 0:
            move = policy_index_to_move(idx, board)
            if move is not None:
                result.append((move, float(masked[idx])))

    return result