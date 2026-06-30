"""Board state and move encoding for the AlphaZero chess engine.

Board Representation (105x8x8):
    Planes 0-95:   8 history frames x 12 piece planes each, most-recent
                   first (frame 0 = current position). Each frame's 12
                   planes are: White P,N,B,R,Q,K then Black P,N,B,R,Q,K.
                   Frames beyond the available game history are zero.
    Plane 96:      Side to move (1.0 if white)
    Plane 97:      En passant square
    Planes 98-101: Castling rights (WK, WQ, BK, BQ)
    Plane 102:     No-progress count (halfmove clock, normalized /100)
    Plane 103:     Repetition count >= 2 (position seen before)
    Plane 104:     Repetition count >= 3 (on verge of 3-fold repetition)
    - Repetition planes can optionally be passed in via `repetition_counts`
      to avoid expensive board.is_repetition() calls.
    - History frames can optionally be passed in via `history` (a list of
      precomputed (12,8,8) piece-plane arrays, most-recent-first) to avoid
      reconstructing them from a move stack on every call.

Move Encoding (8x8x73 = 4672 action space):
    Planes 0-55:  Queen-like moves (8 directions × 7 distances)
    Planes 56-63: Knight moves (8 offsets)
    Planes 64-72: Underpromotions (3 pieces × 3 horizontal offsets)
    - Queen promotions are encoded as queen-like moves (forward direction)

Performance optimisations:
    - MOVE_PLANE_LUT: precomputed 64×64 lookup table mapping (from_sq, to_sq)
      → queen/knight plane index (0-63), or -1 if not a valid queen/knight delta.
      This eliminates all direction-search loops, square rank/file calls, and
      delta arithmetic in the hot path of move_to_policy_index.
    - board_to_tensor uses piece_map() instead of iterating all 64 squares.
    - board_to_tensor accepts optional repetition_counts to avoid expensive
      is_repetition() stack scanning.
    - board_to_tensor accepts optional precomputed history frames to avoid
      repeatedly walking a board's move stack.
"""

import numpy as np
import chess
from typing import Optional, Tuple, List

# Piece type to plane index mapping (within a single 12-plane frame)
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

# ---------------------------------------------------------------------------
# History / plane layout
# ---------------------------------------------------------------------------
NUM_HISTORY_FRAMES = 8          # current position + 7 prior positions
PLANES_PER_FRAME = 12           # 6 white piece types + 6 black piece types
HISTORY_PLANES = NUM_HISTORY_FRAMES * PLANES_PER_FRAME  # 96

# History (96) + side_to_move (1) + en_passant (1) + castling (4)
# + no_progress (1) + repetition>=2 (1) + repetition>=3 (1) = 105
NUM_PLANES = HISTORY_PLANES + 9

NUM_ACTIONS = 8 * 8 * 73  # 4672

# Queen move directions: (dr, dc)
# row 0 = rank 8 (top), col 0 = file a (left)
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


# ---------------------------------------------------------------------------
# Precomputed 64×64 lookup table: MOVE_PLANE_LUT[from_sq][to_sq] → plane (0-63)
# -1 means the delta is not a valid queen or knight move (e.g. same square,
# non-matching ratio, or distance > 7).
# ---------------------------------------------------------------------------
MOVE_PLANE_LUT = np.full((64, 64), -1, dtype=np.int16)

def _build_plane_lut():
    """Fill MOVE_PLANE_LUT for all valid queen-like and knight moves."""
    for from_rank in range(8):
        for from_file in range(8):
            from_sq = chess.square(from_file, from_rank)

            # Queen-like moves
            for d_idx, (qdr, qdc) in enumerate(QUEEN_DIRECTIONS):
                for dist in range(1, 8):
                    to_rank = from_rank + qdr * dist
                    to_file = from_file + qdc * dist
                    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                        break
                    to_sq = chess.square(to_file, to_rank)
                    plane = d_idx * 7 + (dist - 1)
                    MOVE_PLANE_LUT[from_sq, to_sq] = plane

            # Knight moves
            for k_idx, (kdr, kdc) in enumerate(KNIGHT_OFFSETS):
                to_rank = from_rank + kdr
                to_file = from_file + kdc
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = chess.square(to_file, to_rank)
                    MOVE_PLANE_LUT[from_sq, to_sq] = 56 + k_idx

_build_plane_lut()


def piece_planes(board: chess.Board) -> np.ndarray:
    """Encode just the piece placement of `board` as a (12, 8, 8) array.

    This is the per-frame building block used both for the current
    position and for historical frames.
    """
    planes = np.zeros((PLANES_PER_FRAME, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        planes[PIECE_PLANE[(piece.piece_type, piece.color)], rank, file] = 1.0
    return planes


def history_from_board(board: chess.Board,
                        max_frames: int = NUM_HISTORY_FRAMES - 1) -> List[np.ndarray]:
    """Reconstruct up to `max_frames` previous positions as piece-plane
    arrays, most-recent-first, by popping moves off a copy of `board`.

    Requires `board` to carry its full move stack (i.e. not produced via
    `board.copy(stack=False)`). Returns a (possibly shorter) list if fewer
    moves exist in the stack; `board_to_tensor` zero-pads any remainder.

    Args:
        board: The current position. Must have its move stack intact.
        max_frames: Maximum number of prior frames to reconstruct
            (defaults to NUM_HISTORY_FRAMES - 1, i.e. enough to fill out
            the full history alongside the current position).

    Returns:
        List of (12, 8, 8) float32 arrays, most-recent-first.
    """
    frames = []
    if max_frames <= 0:
        return frames
    b = board.copy(stack=True)
    for _ in range(max_frames):
        if not b.move_stack:
            break
        b.pop()
        frames.append(piece_planes(b))
    return frames


def board_to_tensor(board: chess.Board,
                     history: Optional[List[np.ndarray]] = None,
                     repetition_counts: Optional[Tuple[bool, bool]] = None) -> np.ndarray:
    """Encode a chess.Board (+ history) as a (105, 8, 8) float32 numpy array.

    Planes 0-95:   8 history frames x 12 piece planes, most-recent first
                   (frame 0 = current position).
    Plane 96:      Side to move (1.0 if white to move)
    Plane 97:      En passant square
    Planes 98-101: Castling rights (WK, WQ, BK, BQ)
    Plane 102:     No-progress count (halfmove clock, normalized /100)
    Plane 103:     Repetition count >= 2 (position seen before)
    Plane 104:     Repetition count >= 3 (on verge of 3-fold repetition)

    Args:
        board: The chess board to encode (the *current* position).
        history: Optional list of up to (NUM_HISTORY_FRAMES - 1) precomputed
            (12, 8, 8) piece-plane arrays for the immediately preceding
            positions, most-recent-first. If None, all historical frames
            beyond the current position are left as zero (no history
            available/provided). Missing frames (list shorter than
            NUM_HISTORY_FRAMES - 1) are zero-padded.
        repetition_counts: Optional (rep2, rep3) bools specifying repetition
            state. If None, board.is_repetition(2) and is_repetition(3) are
            called (expensive for deep stacks). When called from MCTS with
            tree-based repetition tracking, pass the precomputed values.
    """
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # ---- History frames (current position + up to 7 prior) ----
    frames = [piece_planes(board)]
    if history:
        frames.extend(history[:NUM_HISTORY_FRAMES - 1])
    while len(frames) < NUM_HISTORY_FRAMES:
        frames.append(np.zeros((PLANES_PER_FRAME, 8, 8), dtype=np.float32))

    for i, fp in enumerate(frames[:NUM_HISTORY_FRAMES]):
        tensor[i * PLANES_PER_FRAME:(i + 1) * PLANES_PER_FRAME] = fp

    idx = HISTORY_PLANES

    # Side to move
    if board.turn == chess.WHITE:
        tensor[idx, :, :] = 1.0
    idx += 1

    # En passant
    ep = board.ep_square
    if ep is not None:
        rank = chess.square_rank(ep)
        file = chess.square_file(ep)
        tensor[idx, rank, file] = 1.0
    idx += 1

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[idx, :, :] = 1.0
    idx += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[idx, :, :] = 1.0
    idx += 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[idx, :, :] = 1.0
    idx += 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[idx, :, :] = 1.0
    idx += 1

    # No-progress count (halfmove clock), normalized
    tensor[idx, :, :] = min(board.halfmove_clock, 100) / 100.0
    idx += 1

    # Repetition count planes – use precomputed values if provided
    if repetition_counts is not None:
        rep2, rep3 = repetition_counts
    else:
        rep2 = board.is_repetition(2)
        rep3 = board.is_repetition(3)

    if rep2:
        tensor[idx, :, :] = 1.0
    idx += 1
    if rep3:
        tensor[idx, :, :] = 1.0
    idx += 1

    return tensor


def board_to_tensor_batch(board: chess.Board) -> np.ndarray:
    """Encode board as batch tensor (1, NUM_PLANES, 8, 8)."""
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

    Performance: uses a precomputed MOVE_PLANE_LUT to avoid direction-search
    loops for the hot path (queen/knight moves). Underpromotions still need
    the piece type and direction, but those are rare (~0.7% of moves).
    """
    from_sq = move.from_square

    # Handle underpromotions (rare case, handled explicitly)
    if move.promotion is not None and move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        to_sq = move.to_square
        to_rank = chess.square_rank(to_sq)
        to_file = chess.square_file(to_sq)
        dr = to_rank - from_rank
        dc = to_file - from_file
        if board.turn == chess.WHITE:
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
                dir_idx = 1  # forward-left (from black's perspective)
            elif dr == -1 and dc == -1:
                dir_idx = 2  # forward-right
            else:
                raise ValueError(f"Invalid underpromotion move: {move}")
        plane = _underpromotion_plane(move.promotion, dir_idx)
    else:
        # Fast path: look up in precomputed LUT
        to_sq = move.to_square
        plane = MOVE_PLANE_LUT[from_sq, to_sq]
        if plane == -1:
            raise ValueError(f"Cannot encode move {move} (no LUT entry for {from_sq}→{to_sq})")

    # Flat index: (from_rank * 8 + from_file) * 73 + plane
    # from_sq encodes the same as (rank * 8 + file) on a chess board
    return from_sq * 73 + plane


def _find_queen_move_plane(dr, dc):
    """Find the plane index for a queen-like move with given delta.
    Kept for backward compatibility; not used in hot path.
    """
    if dr == 0 and dc == 0:
        return None

    for d_idx, (qdr, qdc) in enumerate(QUEEN_DIRECTIONS):
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
            if qdr * dc != qdc * dr:
                continue
            if (dr > 0 and qdr < 0) or (dr < 0 and qdr > 0):
                continue
            if (dc > 0 and qdc < 0) or (dc < 0 and qdc > 0):
                continue
            dist = abs(dr) if abs(qdr) == 1 else abs(dr)
            if abs(dr) != abs(dc):
                continue

        if 1 <= dist <= 7:
            return d_idx * 7 + (dist - 1)

    return None


def _find_knight_move_plane(dr, dc):
    """Find the plane index for a knight move with given delta.
    Kept for backward compatibility; not used in hot path.
    """
    for k_idx, (kdr, kdc) in enumerate(KNIGHT_OFFSETS):
        if dr == kdr and dc == kdc:
            return 56 + k_idx
    return None


def policy_index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a flat policy index (0-4671) to a chess.Move.

    Returns None if the move is not valid for the given board state.
    Uses MOVE_PLANE_LUT to validate the queen/knight delta.
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


def policy_to_move_dict(board: chess.Board, policy: np.ndarray, top_k: int = 5, repetition_counts=None):
    """Convert a raw policy vector to the top-k legal moves with probabilities.

    Args:
        board: Current board state
        policy: (4672,) raw policy logits or probabilities
        top_k: Number of top moves to return
        repetition_counts: Optional repetition info to pass to board_to_tensor

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