"""Board state and move encoding for the AlphaZero chess engine.

Board Representation (8x8x20):
    Planes 0-5:   White P, N, B, R, Q, K
    Planes 6-11:  Black P, N, B, R, Q, K
    Plane 12:     Side to move (1.0 if white)
    Planes 13-16: Castling rights (WK, WQ, BK, BQ)
    Plane 17:     En passant square
    Plane 18:     Repetition count >= 2 (position seen before)
    Plane 19:     Repetition count >= 3 (on verge of 3-fold repetition)
    - Repetition planes can optionally be passed in via `repetition_counts`
      to avoid expensive board.is_repetition() calls.

Move Encoding (8x8x73 = 4672 action space):
    Planes 0-55:  Queen-like moves (8 directions x 7 distances)
    Planes 56-63: Knight moves (8 offsets)
    Planes 64-72: Underpromotions (3 pieces x 3 horizontal offsets)
    - Queen promotions are encoded as queen-like moves (forward direction)
"""

import numpy as np
import chess
from typing import Optional, Tuple

# Piece type to plane index mapping (python-chess Piece objects)
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

NUM_PLANES = 20
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

# Underpromotion encoding
UNDERPROMOTION_OFFSETS = {
    "forward":      (1, 0),
    "forward_left": (1, -1),
    "forward_right": (1, +1),
}
UNDERPROMOTION_DIRS = ["forward", "forward_left", "forward_right"]
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _underpromotion_plane(piece_type, dir_idx):
    """Get the plane index for an underpromotion."""
    piece_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[piece_type]
    return 64 + piece_idx * 3 + dir_idx


# Precomputed 64x64 lookup table for queen-like + knight moves
MOVE_PLANE_LUT = np.full((64, 64), -1, dtype=np.int16)

def _build_plane_lut():
    for from_rank in range(8):
        for from_file in range(8):
            from_sq = chess.square(from_file, from_rank)
            base_idx = from_sq * 64
            for d_idx, (qdr, qdc) in enumerate(QUEEN_DIRECTIONS):
                for dist in range(1, 8):
                    to_rank = from_rank + qdr * dist
                    to_file = from_file + qdc * dist
                    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                        break
                    to_sq = chess.square(to_file, to_rank)
                    plane = d_idx * 7 + (dist - 1)
                    MOVE_PLANE_LUT[base_idx + to_sq] = plane
            for k_idx, (kdr, kdc) in enumerate(KNIGHT_OFFSETS):
                to_rank = from_rank + kdr
                to_file = from_file + kdc
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = chess.square(to_file, to_rank)
                    MOVE_PLANE_LUT[base_idx + to_sq] = 56 + k_idx

_build_plane_lut()


# ---------------------------------------------------------------------------
#  Board → Tensor  (pure python‑chess)
# ---------------------------------------------------------------------------
def board_to_tensor(board, repetition_counts: Optional[Tuple[bool, bool]] = None) -> np.ndarray:
    """Convert a chess.Board to a (20,8,8) float32 tensor."""
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Pieces (planes 0–11)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        tensor[PIECE_PLANE[(piece.piece_type, piece.color)], rank, file] = 1.0

    # Side to move (plane 12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Castling rights (planes 13–16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # En passant (plane 17)
    ep = board.ep_square
    if ep is not None:
        tensor[17, chess.square_rank(ep), chess.square_file(ep)] = 1.0

    # Repetition planes 18–19
    if repetition_counts is not None:
        rep2, rep3 = repetition_counts
    else:
        rep2 = board.is_repetition(2)
        rep3 = board.is_repetition(3)
    if rep2:
        tensor[18, :, :] = 1.0
    if rep3:
        tensor[19, :, :] = 1.0

    return tensor


def board_to_tensor_batch(board: chess.Board) -> np.ndarray:
    """Encode board as batch tensor (1, 20, 8, 8)."""
    return board_to_tensor(board)[np.newaxis, ...]


# ---------------------------------------------------------------------------
#  Move → Policy Index  (optimised)
# ---------------------------------------------------------------------------
def move_to_policy_index(move, board: chess.Board) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    prom = move.promotion

    # Fast path: non‑underpromotion (incl. queen promos)
    if prom is not None and prom != chess.QUEEN:
        # Rare underpromotion branch – still fast but kept separate
        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        to_rank,   to_file   = chess.square_rank(to_sq),   chess.square_file(to_sq)
        dr, dc = to_rank - from_rank, to_file - from_file

        turn = board.turn
        if turn == chess.WHITE:
            # white pawn forward: dr == 1
            if dr == 1:
                if dc == 0:       dir_idx = 0
                elif dc == -1:    dir_idx = 1
                elif dc == 1:     dir_idx = 2
                else: raise ValueError(f"Invalid underpromotion move: {move}")
            else:
                raise ValueError(f"Invalid underpromotion move: {move}")
        else:  # BLACK
            if dr == -1:
                if dc == 0:       dir_idx = 0
                elif dc == 1:     dir_idx = 1    # black forward-left
                elif dc == -1:    dir_idx = 2    # black forward-right
                else: raise ValueError(f"Invalid underpromotion move: {move}")
            else:
                raise ValueError(f"Invalid underpromotion move: {move}")

        # Inline the piece index + dir offset
        # piece_idx: knight=0, bishop=1, rook=2
        piece_idx = (prom - chess.KNIGHT)  # if prom is KNIGHT(2), BISHOP(3), ROOK(4)
        # mapping: 2->0, 3->1, 4->2  => (prom - 2)
        plane = 64 + piece_idx * 3 + dir_idx
    else:
        # Queen-like, knight, or queen promotion: flat LUT
        plane = MOVE_PLANE_LUT[from_sq * 64 + to_sq]
        if plane == -1:
            raise ValueError(f"Cannot encode move {move} (no LUT entry for {from_sq}->{to_sq})")

    return from_sq * 73 + plane


def policy_index_to_move(index: int, board: chess.Board):
    """Convert a policy index back to a legal chess.Move (if possible)."""
    source_idx = index // 73
    plane = index % 73
    from_rank, from_file = source_idx // 8, source_idx % 8
    from_square = chess.square(from_file, from_rank)
    piece = board.piece_at(from_square)
    if piece is None:
        return None

    if plane < 56:          # queen‑like
        d_idx, dist = plane // 7, (plane % 7) + 1
        dr, dc = QUEEN_DIRECTIONS[d_idx]
        to_rank = from_rank + dr * dist
        to_file = from_file + dc * dist
    elif plane < 64:        # knight
        k_idx = plane - 56
        dr, dc = KNIGHT_OFFSETS[k_idx]
        to_rank = from_rank + dr
        to_file = from_file + dc
    else:                    # underpromotion
        under_idx = plane - 64
        piece_idx, dir_idx = under_idx // 3, under_idx % 3
        promo_piece = UNDERPROMOTION_PIECES[piece_idx]
        dir_name = UNDERPROMOTION_DIRS[dir_idx]
        dr, dc = UNDERPROMOTION_OFFSETS[dir_name]
        if board.turn == chess.BLACK:
            dr, dc = -dr, -dc
        to_rank = from_rank + dr
        to_file = from_file + dc

    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        return None
    to_square = chess.square(to_file, to_rank)

    # Promotion logic
    promotion = None
    if piece.piece_type == chess.PAWN:
        if (board.turn == chess.WHITE and to_rank == 7) or \
           (board.turn == chess.BLACK and to_rank == 0):
            if plane < 56:
                promotion = chess.QUEEN
            elif plane >= 64:
                under_idx = plane - 64
                promotion = UNDERPROMOTION_PIECES[under_idx // 3]

    move = chess.Move(from_square, to_square, promotion=promotion)
    if move in board.legal_moves:
        return move
    # Sometimes queen promotions are stored as plain moves (without promotion flag)
    if promotion == chess.QUEEN:
        move_no_promo = chess.Move(from_square, to_square)
        if move_no_promo in board.legal_moves:
            return move_no_promo
    return None


# ---------------------------------------------------------------------------
#  Masks / Utilities
# ---------------------------------------------------------------------------
def get_legal_move_mask(board: chess.Board) -> np.ndarray:
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
    result = {}
    for move in board.legal_moves:
        try:
            result[move] = move_to_policy_index(move, board)
        except ValueError:
            continue
    return result


def policy_to_move_dict(board: chess.Board, policy: np.ndarray,
                        top_k: int = 5, repetition_counts=None):
    mask = get_legal_move_mask(board)
    masked = policy * mask
    total = masked.sum()
    if total > 0:
        masked = masked / total
    flat_indices = np.argsort(-masked)[:top_k]
    result = []
    for idx in flat_indices:
        if masked[idx] > 0:
            move = policy_index_to_move(idx, board)
            if move is not None:
                result.append((move, float(masked[idx])))
    return result