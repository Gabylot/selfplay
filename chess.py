"""Drop-in python-chess replacement backed by the fast Rust `fastchess` extension.

This module is a compatibility shim: the existing codebase does
``import chess`` and uses the ``chess.Board`` / ``chess.Move`` /
``chess.Piece`` API.  Everything here re-exports the fast Rust
implementation (shakmaty-based) with the same names/conventions.

The ``USE_FAST_BACKEND`` flag lets you fall back to the original
python-chess package at runtime if something is missing.
"""

import importlib
import numpy as np

# The fast backend is the default
_has_fast = True
try:
    from fastchess import (
        FastBoard as _FastBoard,
        FastMove as _FastMove,
        FastPiece as _FastPiece,
    )
    import fastchess as _fc
except Exception as _e:  # pragma: no cover
    _has_fast = False
    _reason = _e

# ── Constants (python-chess compatible) ────────────────────────────────────
# python-chess: WHITE = True, BLACK = False
WHITE = True
BLACK = False

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

SQUARES = list(range(64))  # A1=0 ... H8=63

# Square name constants (python-chess: chess.A1=0 ... chess.H8=63)
A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A2, B2, C2, D2, E2, F2, G2, H2 = range(8, 16)
A3, B3, C3, D3, E3, F3, G3, H3 = range(16, 24)
A4, B4, C4, D4, E4, F4, G4, H4 = range(24, 32)
A5, B5, C5, D5, E5, F5, G5, H5 = range(32, 40)
A6, B6, C6, D6, E6, F6, G6, H6 = range(40, 48)
A7, B7, C7, D7, E7, F7, G7, H7 = range(48, 56)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)

# ── Square helpers ─────────────────────────────────────────────────────────
def square_rank(square):
    return (square >> 3) & 7


def square_file(square):
    return square & 7


def square(file, rank):
    """python-chess: square(file, rank) -> A1=0 ... H8=63."""
    return rank * 8 + file


def square_name(square):
    f = (square & 7) + ord('a')
    r = (square >> 3) + ord('1')
    return chr(f) + chr(r)


def parse_square(name):
    return square(ord(name[0]) - ord('a'), ord(name[1]) - ord('1'))


# ── Move ────────────────────────────────────────────────────────────────────
class Move:
    """Fast move with python-chess-compatible surface."""

    __slots__ = ('_m',)

    def __init__(self, from_square, to_square, promotion=None):
        # Accept both int square indices and square-name strings
        if isinstance(from_square, str):
            from_square = parse_square(from_square)
        if isinstance(to_square, str):
            to_square = parse_square(to_square)
        self._m = _FastMove(from_square, to_square, promotion)

    @staticmethod
    def from_uci(uci):
        return Move._wrap(_FastMove.from_uci(uci))

    @staticmethod
    def _wrap(fm):
        mv = object.__new__(Move)
        mv._m = fm
        return mv

    @property
    def from_square(self):
        return self._m.from_square

    @property
    def to_square(self):
        return self._m.to_square

    @property
    def promotion(self):
        return self._m.promotion

    def uci(self):
        return self._m.uci()

    def __str__(self):
        return self._m.uci()

    def __repr__(self):
        return f"Move.from_uci('{self.uci()}')"

    def __eq__(self, other):
        if isinstance(other, Move):
            return (self.from_square == other.from_square
                    and self.to_square == other.to_square
                    and self.promotion == other.promotion)
        return NotImplemented

    def __hash__(self):
        return hash((self.from_square, self.to_square, self.promotion))


# ── Piece ───────────────────────────────────────────────────────────────────
class Piece:
    __slots__ = ('_p',)

    def __init__(self, piece_type, color):
        self._p = _FastPiece(piece_type, color)

    @staticmethod
    def from_symbol(symbol):
        c = symbol[0]
        color = WHITE if c.isupper() else BLACK
        pt = {'P': PAWN, 'N': KNIGHT, 'B': BISHOP,
              'R': ROOK, 'Q': QUEEN, 'K': KING}[c.upper()]
        return Piece(pt, color)

    @property
    def piece_type(self):
        return self._p.piece_type

    @property
    def color(self):
        return self._p.color

    def symbol(self):
        return self._p.symbol()

    def __str__(self):
        return self._p.symbol()

    def __repr__(self):
        return f"Piece.from_symbol('{self.symbol()}')"

    def __eq__(self, other):
        if isinstance(other, Piece):
            return (self.piece_type == other.piece_type
                    and self.color == other.color)
        return NotImplemented

    def __hash__(self):
        return hash((self.piece_type, self.color))


# ── Board ───────────────────────────────────────────────────────────────────
class Board:
    __slots__ = ('_b',)

    def __init__(self, fen=None):
        if fen is not None:
            self._b = _FastBoard(fen)
        else:
            self._b = _FastBoard()

    def copy(self, stack=True):
        b = Board.__new__(Board)
        b._b = self._b.copy(stack=stack)
        return b

    def __copy__(self):
        return self.copy(stack=True)

    def __deepcopy__(self, memo):
        return self.copy(stack=True)

    def push(self, move):
        self._b.push(move._m if isinstance(move, Move) else move)

    def pop(self):
        return Move._wrap(self._b.pop())

    # ── Read-only properties ──
    @property
    def turn(self):
        return self._b.turn  # True=White, False=Black (python-chess semantics)

    @property
    def fullmove_number(self):
        return self._b.fullmove_number

    @property
    def halfmove_clock(self):
        return self._b.halfmove_clock

    def ply(self):
        return self._b.ply

    @property
    def legal_moves(self):
        """python-chess compatible: list of Move objects."""
        return [Move._wrap(fm) for fm in self._b.legal_moves]

    @property
    def legal_moves_raw(self):
        """Fast internal access: list of FastMove objects (no wrapping)."""
        return self._b.legal_moves

    @property
    def move_stack(self):
        # python-chess alias
        return self._b.move_stack

    # ── Queries ──
    def fen(self):
        return self._b.fen()

    def piece_at(self, square):
        p = self._b.piece_at(square)
        if p is None:
            return None
        return Piece(p.piece_type, p.color)

    def piece_map(self):
        return {sq: Piece(p.piece_type, p.color)
                for sq, p in self._b.piece_map().items()}

    def piece_arrays(self):
        """Return (pieces, colors) as (64,) uint8 numpy arrays.
        
        pieces: 0=empty, 1=PAWN .. 6=KING
        colors: 1=WHITE, 0=BLACK
        
        Uses the raw FastBoard.piece_map() dict to avoid creating Python Piece objects.
        """
        pieces = np.zeros(64, dtype=np.uint8)
        colors = np.zeros(64, dtype=np.uint8)
        # self._b.piece_map() returns {square: FastPiece} – no wrapper overhead
        for sq, fp in self._b.piece_map().items():
            pieces[sq] = fp.piece_type
            # fp.color is a bool: True=WHITE, False=BLACK
            colors[sq] = 1 if fp.color else 0
        return pieces, colors

    def is_game_over(self, claim_draw=False):
        if self._b.is_game_over(claim_draw=claim_draw):
            return True
        if claim_draw:
            # Claimable draws (python-chess semantics)
            return (self.can_claim_threefold_repetition()
                    or self.can_claim_fifty_moves())
        return False

    def is_checkmate(self):
        return self._b.is_checkmate()

    def is_stalemate(self):
        return self._b.is_stalemate()

    def is_insufficient_material(self):
        return self._b.is_insufficient_material()

    def is_check(self):
        return self._b.is_check()

    def is_fifty_moves(self):
        return self._b.is_fifty_moves()

    def is_seventyfive_moves(self):
        return self._b.is_seventyfive_moves()

    def is_fivefold_repetition(self):
        return self._b.is_fivefold_repetition()

    def is_repetition(self, count):
        return self._b.is_repetition(count)

    def can_claim_threefold_repetition(self):
        return self._b.can_claim_threefold_repetition()

    def can_claim_fifty_moves(self):
        return self._b.can_claim_fifty_moves()

    def result(self, claim_draw=False):
        return self._b.result(claim_draw=claim_draw)

    def has_kingside_castling_rights(self, color):
        return self._b.has_kingside_castling_rights(color)

    def has_queenside_castling_rights(self, color):
        return self._b.has_queenside_castling_rights(color)

    def is_attacked_by(self, color, square):
        return self._b.is_attacked_by(color, square)

    def is_capture(self, move):
        """Return True if the move is a capture (python-chess semantics)."""
        if move.to_square == self.ep_square():
            # En passant capture
            return True
        return self.piece_at(move.to_square) is not None

    def reset(self):
        """Reset the board to the initial position."""
        self._b = _FastBoard()

    def clear(self):
        """Clear the board - set up an empty board (for test setups)."""
        # python-chess clear(): no pieces, white to move, no castling rights
        empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        self._b = _FastBoard(empty_fen)

    def set_piece_at(self, square, piece):
        """Place a piece on the board (python-chess API for test setups)."""
        # Need to implement via FEN reconstruction - get current FEN, modify board
        import re
        fen = self.fen()
        parts = fen.split(' ')
        rows = parts[0].split('/')

        rank = (square >> 3)  # 0 = rank 1, 7 = rank 8
        file_ = square & 7
        row = rows[7 - rank]  # FEN row 0 = rank 8

        # Rebuild the row with the new piece
        new_row = ''
        piece_char = ''
        if piece is not None:
            piece_char = piece.symbol()
            if piece_char.islower():
                piece_char = piece_char.upper() if self.turn == BLACK else piece_char
            else:
                piece_char = piece_char.lower() if self.turn == BLACK else piece_char

        idx = 0
        char_pos = 0
        replaced = False
        for ch in row:
            if ch.isdigit():
                cnt = int(ch)
                if char_pos + cnt <= file_:
                    if char_pos + cnt == file_ and not replaced:
                        # Insert piece before empty run
                        if cnt > 1:
                            new_row += str(cnt - 1)
                        new_row += piece_char or '1'
                        replaced = True
                    else:
                        new_row += ch
                    char_pos += cnt
                else:
                    # Split the run: piece goes inside the digits
                    before = file_ - char_pos
                    after = cnt - before - 1
                    if before > 0:
                        new_row += str(before)
                    if piece_char:
                        new_row += piece_char
                    else:
                        new_row += '1'
                    if after > 0:
                        new_row += str(after)
                    replaced = True
                    char_pos += cnt
            elif char_pos == file_:
                new_row += piece_char or ''
                replaced = True
                char_pos += 1
            else:
                new_row += ch
                char_pos += 1

        if not replaced and char_pos == file_:
            new_row += piece_char or ''
            replaced = True

        rows[7 - rank] = new_row
        parts[0] = '/'.join(rows)
        self._b = _FastBoard(' '.join(parts))

    def ep_square(self):
        return self._b.ep_square()

    def __str__(self):
        return str(self._b)

    def __repr__(self):
        return f"Board('{self.fen()}')"

    def __eq__(self, other):
        if isinstance(other, Board):
            return self.fen() == other.fen()
        return NotImplemented

    def __hash__(self):
        return hash(self.fen())


# ── Convenience re-exports ─────────────────────────────────────────────────
if _has_fast:
    # Add fastchess helper functions at module level
    _square_rank = _fc.square_rank
    _square_file = _fc.square_file
    _square = _fc.square

# ---- Fallback switch -------------------------------------------------------
USE_FAST_BACKEND = _has_fast

__all__ = [
    'Board', 'Move', 'Piece',
    'WHITE', 'BLACK',
    'PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING',
    'SQUARES',
    'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1',
    'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2',
    'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3',
    'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4',
    'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5',
    'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6',
    'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7',
    'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8',
    'square_rank', 'square_file', 'square', 'square_name', 'parse_square',
]
