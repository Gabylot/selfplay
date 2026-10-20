"""Colour-frame consistency tests (frame version 2).

The whole chess pipeline must be colour-neutral: a Black-to-move position
must be indistinguishable from its colour-mirrored White-to-move twin, in
both the input tensor AND the policy labels.  With sigma = rank-flip +
colour-swap:

    encode(sigma P) == encode(P)
    index(sigma m in sigma P) == index(m in P)
    mask(sigma P) == mask(P)

These tests fail LOUDLY if the Rust extension and the Python fallback ever
disagree (either on the encoder or on move indexing) or if any part of the
pipeline stops commuting with sigma.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

import chess
import encoding
from encoding import (
    board_to_tensor, move_to_policy_index, policy_index_to_move,
    get_legal_move_mask, NUM_PLANES, FRAME_VERSION, FrameVersionError,
    NO_PROGRESS_NORM, MOVE_COUNT_NORM,
)


# ---------------------------------------------------------------------------
# sigma = rank-flip + colour-swap, expressed as a FEN transform
# ---------------------------------------------------------------------------

def mirror_fen(fen):
    board_f, turn, castling, ep, half, full = fen.split()
    rows = board_f.split("/")[::-1]                       # rank flip
    rows = ["".join(c.swapcase() if c.isalpha() else c for c in row)
            for row in rows]                              # colour swap
    # rank flip preserves files -> kingside stays kingside; swap colours and
    # re-order to the canonical KQkq order.
    swapped = "".join(c.swapcase() for c in castling)
    castling2 = "".join(c for c in "KQkq" if c in swapped) or "-"
    ep2 = "-" if ep == "-" else ep[0] + str(9 - int(ep[1]))
    turn2 = "b" if turn == "w" else "w"
    return f"{'/'.join(rows)} {turn2} {castling2} {ep2} {half} {full}"


def sigma_square(sq):
    return sq ^ 56


def sigma_move(m):
    return chess.Move(sigma_square(m.from_square), sigma_square(m.to_square),
                      m.promotion)


def ref_python_encode(board):
    """Independent pure-Python encoder used to cross-check the Rust path."""
    t = np.zeros((NUM_PLANES, 8, 8), np.float32)
    turn = board.turn
    flip = (turn == chess.BLACK)
    pieces, colors = board.piece_arrays()
    for sq in range(64):
        pt = int(pieces[sq])
        if pt == 0:
            continue
        col = bool(colors[sq])
        s = sigma_square(sq) if flip else sq
        row, c = s >> 3, s & 7
        plane = (pt - 1) if (col == turn) else (6 + pt - 1)
        t[plane, row, c] = 1.0
    t[112] = 1.0
    p1, p2 = turn, not turn
    t[113] = 1.0 if board.has_kingside_castling_rights(p1) else 0.0
    t[114] = 1.0 if board.has_queenside_castling_rights(p1) else 0.0
    t[115] = 1.0 if board.has_kingside_castling_rights(p2) else 0.0
    t[116] = 1.0 if board.has_queenside_castling_rights(p2) else 0.0
    t[117] = min(board.halfmove_clock / NO_PROGRESS_NORM, 1.0)
    t[118] = board.fullmove_number / MOVE_COUNT_NORM
    return t


# Curated positions: promotions + underpromotions, en passant, castling rights.
CURATED_FENS = [
    "1k4r1/4P3/8/8/8/8/8/4K3 w - - 0 1",             # white pawn promos
    "4k3/8/8/8/8/8/4p3/1K4R1 b - - 0 1",             # black pawn promos
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # ep
    "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2",   # ep next
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",           # castling
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",                         # castling b
]


def random_fens(n_games=6, plies=60, seed=1234):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_games):
        b = chess.Board()
        for _ply in range(plies):
            if b.is_game_over():
                break
            if _ply % 3 == 0:
                out.append(b.fen())
            mv = b.legal_moves
            b.push(mv[int(rng.integers(len(mv)))])
    return out


def all_fens():
    return CURATED_FENS + random_fens()


# ---------------------------------------------------------------------------
# 1. Encoder is colour-neutral: encode(sigma P) == encode(P)
# ---------------------------------------------------------------------------

def test_encode_is_colour_symmetric():
    checked = 0
    for fen in all_fens():
        P = chess.Board(fen)
        try:
            S = chess.Board(mirror_fen(fen))
        except Exception:
            continue
        assert np.array_equal(board_to_tensor(P), board_to_tensor(S)), \
            f"encode(sigma P) != encode(P) for {fen}"
        checked += 1
    assert checked >= 20, f"only {checked} positions checked"


def test_rust_and_python_encoder_agree():
    checked = 0
    for fen in all_fens():
        P = chess.Board(fen)
        rust = board_to_tensor(P)          # Rust fast path
        ref = ref_python_encode(P)         # independent Python reference
        assert np.array_equal(rust, ref), \
            f"Rust vs Python encoder mismatch for {fen}: planes " \
            f"{sorted(set(np.argwhere(rust != ref)[:, 0].tolist()))}"
        checked += 1
    assert checked >= 20


# ---------------------------------------------------------------------------
# 2. Policy indexing commutes with sigma; masks are identical
# ---------------------------------------------------------------------------

def test_policy_index_commutes_with_sigma():
    checked = 0
    for fen in all_fens():
        P = chess.Board(fen)
        try:
            S = chess.Board(mirror_fen(fen))
        except Exception:
            continue
        for m in P.legal_moves:
            pi = move_to_policy_index(m, P)
            si = move_to_policy_index(sigma_move(m), S)
            assert pi == si, \
                f"index mismatch {m.uci()} in {fen}: {pi} vs {si}"
        checked += 1
    assert checked >= 20


def test_legal_mask_is_colour_symmetric():
    checked = 0
    for fen in all_fens():
        P = chess.Board(fen)
        try:
            S = chess.Board(mirror_fen(fen))
        except Exception:
            continue
        assert np.array_equal(get_legal_move_mask(P), get_legal_move_mask(S)), \
            f"legal-move mask differs under sigma for {fen}"
        checked += 1
    assert checked >= 20


def test_legal_move_sets_are_mirrors():
    # sigma must be an isomorphism of the local game tree.
    checked = 0
    for fen in all_fens():
        P = chess.Board(fen)
        try:
            S = chess.Board(mirror_fen(fen))
        except Exception:
            continue
        pm = {(m.from_square, m.to_square, m.promotion) for m in P.legal_moves}
        sm = {(sigma_square(m.from_square), sigma_square(m.to_square),
               m.promotion) for m in S.legal_moves}
        assert pm == sm, f"legal move sets are not mirrors for {fen}"
        checked += 1
    assert checked >= 20


# ---------------------------------------------------------------------------
# 3. index -> move -> index round-trips, both colours (incl. special moves)
# ---------------------------------------------------------------------------

def test_index_roundtrip_both_colours():
    """Every legal move's index decodes back to the same move, and the Rust
    legal-index vector agrees with the Python indexer 1:1."""
    n_moves = 0
    for fen in all_fens():
        P = chess.Board(fen)
        legal = list(P.legal_moves)
        rust_side = P._b.legal_move_policy_indices()
        assert len(rust_side) == len(legal), \
            f"Rust legal-index vector length mismatch for {fen}"
        for m, ridx in zip(legal, rust_side):
            pi = move_to_policy_index(m, P)
            assert ridx is not None and int(ridx) == pi, \
                f"Rust vs Python indexer mismatch {m.uci()} in {fen}: " \
                f"{ridx} vs {pi}"
            back = policy_index_to_move(pi, P)
            assert back is not None and \
                (back.from_square, back.to_square, back.promotion) == \
                (m.from_square, m.to_square, m.promotion), \
                f"index->move round-trip failed {m.uci()} (idx {pi}) in {fen}"
            n_moves += 1
    assert n_moves > 300


def test_promotion_and_underpromotion_roundtrip():
    """Explicit coverage of queen + knight/bishop/rook promotions both sides."""
    for fen in ("1k4r1/4P3/8/8/8/8/8/4K3 w - - 0 1",
                "4k3/8/8/8/8/8/4p3/1K4R1 b - - 0 1"):
        P = chess.Board(fen)
        promos = [m for m in P.legal_moves if m.promotion is not None]
        assert len(promos) >= 4, f"expected promotions in {fen}"
        for m in promos:
            idx = move_to_policy_index(m, P)
            assert policy_index_to_move(idx, P) is not None


# ---------------------------------------------------------------------------
# 4. The old 180-degree rotation must NOT satisfy the invariant
# ---------------------------------------------------------------------------

def test_legacy_rotation_would_break_the_invariant():
    """Canary: the old '63 - sq' rotation maps squares but not consistently
    with a file-preserving mirror, so it must fail the encode invariant.  This
    guards against accidentally reintroducing the old frame."""
    fen = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2"
    P = chess.Board(fen)
    S = chess.Board(mirror_fen(fen))

    def rotate_encode(board):
        t = np.zeros((NUM_PLANES, 8, 8), np.float32)
        turn = board.turn
        rot = (turn == chess.BLACK)
        pieces, colors = board.piece_arrays()
        for sq in range(64):
            pt = int(pieces[sq])
            if pt == 0:
                continue
            s = (63 - sq) if rot else sq
            plane = (pt - 1) if (bool(colors[sq]) == turn) else (6 + pt - 1)
            t[plane, s >> 3, s & 7] = 1.0
        return t

    assert not np.array_equal(rotate_encode(P), rotate_encode(S)), \
        "the legacy rotation unexpectedly satisfied the mirror invariant"


# ---------------------------------------------------------------------------
# 5. Frame-version marker refuses stale checkpoints / buffers
# ---------------------------------------------------------------------------

def test_frame_version_is_two():
    assert FRAME_VERSION == 2


def test_old_checkpoint_is_refused(tmp_path):
    from network import load_checkpoint
    path = str(tmp_path / "old.pt")
    torch.save({'model_state_dict': {}, 'step': 1, 'game_name': 'chess'}, path)
    with pytest.raises(FrameVersionError):
        load_checkpoint(path)


def test_current_checkpoint_loads(tmp_path):
    from network import save_checkpoint, load_checkpoint, create_model_from_config
    from config import get_config
    net = create_model_from_config(get_config())
    path = str(tmp_path / "new.pt")
    save_checkpoint(net, None, path, step=7)
    ck = load_checkpoint(path)
    assert ck['frame_version'] == FRAME_VERSION


def test_old_buffer_is_refused(tmp_path):
    from selfplay import ReplayBuffer
    path = str(tmp_path / "old_buffer.npz")
    np.savez(path, fmt=np.int32(2), values=np.zeros(3, np.float32))
    with pytest.raises(FrameVersionError):
        ReplayBuffer.load(path)


def test_current_buffer_roundtrips(tmp_path):
    from selfplay import ReplayBuffer, NUM_ACTIONS
    import game_api
    buf = ReplayBuffer(max_size=16)
    state = np.zeros((game_api.num_planes(), 8, 8), np.float32)
    state[0, 1, 4] = 1.0
    pol = np.zeros(NUM_ACTIONS, np.float32)
    pol[10] = 1.0
    mask = np.zeros(NUM_ACTIONS, np.float32)
    mask[10] = 1.0
    buf.add_game([(state, pol, 0.5, mask)])
    path = str(tmp_path / "buffer.npz")
    buf.save(path)
    loaded = ReplayBuffer.load(path)
    assert loaded is not None and len(loaded) == len(buf)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
