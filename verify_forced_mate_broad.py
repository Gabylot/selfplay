import warnings, random, chess, sys, time, argparse
warnings.filterwarnings('ignore')

_brute_memo = {}
def brute_can_force(board, plies):
    """True if side to move can force checkmate within `plies` plies (minimax).

    Memoized on (fen, plies) — memoization does not change the predicate,
    it only collapses the exponential search tree's shared subtrees.
    """
    if board.is_game_over() or plies <= 0:
        return False
    key = (board.fen(), plies)
    if key in _brute_memo:
        return _brute_memo[key]
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            _brute_memo[key] = True; return True
        if nb.is_game_over():
            continue
        ok = True
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r)
            if rb.is_game_over():
                # Any game-over reply (stalemate => mover's move didn't force
                # mate; checkmate on the mover => opponent mated us back)
                # defeats the forced mate.
                ok = False; break
            # Each level consumes TWO plies (our move + opponent reply), so
            # the remaining budget in the child position is plies - 2.  The
            # original script used plies - 1, which made brute_can_force
            # check "mate within 2*plies-1" — an apples-to-oranges reference
            # that would flag false mismatches on mate-in-6/7 positions.
            if not brute_can_force(rb, plies - 2):
                ok = False; break
        if ok:
            _brute_memo[key] = True; return True
    _brute_memo[key] = False; return False

def forced_mate(board, plies, _memo={}):
    if board.is_game_over() or plies <= 0:
        return None
    key = (board.fen(), plies)
    if key in _memo:
        return _memo[key]
    if plies == 1:
        for m in board.legal_moves:
            nb = board.copy(stack=False); nb.push(m)
            if nb.is_checkmate():
                _memo[key] = m; return m
        _memo[key] = None; return None
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            _memo[key] = m; return m
        if nb.is_game_over():
            continue
        ok = True
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r)
            if rb.is_game_over():
                ok = False; break
            if forced_mate(rb, plies - 2, _memo) is None:
                ok = False; break
        if ok:
            _memo[key] = m; return m
    _memo[key] = None; return None

# Piece-set pools for 3-8 piece sparse positions (white upper, black lower).
# Each entry is a string of piece symbols: 'KQk' = white K+Q vs black k, etc.
PIECES_POOL = [
    "KQk", "KRk", "KQkr", "KQRk", "KQNk", "KRPk", "KQbk", "KRRk",
    "KBk", "KNk", "KNNk", "KQkn", "KQPk", "KRNk", "KRBk", "KPPk",
]

def _pieces_to_fen(pcs, squares):
    """Build a FEN board string from piece symbols placed on squares."""
    rows = [['1'] * 8 for _ in range(8)]
    for ch, sq in zip(pcs, squares):
        rank = (sq >> 3) & 7
        file_ = sq & 7
        rows[7 - rank][file_] = ch  # rank 8 = row 0
    out_rows = []
    for r in rows:
        out = ''
        run = 0
        for c in r:
            if c == '1':
                run += 1
            else:
                if run:
                    out += str(run)
                    run = 0
                out += c
        if run:
            out += str(run)
        out_rows.append(out)
    return '/'.join(out_rows)

def build_fens(n, seed, legal_min=0, legal_max=60):
    """Directly construct legal sparse FENs by random piece placement.

    Much faster than the original random-play approach (which essentially
    never reaches <=8 pieces).  The custom chess.py shim validates the
    position when Board(fen) is constructed, so we build the full FEN
    string first and hand it over in one call (set_piece_at rejects
    intermediate states and there is no is_valid() in the shim).
    Side-not-to-move-in-check positions and king-adjacency are rejected
    by the backend.
    """
    rng = random.Random(seed + 1)  # +1 distinguishes from verification seed usage
    fens = []
    attempts = 0
    while len(fens) < n and attempts < n * 2000:
        attempts += 1
        pcs = rng.choice(PIECES_POOL)
        squares = rng.sample(range(64), len(pcs))
        # Reject pawns on rank 1/8 (invalid everywhere)
        if any(ch in 'Pp' and (sq >> 3 == 0 or sq >> 3 == 7)
               for ch, sq in zip(pcs, squares)):
            continue
        turn_char = rng.choice(['w', 'b'])
        fen_board = _pieces_to_fen(pcs, squares)
        fen = f"{fen_board} {turn_char} - - 0 1"
        try:
            b = chess.Board(fen)
            # Backend raises on illegal FEN (missing king, adjacent kings,
            # side-not-to-move in check, pawns on back rank, ...)
            if b.is_game_over():
                continue
            n_legal = len(list(b.legal_moves))
            if n_legal == 0:
                continue
            if legal_max is not None and n_legal > legal_max:
                continue
            if legal_min is not None and n_legal < legal_min:
                continue
        except Exception:
            continue
        if fen not in fens:
            fens.append(fen)
    return fens

def main():
    ap = argparse.ArgumentParser(description="Verify forced_mate solver vs naive brute minimax")
    ap.add_argument('--count', type=int, default=50, help="Number of random sparse FENs to check")
    ap.add_argument('--timeout', type=float, default=180.0, help="Wall-clock timeout in seconds")
    ap.add_argument('--seed', type=int, default=1234, help="Random seed for FEN generation")
    ap.add_argument('--legal_max', type=int, default=60, help="Max legal moves per position (skip chaotic ones)")
    args = ap.parse_args()

    print(f"generating {args.count} sparse FENs...", flush=True)
    t0 = time.time()
    fens = build_fens(args.count, args.seed, legal_max=args.legal_max)
    print(f"  generated {len(fens)} FENs in {time.time()-t0:.1f}s", flush=True)
    if len(fens) < args.count:
        print(f"[WARN] only generated {len(fens)}/{args.count} — continuing with what we have",
              file=sys.stderr)

    mismatch = 0
    checked = 0
    t_start = time.time()
    for i, fen in enumerate(fens):
        if time.time() - t_start > args.timeout:
            print(f"[WARN] timeout after {checked}/{len(fens)} positions "
                  f"({args.timeout:.0f}s)", file=sys.stderr)
            break
        b = chess.Board(fen)
        bm_ = brute_can_force(b, 5)
        fm_ = forced_mate(b.copy(), 5) is not None
        checked += 1
        if bm_ != fm_:
            mismatch += 1
            print(f"MISMATCH {fen}  brute={bm_} solver={fm_}")
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(fens)} checked, {mismatch} mismatches "
                  f"({time.time()-t_start:.0f}s)", flush=True)
    print(f"\nchecked {checked} random sparse positions, forced_mate(<=5) vs brute minimax")
    print(f"mismatches: {mismatch}   -> {'FORCED_MATE OK' if mismatch == 0 else 'SOLVER BUG'}")

if __name__ == '__main__':
    main()