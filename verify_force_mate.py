import warnings
warnings.filterwarnings('ignore')
import chess

_memo = {}
def forced_mate(board, plies):
    if board.is_game_over() or plies <= 0:
        return None
    key = (board.fen(), plies)
    if key in _memo:
        return _memo[key]
    if plies == 1:
        for m in board.legal_moves:
            nb = board.copy(stack=False); nb.push(m)
            if nb.is_checkmate():
                _memo[key] = m
                return m
        _memo[key] = None
        return None
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            _memo[key] = m
            return m
        if nb.is_game_over():
            continue
        ok = True
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r)
            if rb.is_game_over():
                ok = False; break
            if forced_mate(rb, plies - 2) is None:
                ok = False; break
        if ok:
            _memo[key] = m
            return m
    _memo[key] = None
    return None

def verify_line(fen, plies):
    """Play the solver's forced line against all replies; confirm mate within plies."""
    b = chess.Board(fen)
    m = forced_mate(b, plies)
    if m is None:
        return True, "no forced mate claimed"
    b.push(m)
    plies_left = plies - 1
    # For every opponent reply, check solver can force mate within remaining plies
    while not b.is_game_over():
        if b.turn == chess.WHITE:
            fm = forced_mate(b, plies_left)
            if fm is None:
                return False, f"FAIL: white to move after {b.fen()}, solver found nothing (plies_left={plies_left})"
            b.push(fm)
            plies_left -= 1
        else:
            # adversary picks reply that maximises survival (any legal move)
            worst = None
            for r in b.legal_moves:
                nb = b.copy(stack=False); nb.push(r)
                fm = forced_mate(nb, plies_left) if not nb.is_game_over() else None
                if fm is None and not nb.is_game_over():
                    # adversary found a reply not forced-mated within remaining plies
                    worst = r
                    break
                if nb.is_game_over() and not nb.is_checkmate():
                    worst = r; break
            if worst is not None:
                return False, f"FAIL: adversary escaped via {worst} after {b.fen()}"
            # all replies forced; pick the first to continue the walk
            r0 = next(iter(b.legal_moves))
            b.push(r0)
            plies_left -= 1
    return b.is_checkmate(), f"game ended: {b.result()} (checkmate={b.is_checkmate()})"

CASES = [
    ("8/8/8/8/2Q5/8/4K3/7k w - - 0 1", 3, "KQ far (claimed mate-in-2)"),
    ("7k/8/8/5K2/8/6Q1/8/8 w - - 0 1", 3, "KQ near (mate-in-2)"),
    ("8/8/8/8/8/3R4/3K4/7k w - - 0 1", 5, "KR (mate-in-3?)"),
    ("7k/8/8/8/8/8/5R2/K5R1 w - - 0 1", 1, "KRR (mate-in-1)"),
    ("8/8/4k3/8/8/3Q4/8/4K3 w - - 0 1", 5, "KQ vs K near (mate-in-3?)"),
    ("8/8/8/8/3k4/8/8/KQ6 w - - 0 1", 5, "KQ vs K corner (mate-in-3?)"),
]
print(f"{'case':<44}{'solver':<10}  verification")
for fen, plies, tag in CASES:
    _memo.clear()
    m = forced_mate(chess.Board(fen), plies)
    ok, msg = verify_line(fen, plies)
    print(f"{tag:<44}{str(m):<10}  {ok}  {msg}")

# Cross-check against naive minimax (no memo) on a couple positions
def brute_can_force(board, plies):
    """white-to-move-or-any: can the side to move force mate within plies plies?"""
    if board.is_game_over() or plies <= 0:
        return False
    for m in board.legal_moves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            return True
        if nb.is_game_over():
            continue
        replies = []
        for r in nb.legal_moves:
            rb = nb.copy(stack=False); rb.push(r)
            replies.append(rb)
        if all(brute_can_force(rb, plies - 1) for rb in replies):
            return True
    return False

print("\nnaive cross-check (mate in 2 / 3):")
for fen, plies in [("8/8/8/8/2Q5/8/4K3/7k w - - 0 1", 3),
                   ("8/8/4k3/8/8/3Q4/8/4K3 w - - 0 1", 5)]:
    _memo.clear()
    sol = forced_mate(chess.Board(fen), plies)
    naive = brute_can_force(chess.Board(fen), plies)
    print(f"  {fen}  solver={sol}  naive_forced_within_{plies}pl={naive}  "
          f"{'MATCH' if (sol is not None) == naive else 'MISMATCH'}")
