import warnings
warnings.filterwarnings('ignore')
import chess

WIN = [
    "7k/8/8/8/8/8/8/KQ6 w - - 0 1",
    "8/8/8/3k4/8/8/8/KQ6 w - - 0 1",
    "8/8/4k3/8/8/3Q4/8/4K3 w - - 0 1",
    "5k2/8/8/8/8/4Q3/8/4K3 w - - 0 1",
    "7k/8/8/8/8/6Q1/8/7K w - - 0 1",
    "7k/8/8/8/8/8/8/R6K w - - 0 1",
    "8/8/4k3/8/8/3R4/8/4K3 w - - 0 1",
    "6k1/8/8/8/8/2R5/8/4K3 w - - 0 1",
    "8/8/8/8/3k4/8/8/1RK5 w - - 0 1",
    "6k1/8/8/8/8/4R3/5R2/K7 w - - 0 1",
    "7k/8/8/8/8/8/5R2/K5R1 w - - 0 1",
    "8/8/8/2k5/8/8/3RR3/3K4 w - - 0 1",
    "6k1/8/8/8/8/8/1B2B3/7K w - - 0 1",
    "8/8/8/3k4/8/8/2B1B3/2K5 w - - 0 1",
    "7k/8/8/8/8/8/5N2/5BK1 w - - 0 1",
    "8/8/8/3k4/8/8/2N1B3/2K5 w - - 0 1",
    "8/8/8/2k5/8/8/3Q2B1/3K4 w - - 0 1",
    "8/8/8/2k5/8/8/2NQ4/2K5 w - - 0 1",
    "8/8/8/2k5/8/8/3R1N2/3K4 w - - 0 1",
    "8/8/8/8/8/4P3/4K3/7k w - - 0 1",
    "8/8/8/8/8/8/2KP4/7k w - - 0 1",
    "8/8/8/8/8/8/2PKP3/k7 w - - 0 1",
]

K = chess.KING; P = chess.PAWN; N = chess.KNIGHT
B = chess.BISHOP; R = chess.ROOK; Q = chess.QUEEN

print(f"{'fen':<42} classification")
for fen in WIN:
    b = chess.Board(fen)
    pm = b.piece_map()
    wt = sorted(p.piece_type for p in pm.values() if p.color == chess.WHITE)
    bt = sorted(p.piece_type for p in pm.values() if p.color == chess.BLACK)
    name = None
    if bt == [K] and wt == [Q, K]:
        name = "KQ vs K (forced win)"
    elif bt == [K] and wt == [R, K]:
        name = "KR vs K (forced win)"
    elif bt == [K] and wt == [R, R, K]:
        name = "KRR vs K (forced win)"
    elif bt == [K] and wt == [B, B, K]:
        name = "KBB vs K (forced win)"
    elif bt == [K] and wt == [N, B, K]:
        name = "KBN vs K (forced win, ~20 moves)"
    elif bt == [N, K] and wt == [Q, K]:
        name = "KQ vs KN (forced win)"
    elif bt == [B, K] and wt == [Q, K]:
        name = "KQ vs KB (forced win)"
    elif bt == [N, K] and wt == [R, K]:
        name = "KR vs KN (forced win)"
    elif bt == [K] and wt == [P, K]:
        pw = [s for s, p in pm.items() if p.piece_type == P and p.color == chess.WHITE][0]
        bk = [s for s, p in pm.items() if p.piece_type == K and p.color == chess.BLACK][0]
        pr, pf = divmod(pw, 8); kr, kf = divmod(bk, 8); n = 7 - pr
        inside = (kr >= pr and pf - n <= kf <= pf + n)
        name = "KP vs K (rule-of-square: %s)" % ("DRAW" if inside else "WIN")
    elif bt == [K] and wt == [P, P, K]:
        name = "KPP vs K (forced win)"
    else:
        name = "UNKNOWN %s vs %s" % (wt, bt)
    print("%-42s %s" % (fen.split()[0], name))
