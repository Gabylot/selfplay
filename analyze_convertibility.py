import warnings, json, collections, sys
warnings.filterwarnings('ignore')
import chess
from measure_maxlen import forced_mate, _memo

PATH = sys.argv[1] if len(sys.argv) > 1 else "output/119/measure_convertibility.jsonl"
GATE_PIECES = 8
MAX_PLIES = 5

games = []
for line in open(PATH):
    try:
        games.append(json.loads(line))
    except Exception:
        pass

def scan(g):
    """Return (convertible, info) for a game: does its trajectory pass through
    a <=8-piece position where the side to move has a forced mate in <=3 moves?"""
    start = chess.Board()  # standard start (32 pieces, never fires)
    fens = [start.fen()] + g["fens"]
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        if len(b.piece_map()) > GATE_PIECES:
            continue
        if b.is_game_over():
            continue
        fm = forced_mate(b, MAX_PLIES)
        if fm is not None:
            return True, {"ply": i, "move": fm.uci(), "plies": None,
                          "fen": fen, "pieces": len(b.piece_map())}
    return False, None

results = []
for g in games:
    conv, info = scan(g)
    results.append((g, conv, info))

stalls = [(g, c, i) for g, c, i in results if g["termination"] != "checkmate"]
checks = [(g, c, i) for g, c, i in results if g["termination"] == "checkmate"]

n = len(games)
conv_stall = sum(1 for _, c, _ in stalls if c)
unconv_stall = len(stalls) - conv_stall

print(f"games: {n}  (checkmate {len(checks)}, non-checkmate {len(stalls)})")
print(f"\nstall rate (non-checkmate): {100*len(stalls)/n:.1f}%")
print(f"  of which override would convert: {conv_stall}/{len(stalls)} "
      f"({100*conv_stall/len(stalls):.0f}%)")
print(f"  remaining after override: {unconv_stall}/{n} "
      f"({100*unconv_stall/n:.1f}% non-checkmate)")
print(f"predicted checkmate rate with override: "
      f"{100*(len(checks)+conv_stall)/n:.1f}%")

print("\nconvertible stalls (would-be conversions):")
for g, c, i in sorted(stalls, key=lambda x: x[1], reverse=True):
    if c:
        print(f"  seed={g['seed']:3d} {g['termination']:20s} len={g['length']:3d} "
              f"-> forced mate at ply {i['ply']} move {i['move']} "
              f"({i['pieces']} pieces)")
print("\nNOT convertible stalls:")
for g, c, i in stalls:
    if not c:
        print(f"  seed={g['seed']:3d} {g['termination']:20s} len={g['length']:3d}")

# sanity: checkmate games that ALSO had a forced-mate opportunity (redundant)
extra = sum(1 for _, c, _ in checks if c)
print(f"\n(checkmate games that also passed through a forced-mate-<=3 position: {extra})")
