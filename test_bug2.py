import chess
from encoding import move_to_policy_index, policy_index_to_move

# Black pawn on e2 capturing to f1 (dr=-1, dc=+1) — the test-reported failing case
b = chess.Board('3k4/8/8/8/8/8/4p3/5K2 b - - 0 1')
print('Board with black pawn e2, white king f1:')
print('turn:', 'white' if b.turn == chess.WHITE else 'black')
for m in list(b.legal_moves):
    if m.promotion:
        i = move_to_policy_index(m, b)
        d = policy_index_to_move(i, b)
        print(f'  {m} idx={i} decoded={d} match={d == m if d else False}')

# Also black pawn on d2 capturing to e1 (dr=-1, dc=+1)
b2 = chess.Board('3k4/8/8/8/8/8/3p4/4K3 b - - 0 1')
print('\nBoard with black pawn d2 forward:')
for m in list(b2.legal_moves):
    if m.promotion:
        i = move_to_policy_index(m, b2)
        d = policy_index_to_move(i, b2)
        print(f'  {m} idx={i} decoded={d} match={d == m if d else False}')

# Black pawn on a2 capturing to b1 (dr=-1, dc=+1)
b3 = chess.Board('3k4/8/8/8/8/8/p7/1K6 b - - 0 1')
print('\nBoard with black pawn a2, white king b1:')
for m in list(b3.legal_moves):
    if m.promotion:
        i = move_to_policy_index(m, b3)
        d = policy_index_to_move(i, b3)
        print(f'  {m} idx={i} decoded={d} match={d == m if d else False}')