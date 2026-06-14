import chess
import sys
from encoding import move_to_policy_index, policy_index_to_move

b = chess.Board('3k4/8/8/8/8/8/1p6/4K3 b - - 0 1')
print('turn:', b.turn)
for m in list(b.legal_moves):
    print('move:', m, 'promo:', m.promotion)
    if m.promotion:
        i = move_to_policy_index(m, b)
        d = policy_index_to_move(i, b)
        print('  idx:', i, 'decoded:', d, 'match:', d == m)