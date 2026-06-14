import chess
from encoding import UNDERPROMOTION_DIRS, UNDERPROMOTION_OFFSETS

# Trace the specific index 947 for e2f1r
b = chess.Board('3k4/8/8/8/8/8/4p3/5K2 b - - 0 1')
idx = 947
source_idx = idx // 73
plane = idx % 73
from_rank = source_idx // 8
from_file = source_idx % 8
print(f'source_idx={source_idx}, plane={plane}, from: ({from_rank},{from_file})')

under_idx = plane - 64
piece_idx = under_idx // 3
dir_idx = under_idx % 3
print(f'under_idx={under_idx}, piece_idx={piece_idx}, dir_idx={dir_idx}')

dir_name = UNDERPROMOTION_DIRS[dir_idx]
dr, dc = UNDERPROMOTION_OFFSETS[dir_name]
print(f'raw ({dr},{dc}) for {dir_name}')

if b.turn == chess.BLACK:
    dr2 = -dr
    dc2 = dc
    print(f'After BLACK adjust (current): dr={dr2}, dc={dc2}')
    print(f'to: ({from_rank+dr2},{from_file+dc2})')
    print(f'expected to: (0, 5) for f1')
    
    dr3 = -dr
    dc3 = -dc
    print(f'After BLACK adjust (fixed):   dr={dr3}, dc={dc3}')
    print(f'to: ({from_rank+dr3},{from_file+dc3})')