import warnings; warnings.filterwarnings('ignore')
import torch, numpy as np, chess
from network import AlphaZeroNet
from encoding import board_to_tensor

def load(path):
    net = AlphaZeroNet(num_residual_blocks=8, num_filters=64, num_policy_channels=64,
                       num_value_channels=1, value_fc_size=128)
    sd = torch.load(path, map_location='cpu'); sd = sd.get('model_state_dict', sd)
    net.load_state_dict(sd); net.eval(); return net

start = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
recip = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1')
ta = board_to_tensor(start)   # (119,8,8)
tb = board_to_tensor(recip)

for path in ['output/119/checkpoints/step_200.pt',
             'output/119/checkpoints/step_800.pt',
             'output/119/checkpoints/step_2000.pt']:
    try:
        net = load(path)
        with torch.no_grad():
            _, va = net.predict(ta); _, vb = net.predict(tb)
        print(f"{path}: v(white-tm)={va:+.4f}  v(recip black-tm)={vb:+.4f}  diff={va-vb:+.4f}")
    except Exception as e:
        print(path, 'ERR', e)