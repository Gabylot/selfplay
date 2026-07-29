"""Diagnostic with a trained network (step 200) to compare repetition behavior."""
import sys
import numpy as np
import chess
import torch
from encoding import board_to_tensor
from network import AlphaZeroNet, load_checkpoint
from mcts import MCTS

# Load trained network
device = torch.device('cpu')
net = AlphaZeroNet(num_residual_blocks=10, num_filters=128,
                   num_policy_channels=128, num_value_channels=128,
                   value_fc_size=256)
net.eval()

checkpoint = torch.load(
    r'F:\python\selfplay\output\default\checkpoints\step_1800.pt',
    map_location=device,
    weights_only=False
)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
print(f"Loaded trained network from step {checkpoint.get('step', '?')}")

# Create position with is_repetition(2) - one more cycle = draw
b = chess.Board()
for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
    b.push(chess.Move.from_uci(uci))
    
print(f"\nROOT: rep2={b.is_repetition(2)} rep3={b.is_repetition(3)}")
print(f"FEN: {b.fen()}")

# Check trained network value for current position
t = board_to_tensor(b)
_, root_value = net.predict(t)
print(f"Root value from trained net: {root_value:+.4f}")

# Check each child
print("\nChild analysis (move -> rep3, value):")
for move in list(b.legal_moves):
    b2 = b.copy(stack=True)
    b2.push(move)
    t2 = board_to_tensor(b2)
    _, v2 = net.predict(t2)
    rep3 = b2.is_repetition(3)
    print(f"  {move.uci():6s} rep3={rep3} value={v2:+.4f}")

# Run MCTS
mcts = MCTS(network=net, num_simulations=200, batch_size=1,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
root = mcts.get_root(b)
visit_policy, best_move, stats = mcts.search(root)

print(f"\nMCTS: best={best_move.uci() if best_move else 'None'}, avg_depth={stats['avg_depth']:.1f}")
children = mcts.get_root_child_stats(root)
print(f"{'Move':8s} {'N':6s} {'Q':8s} {'W':8s}")
for c in children:
    print(f"{c['move']:8s} {c['N']:6d} {c['Q']:+.4f} {c['W']:+.4f}")

# Check deeper: after Nf3, what does trained net say?
print("\nAfter Nf3 (g1f3) - check deep path to 3-fold:")
b2 = b.copy(stack=True)
b2.push(chess.Move.from_uci('g1f3'))
print(f"  rep2={b2.is_repetition(2)} rep3={b2.is_repetition(3)} value={net.predict(board_to_tensor(b2))[1]:+.4f}")

# Check value of the 3-fold position itself
b3 = b2.copy(stack=True)
b3.push(chess.Move.from_uci('g8f6'))
print(f"  After Nf6 (g8f6): rep2={b3.is_repetition(2)} rep3={b3.is_repetition(3)} value={net.predict(board_to_tensor(b3))[1]:+.4f}")

b4 = b3.copy(stack=True)
b4.push(chess.Move.from_uci('f3g1'))
print(f"  After Ng1 (f3g1): rep2={b4.is_repetition(2)} rep3={b4.is_repetition(3)} value={net.predict(board_to_tensor(b4))[1]:+.4f}")

# Create a normal opening (no repetition) for comparison
print("\n--- Normal opening (no repetition) ---")
b_normal = chess.Board()
for uci in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6']:
    b_normal.push(chess.Move.from_uci(uci))
t_normal = board_to_tensor(b_normal)
_, v_normal = net.predict(t_normal)
print(f"Position: {b_normal.fen()}")
print(f"rep2={b_normal.is_repetition(2)}, value={v_normal:+.4f}")