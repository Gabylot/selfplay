"""Diagnostic: check if the MCTS correctly handles repetition Q values."""
import sys
import numpy as np
import chess
from encoding import board_to_tensor
from network import AlphaZeroNet
from mcts import MCTS

net = AlphaZeroNet(num_residual_blocks=10, num_filters=128,
                   num_policy_channels=128, num_value_channels=128,
                   value_fc_size=256)
net.eval()

# Create position with is_repetition(2) - one more cycle = draw
b = chess.Board()
for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
    b.push(chess.Move.from_uci(uci))
    
print(f"ROOT: rep2={b.is_repetition(2)} rep3={b.is_repetition(3)}")

# Check each child: does a move create 3-fold? What's the value?
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
for c in children[:5]:
    print(f"{c['move']:8s} {c['N']:6d} {c['Q']:+.4f} {c['W']:+.4f}")

# Check deeper: after Nf3, what happens?
print("\nAfter Nf3 (g1f3):")
b2 = b.copy(stack=True)
b2.push(chess.Move.from_uci('g1f3'))
print(f"  rep2={b2.is_repetition(2)} rep3={b2.is_repetition(3)}")
for move in list(b2.legal_moves)[:5]:
    b3 = b2.copy(stack=True)
    b3.push(move)
    print(f"  After {move.uci():6s} rep2={b3.is_repetition(2)} rep3={b3.is_repetition(3)}")
    # And one more level
    for m2 in list(b3.legal_moves)[:3]:
        b4 = b3.copy(stack=True)
        b4.push(m2)
        if b4.is_repetition(3):
            print(f"    -> {m2.uci():6s} -> 3-FOLD!")