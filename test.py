import numpy as np
from mcts import MCTSNode, MCTS

# Fake root with 30 children, mimicking your screenshot's prior shape
root = MCTSNode(board=None)  # board unused for this test
priors = [0.743] + [0.257/29] * 29
for i, p in enumerate(priors):
    child = MCTSNode(board=None, parent=root, prior=p)
    root.children[i] = child

print("Before noise:", [round(c.P, 4) for c in root.children.values()])

# Use a dummy MCTS instance just for the noise method
mcts = MCTS(network=None, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
mcts._add_dirichlet_noise(root)

print("After noise: ", [round(c.P, 4) for c in root.children.values()])