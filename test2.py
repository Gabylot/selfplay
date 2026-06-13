from mcts import MCTSNode, MCTS

mcts = MCTS(network=None, c_puct=1.5)

root = MCTSNode(board=None)
root.N = 0

# Child A: dominant prior (0.743), Child B: typical alternative (0.0086)
child_a = MCTSNode(board=None, parent=root, prior=0.743)
child_b = MCTSNode(board=None, parent=root, prior=0.0086)
root.children = {0: child_a, 1: child_b}

# Simulate several rounds, manually updating N/Q as if simulations ran
for round_num in range(10):
    selected = mcts._select_child(root)
    label = 'A' if selected is child_a else 'B'
    print(f"Round {round_num}: selected {label}, "
          f"A(N={child_a.N},Q={child_a.Q:.3f}) "
          f"B(N={child_b.N},Q={child_b.Q:.3f}) "
          f"root.N={root.N}")
    
    # Simulate: pretend the selected child gets visited, Q stays near 0
    selected.N += 1
    root.N += 1
    # Q update: assume value ~0 (untrained network, neutral eval)
    selected.W += 0.0
    selected.Q = selected.W / selected.N