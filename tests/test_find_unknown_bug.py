"""Run this file. It will either:
  - print "ALL GOOD" if nothing breaks, or
  - print "FOUND IT" with details about the exact position and state
    where best_move becomes None despite children existing.

Just run: python tests/test_find_unknown_bug.py
and send back everything it prints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import chess

from encoding import board_to_tensor, NUM_ACTIONS, policy_index_to_move
from mcts import MCTS, MCTSNode


class MockNetwork:
    """Returns random policy + value, like an untrained network."""
    def predict(self, state):
        policy = np.random.dirichlet(np.ones(NUM_ACTIONS)) * 0.5
        value = np.random.uniform(-0.1, 0.1)
        return policy, value

    def predict_batch(self, states):
        n = states.shape[0]
        policies = np.stack([np.random.dirichlet(np.ones(NUM_ACTIONS)) * 0.5 for _ in range(n)])
        values = np.random.uniform(-0.1, 0.1, size=n)
        return policies, values


def check_one_search(mcts, board, game_num, move_num):
    """Run one MCTS search and check for the bug. Returns True if bug found."""
    root = mcts.get_root(board)
    visit_policy, best_move, stats = mcts.search(root)

    if not root.children:
        # No children at all — different issue, but let's note it
        return False

    total_visits = sum(c.N for c in root.children.values())
    if total_visits == 0:
        return False

    best_idx = int(np.argmax(visit_policy))

    if best_move is None:
        print("\n" + "=" * 60)
        print("FOUND IT")
        print("=" * 60)
        print(f"Game {game_num}, move {move_num}")
        print(f"FEN: {board.fen()}")
        print(f"Number of children: {len(root.children)}")
        print(f"Total visits: {total_visits}")
        print(f"best_idx (argmax of visit_policy): {best_idx}")
        print(f"Is best_idx a key in root.children? {best_idx in root.children}")

        if best_idx in root.children:
            child = root.children[best_idx]
            print(f"Stored child.move (from expansion): {child.move}")
            print(f"Stored child.N: {child.N}")
            decoded = policy_index_to_move(best_idx, board)
            print(f"policy_index_to_move(best_idx, board): {decoded}")
            print(f"Is child.move in board.legal_moves? {child.move in board.legal_moves}")
            if decoded is not None:
                print(f"Is decoded in board.legal_moves? {decoded in board.legal_moves}")
        else:
            print("best_idx is NOT a key in root.children at all.")
            print(f"All children action_idx values: {sorted(root.children.keys())}")
            print(f"visit_policy[best_idx] = {visit_policy[best_idx]}")
            # Show what visit_policy looks like at the actual children's indices
            for idx, child in root.children.items():
                print(f"  child idx={idx}, N={child.N}, visit_policy[idx]={visit_policy[idx]}")

        print("=" * 60)
        return True

    return False


if __name__ == "__main__":
    np.random.seed(0)

    mock_net = MockNetwork()

    # Test with a few different simulation counts and batch sizes,
    # since the real bug may depend on these settings.
    configs = [
        dict(num_simulations=200, c_puct=3.0, dirichlet_alpha=0.3,
             dirichlet_epsilon=0.4, batch_size=8),
        dict(num_simulations=10, c_puct=1.5, dirichlet_alpha=0.3,
             dirichlet_epsilon=0.25, batch_size=1),
    ]

    found_bug = False

    for cfg_idx, cfg in enumerate(configs):
        print(f"\n--- Config {cfg_idx}: {cfg} ---")
        mcts = MCTS(mock_net, **cfg)

        # Play several games, checking every position along the way
        for game_num in range(5):
            board = chess.Board()
            move_num = 0

            while not board.is_game_over() and move_num < 80:
                if check_one_search(mcts, board, game_num, move_num):
                    found_bug = True
                    break

                # Play a random legal move to advance the game
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = legal[np.random.randint(len(legal))]
                board.push(move)
                move_num += 1

            if found_bug:
                break
        if found_bug:
            break

    if not found_bug:
        print("\nALL GOOD — no instance of best_move=None with children present, "
              "across all configs and games tested.")
        print("If the real bug still happens during actual self-play, it likely")
        print("depends on something this test doesn't reproduce yet — e.g. the")
        print("real (trained) network's policy output shape/values, or a specific")
        print("position that random play didn't reach. Let me know and we'll dig")
        print("into selfplay.py's game loop directly instead.")