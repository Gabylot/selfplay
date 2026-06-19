"""Quick validation that RustBoard works with encoding, MCTS, and selfplay."""
import chess
from encoding import board_to_tensor, move_to_policy_index
from mcts import MCTS, MCTSNode, USE_RUST
from network import AlphaZeroNet   # adjust if needed
from selfplay import play_one_game, self_play_game
import numpy as np

# 1. Basic RustBoard
if USE_RUST:
    from chess_rust import RustBoard, RustMove
    board = RustBoard()
    print("FEN start:", board.fen())
    print("Turn (True=white):", board.turn)
    print("Legal moves count:", len(board.legal_moves))
    board.push("e2e4")
    print("After e4:", board.fen())
    board.push("e7e5")
    print("After e5:", board.fen())
    print("Game over:", board.is_game_over())
    print("Halfmove clock:", board.halfmove_clock)
    print("Fullmove number:", board.fullmove_number)
    print("Ply:", board.ply)
    print("Repetition(2):", board.is_repetition(2))
    print("Repetition(3):", board.is_repetition(3))
    print("Result:", board.result())
    print()

# 2. board_to_tensor (fast Rust path)
tensor = board_to_tensor(board)
print("Tensor shape:", tensor.shape)
print("Plane 12 (side to move) sum:", tensor[12].sum())
print("Plane 0 (White Pawns) sum:", tensor[0].sum())
print()

# 3. move_to_policy_index
for move in list(board.legal_moves)[:5]:
    idx = move_to_policy_index(move, board)
    print(f"Move {move.uci} -> index {idx}")
print()

# 4. MCTS node creation (uses Rust board)
root = MCTSNode(board, ply=board.ply)
print("Root board FEN:", root.board.fen())
# Test lazy child materialisation (will be done by MCTS search)
print()

# 5. Run a self-play game with a real network (if available)
# If you have a trained network object, you can run a full game.
# For a minimal check, we'll simulate a self-play game using the Rust board directly.
from selfplay import MCTS as MCTSEngine, adjudicate_by_material

# Create a dummy network (or load your real one)
class DummyNet:
    def predict(self, x):
        # Return uniform policy and 0 value
        return np.ones(4672)/4672, 0.0
    def predict_batch(self, x):
        return np.ones((x.shape[0], 4672))/4672, np.zeros(x.shape[0])

# This dummy will produce random moves; it's just to test the pipeline.
eng = MCTSEngine(
    network=DummyNet(),
    num_simulations=8,
    batch_size=1,
    max_game_length=150,
    adjudicate_material=True,
)

game_data, info = play_one_game(
    eng,
    max_game_length=150,
    adjudicate_material=True,
    temp_threshold=30,
    temp_high=1.0,
    temp_low=0.1,
)
print("Game result:", info['result_str'], "length:", info['length'])
print("Termination:", info['termination'])
print("Number of positions:", len(game_data))