import cProfile, pstats, time
import chess
from mcts import MCTS, MCTSNode
from network import AlphaZeroNet
from config import Config  # if you have a config module; otherwise replace with values

# Build network exactly as your self-play does
net = AlphaZeroNet(
    num_residual_blocks=4,    # change to your actual config
    num_filters=64,
    num_policy_channels=64,
    num_value_channels=64,
    value_fc_size=128,
)
net.eval()

# Build MCTS with the same parameters you normally use
mcts = MCTS(
    network=net,
    num_simulations=800,
    c_puct=1.5,
    dirichlet_alpha=0.25,
    dirichlet_epsilon=0.25,
    batch_size=64,            # match your inference server's max_batch
    c_virtual_loss=0.5,
    max_game_length=200,
    adjudicate_material=True,
    piece_values={'P':1,'N':3,'B':3,'R':5,'Q':9},
    adjudicate_graded=True,
    adjudicate_scaling=9.0,
)

board = chess.Board()
root = mcts.get_root(board)

prof = cProfile.Profile()
prof.enable()
t0 = time.perf_counter()
visit_policy, best_move, stats = mcts.search(root)
t1 = time.perf_counter()
prof.disable()

print(f"\nSearch took {t1-t0:.3f}s ({stats.get('num_simulations',800)/(t1-t0):.0f} sims/s)")
print(f"Avg depth: {stats.get('avg_depth',0):.1f}, Max depth: {stats.get('max_depth',0)}")

# Save raw profiling data
prof.dump_stats("mcts_profile.prof")

# Print top 20 functions by cumulative time
p = pstats.Stats(prof)
print("\nTop 20 functions by cumulative time:")
p.sort_stats('cumulative').print_stats(20)

# Also print by internal time (excluding calls to other functions)
print("\nTop 10 functions by internal time:")
p.sort_stats('tottime').print_stats(10)