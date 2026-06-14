"""Self-play game generation and replay buffer for AlphaZero chess.

Generates training games using the current network + MCTS.
Stores (board_state, policy_target, value_target) tuples in a FIFO replay buffer.
No data augmentation (valid for chess — rotations don't preserve legality).

Design choice: Self-play always uses the *latest* network, not the gated "best".
Gating/Elo are purely monitoring signals. This prevents stagnation from unlucky
variance in evaluation matches early in training.
"""

import numpy as np
import chess
from collections import deque
from typing import List, Tuple, Optional

from encoding import board_to_tensor, get_legal_move_mask
from network import AlphaZeroNet
from mcts import MCTS


class ReplayBuffer:
    """Fixed-size FIFO replay buffer for self-play positions."""
    
    def __init__(self, max_size: int = 100000):
        """
        Args:
            max_size: Maximum number of positions to store.
                     Older games are evicted when full.
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.total_games = 0
        self.total_positions = 0
    
    def add_game(self, game_data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add a completed game's positions to the buffer.
        
        Args:
            game_data: List of (state, policy, value) tuples.
                      state: (18, 8, 8) numpy array
            policy: (4672,) numpy array (MCTS visit distribution)
                      value: float in [-1, 1] (game outcome from player's perspective)
        """
        for state, policy, value in game_data:
            self.buffer.append((state, policy, value))
            self.total_positions += 1
        self.total_games += 1
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch from the buffer.
        
        Returns:
            states: (batch, 18, 8, 8)
            policies: (batch, 4672)
            values: (batch, 1)
        """
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                   replace=False)
        states = np.array([self.buffer[i][0] for i in indices])
        policies = np.array([self.buffer[i][1] for i in indices])
        values = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)
    
    def get_outcome_distribution(self) -> dict:
        """Get the distribution of outcomes currently in the buffer."""
        if len(self.buffer) == 0:
            return {'white_wins': 0, 'black_wins': 0, 'draws': 0}
        
        white_wins = 0
        black_wins = 0
        draws = 0
        
        # Sample to estimate distribution (full scan is expensive for large buffers)
        sample_size = min(1000, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=sample_size, replace=False)
        
        for i in indices:
            v = self.buffer[i][2]
            if v > 0.5:
                white_wins += 1
            elif v < -0.5:
                black_wins += 1
            else:
                draws += 1
        
        scale = len(self.buffer) / sample_size
        return {
            'white_wins': int(white_wins * scale),
            'black_wins': int(black_wins * scale),
            'draws': int(draws * scale),
        }


def get_temperature(move_number: int, threshold: int = 30,
                     temp_high: float = 1.0, temp_low: float = 0.1) -> float:
    """Get temperature for move selection based on move number.
    
    Args:
        move_number: Current move number (0-indexed half-move count)
        threshold: Move number to switch from high to low temperature
        temp_high: Temperature for opening moves
        temp_low: Temperature for later moves
    
    Returns:
        Temperature value
    """
    if move_number < threshold:
        return temp_high
    else:
        return temp_low


def adjudicate_by_material(board: chess.Board, piece_values: dict) -> Optional[float]:
    """Adjudicate a game based on material count when max length is hit.
    
    Args:
        board: Current board state
        piece_values: Dict mapping piece character to value
    
    Returns:
        1.0 if white has more material, -1.0 if black has more, 0.0 if equal.
    """
    white_material = 0
    black_material = 0
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        
        val = piece_values.get(piece.symbol().upper(), 0)
        if piece.color == chess.WHITE:
            white_material += val
        else:
            black_material += val
    
    if white_material > black_material:
        return 1.0
    elif black_material > white_material:
        return -1.0
    else:
        return 0.0


def play_one_game(mcts_engine: MCTS,
                   max_game_length: int = 150,
                   adjudicate_material: bool = True,
                   piece_values: dict = None,
                   temp_threshold: int = 30,
                   temp_high: float = 1.0,
                   temp_low: float = 0.1,
                   verbose: bool = False,
                   on_move=None) -> Tuple[List[Tuple], dict]:
    """Play a single self-play game using MCTS.
    
    Args:
        mcts_engine: MCTS instance with the current network
        max_game_length: Maximum half-moves before adjudication
        adjudicate_material: Whether to adjudicate by material at max length
        piece_values: Piece values for material adjudication
        temp_threshold: Move number to switch temperature
        temp_high: High temperature
        temp_low: Low temperature
        verbose: Print progress
        on_move: Optional callback called after each move with
                 (board_fen: str, move_uci: str, move_number: int)
    
    Returns:
        game_data: List of (state, policy, value) tuples for each position
        game_info: Dict with game metadata (result, length, termination, stats)
    """
    if piece_values is None:
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    
    board = chess.Board()
    game_states = []  # (state_tensor, policy, current_player)
    mcts_stats_list = []
    move_count = 0
    termination = "unknown"
    outcome = 0.0  # Default to draw if something goes wrong
    
    while not board.is_game_over() and move_count < max_game_length:
        # Run MCTS
        root = mcts_engine.get_root(board)
        visit_policy, best_move, stats = mcts_engine.search(root)
        
        # Capture all root child stats (all candidates with visits)
        move_candidates = mcts_engine.get_root_child_stats(root)
        
        # Select move with temperature
        temperature = get_temperature(move_count, temp_threshold, temp_high, temp_low)
        visit_policy, selected_move = mcts_engine.select_move_with_temperature(root, temperature)
        
        if selected_move is None:
            # Fallback — shouldn't happen
            selected_move = best_move
            if selected_move is None:
                break
        
        # Store the position, policy, and which player is to move
        state_tensor = board_to_tensor(board)
        current_player = 1.0 if board.turn == chess.WHITE else -1.0
        game_states.append((state_tensor, visit_policy.copy(), current_player))
        mcts_stats_list.append(stats)
        
        # Store per-move MCTS candidate stats
        mcts_move_data = {
            'selected_move': selected_move.uci(),
            'candidates': move_candidates,
        }
        
        # Make the move
        board.push(selected_move)
        move_count += 1
        
        # Notify caller of the new position
        if on_move is not None:
            on_move(board.fen(), selected_move.uci(), move_count, mcts_move_data)
        
        if verbose and move_count % 10 == 0:
            print(f"  Move {move_count}: {selected_move} (visits={int(visit_policy.max() * sum(1 for c in root.children.values() for _ in range(c.N)))}...)")
    
    # Determine game result
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            outcome = 1.0
            termination = "checkmate" if board.is_checkmate() else "other"
        elif result == "0-1":
            outcome = -1.0
            termination = "checkmate" if board.is_checkmate() else "other"
        else:
            outcome = 0.0
            if board.is_repetition():
                termination = "repetition"
            elif board.is_fifty_moves():
                termination = "fifty_moves"
            elif board.is_insufficient_material():
                termination = "insufficient_material"
            else:
                termination = "stalemate"
    elif move_count >= max_game_length:
        termination = "max_length"
        if adjudicate_material:
            outcome = adjudicate_by_material(board, piece_values)
            if outcome > 0:
                termination = "material_white"
            elif outcome < 0:
                termination = "material_black"
        else:
            outcome = 0.0  # Draw by max length
    else:
        # Game ended prematurely (e.g. MCTS could not find a move)
        # Determine result from current board state
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                outcome = 1.0
                termination = "checkmate" if board.is_checkmate() else "other"
            elif result == "0-1":
                outcome = -1.0
                termination = "checkmate" if board.is_checkmate() else "other"
            else:
                outcome = 0.0
                if board.is_repetition():
                    termination = "repetition"
                elif board.is_fifty_moves():
                    termination = "fifty_moves"
                elif board.is_insufficient_material():
                    termination = "insufficient_material"
                else:
                    termination = "stalemate"
        else:
            # Board is not in game-over state; force draw with proper termination
            outcome = 0.0
            if move_count >= max_game_length:
                termination = "max_length"
            else:
                termination = "unknown"
                # --- DIAGNOSTIC LOGGING ---
                with open("unknown_termination_log.txt", "a") as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"UNKNOWN TERMINATION at move_count={move_count}\n")
                    f.write(f"FEN: {board.fen()}\n")
                    f.write(f"board.is_game_over(): {board.is_game_over()}\n")
                    f.write(f"Legal moves count: {len(list(board.legal_moves))}\n")
                    f.write(f"Legal moves: {[m.uci() for m in board.legal_moves]}\n")
    
                    # Re-run MCTS one more time on this exact position so we
                    # can inspect what happened (note: this may give slightly
                    # different results due to randomness, but the position
                    # and root.children should reveal the structural issue).
                    root = mcts_engine.get_root(board)
                    visit_policy, best_move, stats = mcts_engine.search(root)
    
                    f.write(f"root.children count: {len(root.children)}\n")
                    f.write(f"root.children keys: {sorted(root.children.keys())}\n")
                    total_visits = sum(c.N for c in root.children.values())
                    f.write(f"total visits across children: {total_visits}\n")
    
                    for idx, child in root.children.items():
                        f.write(
                            f"  child idx={idx}, N={child.N}, P={child.P:.4f}, "
                            f"Q={child.Q:.4f}, move={child.move}, "
                            f"move_in_legal={child.move in board.legal_moves}\n"
                        )
    
                    f.write(f"search() returned best_move: {best_move}\n")
    
                    best_idx = int(np.argmax(visit_policy)) if visit_policy.sum() > 0 else None
                    f.write(f"argmax(visit_policy): {best_idx}\n")
                    if best_idx is not None:
                        f.write(f"best_idx in root.children: {best_idx in root.children}\n")
                        from encoding import policy_index_to_move
                        decoded = policy_index_to_move(best_idx, board)
                        f.write(f"policy_index_to_move(best_idx, board): {decoded}\n")
    
                    f.write(f"select_move_with_temperature was called with selected_move=None\n")
                    f.write(f"visit_policy.sum(): {visit_policy.sum()}\n")
                    f.write(f"visit_policy.max(): {visit_policy.max()}\n")
                    f.write("=" * 60 + "\n\n")
                # --- END DIAGNOSTIC LOGGING ---

    
    # Assign values to all positions from each player's perspective
    game_data = []
    for state_tensor, policy, player in game_states:
        # Value is from the perspective of the player who was to move
        value = outcome * player
        game_data.append((state_tensor, policy, value))
    
    # Compute aggregate stats
    avg_depth = np.mean([s.get('avg_depth', 0) for s in mcts_stats_list]) if mcts_stats_list else 0
    
    game_info = {
        'result': outcome,  # 1.0 = white wins, -1.0 = black wins, 0.0 = draw
        'result_str': board.result() if board.is_game_over() else '*',
        'length': move_count,
        'termination': termination,
        'avg_mcts_depth': float(avg_depth),
        'num_positions': len(game_data),
    }
    
    if verbose:
        print(f"Game finished: {termination}, result={board.result()}, length={move_count}")
    
    return game_data, game_info


def self_play_game(network: AlphaZeroNet, config, on_move=None) -> Tuple[List[Tuple], dict]:
    """Play a single self-play game with settings from config.
    
    Convenience wrapper around play_one_game that reads config.
    
    Args:
        network: The neural network for position evaluation
        config: Config object with self-play settings
        on_move: Optional callback called after each move with
                 (board_fen, move_uci, move_number)
    """
    mcts_engine = MCTS(
        network=network,
        num_simulations=config.mcts.num_simulations,
        c_puct=config.mcts.c_puct,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        dirichlet_epsilon=config.mcts.dirichlet_epsilon,
        batch_size=getattr(config.mcts, 'batch_size', 1),
        c_virtual_loss=getattr(config.mcts, 'c_virtual_loss', 0.5),
    )
    
    return play_one_game(
        mcts_engine=mcts_engine,
        max_game_length=config.selfplay.max_game_length,
        adjudicate_material=config.selfplay.adjudicate_material,
        piece_values=config.selfplay.piece_values,
        temp_threshold=config.selfplay.temperature_threshold,
        temp_high=config.selfplay.temperature_high,
        temp_low=config.selfplay.temperature_low,
        on_move=on_move,
    )
