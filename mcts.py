"""Monte Carlo Tree Search with PUCT selection for AlphaZero chess.

Implements the AlphaZero PUCT variant:
- Each node tracks: visit count N, total value W, mean value Q, prior P
- Dirichlet noise at root for exploration
- Temperature-based move selection
"""

import math
import numpy as np
import chess
from typing import Optional, Tuple, List, Dict

from encoding import (
    board_to_tensor, get_legal_move_mask, move_to_policy_index,
    policy_index_to_move, NUM_ACTIONS
)
from network import AlphaZeroNet


class MCTSNode:
    """A node in the MCTS tree."""
    
    __slots__ = ['board', 'parent', 'move', 'children', 'N', 'W', 'Q', 'P', 
                 'is_expanded', 'legal_moves_cached', 'visit_count']
    
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children: Dict[int, MCTSNode] = {}  # policy_index -> child
        self.N = 0          # Visit count
        self.W = 0.0        # Total value
        self.Q = 0.0        # Mean value (W / N)
        self.P = prior      # Prior probability from network
        self.is_expanded = False
        self.legal_moves_cached: Optional[List[chess.Move]] = None
        self.visit_count = 0


class MCTS:
    """Monte Carlo Tree Search with PUCT selection."""
    
    def __init__(self, 
                 network: AlphaZeroNet,
                 num_simulations: int = 200,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        """
        Args:
            network: The neural network for position evaluation
            num_simulations: Number of MCTS simulations per move
            c_puct: PUCT exploration constant
            dirichlet_alpha: Dirichlet noise parameter
            dirichlet_epsilon: Weight of Dirichlet noise vs network prior
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
    
    def get_root(self, board: chess.Board) -> MCTSNode:
        """Create a root node for the given board."""
        return MCTSNode(board.copy())
    
    def search(self, root: MCTSNode) -> Tuple[np.ndarray, chess.Move, dict]:
        """Run MCTS from root and return visit distribution, best move, and stats.
        
        Args:
            root: Root node of the search tree
        
        Returns:
            visit_policy: (4672,) visit distribution as a probability vector
            best_move: The selected move
            stats: Dictionary with search statistics (avg_depth, etc.)
        """
        # Expand root node first
        if not root.is_expanded:
            self._expand_node(root)
        
        # Add Dirichlet noise to root priors
        self._add_dirichlet_noise(root)
        
        # Run simulations
        max_depth = 0
        total_depth = 0
        
        for _ in range(self.num_simulations):
            node = root
            depth = 0
            
            # Selection: traverse tree using PUCT
            while node.is_expanded and node.children:
                node = self._select_child(node)
                depth += 1
            
            # Expansion and evaluation
            if not node.is_expanded:
                value = self._expand_node(node)
            else:
                # Terminal node — value is the game result
                value = self._get_terminal_value(node)
            
            # Backpropagation
            self._backpropagate(node, value)
            
            max_depth = max(max_depth, depth)
            total_depth += depth
        
        # Compute visit distribution
        visit_policy, best_move = self._get_visit_policy(root)
        
        stats = {
            'avg_depth': total_depth / self.num_simulations if self.num_simulations > 0 else 0,
            'max_depth': max_depth,
            'num_simulations': self.num_simulations,
        }
        
        return visit_policy, best_move, stats
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with highest PUCT score."""
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_n = math.sqrt(node.N + 1)
        
        for action_idx, child in node.children.items():
            # PUCT formula: Q + c_puct * P * sqrt(parent_N) / (1 + child_N)
            ucb = child.Q + self.c_puct * child.P * sqrt_parent_n / (1 + child.N)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        
        return best_child
    
    def _expand_node(self, node: MCTSNode) -> float:
        """Expand a node using the network. Returns the value estimate.
        
        If the node is a terminal position, returns the game result
        without querying the network.
        """
        # Check for terminal position
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":
                return 1.0
            elif result == "0-1":
                return -1.0
            else:
                return 0.0  # Draw
        
        # Get network prediction
        state = board_to_tensor(node.board)
        policy, value = self.network.predict(state)
        
        # Get legal moves and their policy indices
        legal_moves = list(node.board.legal_moves)
        node.legal_moves_cached = legal_moves
        
        if not legal_moves:
            # No legal moves — shouldn't happen if game_over check above works
            return 0.0
        
        # Mask illegal moves and renormalize
        mask = get_legal_move_mask(node.board)
        legal_policy = policy * mask
        legal_sum = legal_policy.sum()
        
        if legal_sum > 0:
            legal_policy = legal_policy / legal_sum
        else:
            # Uniform over legal moves if network gives zero probability to all
            legal_policy = mask / mask.sum()
        
        # Create child nodes
        for move in legal_moves:
            try:
                action_idx = move_to_policy_index(move, node.board)
            except ValueError:
                continue
            
            prior = float(legal_policy[action_idx])
            
            # Make the move on a new board for the child
            child_board = node.board.copy()
            child_board.push(move)
            
            child = MCTSNode(child_board, parent=node, move=move, prior=prior)
            node.children[action_idx] = child
        
        node.is_expanded = True
        
        return float(value)
    
    def _get_terminal_value(self, node: MCTSNode) -> float:
        """Get the value of a terminal node from the current player's perspective."""
        result = node.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree, flipping sign each ply."""
        current = node
        v = value
        while current is not None:
            current.N += 1
            current.W += v
            current.Q = current.W / current.N
            current = current.parent
            v = -v  # Flip value for opponent's perspective
    
    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root node's priors for exploration."""
        if not node.children:
            return
        
        num_children = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)
        
        for i, action_idx in enumerate(node.children):
            child = node.children[action_idx]
            child.P = (1 - self.dirichlet_epsilon) * child.P + \
                      self.dirichlet_epsilon * noise[i]
    
    def _get_visit_policy(self, root: MCTSNode) -> Tuple[np.ndarray, chess.Move]:
        """Compute visit count distribution and select move.
        
        Uses temperature-based selection:
        - If max visit count > threshold, select greedily
        - Otherwise, sample proportionally to visit counts^temperature
        
        Returns:
            visit_policy: (4672,) normalized visit distribution
            best_move: Selected move
        """
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        
        if not root.children:
            # No children — return empty
            return visit_policy, None
        
        # Fill visit counts
        total_visits = 0
        for action_idx, child in root.children.items():
            visit_policy[action_idx] = child.N
            total_visits += child.N
        
        if total_visits == 0:
            return visit_policy, None
        
        # Normalize to get probability distribution
        visit_probs = visit_policy / total_visits
        
        # Select move based on temperature
        # High temperature: sample proportionally
        # Low temperature: select most visited
        # Temperature is applied via softmax of log(visit_count) / tau
        
        # Find the move with most visits for greedy selection
        best_idx = int(np.argmax(visit_policy))
        best_move = policy_index_to_move(best_idx, root.board)
        
        return visit_probs, best_move
    
    def get_root_child_stats(self, root: MCTSNode) -> list:
        """Get stats for all root children that got at least one visit.
        
        Returns:
            List of dicts sorted by visit count descending, each with:
                move: UCI string of the move
                N: visit count
                W: total value
                Q: mean value
                P: prior probability
        """
        stats = []
        for action_idx, child in root.children.items():
            if child.N > 0:
                stats.append({
                    'move': child.move.uci() if child.move else None,
                    'N': child.N,
                    'W': child.W,
                    'Q': child.Q,
                    'P': child.P,
                })
        return sorted(stats, key=lambda x: x['N'], reverse=True)

    def select_move_with_temperature(self, root: MCTSNode, temperature: float) -> Tuple[np.ndarray, chess.Move]:
        """Select a move using the given temperature.
        
        Args:
            root: The root node after search
            temperature: Temperature parameter. 1.0 = proportional, 0.0 = greedy
        
        Returns:
            visit_policy: (4672,) visit distribution
            move: Selected move
        """
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        
        if not root.children:
            return visit_policy, None
        
        # Get visit counts
        visit_counts = {}
        for action_idx, child in root.children.items():
            visit_counts[action_idx] = child.N
        
        total = sum(visit_counts.values())
        if total == 0:
            return visit_policy, None
        
        if temperature < 1e-8:
            # Greedy selection
            best_idx = max(visit_counts, key=visit_counts.get)
            visit_policy[best_idx] = 1.0
            move = policy_index_to_move(best_idx, root.board)
        else:
            # Sample proportionally to visit_count^(1/temperature)
            indices = list(visit_counts.keys())
            counts = np.array([visit_counts[i] for i in indices], dtype=np.float64)
            
            # Apply temperature
            probs = counts ** (1.0 / temperature)
            probs = probs / probs.sum()
            
            # Sample
            chosen = np.random.choice(len(indices), p=probs)
            chosen_idx = indices[chosen]
            
            visit_policy[chosen_idx] = 1.0
            move = policy_index_to_move(chosen_idx, root.board)
            
            # Also store the full normalized visit distribution
            for idx in indices:
                visit_policy[idx] = visit_counts[idx] / total
        
        return visit_policy, move