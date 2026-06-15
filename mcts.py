"""Monte Carlo Tree Search with PUCT selection for AlphaZero chess.

Implements the AlphaZero PUCT variant with optional batched inference:
- Each node tracks: visit count N, total value W, mean value Q, prior P
- Dirichlet noise at root for exploration
- Temperature-based move selection
- Virtual loss for tree-parallel batched inference (batch_size > 1)
"""

import math
import numpy as np
import chess
from typing import Optional, Tuple, List, Dict

from encoding import (
    board_to_tensor, get_legal_move_mask, get_legal_move_mask_from_moves,
    move_to_policy_index, policy_index_to_move, NUM_ACTIONS
)
from network import AlphaZeroNet


class MCTSNode:
    """A node in the MCTS tree."""
    
    __slots__ = ['board', 'parent', 'move', 'children', 'N', 'W', 'Q', 'P',
                 'is_expanded', 'legal_moves_cached', 'visit_count', 'virtual_loss']
    
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
        self.visit_count = 0  # Duplicate of N, kept for backward compatibility
        self.virtual_loss = 0  # Virtual loss counter for batched inference


class MCTS:
    """Monte Carlo Tree Search with PUCT selection.
    
    Supports both sequential (batch_size=1) and batched (batch_size>1) modes.
    Batched mode uses virtual loss to collect multiple leaf nodes before
    evaluating them together in a single network forward pass.
    """
    
    def __init__(self,
                 network: AlphaZeroNet,
                 num_simulations: int = 200,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 batch_size: int = 1,
                 c_virtual_loss: float = 0.5):
        """
        Args:
            network: The neural network for position evaluation
            num_simulations: Number of MCTS simulations per move
            c_puct: PUCT exploration constant
            dirichlet_alpha: Dirichlet noise parameter
            dirichlet_epsilon: Weight of Dirichlet noise vs network prior
            batch_size: Number of leaves to collect before network eval.
                       1 = sequential (no virtual loss). >1 = batched inference.
            c_virtual_loss: Virtual loss penalty constant for batched mode.
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.c_virtual_loss = c_virtual_loss
    
    def get_root(self, board: chess.Board) -> MCTSNode:
        """Create a root node for the given board."""
        return MCTSNode(board.copy(stack=False))
    
    def search(self, root: MCTSNode) -> Tuple[np.ndarray, chess.Move, dict]:
        """Run MCTS from root and return visit distribution, best move, and stats.
        
        Uses batched inference if self.batch_size > 1, otherwise falls back
        to the standard sequential search.
        
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
        
        if self.batch_size > 1:
            # Batched mode: collect leaves, evaluate in batch, backprop
            sims_done = 0
            while sims_done < self.num_simulations:
                # Determine batch size for this iteration
                bs = min(self.batch_size, self.num_simulations - sims_done)
                
                # Collect leaf nodes via selection with virtual loss
                leaf_nodes = self._collect_batch(root, bs)
                
                if not leaf_nodes:
                    # Shouldn't happen, but safety check
                    break
                
                # Evaluate all leaves in a single batch network call
                values = self._evaluate_batch(leaf_nodes)
                
                # Backpropagate each leaf, removing virtual losses
                for leaf, value in zip(leaf_nodes, values):
                    self._backpropagate_with_virtual_loss(leaf, value)
                    
                    # Track depth
                    depth = self._node_depth(leaf)
                    max_depth = max(max_depth, depth)
                    total_depth += depth
                
                sims_done += bs
        else:
            # Sequential mode (original behavior, no virtual loss overhead)
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
        """Select the child with highest PUCT score, accounting for virtual losses."""
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_n = math.sqrt(node.N + 1)
        
        for action_idx, child in node.children.items():
            # Effective visit count includes virtual losses
            effective_N = child.N + child.virtual_loss
            
            # PUCT formula with virtual loss penalty
            # Q term: uses actual Q (not affected by virtual loss)
            # Exploration term: penalizes nodes with virtual losses
            # Direct penalty term: - c_virtual_loss * virtual_loss
            ucb = child.Q + self.c_puct * child.P * sqrt_parent_n / (1 + effective_N) \
                  - self.c_virtual_loss * child.virtual_loss
            if ucb > best_score:
                best_score = ucb
                best_child = child
        
        return best_child
    
    def _collect_batch(self, root: MCTSNode, batch_size: int) -> List[MCTSNode]:
        """Run batch_size selections, applying virtual loss along each path.
        
        Each selection traverses from root to a leaf, incrementing virtual_loss
        by 1 on each node visited. This discourages multiple selections from
        choosing the same path.
        
        Args:
            root: Root node
            batch_size: Number of leaf nodes to collect
        
        Returns:
            List of leaf MCTSNode objects (may include duplicates if all
            paths converge to the same terminal).
        """
        leaves = []
        
        for _ in range(batch_size):
            node = root
            
            # Selection with virtual loss
            while node.is_expanded and node.children:
                # Apply virtual loss to this node before selecting child
                node.virtual_loss += 1
                node = self._select_child(node)
            
            # Apply virtual loss to the leaf
            node.virtual_loss += 1
            leaves.append(node)
        
        return leaves
    
    def _evaluate_batch(self, leaf_nodes: List[MCTSNode]) -> List[float]:
        """Evaluate a batch of leaf nodes.
        
        For each leaf:
        - If terminal, returns game result directly (skip network)
        - If not expanded yet, queues for network eval
        - If already expanded (terminal that was previously visited), 
          returns the terminal value (shouldn't happen in normal flow)
        
        Non-terminal leaves are evaluated together in a single batched
        network call.
        
        Args:
            leaf_nodes: List of leaf nodes to evaluate
        
        Returns:
            List of value estimates, one per leaf
        """
        # Separate terminal and expandable leaves
        terminal_values = {}
        expandable_indices = []
        expandable_nodes = []
        
        for i, node in enumerate(leaf_nodes):
            if node.board.is_game_over():
                terminal_values[i] = self._get_terminal_value(node)
            elif not node.is_expanded:
                expandable_indices.append(i)
                expandable_nodes.append(node)
            else:
                # Already expanded but no children — shouldn't happen
                # in normal flow, but handle gracefully
                terminal_values[i] = self._get_terminal_value(node)
        
        # For expandable leaves, we need network eval
        if expandable_nodes:
            # Collect states
            states_list = []
            for node in expandable_nodes:
                state = board_to_tensor(node.board)
                states_list.append(state)
            
            # Stack into batch
            states_batch = np.stack(states_list, axis=0)  # (batch, 18, 8, 8)
            
            # Single batched network call
            policies_batch, values_batch = self.network.predict_batch(states_batch)
            
            # Expand each node with its prediction
            for idx, node in enumerate(expandable_nodes):
                self._expand_node_with_data(node, policies_batch[idx], values_batch[idx])
                terminal_values[expandable_indices[idx]] = float(values_batch[idx])
        
        # Build result list in original order
        results = []
        for i in range(len(leaf_nodes)):
            if i in terminal_values:
                results.append(terminal_values[i])
            else:
                # Fallback — should not reach here
                results.append(0.0)
        
        return results
    
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
        
        # Complete expansion using the prediction data
        return self._expand_node_with_data(node, policy, value)
    
    def _expand_node_with_data(self, node: MCTSNode,
                                policy: np.ndarray, value: float) -> float:
        """Expand a node using precomputed policy and value.
        
        Shared by both sequential and batched expansion paths.
        
        Args:
            node: Node to expand
            policy: (4672,) policy probability vector from network
            value: Scalar value estimate from network
        
        Returns:
            value: The value estimate (same as input)
        """
        # Get legal moves (only once)
        legal_moves = list(node.board.legal_moves)
        node.legal_moves_cached = legal_moves
        
        if not legal_moves:
            # No legal moves — shouldn't happen if game_over check above works
            return 0.0
        
        # Build legal move mask from the already-computed legal_moves list
        mask = get_legal_move_mask_from_moves(legal_moves, node.board)
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
            
            # Use stack=False for cheaper board copy (no move history needed)
            child_board = node.board.copy(stack=False)
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
    
    def _backpropagate_with_virtual_loss(self, node: MCTSNode, value: float):
        """Backpropagate and remove virtual loss along the path.
        
        - Decrements virtual_loss by 1 on each visited node
        - Updates N, W, Q as in standard backpropagation
        """
        current = node
        v = value
        while current is not None:
            # Remove the virtual loss we applied during collection
            current.virtual_loss -= 1
            
            # Backprop value (same as standard)
            current.N += 1
            current.W += v
            current.Q = current.W / current.N
            current = current.parent
            v = -v  # Flip value for opponent's perspective
    
    def _node_depth(self, node: MCTSNode) -> int:
        """Compute the depth of a node from root."""
        depth = 0
        current = node.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root node's priors for exploration.
        
        If dirichlet_alpha <= 0, noise is skipped (used in evaluation mode).
        """
        if not node.children:
            return
        
        if self.dirichlet_alpha <= 0 or self.dirichlet_epsilon <= 0:
            return
        
        num_children = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)
        
        for i, action_idx in enumerate(node.children):
            child = node.children[action_idx]
            child.P = (1 - self.dirichlet_epsilon) * child.P + \
                      self.dirichlet_epsilon * noise[i]
    
    def _find_checkmate_child(self, root: MCTSNode) -> Optional[chess.Move]:
        """Check if any root child delivers checkmate.
        
        A child whose board is in checkmate means the move leading to it
        wins the game immediately. This should always be played.
        
        Args:
            root: The root node after search
        
        Returns:
            The checkmate move if found, None otherwise.
        """
        for child in root.children.values():
            if child.board.is_checkmate():
                return child.move
        return None
    
    def _get_visit_policy(self, root: MCTSNode) -> Tuple[np.ndarray, chess.Move]:
        """Compute visit count distribution and select move.
        
        Uses temperature-based selection:
        - If max visit count > threshold, select greedily
        - Otherwise, sample proportionally to visit counts^temperature
        
        Returns:
            visit_policy: (4672,) normalized visit distribution
            best_move: Selected move (most visited)
        """
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        
        if not root.children:
            # No children — return empty
            return visit_policy, None
        
        # Force checkmate move if MCTS found one
        checkmate_move = self._find_checkmate_child(root)
        if checkmate_move is not None:
            idx = move_to_policy_index(checkmate_move, root.board)
            visit_policy[idx] = 1.0
            return visit_policy, checkmate_move
        
        # Fill visit counts
        total_visits = 0
        for action_idx, child in root.children.items():
            visit_policy[action_idx] = child.N
            total_visits += child.N
        
        if total_visits == 0:
            return visit_policy, None
        
        # Normalize to get probability distribution
        visit_probs = visit_policy / total_visits
        
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
        
        # Force checkmate move if MCTS found one
        checkmate_move = self._find_checkmate_child(root)
        if checkmate_move is not None:
            idx = move_to_policy_index(checkmate_move, root.board)
            visit_policy[idx] = 1.0
            return visit_policy, checkmate_move
        
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
