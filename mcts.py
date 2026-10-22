"""Monte Carlo Tree Search with PUCT selection for AlphaZero games.

Implements the AlphaZero PUCT variant with optional batched inference:
- Each node tracks: visit count N, total value W, mean value Q, prior P
- Dirichlet noise at root for exploration
- Temperature-based move selection
- Virtual loss for tree-parallel batched inference (batch_size > 1)
- Lazy state construction: child boards/states are only copied+push'ed on
  first access, saving ~24% CPU by avoiding copies for never-visited
  children.
- Cached terminal status and value on each node to eliminate redundant
  game-state queries during repeated leaf evaluations.
- **NEW**: respects `max_game_length` with configurable adjudication.

The search drives whatever game is active in ``game_api`` (chess or
connect4); none of the board-specific calls here are game-aware.
"""

import math
import time
import numpy as np
import chess
from typing import Optional, Tuple, List, Dict

import game_api
from encoding import (
    board_to_tensor, get_legal_move_mask, get_legal_move_mask_from_moves,
    move_to_policy_index, policy_index_to_move, NUM_ACTIONS
)
from network import AlphaZeroNet


class MCTSNode:
    """A node in the MCTS tree.

    Board is lazily constructed on first access via the ``board`` property.
    This avoids expensive ``board.copy(stack=True)`` for child nodes that are
    never visited during the search.

    ``is_game_over`` and the terminal value are cached once computed, saving
    redundant python-chess calls when the same leaf node is evaluated in
    multiple simulation iterations (common in batched mode).
    """

    __slots__ = ['_board', '_board_ready', 'parent', 'move', 'children',
                 'N', 'W', 'Q', 'P', 'P_orig', 'is_expanded',
                 'legal_moves_cached', 'visit_count', 'virtual_loss',
                 '_game_over_cached', '_terminal_value_cached',
                 '_mate_seeded', '_position_hash', 'depth', 'rs_id']

    def __init__(self, board: Optional[chess.Board] = None,
                 parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None,
                 prior: float = 0.0):
        # Eagerly set if board is provided (root nodes), lazy otherwise (children)
        self._board = board
        self._board_ready = board is not None
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children: Dict[int, MCTSNode] = {}  # policy_index -> child
        self.N = 0          # Visit count
        self.W = 0.0        # Total value
        self.Q = 0.0        # Mean value (W / N)
        self.P = prior      # Prior probability from network
        self.P_orig = prior  # Original prior before Dirichlet noise (for tree recycling)
        self.is_expanded = False
        self.legal_moves_cached: Optional[List[chess.Move]] = None
        self.visit_count = 0  # Duplicate of N, kept for backward compatibility
        self.virtual_loss = 0  # Virtual loss counter for batched inference
        self._game_over_cached: Optional[bool] = None  # None = not yet checked
        self._terminal_value_cached: Optional[float] = None  # Cached terminal value
        self._mate_seeded: bool = False  # Root already received mate-seeding visits
        self._position_hash: Optional[int] = None  # Zobrist hash, cached on first materialisation
        # Cached depth from root — avoids O(depth) walk in _node_depth()
        self.depth = 0 if parent is None else parent.depth + 1

    @staticmethod
    def new_child(parent: 'MCTSNode', move, prior: float) -> 'MCTSNode':
        """Fast-path constructor for child nodes (the search hot loop).

        Avoids the ~22 attribute stores of ``__init__`` for the large
        majority of nodes that start with default values.  Semantically
        identical: children get the same initial state via ``__init__``.
        """
        node = object.__new__(MCTSNode)
        node._board = None
        node._board_ready = False
        node.parent = parent
        node.move = move
        node.children = {}
        node.N = 0
        node.W = 0.0
        node.Q = 0.0
        node.P = prior
        node.P_orig = prior
        node.is_expanded = False
        node.legal_moves_cached = None
        node.visit_count = 0
        node.virtual_loss = 0
        node._game_over_cached = None
        node._terminal_value_cached = None
        node._mate_seeded = False
        node._position_hash = None
        node.depth = parent.depth + 1
        return node

    @property
    def board(self) -> chess.Board:
        """Lazily materialize the board on first access.

        Walks up via parent, copies with ``stack=True`` to preserve the
        full game history for threefold-repetition detection, then pushes
        the move.  The result is cached so subsequent accesses are free.
        """
        if not self._board_ready:
            # Build from parent once, then cache
            parent_board = self.parent.board
            b = parent_board.copy(stack=True)
            b.push(self.move)
            self._board = b
            self._board_ready = True
        return self._board

    @board.setter
    def board(self, value: chess.Board):
        """Allow direct assignment (used by get_root / recycle_tree)."""
        self._board = value
        self._board_ready = True


class MCTS:
    """Monte Carlo Tree Search with PUCT selection.

    Supports both sequential (batch_size=1) and batched (batch_size>1) modes.
    Batched mode uses virtual loss to collect multiple leaf nodes before
    evaluating them together in a single network forward pass.

    **NEW** parameters for respecting self-play game-length limit:
        max_game_length: int - number of half-moves after which the game is
                          adjudicated (material or draw).
        adjudicate_material: bool - if True, use material to decide winner;
                               if False, treat as draw (0.0).
        piece_values: dict - mapping from piece symbol to material points.
        adjudicate_graded: bool - if True, use tanh scaling; else flat +/-1.
        adjudicate_scaling: float - tanh denominator (only if graded).
    """

    def __init__(self,
                 network: AlphaZeroNet,
                 num_simulations: int = 200,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 batch_size: int = 1,
                 c_virtual_loss: float = 0.5,
                 # ---- NEW parameters ----
                 max_game_length: int = 150,
                 adjudicate_material: bool = True,
                 piece_values: Optional[dict] = None,
                 adjudicate_graded: bool = True,
                 adjudicate_scaling: float = 9.0,
                 mate_seeding: bool = False,
                 min_fresh_sims: int = 64,
                 inference_pipeline: int = 1):
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
            max_game_length: Maximum half-moves before adjudication.
            adjudicate_material: Whether to decide winner by material at limit.
            piece_values: Material values for adjudication.
            adjudicate_graded: Use tanh grading (True) or +/-1 (False).
            adjudicate_scaling: Scaling for tanh (if graded).
            mate_seeding: Give every root child that delivers checkmate one
                terminal pre-visit before the normal search.  The seed updates
                visit counts and values only; it does not override the visit
                policy, move choice, or temperature sampling.
            min_fresh_sims: Minimum number of new simulations to run after
                tree reuse, even when the recycled root already has the full
                visit budget.  0 reproduces the historical
                ``max(0, num_simulations - root.N)`` behaviour.
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.c_virtual_loss = c_virtual_loss

        # ---- NEW attributes ----
        self.max_game_length = max_game_length
        self.adjudicate_material = adjudicate_material
        self.piece_values = piece_values or {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
        self.adjudicate_graded = adjudicate_graded
        self.adjudicate_scaling = adjudicate_scaling
        self.mate_seeding = mate_seeding
        self.min_fresh_sims = max(0, int(min_fresh_sims))

        # Async inference pipelining: number of leaf-batches kept in flight so
        # tree work (selection / expansion / backprop) overlaps the network
        # round-trip.  Only active when batch_size > 1 AND the injected
        # "network" exposes the async API (predict_batch_async + wait_result),
        # i.e. the GPU InferenceClient.  Values > 1 change WHEN leaves are
        # collected (a later batch is gathered while an earlier one is in
        # flight), which slightly alters search dynamics — comparable to
        # increasing batch_size — but the per-leaf math is unchanged.
        self.inference_pipeline = max(1, int(inference_pipeline))

        # Profiling: when True, search() collects per-phase timing into stats['profiling']
        self.profile = False

    def get_root(self, board) -> MCTSNode:
        """Create a root node for the given game state (eagerly stored)."""
        return MCTSNode(board.copy())

    def recycle_tree(self, root: MCTSNode, move) -> Optional[MCTSNode]:
        """Promote the child of `root` that corresponds to `move` to a new root.

        Instead of discarding the entire search tree after each move, this
        method promotes the selected child to be the new root, preserving
        its subtree (expanded children, Q values, visit counts) for the
        next search.

        The promoted node's parent pointer is cleared and its state is
        eagerly materialized (since the parent relationship is severed).

        Args:
            root: The current root node (already searched)
            move: The move that was selected

        Returns:
            The promoted child as a new root, or None if the child
            wasn't found.
        """
        # Compare through the game's canonical move key so any move object
        # (chess.Move, FastMove, or a bare connect4 column int) works.
        selected_key = game_api.move_key(move)
        b = root.board

        for action_idx, child in root.children.items():
            if game_api.move_key(child.move) != selected_key:
                continue
            # Detach from parent
            child.parent = None

            # Eagerly materialize the state since the parent is gone
            if not child._board_ready:
                nb = b.copy(stack=True)
                game_api.apply(nb, move)
                child._board = nb
                child._board_ready = True

            # Reset depth since this is now a root node
            child.depth = 0

            # Clear visit_count (backward-compat duplicate of N).
            # N/W/Q are intentionally kept so the next search
            # benefits from accumulated visit information.
            child.visit_count = 0
            child._mate_seeded = False

            return child

        return None

    def search(self, root: MCTSNode) -> Tuple[np.ndarray, chess.Move, dict]:
        """Run MCTS from root and return visit distribution, best move, and stats.

        Uses batched inference if self.batch_size > 1, otherwise falls back
        to the standard sequential search.

        When ``self.profile`` is ``True``, the returned ``stats`` dict includes
        a ``'profiling'`` sub-dict with detailed per-phase timing.

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

        inherited_root_n = int(root.N)

        # Give mating root moves a terminal pre-visit before the normal search.
        self._seed_mate_children(root)

        # Add Dirichlet noise to root priors
        self._add_dirichlet_noise(root)

        reused_root = inherited_root_n > 0
        budget_new_sims = max(0, self.num_simulations - int(root.N))
        target_new_sims = (max(budget_new_sims, self.min_fresh_sims)
                           if reused_root else budget_new_sims)
        floor_fired = reused_root and budget_new_sims < self.min_fresh_sims

        # Profiling accumulators
        if self.profile:
            pf = {
                'selection_total': 0.0,
                'expansion_total': 0.0,
                'backprop_total': 0.0,
                'expand_legal_moves': 0.0,
                'expand_policy_indices': 0.0,
                'expand_mask_renorm': 0.0,
                'expand_child_creation': 0.0,
                'network_predict': 0.0,
                'network_batch_predict': 0.0,
                'collection_vl_mgmt': 0.0,
                'network_calls': 0,
                'network_batch_calls': 0,
                'expansions': 0,
                'total_nodes_created': 0,
            }
        else:
            pf = None

        # Run simulations
        max_depth = 0
        total_depth = 0
        sims_done = 0

        if self.batch_size > 1:
            # Batched mode: collect leaves, evaluate in batch, backprop
            use_pipeline = (self.inference_pipeline > 1
                            and hasattr(self.network, 'predict_batch_async')
                            and hasattr(self.network, 'wait_result'))

            if not use_pipeline:
                while sims_done < target_new_sims:
                    # Determine batch size for this iteration
                    bs = min(self.batch_size, target_new_sims - sims_done)

                    # Collect leaf nodes via selection with virtual loss.
                    leaf_depth_pairs = self._collect_batch(root, bs)

                    if not leaf_depth_pairs:
                        break

                    leaf_nodes = [ld[0] for ld in leaf_depth_pairs]

                    # Evaluate all leaves in a single batch network call
                    t0 = time.perf_counter() if pf else 0
                    values = self._evaluate_batch(leaf_nodes)
                    if pf:
                        pf['network_batch_predict'] += time.perf_counter() - t0
                        pf['network_batch_calls'] += 1

                    # Backpropagate each leaf, removing virtual losses
                    for (leaf, depth), value in zip(leaf_depth_pairs, values):
                        t0 = time.perf_counter() if pf else 0
                        self._backpropagate_with_virtual_loss(leaf, value)
                        if pf:
                            pf['backprop_total'] += time.perf_counter() - t0
                        max_depth = max(max_depth, depth)
                        total_depth += depth

                    sims_done += len(leaf_nodes)
            else:
                # ── Pipelined batched mode ──
                # Keep up to `inference_pipeline` batches in flight so tree
                # work overlaps the network round-trip.  While batch k is on
                # the GPU we already collect + dispatch batch k+1; results
                # are applied strictly in submission order (FIFO).
                from collections import deque
                inflight: deque = deque()   # (req_id|None, pairs, tv, ex_idx, ex_nodes, ids)
                outstanding = 0             # leaves dispatched but not applied

                while True:
                    # Top the pipeline up.
                    while (len(inflight) < self.inference_pipeline
                           and sims_done + outstanding < target_new_sims):
                        bs = min(self.batch_size,
                                 target_new_sims - sims_done - outstanding)

                        inflight_ids: set = set()
                        for entry in inflight:
                            inflight_ids.update(entry[5])

                        pairs = self._collect_batch(root, bs,
                                                    excluded_ids=inflight_ids)
                        if not pairs:
                            break

                        leaf_nodes = [ld[0] for ld in pairs]
                        tv, ex_idx, ex_nodes = self._split_terminal_leaves(leaf_nodes)

                        if ex_nodes:
                            states = np.stack(
                                [game_api.encode(n.board) for n in ex_nodes],
                                axis=0)
                            t0 = time.perf_counter() if pf else 0
                            req_id = self.network.predict_batch_async(states)
                            if pf:
                                pf['collection_vl_mgmt'] += time.perf_counter() - t0
                        else:
                            req_id = None

                        pair_ids = {id(n) for n in leaf_nodes}
                        inflight.append((req_id, pairs, tv, ex_idx,
                                         ex_nodes, pair_ids))
                        outstanding += len(pairs)

                    if not inflight:
                        break

                    # Apply the oldest batch's results (FIFO keeps VL bookkeeping
                    # consistent with collection order).
                    (req_id, pairs, tv, ex_idx,
                     ex_nodes, pair_ids) = inflight.popleft()
                    outstanding -= len(pairs)

                    t0 = time.perf_counter() if pf else 0
                    if req_id is not None:
                        policies, vals = self.network.wait_result(req_id)
                        values = self._apply_network_results(
                            [ld[0] for ld in pairs], tv, ex_idx, ex_nodes,
                            policies, vals)
                    else:
                        values = [tv.get(i, 0.0) for i in range(len(pairs))]
                    if pf:
                        pf['network_batch_predict'] += time.perf_counter() - t0
                        pf['network_batch_calls'] += 1

                    # Backpropagate each leaf, removing virtual losses
                    for (leaf, depth), value in zip(pairs, values):
                        t0 = time.perf_counter() if pf else 0
                        self._backpropagate_with_virtual_loss(leaf, value)
                        if pf:
                            pf['backprop_total'] += time.perf_counter() - t0
                        max_depth = max(max_depth, depth)
                        total_depth += depth

                    sims_done += len(pairs)
        else:
            # Sequential mode (original behavior, no virtual loss overhead)
            for _ in range(target_new_sims):
                node = root
                depth = 0

                # Selection: traverse tree using PUCT
                t0 = time.perf_counter() if pf else 0
                while node.is_expanded and node.children:
                    node = self._select_child(node)
                    depth += 1
                if pf:
                    pf['selection_total'] += time.perf_counter() - t0

                # Expansion and evaluation
                if not node.is_expanded:
                    t0 = time.perf_counter() if pf else 0
                    value = self._expand_node(node)
                    if pf:
                        pf['expansion_total'] += time.perf_counter() - t0
                        pf['expansions'] += 1
                else:
                    value = self._get_terminal_value(node)

                # Backpropagation
                t0 = time.perf_counter() if pf else 0
                self._backpropagate(node, value)
                if pf:
                    pf['backprop_total'] += time.perf_counter() - t0

                max_depth = max(max_depth, depth)
                total_depth += depth
                sims_done += 1

        # Compute visit distribution
        visit_policy, best_move = self._get_visit_policy(root)

        stats = {
            'avg_depth': total_depth / sims_done if sims_done > 0 else 0,
            'max_depth': max_depth,
            'num_simulations': sims_done,
            'inherited_root_n': inherited_root_n,
            'budget_new_sims': budget_new_sims,
            'target_new_sims': target_new_sims,
            'min_fresh_sims': self.min_fresh_sims,
            'reused_root': bool(reused_root),
            'floor_fired': bool(floor_fired),
        }

        if pf:
            stats['profiling'] = pf

        return visit_policy, best_move, stats

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with highest PUCT score, accounting for virtual losses.

        Virtual loss is implemented as a coherent, single-mechanism correction:
        we pretend each in-flight evaluation returned a win for the child's side.
        This inflates BOTH N (shrinking the exploration bonus) AND W (raising Q
        from the child's perspective, which lowers -Q from the parent's perspective),
        making the child less attractive to re-select within the same batch.

        NOTE: ``self.c_virtual_loss`` is a dead parameter for this function - it
        no longer appears in the UCB formula. The virtual loss now enters solely
        through ``effective_N`` and ``effective_W``. The parameter is preserved
        in the constructor for backward compatibility but has no effect here.
        """
        best_score = -float('inf')
        best_child = None

        sqrt_parent_n = math.sqrt(node.N + 1)

        for action_idx, child in node.children.items():
            # Virtual loss: pretend in-flight evaluations already returned a
            # win for the child's side. This inflates N (shrinking exploration
            # bonus) AND inflates W (raising Q from the child's perspective,
            # which lowers -Q from the parent's perspective, making the child
            # less attractive to select again).
            effective_N = child.N + child.virtual_loss
            effective_W = child.W + child.virtual_loss
            effective_Q = effective_W / effective_N if effective_N > 0 else 0.0

            # PUCT formula (standard AlphaZero style - no separate penalty term).
            # -effective_Q: Q is stored from the child's side-to-move perspective,
            # so we negate it for the parent's selection decision.
            ucb = -effective_Q + self.c_puct * child.P * sqrt_parent_n / (1 + effective_N)

            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child

    def _collect_batch(self, root: MCTSNode, batch_size: int,
                       excluded_ids: Optional[set] = None) -> List[Tuple[MCTSNode, int]]:
        """Run batch_size selections, applying virtual loss along each path.

        Each selection traverses from root to a leaf, incrementing virtual_loss
        by 1 on each node visited. This discourages multiple selections from
        choosing the same path.

        **Deduplication**: If a leaf is selected for the second time within
        the same batch (before it has been evaluated and expanded), the
        duplicate is skipped and its virtual losses are rolled back. This
        prevents:
        1. Wasted network evaluations of the same position
        2. Double-expansion of the same node (``_expand_node_with_data``
           would create duplicate children on the second call)
        3. Double-backprop of values from the same leaf

        ``excluded_ids`` additionally lists ``id(node)`` values of leaves that
        are already dispatched but not yet applied (async pipelining); those
        are treated as duplicates too, so a later batch never selects a leaf
        whose evaluation is still in flight.

        If the tree is exhausted and every path leads to already-selected
        leaves, fewer than ``batch_size`` leaves may be returned. The caller
        must handle this (``search()`` uses ``len(leaf_nodes)`` for the
        simulation counter in the batched path).

        Args:
            root: Root node
            batch_size: Target number of leaf nodes to collect
            excluded_ids: Optional set of node ids currently in flight

        Returns:
            List of ``(leaf_node, depth)`` tuples where depth is the number
            of selection steps from root to the leaf (always correct,
            regardless of tree recycling). May be shorter than batch_size.
        """
        leaves = []
        selected_leaf_ids: set = set(excluded_ids) if excluded_ids else set()

        attempts = 0
        max_attempts = batch_size * 4 + 8  # guard against pathological loops

        while len(leaves) < batch_size and attempts < max_attempts:
            attempts += 1
            node = root
            path: list = []  # nodes we applied VL to (for rollback if duplicate)

            # Selection with virtual loss
            while node.is_expanded and node.children:
                # Apply virtual loss to this node before selecting child
                node.virtual_loss += 1
                path.append(node)
                node = self._select_child(node)

            # If this leaf was already selected earlier in this batch (or is
            # in flight from an earlier batch), roll back and skip.
            #  (``virtual_loss`` on the leaf itself hasn't been incremented
            #   yet, so we check ids rather than VL.)
            if id(node) in selected_leaf_ids:
                # Undo virtual losses applied to interior nodes along the path
                for n in path:
                    n.virtual_loss -= 1
                # Don't count this iteration - caller handles shorter returns
                continue

            # Apply virtual loss to the leaf and record it
            node.virtual_loss += 1
            selected_leaf_ids.add(id(node))
            # Depth = number of interior nodes traversed from root to this leaf.
            # This is always correct, unlike the cached ``node.depth`` field
            # which becomes stale after tree recycling (recycle_tree only
            # resets the promoted root's depth, not its descendants').
            leaves.append((node, len(path)))

        return leaves

    def _split_terminal_leaves(self, leaf_nodes: List[MCTSNode]):
        """Classify a batch of leaves into terminal vs expandable.

        Terminal leaves get their value computed locally (cached); expandable
        leaves need one network evaluation each.  Returns
        ``(terminal_values, expandable_indices, expandable_nodes)`` where
        ``terminal_values`` maps leaf index -> value.
        """
        terminal_values = {}
        expandable_indices = []
        expandable_nodes = []

        for i, node in enumerate(leaf_nodes):
            # ---- Terminal check: ACTUAL game-ending + max_length ----
            if node._game_over_cached is None:
                node._game_over_cached = self._is_terminal_state(node)

            if node._game_over_cached:
                terminal_values[i] = self._get_terminal_value(node)
            elif not node.is_expanded:
                expandable_indices.append(i)
                expandable_nodes.append(node)
            else:
                # Node is expanded but terminal (should not happen normally)
                terminal_values[i] = self._get_terminal_value(node)

        return terminal_values, expandable_indices, expandable_nodes

    def _apply_network_results(self, leaf_nodes, terminal_values,
                               expandable_indices, expandable_nodes,
                               policies_batch, values_batch) -> List[float]:
        """Expand leaves from network outputs; return values aligned to leaves."""
        if expandable_nodes:
            for k, node in enumerate(expandable_nodes):
                self._expand_node_with_data(node, policies_batch[k],
                                            float(values_batch[k]))
                terminal_values[expandable_indices[k]] = float(values_batch[k])

        # Reconstruct results in the original leaf order
        return [terminal_values.get(i, 0.0) for i in range(len(leaf_nodes))]

    def _evaluate_batch(self, leaf_nodes: List[MCTSNode]) -> List[float]:
        """Evaluate a batch of leaf nodes synchronously.

        Terminal nodes are those where the game is over, a threefold repetition
        has occurred, the 50-move rule has been exceeded, **or** the maximum
        game length has been reached (adjudicated according to config).

        Non-terminal, unexpanded leaves are stacked and evaluated in one
        batched network call.
        """
        terminal_values, expandable_indices, expandable_nodes = \
            self._split_terminal_leaves(leaf_nodes)

        # ---- Batched network evaluation for expandable leaves ----
        policies_batch = values_batch = None
        if expandable_nodes:
            states_list = [game_api.encode(n.board) for n in expandable_nodes]
            states_batch = np.stack(states_list, axis=0)
            policies_batch, values_batch = self.network.predictBatch(states_batch)

        return self._apply_network_results(
            leaf_nodes, terminal_values,
            expandable_indices, expandable_nodes,
            policies_batch, values_batch)

    def _expand_node(self, node: MCTSNode) -> float:
        """Expand a node using the network. Returns the value estimate.

        Terminal state is cached on the node to avoid redundant
        ``is_game_over()`` calls.
        """
        if node._game_over_cached is None:
            node._game_over_cached = self._is_terminal_state(node)
        if node._game_over_cached:
            return self._get_terminal_value(node)

        # Get network prediction
        state = game_api.encode(node.board)
        policy, value = self.network.predict(state)

        # Delegates to the optimized expansion using raw FastMove objects
        return self._expand_node_with_data(node, policy, value)

    def _is_terminal_state(self, node) -> bool:
        """Whether the node is terminal inside the search tree.

        Matches the original chess behavior: actually game over, threefold
        repetition, fifty-move rule (chess only), or max-game-length.
        """
        b = node.board
        return (game_api.is_terminal(b)
                or game_api.is_draw_claim(b)
                or game_api.is_fifty_moves(b)
                or game_api.state_ply(b) >= self.max_game_length)

    def _expand_node_with_data(self, node: MCTSNode,
                                policy: np.ndarray, value: float) -> float:
        """Expand a node using precomputed policy and value.

        The chess backend provides an optimized fast path over raw
        FastMove objects (legacy).  Other games fall back to the generic
        per-move index path, which is all the game API needs.
        """
        if not hasattr(node.board, "legal_moves_raw"):
            return self._expand_node_generic(node, policy, value)
        return self._expand_node_chess_raw(node, policy, value)

    def _expand_node_generic(self, node: MCTSNode,
                             policy: np.ndarray, value: float) -> float:
        """Generic expansion via the game API (connect4 and friends)."""
        b = node.board
        moves = list(game_api.legal_moves(b))
        if not moves:
            return 0.0

        indices = [game_api.to_index(b, m) for m in moves]
        legal = policy[np.asarray(indices, dtype=np.intp)]
        legal_sum = float(legal.sum())
        if legal_sum > 0:
            priors = legal / legal_sum
        else:
            priors = np.full(len(indices), 1.0 / len(indices),
                             dtype=np.float32)

        for move, action_idx, prior in zip(moves, indices, priors):
            child = MCTSNode.new_child(node, move, float(prior))
            node.children[action_idx] = child

        node.is_expanded = True
        return float(value)

    def _expand_node_chess_raw(self, node: MCTSNode,
                               policy: np.ndarray, value: float) -> float:
        """Expand a node using precomputed policy and value.

        Uses the raw FastMove list (``board.legal_moves_raw``) to avoid
        wrapping every legal move into a ``chess.Move`` object.  This saves
        ~20 Python object creations per expansion with no loss of information.
        """
        # ── Legal moves: raw FastMove list from Rust backend ──
        raw_moves = node.board.legal_moves_raw   # list of FastMove objects

        if not raw_moves:
            return 0.0

        # ── Compute policy indices in ONE Rust call (no per-move Python
        # loop).  Falls back to the Python path if the board lacks the
        # fast method (e.g. python-chess fallback boards).
        rust_idxs = node.board._b.legal_move_policy_indices() \
            if hasattr(node.board._b, 'legal_move_policy_indices') else None
        if rust_idxs is not None and len(rust_idxs) == len(raw_moves):
            move_indices = rust_idxs
        else:
            move_indices = []
            for raw_move in raw_moves:
                try:
                    action_idx = move_to_policy_index(raw_move, node.board)
                    move_indices.append(action_idx)
                except ValueError as e:
                    with open("encoding_failures.log", "a") as f:
                        f.write(f"FEN: {node.board.fen()}\n"
                                f"Move: {raw_move.uci()}\n"
                                f"Error: {e}\n\n")
                    move_indices.append(None)

        # ── Renormalize over ONLY the legal children, not the full 4672 --
        # vector (gather is far cheaper than building + reducing a mask).
        valid_idx = [m for m in move_indices if m is not None]
        if valid_idx:
            legal = policy[np.asarray(valid_idx, dtype=np.intp)]
            legal_sum = legal.sum()
            if legal_sum > 0:
                priors = legal / legal_sum
            else:
                # All-legal-zero policy: fall back to uniform (matches the
                # old mask/reduce path, which used mask / mask.sum()).
                priors = np.full(len(valid_idx), 1.0 / len(valid_idx),
                                 dtype=np.float32)
        else:
            priors = None

        # ── Create children, storing the raw FastMove directly ──
        k = 0
        for raw_move, action_idx in zip(raw_moves, move_indices):
            if action_idx is None:
                continue
            prior = float(priors[k]) if priors is not None else 0.0
            k += 1

            # FastMove supports .from_square, .to_square, .promotion, .uci(),
            # so the child node can use it just like a chess.Move.
            child = MCTSNode.new_child(node, raw_move, prior)
            node.children[action_idx] = child

        if not node.children:
            print(f"WARNING: No children created for node with {len(raw_moves)} legal moves")

        node.is_expanded = True
        return float(value)

    def _get_terminal_value(self, node):
        """Return the terminal value of a node.

        Three cases:
          1. The game is ACTUALLY over (checkmate, stalemate,
             insufficient material) — use board.result().
          2. The game reached max_game_length — adjudicate by
             material (configurable) or return 0.0 (draw).
          3. The game is claimably drawn (threefold repetition,
             fifty-move rule) — return 0.0.
        """
        if node._terminal_value_cached is not None:
            return node._terminal_value_cached

        # ── Case 1: actually over (checkmate, stalemate, etc.) ──
        tv = game_api.terminal_value(node.board)
        if tv is not None:
            node._terminal_value_cached = tv
            return tv

        # ── Case 2: hit max game length (adjudicate) ──
        if game_api.state_ply(node.board) >= self.max_game_length:
            if self.adjudicate_material:
                # ``adjudicate_value`` returns a WHITE-perspective material
                # advantage (see games/chess.py), but the tree convention
                # (terminal_value / _backpropagate) is side-to-move.  Convert
                # so a Black-to-move node does not get a sign-flipped value.
                val = game_api.adjudicate_value(
                    node.board,
                    piece_values=self.piece_values,
                    graded=self.adjudicate_graded,
                    scaling=self.adjudicate_scaling
                )
                val *= game_api.player_sign(node.board)
            else:
                val = 0.0
            node._terminal_value_cached = val
            return val

        # ── Case 3: claimable draws (repetition, 50-move) ──
        # terminal_value() returns None for these, so they fall through to
        # here.  0.0 is the correct value.
        node._terminal_value_cached = 0.0
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
        """Compute the depth of a node from root.

        Uses the cached ``depth`` field (O(1)) when available.
        Falls back to walking up the tree for nodes created before
        the depth field was added.
        """
        # Fast path: use cached depth
        if hasattr(node, 'depth'):
            return node.depth
        # Fallback: walk up the tree
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
            child.P = (1 - self.dirichlet_epsilon) * child.P_orig + \
                      self.dirichlet_epsilon * noise[i]

    def _seed_mate_children(self, root: MCTSNode) -> None:
        """Give every root child that delivers checkmate one terminal visit.

        Runs once per root, after root expansion and before the normal search.
        No network call is made: the mating child is known to be terminal, so
        its side-to-move value is -1 (the mover scored +1).  Because this uses
        ordinary backpropagation, the matching root visit/value is updated too.

        This is only a visit-count/value nudge.  It does not touch the visit
        policy, the selected move, or temperature sampling; those still come
        from the normal search distribution.
        """
        if not self.mate_seeding or not game_api.is_game("chess"):
            return
        if root._mate_seeded:
            return
        root._mate_seeded = True

        for child in root.children.values():
            if not child.board.is_checkmate():
                continue
            # Terminal values are stored from the child's side-to-move
            # perspective, so a checkmated child is -1; the parent (the mover)
            # receives +1 as the value flips sign on the way up.
            child._game_over_cached = True
            child._terminal_value_cached = -1.0
            self._backpropagate(child, -1.0)

    def _get_visit_policy(self, root: MCTSNode) -> Tuple[np.ndarray, chess.Move]:
        """Compute visit count distribution and select move.

        Uses temperature-based selection:
        - If max visit count > threshold, select greedily
        - Otherwise, sample proportionally to visit counts^temperature

        Returns:
            visit_policy: (4672,) normalized visit distribution
            best_move: Selected move (most visited)
        """
        visit_policy = np.zeros(game_api.num_actions(), dtype=np.float32)

        if not root.children:
            # No children -- return empty
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

        # Find the move with most visits for greedy selection
        best_idx = int(np.argmax(visit_policy))
        best_move = game_api.index_to_move(root.board, best_idx)

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
                    'move': game_api.move_str(child.move) if child.move else None,
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
        visit_policy = np.zeros(game_api.num_actions(), dtype=np.float32)

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
            move = game_api.index_to_move(root.board, best_idx)
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
            move = game_api.index_to_move(root.board, chosen_idx)

            # Also store the full normalized visit distribution
            for idx in indices:
                visit_policy[idx] = visit_counts[idx] / total

        return visit_policy, move