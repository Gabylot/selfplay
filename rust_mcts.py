"""Rust-backed MCTS engine.

The Rust arena owns the numeric search state (``N``, ``W``, ``Q``, ``policy``,
``virtual_loss``) and runs the hot loops: PUCT selection / batch collection /
back-propagation.  Python keeps what needs chess rules and the network:

  - board materialisation and legal-move lookup (via the Rust ``fastchess`` shim)
  - ``board_to_tensor`` encoding and the GPU inference server round-trip
  - Dirichlet noise and move selection (reads stats back from the arena)

This mirrors the batched path of ``mcts.MCTS.search`` exactly, so games played
with ``RustMCTS`` should match the pure-Python engine (verified for the arena
core in ``tests/test_mcts_rs.py`` and for the full engine in
``tests/test_engine_equivalence.py``).

Public driving surface (subset of ``mcts.MCTS``):
``get_root``, ``search``, ``recycle_tree``, ``select_move_with_temperature``,
``get_root_child_stats``.
"""

import os
import ctypes

import numpy as np
import chess

from encoding import NUM_ACTIONS, move_to_policy_index, policy_index_to_move
from mcts import MCTSNode, adjudicate_by_material


_DEFAULT_DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'mcts_rs', 'target', 'release', 'mcts_rs.dll')


class RustMCTS:
    def __init__(self, network, num_simulations=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25, batch_size=1,
                 max_game_length=150, piece_values=None, dll_path=None):
        self.network = network
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.batch_size = int(batch_size)
        self.max_game_length = int(max_game_length)
        self.piece_values = piece_values or {'P':1,'N':3,'B':3,'R':5,'Q':9}
        self._node = {}          # rs_id -> MCTSNode mirror
        self._lib = ctypes.CDLL(dll_path or _DEFAULT_DLL)
        self._bind()
        self._node.clear()
        self._lib.mcts_rs_reset()
        self.profile = False

    # ── FFI ─────────────────────────────────────────────────────────────────
    def _bind(self):
        L = self._lib
        L.mcts_rs_init()
        L.mcts_rs_reset()
        L.mcts_rs_root.restype = ctypes.c_int
        L.mcts_rs_add_child.argtypes = [ctypes.c_int, ctypes.c_float]
        L.mcts_rs_add_child.restype = ctypes.c_int
        L.mcts_rs_mark_expanded.argtypes = [ctypes.c_int, ctypes.c_bool]
        L.mcts_rs_select_batch.argtypes = [
            ctypes.c_int, ctypes.c_float, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int)]
        L.mcts_rs_select_batch.restype = ctypes.c_int
        L.mcts_rs_backprop.argtypes = [ctypes.c_int, ctypes.c_double]
        L.mcts_rs_get_n.argtypes = [ctypes.c_int]
        L.mcts_rs_get_n.restype = ctypes.c_longlong
        L.mcts_rs_get_w.argtypes = [ctypes.c_int]
        L.mcts_rs_get_w.restype = ctypes.c_double
        L.mcts_rs_get_policy.argtypes = [ctypes.c_int]
        L.mcts_rs_get_policy.restype = ctypes.c_float
        L.mcts_rs_children.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                       ctypes.c_size_t]
        L.mcts_rs_children.restype = ctypes.c_int
        L.mcts_rs_recycle.argtypes = [ctypes.c_int]
        L.mcts_rs_reset()

    # ── mirror helpers ──────────────────────────────────────────────────────
    def _mk(self, rs_id, board=None, parent=None, move=None, prior=0.0):
        n = MCTSNode(board=board, parent=parent, move=move, prior=prior)
        n.rs_id = rs_id
        self._node[rs_id] = n
        return n

    def _N(self, rs_id):
        return int(self._lib.mcts_rs_get_n(rs_id))

    def _W(self, rs_id):
        return float(self._lib.mcts_rs_get_w(rs_id))

    # ── public API (mirrors MCTS) ───────────────────────────────────────────
    def get_root(self, board):
        rs = self._lib.mcts_rs_root()
        return self._mk(rs, board=board.copy())

    def recycle_tree(self, root, move):
        child = self._find_child_by_move(root, move)
        if child is None:
            return None
        child.parent = None
        if not child._board_ready:
            b = root.board.copy(stack=True)
            b.push(move)
            child._board = b
            child._board_ready = True
        self._lib.mcts_rs_recycle(child.rs_id)
        return child

    def _find_child_by_move(self, root, move):
        fs, ts, pr = move.from_square, move.to_square, move.promotion
        for child in root.children.values():
            cm = child.move
            if (cm.from_square, cm.to_square, cm.promotion) == (fs, ts, pr):
                return child
        return None

    def search(self, root):
        """Run MCTS from root. Returns (visit_policy, best_move, stats)."""
        if not root.is_expanded:
            self._expand_node(root)
        self._add_dirichlet_noise(root)

        target_new = max(0, self.num_simulations - self._N(root.rs_id))
        sims = 0
        total_depth = 0
        max_depth = 0
        while sims < target_new:
            bs = min(self.batch_size, target_new - sims)
            leaves, depths = self._batch_select(root.rs_id, bs)
            if not leaves:
                break
            values = self._evaluate_batch(leaves)
            for rs, value in zip(leaves, values):
                self._lib.mcts_rs_backprop(rs, float(value))
                max_depth = max(max_depth, depths.get(rs, 0))
                total_depth += depths.get(rs, 0)
            sims += len(leaves)

        visit_policy, best_move = self._get_visit_policy(root)
        stats = {
            'num_simulations': sims,
            'max_depth': max_depth,
            'avg_depth': total_depth / sims if sims else 0.0,
        }
        return visit_policy, best_move, stats

    def _batch_select(self, rs_id, batch_size):
        arr = (ctypes.c_int * batch_size)()
        depths = (ctypes.c_int * batch_size)()
        n = self._lib.mcts_rs_select_batch(
            rs_id, self.c_puct, batch_size, arr, batch_size, depths)
        leaves = [arr[i] for i in range(n)]
        dmap = {arr[i]: int(depths[i]) for i in range(n)}
        return leaves, dmap

    # alias for internal use
    _batch_collect = _batch_select

    def _batch_size(self, leaves):
        pass

    # ── expansion / evaluation (Python, chess + network) ───────────────────
    def _expand_node(self, node):
        from encoding import board_to_tensor
        if node._game_over_cached is None:
            node._game_over_cached = (
                node.board.is_game_over()
                or node.board.is_repetition(3)
                or node.board.is_fifty_moves()
                or node.board.ply() >= self.max_game_length)
        if node._game_over_cached:
            return self._get_terminal_value(node)
        state = board_to_tensor(node.board)
        policy, value = self.network.predict(state)
        self._expand_with_data(node, np.asarray(policy, dtype=np.float32),
                               float(value))
        return float(value)

    def _expand_with_data(self, node, policy, value):
        raw_moves = node.board.legal_moves_raw
        if not raw_moves:
            return 0.0
        move_indices = []
        for m in raw_moves:
            try:
                move_indices.append(move_to_policy_index(m, node.board))
            except ValueError:
                move_indices.append(None)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in move_indices:
            if a is not None:
                mask[a] = 1.0
        legal_policy = policy * mask
        s = legal_policy.sum()
        legal_policy = (legal_policy / s) if s > 0 else (mask / mask.sum())
        for m, a in zip(raw_moves, move_indices):
            if a is None:
                continue
            prior = float(legal_policy[a])
            rs = self._lib.mcts_rs_add_child(node.rs_id, prior)
            child = self._mk(rs, parent=node, move=m, prior=prior)
            node.children[a] = child
        self._lib.mcts_rs_mark_expanded(node.rs_id, True)
        node.is_expanded = True
        return 0.0

    def _evaluate_batch(self, leaf_rs_ids):
        from encoding import board_to_tensor
        values = {}
        expandable = []
        expandable_idx = []
        for i, rs in enumerate(leaf_rs_ids):
            node = self._node[rs]
            if node._game_over_cached is None:
                node._game_over_cached = (
                    node.board.is_game_over()
                    or node.board.is_repetition(3)
                    or node.board.is_fifty_moves()
                    or node.board.ply() >= self.max_game_length)
            if node._game_over_cached:
                values[i] = self._get_terminal_value(node)
            elif not node.is_expanded:
                expandable_idx.append(i)
                expandable.append(node)
            else:
                values[i] = self._get_terminal_value(node)

        if expandable:
            states = np.stack([board_to_tensor(n.board) for n in expandable],
                              axis=0)
            policies, vals = self.network.predictBatch(states)
            policies = np.asarray(policies, dtype=np.float32)
            for k, (idx, node) in enumerate(zip(expandable_idx, expandable)):
                self._expand_with_data(node, policies[k],
                                       float(vals[k]))
                values[idx] = float(vals[k])

        return [values[i] for i in range(len(leaf_rs_ids))]

    def _get_terminal_value(self, node):
        if node._terminal_value_cached is not None:
            return node._terminal_value_cached
        result = node.board.result()
        if result != "*":
            if result == "1-0":
                val = 1.0 if node.board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                val = -1.0 if node.board.turn == chess.WHITE else 1.0
            else:
                val = 0.0
            node._terminal_value_cached = val
            return val
        node._terminal_value_cached = 0.0
        return 0.0

    def _add_dirichlet_noise(self, root):
        if not root.children:
            return
        if self.dirichlet_alpha <= 0 or self.dirichlet_epsilon <= 0:
            return
        num_children = len(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)
        for i, child in enumerate(root.children.values()):
            orig = float(self._lib.mcts_rs_get_policy(child.rs_id))
            newp = (1 - self.dirichlet_epsilon) * orig + self.dirichlet_epsilon * noise[i]
            self._lib.mcts_rs_set_policy(child.rs_id, newp)

    # ── move selection / stats (reads from arena) ───────────────────────────
    def _get_visit_policy(self, root):
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if not root.children:
            return visit_policy, None
        total = 0
        best_idx = None
        best_n = -1
        for a, child in root.children.items():
            n = self._N(child.rs_id)
            visit_policy[a] = n
            total += n
            if n > best_n:
                best_n = n
                best_idx = a
        if total == 0:
            return visit_policy, None
        visit_probs = visit_policy / total
        best_move = policy_index_to_move(best_idx, root.board)
        return visit_probs, best_move

    def select_move_with_temperature(self, root, temperature):
        """Same contract as MCTS.select_move_with_temperature."""
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if not root.children:
            return visit_policy, None
        counts = {}
        for a, child in root.children.items():
            counts[a] = self._N(child.rs_id)
        total = sum(counts.values())
        if total == 0:
            return visit_policy, None
        if temperature < 1e-8:
            best_idx = max(counts, key=counts.get)
            visit_policy[best_idx] = 1.0
            move = policy_index_to_move(best_idx, root.board)
        else:
            idxs = list(counts.keys())
            c = np.array([counts[i] for i in idxs], dtype=np.float64)
            probs = c ** (1.0 / temperature)
            probs = probs / probs.sum()
            chosen = int(np.random.choice(len(idxs), p=probs))
            chosen_idx = idxs[chosen]
            visit_policy[chosen_idx] = 1.0
            move = policy_index_to_move(chosen_idx, root.board)
        return visit_policy, move

    def get_root_child_stats(self, root):
        stats = []
        for a, child in root.children.items():
            n = self._N(child.rs_id)
            if n > 0:
                stats.append({
                    'move': child.move.uci() if child.move else None,
                    'N': n,
                    'W': self._W(child.rs_id),
                    'Q': self._W(child.rs_id) / n if n else 0.0,
                    'P': float(self._lib.mcts_rs_get_policy(child.rs_id)),
                })
        return sorted(stats, key=lambda x: x['N'], reverse=True)