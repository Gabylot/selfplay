"""Rust-backed MCTS engine.

The Rust arena owns *all* per-node search state (``N``, ``W``, ``Q``, ``policy``,
``virtual_loss``, parent link, move action) and runs the hot loops: PUCT
selection / batch collection / back-propagation.  Unlike the earlier driver,
Python does **not** mirror every child as an ``MCTSNode`` -- that object churn
(1.5M objects per self-play pass) was the dominant cost.  Python now keeps only:

  - a board cache (lazily materialised, one copy+push per visited node, exactly
    as ``mcts.MCTS`` does via ``MCTSNode.board``),
  - the move object per node (needed to materialise boards),
  - terminal/expanded caches.

Everything numeric lives in the Rust arena and is created with a single batched
``mcts_rs_expand_batch`` FFI call per expansion (previously one locked FFI call
*per child*).

This mirrors the batched path of ``mcts.MCTS.search`` exactly, so games played
with ``RustMCTS`` should match the pure-Python engine (verified by
``tests/test_engine_equivalence.py`` and ``tests/test_mcts_rs.py``).

Public driving surface (subset of ``mcts.MCTS``):
``get_root``, ``search``, ``recycle_tree``, ``select_move_with_temperature``,
``get_root_child_stats``.
"""

import os
import ctypes

import numpy as np
import chess

import game_api
from encoding import (NUM_ACTIONS, board_to_tensor, move_to_policy_index)


_DEFAULT_DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'mcts_rs', 'target', 'release', 'mcts_rs.dll')

_NONE = -1
# Upper bound on legal chess moves in any position (max is 218).
_MAX_MOVES = 256


class _RNode:
    """Thin handle for the search root and its exposed children.

    Only the root (and its direct children, for stats/noise/recycle) ever need
    a Python handle; the arena owns everything else.
    """

    __slots__ = ('rs_id', '_eng', '_children')

    def __init__(self, eng, rs_id):
        self._eng = eng
        self.rs_id = rs_id
        self._children = None

    @property
    def children(self):
        if self._children is None:
            self._children = self._eng._make_children(self.rs_id)
        return self._children

    @property
    def board(self):
        return self._eng.board_of(self.rs_id)

    @property
    def is_expanded(self):
        return bool(self._eng._lib.mcts_rs_is_expanded(self.rs_id))


class RustMCTS:
    def __init__(self, network, num_simulations=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25, batch_size=1,
                 c_virtual_loss=0.5, max_game_length=150,
                 adjudicate_material=True, piece_values=None,
                 adjudicate_graded=True, adjudicate_scaling=9.0,
                 force_mate_in_one=True, inference_pipeline=1,
                 dll_path=None, **kwargs):
        self.network = network
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.batch_size = int(batch_size)
        self.max_game_length = int(max_game_length)
        self.adjudicate_material = bool(adjudicate_material)
        self.adjudicate_graded = bool(adjudicate_graded)
        self.adjudicate_scaling = float(adjudicate_scaling)
        self.piece_values = piece_values or {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
        # Accepted for drop-in compatibility with mcts.MCTS (unused here).
        self.c_virtual_loss = c_virtual_loss
        self.force_mate_in_one = force_mate_in_one
        self.inference_pipeline = max(1, int(inference_pipeline))
        self.profile = False

        self._lib = ctypes.CDLL(dll_path or _DEFAULT_DLL)
        self._bind()

        # Per-search Python-side caches (cleared on reset()).
        self._boards = {}     # rs_id -> chess.Board
        self._moves = {}      # rs_id -> move object (FastMove / chess.Move)
        self._rnodes = {}     # rs_id -> _RNode
        self._terminal = {}   # rs_id -> bool
        self._tvalue = {}     # rs_id -> float

        self._reset()

    # ── FFI ─────────────────────────────────────────────────────────────────
    def _bind(self):
        L = self._lib
        ci, cf, cd, cll, csz = (ctypes.c_int, ctypes.c_float, ctypes.c_double,
                                ctypes.c_longlong, ctypes.c_size_t)
        pi, pf = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
        L.mcts_rs_init()
        L.mcts_rs_root.restype = ci
        L.mcts_rs_reset.restype = None
        L.mcts_rs_expand_batch.argtypes = [ci, pi, pf, csz, pi]
        L.mcts_rs_expand_batch.restype = ci
        L.mcts_rs_is_expanded.argtypes = [ci]
        L.mcts_rs_is_expanded.restype = ctypes.c_bool
        L.mcts_rs_get_parent.argtypes = [ci]
        L.mcts_rs_get_parent.restype = ci
        L.mcts_rs_get_action.argtypes = [ci]
        L.mcts_rs_get_action.restype = ci
        L.mcts_rs_children2.argtypes = [ci, pi, pi, csz]
        L.mcts_rs_children2.restype = ci
        L.mcts_rs_select_batch.argtypes = [ci, cf, ci, pi, csz, pi]
        L.mcts_rs_select_batch.restype = ci
        L.mcts_rs_backprop.argtypes = [ci, cd]
        L.mcts_rs_get_n.argtypes = [ci]
        L.mcts_rs_get_n.restype = cll
        L.mcts_rs_get_w.argtypes = [ci]
        L.mcts_rs_get_w.restype = cd
        L.mcts_rs_get_policy.argtypes = [ci]
        L.mcts_rs_get_policy.restype = cf
        L.mcts_rs_set_policy.argtypes = [ci, cf]
        L.mcts_rs_set_policy.restype = None
        L.mcts_rs_recycle.argtypes = [ci]
        L.mcts_rs_recycle.restype = None

    def _reset(self):
        self._lib.mcts_rs_reset()
        self._boards.clear()
        self._moves.clear()
        self._rnodes.clear()
        self._terminal.clear()
        self._tvalue.clear()

    # ── board / move lookups ────────────────────────────────────────────────
    def board_of(self, rs_id):
        """Return (and cache) the board for a node, walking from the nearest
        materialised ancestor (mirrors ``MCTSNode.board``)."""
        b = self._boards.get(rs_id)
        if b is not None:
            return b
        path = []
        cur = rs_id
        while cur not in self._boards:
            p = self._lib.mcts_rs_get_parent(cur)
            if p < 0:
                break
            path.append(cur)
            cur = p
        base = self._boards[cur]
        for nid in reversed(path):
            nb = base.copy(stack=True)
            nb.push(self._moves[nid])
            self._boards[nid] = nb
            base = nb
        return self._boards[rs_id]

    def _rnode(self, rs_id):
        n = self._rnodes.get(rs_id)
        if n is None:
            n = _RNode(self, rs_id)
            self._rnodes[rs_id] = n
        return n

    def _make_children(self, rs_id):
        ids = (ctypes.c_int * _MAX_MOVES)()
        acts = (ctypes.c_int * _MAX_MOVES)()
        n = self._lib.mcts_rs_children2(rs_id, ids, acts, _MAX_MOVES)
        return {int(acts[i]): self._rnode(int(ids[i])) for i in range(n)}

    def _N(self, rs_id):
        return int(self._lib.mcts_rs_get_n(rs_id))

    def _W(self, rs_id):
        return float(self._lib.mcts_rs_get_w(rs_id))

    # ── terminal handling (matches mcts.MCTS) ───────────────────────────────
    def _is_terminal(self, rs_id):
        t = self._terminal.get(rs_id)
        if t is not None:
            return t
        b = self.board_of(rs_id)
        t = (game_api.is_terminal(b)
             or game_api.is_draw_claim(b)
             or game_api.is_fifty_moves(b)
             or game_api.state_ply(b) >= self.max_game_length)
        self._terminal[rs_id] = t
        return t

    def _get_terminal_value(self, rs_id):
        v = self._tvalue.get(rs_id)
        if v is not None:
            return v
        b = self.board_of(rs_id)
        tv = game_api.terminal_value(b)
        if tv is not None:
            self._tvalue[rs_id] = tv
            return tv
        if game_api.state_ply(b) >= self.max_game_length:
            if self.adjudicate_material:
                val = game_api.adjudicate_value(
                    b, piece_values=self.piece_values,
                    graded=self.adjudicate_graded, scaling=self.adjudicate_scaling)
                val *= game_api.player_sign(b)
            else:
                val = 0.0
            self._tvalue[rs_id] = val
            return val
        self._tvalue[rs_id] = 0.0
        return 0.0

    # ── public API (mirrors MCTS) ───────────────────────────────────────────
    def get_root(self, board):
        rs = self._lib.mcts_rs_root()
        self._boards[rs] = board.copy()
        return self._rnode(rs)

    def recycle_tree(self, root, move):
        child = self._child_for_move(root, move)
        if child is None:
            return None
        self._lib.mcts_rs_recycle(child.rs_id)
        # Materialise the promoted node's board while the parent link is still
        # intact (board_of walks parents).
        self.board_of(child.rs_id)
        return child

    def _child_for_move(self, root, move):
        try:
            action = move_to_policy_index(move, root.board)
        except (ValueError, AttributeError):
            action = None
        if action is not None:
            child = root.children.get(action)
            if child is not None:
                return child
        # Fallback: compare from/to/promotion.
        fs, ts, pr = move.from_square, move.to_square, move.promotion
        for child in root.children.values():
            m = self._moves.get(child.rs_id)
            if m is not None and (m.from_square, m.to_square, m.promotion) == (fs, ts, pr):
                return child
        return None

    def search(self, root):
        """Run MCTS from root. Returns (visit_policy, best_move, stats)."""
        if not root.is_expanded:
            self._expand_node(root.rs_id)

        self._add_dirichlet_noise(root)

        target_new = max(0, self.num_simulations - self._N(root.rs_id))
        sims = 0
        total_depth = 0
        max_depth = 0
        while sims < target_new:
            bs = min(self.batch_size, target_new - sims)
            leaves = self._batch_select(root.rs_id, bs)
            if not leaves:
                break
            values = self._evaluate_batch([rs for rs, _d in leaves])
            for (rs, depth), value in zip(leaves, values):
                self._lib.mcts_rs_backprop(rs, float(value))
                if depth > max_depth:
                    max_depth = depth
                total_depth += depth
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
        return [(int(arr[i]), int(depths[i])) for i in range(n)]

    # ── expansion / evaluation ──────────────────────────────────────────────
    def _expand_node(self, rs_id):
        if self._is_terminal(rs_id):
            return self._get_terminal_value(rs_id)
        state = board_to_tensor(self.board_of(rs_id))
        policy, value = self.network.predict(state)
        self._expand_with_data(rs_id, np.asarray(policy, dtype=np.float32),
                               float(value))
        return float(value)

    def _expand_with_data(self, rs_id, policy, value):
        board = self.board_of(rs_id)
        raw = board.legal_moves_raw
        if not raw:
            self._lib.mcts_rs_expand_batch(
                rs_id, None, None, 0, None)
            return 0.0

        rust_idxs = (board._b.legal_move_policy_indices()
                     if hasattr(board._b, 'legal_move_policy_indices') else None)

        valid_moves = []
        valid_idx = []
        if rust_idxs is not None and len(rust_idxs) == len(raw):
            for m, a in zip(raw, rust_idxs):
                if a is None:
                    continue
                valid_moves.append(m)
                valid_idx.append(a)
        else:
            for m in raw:
                try:
                    valid_moves.append(m)
                    valid_idx.append(move_to_policy_index(m, board))
                except ValueError:
                    continue

        if not valid_idx:
            self._lib.mcts_rs_expand_batch(rs_id, None, None, 0, None)
            return 0.0

        idx_arr = np.asarray(valid_idx, dtype=np.int32)
        legal = policy[idx_arr]
        s = float(legal.sum())
        if s > 0:
            priors = np.ascontiguousarray(legal / s, dtype=np.float32)
        else:
            priors = np.full(len(valid_idx), 1.0 / len(valid_idx), dtype=np.float32)
        out = np.empty(len(valid_idx), dtype=np.int32)

        self._lib.mcts_rs_expand_batch(
            rs_id,
            idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            priors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(valid_idx),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

        for k, rid in enumerate(out.tolist()):
            self._moves[rid] = valid_moves[k]
        return 0.0

    def _evaluate_batch(self, leaf_rs_ids):
        values = [0.0] * len(leaf_rs_ids)
        expandable = []          # (index, rs_id)
        for i, rs in enumerate(leaf_rs_ids):
            if self._is_terminal(rs):
                values[i] = self._get_terminal_value(rs)
            elif not self._lib.mcts_rs_is_expanded(rs):
                expandable.append((i, rs))
            else:
                values[i] = self._get_terminal_value(rs)

        if expandable:
            states = np.stack(
                [board_to_tensor(self.board_of(rs)) for _i, rs in expandable],
                axis=0)
            policies, vals = self.network.predictBatch(states)
            policies = np.asarray(policies, dtype=np.float32)
            for k, (i, rs) in enumerate(expandable):
                self._expand_with_data(rs, policies[k], float(vals[k]))
                values[i] = float(vals[k])

        return values

    def _add_dirichlet_noise(self, root):
        if self.dirichlet_alpha <= 0 or self.dirichlet_epsilon <= 0:
            return
        children = root.children
        if not children:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        for i, child in enumerate(children.values()):
            orig = float(self._lib.mcts_rs_get_policy(child.rs_id))
            newp = (1 - self.dirichlet_epsilon) * orig + self.dirichlet_epsilon * noise[i]
            self._lib.mcts_rs_set_policy(child.rs_id, newp)

    # ── move selection / stats (reads from arena) ───────────────────────────
    def _get_visit_policy(self, root):
        children = root.children
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if not children:
            return visit_policy, None
        total = 0
        best_idx = None
        best_n = -1
        for a, child in children.items():
            n = self._N(child.rs_id)
            visit_policy[a] = n
            total += n
            if n > best_n:
                best_n = n
                best_idx = a
        if total == 0:
            return visit_policy, None
        return visit_policy / total, self._moves.get(children[best_idx].rs_id)

    def select_move_with_temperature(self, root, temperature):
        """Same contract as MCTS.select_move_with_temperature."""
        children = root.children
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if not children:
            return visit_policy, None
        counts = {a: self._N(c.rs_id) for a, c in children.items()}
        total = sum(counts.values())
        if total == 0:
            return visit_policy, None
        if temperature < 1e-8:
            best_idx = max(counts, key=counts.get)
        else:
            idxs = list(counts.keys())
            c = np.array([counts[i] for i in idxs], dtype=np.float64)
            probs = c ** (1.0 / temperature)
            probs = probs / probs.sum()
            best_idx = idxs[int(np.random.choice(len(idxs), p=probs))]
        visit_policy[best_idx] = 1.0
        return visit_policy, self._moves.get(children[best_idx].rs_id)

    def get_root_child_stats(self, root):
        stats = []
        for a, child in root.children.items():
            n = self._N(child.rs_id)
            if n > 0:
                m = self._moves.get(child.rs_id)
                stats.append({
                    'move': m.uci() if m is not None else None,
                    'N': n,
                    'W': self._W(child.rs_id),
                    'Q': self._W(child.rs_id) / n if n else 0.0,
                    'P': float(self._lib.mcts_rs_get_policy(child.rs_id)),
                })
        return sorted(stats, key=lambda x: x['N'], reverse=True)
