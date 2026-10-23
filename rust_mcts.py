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
import weakref

import numpy as np
import chess

import game_api
from encoding import board_to_tensor, move_to_policy_index


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

    __slots__ = ('rs_id', '_eng_ref', '_children', 'mate_seeded')

    def __init__(self, eng, rs_id):
        # Weak back-reference: the engine owns its _RNodes, so a strong ref
        # here would form a cycle and stop the engine (and its arena) from
        # being freed promptly between games.
        self._eng_ref = weakref.ref(eng)
        self.rs_id = rs_id
        self._children = None
        self.mate_seeded = False

    @property
    def _eng(self):
        return self._eng_ref()

    @property
    def children(self):
        if self._children is None:
            eng = self._eng
            eng._use()          # select the owning engine's thread-local arena
            self._children = eng._make_children(self.rs_id)
        return self._children

    @property
    def board(self):
        eng = self._eng
        eng._use()
        return eng.board_of(self.rs_id)

    @property
    def is_expanded(self):
        eng = self._eng
        eng._use()
        return bool(eng._lib.mcts_rs_is_expanded(self.rs_id))


class RustMCTS:
    def __init__(self, network, num_simulations=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25, batch_size=1,
                 c_virtual_loss=0.5, max_game_length=150,
                 adjudicate_material=True, piece_values=None,
                 adjudicate_graded=True, adjudicate_scaling=9.0,
                 mate_seeding=False, min_fresh_sims=64,
                 inference_pipeline=1, dll_path=None, **kwargs):
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
        self.mate_seeding = mate_seeding
        self.min_fresh_sims = max(0, int(min_fresh_sims))
        self.inference_pipeline = max(1, int(inference_pipeline))
        self.profile = False

        self._lib = ctypes.CDLL(dll_path or _DEFAULT_DLL)
        self._bind()

        # Per-engine arena handle (allows several engines per process/thread).
        self._h = self._lib.mcts_rs_new_arena()

        # Per-search Python-side caches (bounded per game).
        self._boards = {}     # rs_id -> chess.Board
        self._moves = {}      # rs_id -> move object (FastMove / chess.Move)
        self._rnodes = {}     # rs_id -> _RNode
        self._terminal = {}   # rs_id -> bool
        self._tvalue = {}     # rs_id -> float

    # ── FFI ─────────────────────────────────────────────────────────────────
    def _bind(self):
        L = self._lib
        ci, cf, cd, cll, csz = (ctypes.c_int, ctypes.c_float, ctypes.c_double,
                                ctypes.c_longlong, ctypes.c_size_t)
        pi, pf = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
        L.mcts_rs_init()
        L.mcts_rs_new_arena.restype = ci
        L.mcts_rs_use_arena.argtypes = [ci]
        L.mcts_rs_use_arena.restype = None
        L.mcts_rs_free_arena.argtypes = [ci]
        L.mcts_rs_free_arena.restype = None
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
        L.mcts_rs_select_batch_ex.argtypes = [ci, cf, ci, pi, csz, pi, pi, csz]
        L.mcts_rs_select_batch_ex.restype = ci
        L.mcts_rs_backprop.argtypes = [ci, cd]
        L.mcts_rs_get_n.argtypes = [ci]
        L.mcts_rs_get_n.restype = cll
        L.mcts_rs_get_w.argtypes = [ci]
        L.mcts_rs_get_w.restype = cd
        # Batched visit-count read: one FFI call returns N for all children of
        # a node (replaces one _N() call per child).
        L.mcts_rs_get_n_batch.argtypes = [ci, ctypes.POINTER(ctypes.c_int64), csz]
        L.mcts_rs_get_n_batch.restype = ci
        L.mcts_rs_get_policy.argtypes = [ci]
        L.mcts_rs_get_policy.restype = cf
        L.mcts_rs_get_policy_orig.argtypes = [ci]
        L.mcts_rs_get_policy_orig.restype = cf
        L.mcts_rs_set_policy.argtypes = [ci, cf]
        L.mcts_rs_set_policy.restype = None
        L.mcts_rs_recycle.argtypes = [ci]
        L.mcts_rs_recycle.restype = None

    def _use(self):
        """Select this engine's arena on the calling thread."""
        self._lib.mcts_rs_use_arena(self._h)

    def close(self):
        """Release this engine's arena back to the pool."""
        if getattr(self, '_h', None) is not None:
            try:
                self._lib.mcts_rs_free_arena(self._h)
            except Exception:
                pass
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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

    def _N_batch(self, root):
        """Visit counts for all children of ``root`` in a single FFI call.

        Returns {action: N} — one FFI call instead of one per child.
        """
        children = root.children
        if not children:
            return {}
        n_children = len(children)
        out = (ctypes.c_int64 * n_children)()
        got = self._lib.mcts_rs_get_n_batch(root.rs_id, out, n_children)
        result = {}
        child_items = list(children.items())
        for i in range(got):
            result[child_items[i][0]] = int(out[i])
        return result

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
        self._use()
        rs = self._lib.mcts_rs_root()
        self._boards[rs] = board.copy()
        return self._rnode(rs)

    def recycle_tree(self, root, move):
        self._use()
        child = self._child_for_move(root, move)
        if child is None:
            return None
        self._lib.mcts_rs_recycle(child.rs_id)
        # Materialise the promoted node's board while the parent link is still
        # intact (board_of walks parents).
        self.board_of(child.rs_id)
        child.mate_seeded = False
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
        self._use()
        if not root.is_expanded:
            self._expand_node(root.rs_id)

        inherited_root_n = self._N(root.rs_id)

        self._seed_mate_children(root)

        self._add_dirichlet_noise(root)

        reused_root = inherited_root_n > 0
        budget_new_sims = max(0, self.num_simulations - self._N(root.rs_id))
        target_new = (max(budget_new_sims, self.min_fresh_sims)
                      if reused_root else budget_new_sims)
        floor_fired = reused_root and budget_new_sims < self.min_fresh_sims
        use_pipeline = (self.inference_pipeline > 1
                        and hasattr(self.network, 'predict_batch_async')
                        and hasattr(self.network, 'wait_result'))

        if self.batch_size > 1 and use_pipeline:
            sims, total_depth, max_depth = self._run_pipelined(root.rs_id, target_new)
        else:
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
            'inherited_root_n': inherited_root_n,
            'budget_new_sims': budget_new_sims,
            'target_new_sims': target_new,
            'min_fresh_sims': self.min_fresh_sims,
            'reused_root': bool(reused_root),
            'floor_fired': bool(floor_fired),
        }
        return visit_policy, best_move, stats

    def _run_pipelined(self, root_id, target_new):
        """Async-inference batched search (mirrors MCTS's pipelined mode).

        Keeps up to ``inference_pipeline`` evaluation batches in flight so tree
        work (selection + backprop of older batches) overlaps the network
        round-trip.  Results are applied strictly in submission order (FIFO),
        which keeps virtual-loss bookkeeping consistent with collection order.
        """
        from collections import deque
        P = max(1, self.inference_pipeline)
        # Gathered-resolution mode: request server-side packing of ONLY the
        # legal priors (identical search semantics, ~100x smaller download).
        gather_api = hasattr(self.network, 'predict_batch_gathered_async')
        inflight = deque()      # (req_id, leaves, tv, ex_idx, ex_nodes, ids,
                                #  targets or None)
        outstanding = 0         # leaves dispatched but not yet applied
        sims = 0
        total_depth = 0
        max_depth = 0

        while True:
            # Top the pipeline up.
            while len(inflight) < P and sims + outstanding < target_new:
                bs = min(self.batch_size, target_new - sims - outstanding)
                excluded = set()
                for entry in inflight:
                    excluded.update(entry[5])
                leaves = self._batch_select(root_id, bs, exclude=excluded)
                if not leaves:
                    break

                tv, ex_idx, ex_nodes = self._split_leaves(leaves)
                targets = None
                if ex_nodes:
                    states = np.stack(
                        [board_to_tensor(self.board_of(rs)) for rs in ex_nodes],
                        axis=0)
                    if gather_api:
                        targets = [self._legal_targets(rs) for rs in ex_nodes]
                        legal_indices = [
                            np.asarray(vi, dtype=np.int64)
                            for _, vi in targets
                        ]
                        req_id = self.network.predict_batch_gathered_async(
                            states, legal_indices)
                    else:
                        req_id = self.network.predict_batch_async(states)
                else:
                    req_id = None
                ids = {rs for rs, _d in leaves}
                inflight.append(
                    (req_id, leaves, tv, ex_idx, ex_nodes, ids, targets))
                outstanding += len(leaves)

            if not inflight:
                break

            req_id, leaves, tv, ex_idx, ex_nodes, _ids, targets \
                = inflight.popleft()
            outstanding -= len(leaves)

            if req_id is not None:
                policies, vals = self.network.wait_result(req_id)
                if targets is not None:
                    values = self._apply_network_results_gathered(
                        leaves, tv, ex_idx, ex_nodes, policies, targets, vals)
                else:
                    values = self._apply_network_results(
                        leaves, tv, ex_idx, ex_nodes, policies, vals)
            else:
                values = [tv.get(i, 0.0) for i in range(len(leaves))]

            for (rs, depth), value in zip(leaves, values):
                self._lib.mcts_rs_backprop(rs, float(value))
                if depth > max_depth:
                    max_depth = depth
                total_depth += depth
            sims += len(leaves)

        return sims, total_depth, max_depth

    def _batch_select(self, rs_id, batch_size, exclude=None):
        arr = (ctypes.c_int * batch_size)()
        depths = (ctypes.c_int * batch_size)()
        if exclude:
            ex = (ctypes.c_int * len(exclude))(*exclude)
            n = self._lib.mcts_rs_select_batch_ex(
                rs_id, self.c_puct, batch_size, arr, batch_size, depths,
                ex, len(exclude))
        else:
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

    def _legal_targets(self, rs_id):
        """(valid_moves, valid_idx) for a node's legal moves.

        ``valid_idx`` are action indices in policy space; index-None moves
        (not representable in the policy layout) are filtered out.  Shared
        by the full-policy and gathered-response expansion paths.
        """
        board = self.board_of(rs_id)
        raw = board.legal_moves_raw
        if not raw:
            return [], []
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
        return valid_moves, valid_idx

    def _expand_with_data(self, rs_id, policy, value):
        valid_moves, valid_idx = self._legal_targets(rs_id)

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

    def _expand_with_priors(self, rs_id, valid_moves, valid_idx, priors,
                            value):
        """Expand a node from PRE-GATHERED legal priors (gathered response).

        ``priors`` must align with ``valid_idx`` (the server packs them in
        the request's legal-index order).  Renormalization/collapse rules
        mirror ``_expand_with_data``.
        """
        if not valid_idx:
            self._lib.mcts_rs_expand_batch(rs_id, None, None, 0, None)
            return 0.0

        priors = np.asarray(priors, dtype=np.float32)
        if len(priors) != len(valid_idx):
            raise RuntimeError(
                f"gathered response width {len(priors)} != legal count "
                f"{len(valid_idx)} for node {rs_id}"
            )
        s = float(priors.sum())
        if s > 0:
            priors = np.ascontiguousarray(priors / s, dtype=np.float32)
        else:
            priors = np.full(len(valid_idx), 1.0 / len(valid_idx),
                             dtype=np.float32)
        idx_arr = np.asarray(valid_idx, dtype=np.int32)
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

    def _split_leaves(self, leaves):
        """Classify selected leaves into terminal vs expandable.

        Mirrors ``mcts.MCTS._split_terminal_leaves``.  Returns
        ``(terminal_values, expandable_idx, expandable_rs)`` where
        ``terminal_values`` maps a leaf index to its value.
        """
        tv = {}
        ex_idx = []
        ex_rs = []
        for i, (rs, _d) in enumerate(leaves):
            if self._is_terminal(rs):
                tv[i] = self._get_terminal_value(rs)
            elif not self._lib.mcts_rs_is_expanded(rs):
                ex_idx.append(i)
                ex_rs.append(rs)
            else:
                tv[i] = self._get_terminal_value(rs)
        return tv, ex_idx, ex_rs

    def _apply_network_results(self, leaves, tv, ex_idx, ex_rs, policies, vals):
        """Expand leaves with network outputs; return values in leaf order."""
        policies = np.asarray(policies, dtype=np.float32)
        for k, rs in enumerate(ex_rs):
            self._expand_with_data(rs, policies[k], float(vals[k]))
            tv[ex_idx[k]] = float(vals[k])
        return [tv.get(i, 0.0) for i in range(len(leaves))]

    def _apply_network_results_gathered(self, leaves, tv, ex_idx, ex_nodes,
                                        gathered, targets, vals):
        """Expand leaves from gathered priors; return values in leaf order.

        ``gathered`` is the list of per-sample prior arrays returned by the
        gathered API; ``targets`` is the (valid_moves, valid_idx) pair each
        sample was submitted with.
        """
        for k, rs in enumerate(ex_nodes):
            valid_moves, valid_idx = targets[k]
            self._expand_with_priors(rs, valid_moves, valid_idx,
                                     gathered[k], float(vals[k]))
            tv[ex_idx[k]] = float(vals[k])
        return [tv.get(i, 0.0) for i in range(len(leaves))]

    def _evaluate_batch(self, leaf_rs_ids):
        leaves = [(rs, 0) for rs in leaf_rs_ids]
        tv, ex_idx, ex_rs = self._split_leaves(leaves)
        if ex_rs:
            states = np.stack(
                [board_to_tensor(self.board_of(rs)) for rs in ex_rs], axis=0)
            policies, vals = self._predict_batch_network(states)
            return self._apply_network_results(
                leaves, tv, ex_idx, ex_rs, policies, vals)
        return [tv.get(i, 0.0) for i in range(len(leaves))]

    def _predict_batch_network(self, states):
        """Run a batched forward pass, supporting both the CPU AlphaZeroNet
        (``predictBatch``) and the GPU InferenceClient (``predict_batch``)."""
        net = self.network
        if hasattr(net, 'predict_batch'):
            return net.predict_batch(states)
        return net.predictBatch(states)

    def _add_dirichlet_noise(self, root):
        if self.dirichlet_alpha <= 0 or self.dirichlet_epsilon <= 0:
            return
        children = root.children
        if not children:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        for i, child in enumerate(children.values()):
            orig = float(self._lib.mcts_rs_get_policy_orig(child.rs_id))
            newp = (1 - self.dirichlet_epsilon) * orig + self.dirichlet_epsilon * noise[i]
            self._lib.mcts_rs_set_policy(child.rs_id, newp)

    # ── mate seeding / move selection / stats (reads from arena) ────────────
    def _seed_mate_children(self, root):
        """Give every root child that delivers checkmate one terminal visit.

        Mirrors ``mcts.MCTS._seed_mate_children``.  Runs once per root after
        expansion and before search, makes no network call, and only updates
        visit/value counts.  The normal visit policy, move choice, and
        temperature sampling are left untouched.
        """
        if not self.mate_seeding or not game_api.is_game("chess"):
            return
        if root.mate_seeded:
            return
        root.mate_seeded = True

        for child in root.children.values():
            if not self.board_of(child.rs_id).is_checkmate():
                continue
            # The node is terminal from the child's side-to-move perspective
            # (-1); backprop flips the sign so the mover's root receives +1.
            self._terminal[child.rs_id] = True
            self._tvalue[child.rs_id] = -1.0
            self._lib.mcts_rs_backprop(child.rs_id, -1.0)

    def _get_visit_policy(self, root):
        children = root.children
        visit_policy = np.zeros(game_api.num_actions(), dtype=np.float32)
        if not children:
            return visit_policy, None

        counts = self._N_batch(root)
        total = 0
        best_idx = None
        best_n = -1
        for a, n in counts.items():
            visit_policy[a] = n
            total += n
            if n > best_n:
                best_n = n
                best_idx = a
        if total == 0:
            return visit_policy, None
        return visit_policy / total, self._moves.get(children[best_idx].rs_id)

    def select_move_with_temperature(self, root, temperature):
        """Same contract as MCTS.select_move_with_temperature.

        - Greedy (``temperature < 1e-8``): one-hot on the most-visited child.
        - Otherwise: sample proportionally to ``N ** (1/T)`` for the chosen
          move, but return the *full normalised visit distribution* as the
          policy target (matches MCTS).
        """
        self._use()
        visit_policy = np.zeros(game_api.num_actions(), dtype=np.float32)
        children = root.children
        if not children:
            return visit_policy, None

        counts = self._N_batch(root)
        total = sum(counts.values())
        if total == 0:
            return visit_policy, None

        if temperature < 1e-8:
            best_idx = max(counts, key=counts.get)
            visit_policy[best_idx] = 1.0
            return visit_policy, self._moves.get(children[best_idx].rs_id)

        idxs = list(counts.keys())
        c = np.array([counts[i] for i in idxs], dtype=np.float64)
        probs = c ** (1.0 / temperature)
        probs = probs / probs.sum()
        chosen_idx = idxs[int(np.random.choice(len(idxs), p=probs))]
        for idx in idxs:
            visit_policy[idx] = counts[idx] / total
        return visit_policy, self._moves.get(children[chosen_idx].rs_id)

    def get_root_child_stats(self, root):
        self._use()
        stats = []
        counts = self._N_batch(root)
        for a, child in root.children.items():
            n = counts.get(a, 0)
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
