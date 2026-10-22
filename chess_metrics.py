"""Rolling chess-quality metrics computed in the MAIN process only.

Everything here is derived from data the training loop already holds per
finished self-play game:

  * the per-game ``game_info`` dict returned by the worker
    (``result``/``outcome``, ``termination``, ``length``), and
  * the in-order list of stored positions (``game_data``): each entry is
    ``(state, visit_policy, value_target, legal_mask)``.

No worker, search, training or buffer code path is touched, and nothing here
runs inside a per-move hot loop -- it is called at most once per finished game
(plus a batch of network inference every ``log_every`` games).  The network is
used only to *re-predict* values/priors for positions we already stored; the
search is never involved.

Metrics and their exact provenance:

  white share (overall / checkmate / material)   <- outcome + termination
  termination mix                                <- termination
  game length median / p10 / p90                 <- length
  masked policy entropy                          <- stored state (re-predict prior) + mask
  root KL(net prior || MCTS visit policy)        <- stored visit_policy + re-predicted prior + mask
  value error at ply 20 / 60 / 100               <- stored state + value target at that ply

The visit policy stored in the buffer is the *normalised MCTS visit
distribution* only while the self-play temperature is > 0 (the opening); later
plies (and forced mate-in-1 moves) are one-hot greedy targets.  The KL is
therefore computed only over positions whose target has >1 non-zero entry, so
it is a root-KL proxy on the high-temperature opening rather than a whole-game
average.  This is documented on the log group.
"""

from collections import Counter, deque

import numpy as np

WINDOW = 500
LOG_EVERY = 100
PHASE_PLIES = (20, 60, 100)


# ── pure aggregations over the per-game window ──────────────────────────────

def white_share(games):
    """Share of *decisive* games won by White, split by termination family."""
    def _share(pred):
        dec = [g for g in games if g['result'] != 0 and pred(g)]
        if not dec:
            return None
        return sum(1 for g in dec if g['result'] > 0) / len(dec)

    return {
        'white_share_overall': _share(lambda g: True),
        'white_share_checkmate': _share(lambda g: g['termination'] == 'checkmate'),
        'white_share_material': _share(
            lambda g: g['termination'] in ('material_white', 'material_black')),
    }


def termination_mix(games):
    """Fraction of games ending for each termination reason."""
    if not games:
        return {}
    n = len(games)
    c = Counter(g['termination'] for g in games)
    return {f'termination_share_{k}': v / n for k, v in c.items()}


def length_stats(games):
    """Median and 10th/90th percentile of game length (full moves)."""
    if not games:
        return {}
    L = np.asarray([g['length'] for g in games], dtype=np.float64)
    return {
        'length_median': float(np.median(L)),
        'length_p10': float(np.percentile(L, 10)),
        'length_p90': float(np.percentile(L, 90)),
    }


def fresh_sim_stats(games):
    """Rolling fresh-simulation diagnostics across all moves in the window.

    ``fresh_sims`` is the number of NEW simulations run for a move (after the
    reused-root visit count is subtracted and after any min_fresh_sims floor).
    ``floor_fired`` is whether the old budget
    ``num_simulations - root.N`` was below ``min_fresh_sims``.
    """
    fresh = []
    floor_moves = 0
    total_moves = 0
    for g in games:
        counts = g.get('fresh_sims') or []
        flags = g.get('floor_fired') or []
        fresh.extend(int(x) for x in counts)
        total_moves += len(counts)
        floor_moves += sum(1 for x in flags if x)
    if not fresh:
        return {}
    a = np.asarray(fresh, dtype=np.float64)
    return {
        'fresh_sims_mean': float(a.mean()),
        'fresh_sims_p10': float(np.percentile(a, 10)),
        'fresh_floor_fired_share': float(floor_moves / total_moves)
            if total_moves else 0.0,
        'fresh_sims_below25_share': float(np.mean(a < 25)),
        'fresh_sims_zero_share': float(np.mean(a == 0)),
    }


# ── policy metrics (prior vs visit) ─────────────────────────────────────────

def _log_pos(x):
    """log(x) with log(0) -> 0 (so 0*log0 contributes nothing)."""
    return np.log(x, out=np.zeros_like(x), where=(x > 0))


def policy_kl_entropy(policies, priors, masks):
    """Masked policy entropy and root KL(net prior || MCTS visit policy).

    ``policies``: stored (temperature-processed) visit targets, (N, A).
    ``priors``:   re-predicted net softmax, (N, A).
    ``masks``:    per-position legal-move masks, (N, A) 0/1.

    Vectorised float32, one pass over the batch.
    """
    policies = np.asarray(policies, dtype=np.float32)
    priors = np.asarray(priors, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if priors.ndim == 1:
        priors = priors[None, :]
        masks = masks[None, :]
        policies = policies[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        q = priors * masks
        qs = q.sum(axis=1)
        valid = qs > 0
        if not valid.any():
            return {}
        q[valid] /= qs[valid, None]
        ent = -np.einsum('ij,ij->i', q, _log_pos(q))

        out = {'masked_policy_entropy': float(ent[valid].mean())}

        # KL is only meaningful where the stored target is a real visit
        # distribution (>1 non-zero), not a one-hot greedy / mate-in-1 target.
        sel = valid & ((policies > 0).sum(axis=1) >= 2)
        if sel.any():
            p = policies[sel] * masks[sel]
            ps = p.sum(axis=1)
            ok = ps > 0
            if ok.any():
                p, ps, qq = p[ok], ps[ok], q[sel][ok]
                p = p / ps[:, None]
                kl = np.einsum('ij,ij->i', p, _log_pos(p) - np.log(np.maximum(qq, 1e-12)))
                out['root_kl_visit_prior'] = float(kl.mean())
                out['root_kl_positions'] = float(kl.size)
    return out


# ── value-head error by game phase ──────────────────────────────────────────

def value_error_by_phase(phase_records, predict_values, max_per_phase=64):
    """Mean |V_net - V_target| at the fixed phase plies.

    ``phase_records``: iterable of per-game lists ``[(ply, state, target), ...]``.
    ``predict_values``: callable taking an (N, P, H, W) float array -> (N,) values.
    ``max_per_phase``: cap on states forwarded per phase (evenly subsampled) so
    the metric stays a single small batch regardless of window size.
    """
    plies = set(PHASE_PLIES)
    for rec in phase_records:
        for ply, _state, _target in rec:
            plies.add(ply)
    buckets = {p: ([], []) for p in plies}
    for rec in phase_records:
        for ply, state, target in rec:
            buckets[ply][0].append(state)
            buckets[ply][1].append(target)

    # Gather (capped) per-phase states into ONE batch so the value head is
    # queried with a single forward pass regardless of the number of phases.
    spans = []      # (ply, start, end)
    all_states, all_targets = [], []
    for ply in sorted(buckets):
        states, targets = buckets[ply]
        if not states:
            continue
        if max_per_phase and len(states) > max_per_phase:
            idx = np.linspace(0, len(states) - 1, max_per_phase).astype(int)
            states = [states[i] for i in idx]
            targets = [targets[i] for i in idx]
        start = len(all_states)
        all_states.extend(states)
        all_targets.extend(targets)
        spans.append((ply, start, len(all_states)))

    if not all_states:
        return {}

    preds = []
    for i in range(0, len(all_states), 256):
        chunk = np.stack(all_states[i:i + 256]).astype(np.float32)
        preds.append(np.asarray(predict_values(chunk), dtype=np.float32).reshape(-1))
    preds = np.concatenate(preds)
    targets = np.asarray(all_targets, dtype=np.float32)

    out = {}
    for ply, start, end in spans:
        out[f'value_error_ply{ply}'] = float(
            np.mean(np.abs(preds[start:end] - targets[start:end])))
    return out


# ── orchestrator ────────────────────────────────────────────────────────────

class RollingChessStats:
    """Holds the rolling windows and coordinates computation + logging.

    Deliberately holds *references* to already-created per-game data only
    (no copying, no per-move state).  ``add_game`` is O(len(PHASE_PLIES)).
    """

    def __init__(self, window=WINDOW, log_every=LOG_EVERY,
                 phase_plies=PHASE_PLIES):
        self.window = int(window)
        self.log_every = int(log_every)
        self.phase_plies = tuple(phase_plies)
        self._games = deque(maxlen=self.window)
        self._phase = deque(maxlen=self.window)
        self._since_log = 0

    def add_game(self, game_info, game_data):
        """Record one finished self-play game.  Cheap: O(#phase plies)."""
        self._games.append({
            'result': float(game_info['result']),
            'termination': game_info['termination'],
            'length': int(game_info['length']),
            'fresh_sims': list(game_info.get('fresh_sims') or []),
            'floor_fired': list(game_info.get('floor_fired') or []),
        })
        rec = []
        for ply in self.phase_plies:
            k = ply - 1
            if 0 <= k < len(game_data):
                state, _pol, value, _mask = game_data[k]
                rec.append((ply, state, float(value)))
        self._phase.append(rec)
        self._since_log += 1

    def due(self):
        return self._since_log >= self.log_every and len(self._games) > 0

    def reset_counter(self):
        self._since_log = 0

    def game_metrics(self):
        """Window-only metrics (no network needed)."""
        out = {}
        out.update(white_share(self._games))
        out.update(termination_mix(self._games))
        out.update(length_stats(self._games))
        out.update(fresh_sim_stats(self._games))
        return out

    def compute(self, predict_batch, sample_batch):
        """Full metric dict.

        ``sample_batch(n)`` -> (states, policies, values, masks) sampled from
        the replay buffer; ``predict_batch(states)`` -> (priors, values).
        """
        out = self.game_metrics()

        games = list(self._games)
        # policy metrics from a buffer sample
        try:
            n = min(128, max(1, len(games)))
            states, policies, _values, masks = sample_batch(n)
            states = np.asarray(states, dtype=np.float32)
            priors, _vals = predict_batch(states)
            out.update(policy_kl_entropy(policies, priors, masks))
        except Exception as exc:  # never let diagnostics break training
            print(f"[WARN] chess policy metrics skipped: {exc}")

        # value error by phase
        try:
            out.update(value_error_by_phase(
                list(self._phase),
                lambda s: predict_batch(s)[1]))
        except Exception as exc:
            print(f"[WARN] chess value metrics skipped: {exc}")

        return out
