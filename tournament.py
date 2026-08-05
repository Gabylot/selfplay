"""Tournament mode for the AlphaZero chess engine.

Runs a single-elimination-free competition between invited checkpoints
(models) using the existing worker pipeline (ParallelSelfPlay + the
network-vs-network "gating" eval task).  Only one game is played at a
time; every move is streamed to a dedicated LiveGameState so the GUI's
tournament view can follow the game live with MCTS statistics, while a
standings table is kept up to date after every game.

Supported systems:
  - round_robin: every pair of models plays ``games_per_pair`` games
    (colors alternate), exactly once per pair.
  - swiss: a configurable number of rounds; each round pairs players
    with equal (or closest) scores, avoiding repeated pairings, with
    color balance tracked per player.  A bye is awarded when the field
    is odd.
"""

import io
import threading
from pathlib import Path

import torch

from network import load_checkpoint, create_model_from_config
from selfplay import ParallelSelfPlay
from evaluation import elo_expected_score


# ─────────────────────────────────────────────────────────────────────────────
# Model loading / player registry
# ─────────────────────────────────────────────────────────────────────────────

def _state_bytes(path: Path) -> bytes:
    """Return the serialized raw state_dict bytes for a checkpoint file.

    Accepts both full checkpoints (with a 'model_state_dict' key) and
    bare state dicts.  The bytes are what workers load into their own
    AlphaZeroNet instances.
    """
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()


def discover_models(model_dir=None, model_paths=None):
    """Build the list of invited players.

    Args:
        model_dir: Directory scanned for ``*.pt`` checkpoints.
        model_paths: Optional list of ``"name:path"`` or plain ``path``
            entries for explicitly invited models.

    Returns:
        List of player dicts: {name, path, weights, elo, wins, losses,
        draws, pts, games, white_games, black_games, bye}.
    """
    entries = []  # (name, path)

    if model_dir:
        d = Path(model_dir)
        if not d.exists():
            raise FileNotFoundError(f"Model directory not found: {d}")
        for p in sorted(d.glob("*.pt")):
            entries.append((p.stem, p))

    for raw in (model_paths or []):
        raw = raw.strip()
        if not raw:
            continue
        if ':' in raw:
            name, path = raw.split(':', 1)
            name, path = name.strip(), Path(path.strip())
        else:
            path = Path(raw)
            name = path.stem
        if not path.exists():
            raise FileNotFoundError(f"Invited model not found: {path}")
        entries.append((name, path))

    # De-duplicate names by appending an index.
    used = {}
    players = []
    for name, path in entries:
        base, k = name, 0
        while base in used:
            k += 1
            base = f"{name}#{k}"
        used[base] = True
        players.append({
            'name': base,
            'path': str(path),
            'weights': None,
            'elo': 1000.0,
            'wins': 0, 'losses': 0, 'draws': 0,
            'pts': 0.0,
            'games': 0, 'white_games': 0, 'black_games': 0,
            'bye': 0,
        })

    if not players:
        raise ValueError("No models invited — provide --model-dir or --models")

    for pl in players:
        pl['weights'] = _state_bytes(Path(pl['path']))
    return players


# ─────────────────────────────────────────────────────────────────────────────
# Scheduling
# ─────────────────────────────────────────────────────────────────────────────

def round_robin_rounds(players, games_per_pair):
    """Round-robin: one round containing every unordered pair once.

    Each entry is {'a': player, 'b': player}.  Colors alternate across
    the games_per_pair games in each pair.
    """
    round1 = []
    for i, a in enumerate(players):
        for b in players[i + 1:]:
            round1.append({'a': a, 'b': b})
    return [round1]


def _color_sequence(a, b, games_per_pair):
    """Return list of (white_player, black_player) for a pairing."""
    seq = []
    for gi in range(games_per_pair):
        if gi % 2 == 0:
            seq.append((a, b))
        else:
            seq.append((b, a))
    return seq


class SwissPairer:
    """Generates Swiss-system pairings one round at a time.

    Pairings are computed from the *current* standings, so scores feed
    back between rounds.  A player is never paired with the same model
    twice, and the bye goes to the lowest-standing player who has not
    yet received one (with the lowest overall as a fallback).
    """

    def __init__(self, players):
        self.players = players
        self.played = set()

    def pair(self, round_idx):
        n = len(self.players)
        # Sort strongest first: points desc, then fewer games, then index.
        order = sorted(range(n),
                       key=lambda i: (-self.players[i]['pts'],
                                      self.players[i]['games'], i))
        used = [False] * n
        entries = []
        bye = None

        # Odd field → award a bye to the weakest player who has not yet
        # received one, lowest remaining as fallback.
        if n % 2 == 1:
            chosen = None
            for i in reversed(order):
                if not used[i] and self.players[i]['bye'] == 0:
                    chosen = i
                    break
            if chosen is None:
                chosen = order[-1]
            used[chosen] = True
            bye = self.players[chosen]

        # Pair every remaining player exactly once.  Prefer an opponent of
        # equal score who hasn't been played yet; fall back to any other
        # opponent rather than leaving a player stranded (which would
        # create spurious extra byes).
        for idx in order:
            if used[idx]:
                continue
            a = self.players[idx]
            best = None
            best_dist = float('inf')
            for j in order:
                if j == idx or used[j]:
                    continue
                if frozenset((a['name'], self.players[j]['name'])) in self.played:
                    continue
                dist = abs(a['pts'] - self.players[j]['pts'])
                if dist < best_dist:
                    best_dist = dist
                    best = j
            if best is None:
                # Orphan fallback: take the first available opponent even if
                # this creates a repeat pairing — never strand a player.
                for j in order:
                    if j != idx and not used[j]:
                        best = j
                        break
                if best is None:
                    used[idx] = True
                    entries.append({'bye': a})
                    continue
            b = self.players[best]
            used[idx] = used[best] = True
            self.played.add(frozenset((a['name'], b['name'])))
            # Prefer giving White to the player with fewer white games;
            # tie-break by round parity.
            if a['white_games'] != b['white_games']:
                white, black = (a, b) if a['white_games'] < b['white_games'] else (b, a)
            else:
                white, black = (a, b) if round_idx % 2 == 0 else (b, a)
            entries.append({'a': white, 'b': black})

        if bye is not None:
            entries.append({'bye': bye})
        return entries


def build_schedule(config, players):
    """Build the ordered round list for round-robin, or a SwissPairer.

    Round-robin returns a list of rounds (a single round with every
    unordered pair).  For Swiss it returns ``None`` — pairings are meant
    to be generated dynamically during the run from live standings.
    """
    sysname = str(config.tournament.system).lower()
    if sysname in ('round_robin', 'round-robin', 'rr'):
        return round_robin_rounds(players, int(config.tournament.games_per_pair))
    if sysname in ('swiss', 'sw'):
        return None
    raise ValueError(f"Unknown tournament system: {sysname}")


# ─────────────────────────────────────────────────────────────────────────────
# Tournament controller
# ─────────────────────────────────────────────────────────────────────────────

class Tournament:
    """Holds tournament state and exposes it to the GUI (standings/status)."""

    def __init__(self, config, players, schedule):
        self.config = config
        self.players = players
        self.schedule = schedule  # None for swiss (generated per-round)
        self.live_game = None  # set by the runner / GUI wiring

        self.system = str(config.tournament.system).lower()
        self.games_per_pair = int(config.tournament.games_per_pair)
        self.elo_k = float(config.tournament.elo_k)
        self.initial_elo = float(config.tournament.initial_elo)
        self.swiss_rounds = int(config.tournament.swiss_rounds)

        self.round_idx = 0
        self.pair_idx = 0
        self.game_idx = 0
        self.phase = 'idle'          # idle | running | finished
        self.current_white = None
        self.current_black = None

        self._swiss_pairer = SwissPairer(self.players) if self.system in ('swiss', 'sw') else None
        self.total_planned = self._count_planned_games()

        for pl in self.players:
            pl['elo'] = self.initial_elo

        self._lock = threading.Lock()

    def _count_planned_games(self):
        if self._swiss_pairer is not None:
            games_per_round = (len(self.players) // 2) * self.games_per_pair
            return games_per_round * self.swiss_rounds
        total = 0
        for rnd in self.schedule:
            for entry in rnd:
                if 'bye' in entry:
                    continue
                total += self.games_per_pair
        return total

    def round_entries(self, round_idx):
        """Return the pairings for a round (Swiss pairs are generated here)."""
        if self._swiss_pairer is not None:
            return self._swiss_pairer.pair(round_idx)
        return self.schedule[round_idx]

    def _by_name(self, name):
        for pl in self.players:
            if pl['name'] == name:
                return pl
        return None

    def record_result(self, white_name, black_name, result_str):
        """Update standings for a completed game.

        result_str: '1-0' (white wins), '0-1' (black wins), else draw.
        """
        w = self._by_name(white_name)
        b = self._by_name(black_name)
        if w is None or b is None:
            return
        if result_str == '1-0':
            w['wins'], w['pts'] = w['wins'] + 1, w['pts'] + 1.0
            b['losses'] = b['losses'] + 1
        elif result_str == '0-1':
            b['wins'], b['pts'] = b['wins'] + 1, b['pts'] + 1.0
            w['losses'] = w['losses'] + 1
        else:
            w['draws'], b['draws'] = w['draws'] + 1, b['draws'] + 1
            w['pts'], b['pts'] = w['pts'] + 0.5, b['pts'] + 0.5
        w['games'] += 1
        b['games'] += 1
        w['white_games'] += 1
        b['black_games'] += 1

        # Elo update (from white's perspective).
        ea = elo_expected_score(w['elo'], b['elo'])
        score = 1.0 if result_str == '1-0' else (0.0 if result_str == '0-1' else 0.5)
        w['elo'] += self.elo_k * (score - ea)
        b['elo'] += self.elo_k * ((1.0 - score) - (1.0 - ea))

    def record_bye(self, name):
        pl = self._by_name(name)
        if pl is None:
            return
        pl['wins'] += 1
        pl['pts'] += 1.0
        pl['games'] += 1
        pl['bye'] += 1

    def standings(self):
        """Standings table rows, sorted by points desc."""
        with self._lock:
            rows = [dict(pl) for pl in self.players]
        rows.sort(key=lambda r: (-r['pts'], -r['wins'], r['games']))
        for rank, r in enumerate(rows, start=1):
            r['rank'] = rank
            r['score_pct'] = (r['pts'] / r['games'] * 100.0) if r['games'] else 0.0
        return rows

    def status(self):
        with self._lock:
            return {
                'system': self.system,
                'phase': self.phase,
                'round_idx': self.round_idx,
                'pair_idx': self.pair_idx,
                'game_idx': self.game_idx,
                'total_games': self.total_planned,
                'num_players': len(self.players),
                'games_per_pair': self.games_per_pair,
                'current_white': self.current_white,
                'current_black': self.current_black,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _play_one_game(psp, white, black, live_game, game_id, label):
    """Play a single network-vs-network game through the worker pool.

    Streams live_start/live_move/live_end messages into live_game and
    returns the final result string ('1-0', '0-1', '1/2-1/2' or '*').
    """
    if live_game is not None:
        live_game.start_game(game_id, 0, game_type='tournament', match_info=label)

    if psp._use_gpu:
        # network_id 0 = white, network_id 1 = black on the GPU server.
        psp.push_eval_weights(white['weights'], black['weights'])

    task = {
        'type': 'eval',
        'eval_type': 'gating',
        'weights_a': white['weights'],
        'weights_b': black['weights'],
        'a_is_white': True,
        'game_label': label,
    }
    dispatched = psp.dispatch_eval_games([task])
    if dispatched == 0:
        return '*'

    result = '*'
    while True:
        r = psp.collect_one(timeout=300.0)
        if r is None:
            print("[WARN] Tournament: no result in 5 min — skipping game")
            break
        if r.get('done'):
            break
        rt = r.get('type')
        if rt == 'live_move' and live_game is not None:
            live_game.update(r['fen'], r['move'], r['move_number'],
                             mcts_stats=r.get('mcts_stats'))
        elif rt == 'live_end' and live_game is not None:
            live_game.game_over(r['result'], r.get('termination', ''))
        elif rt == 'eval':
            result = r['result']
            break
    return result


def run_tournament(config, live_game=None, tournament=None, num_workers=1,
                   progress_cb=None):
    """Run the tournament until completion.

    Args:
        config: Config object.
        live_game: LiveGameState the tournament streams games into.
        tournament: Pre-built Tournament (state provider for the GUI).
            If None, a fresh one is built.
        num_workers: Worker pool size.  Tournament plays one game at a
            time, so 1 worker is sufficient.
        progress_cb: Optional callback(tournament, result) invoked after
            each finished game (used by main.py for console output).
    """
    import multiprocessing as _mp
    if _mp.get_start_method(allow_none=True) is None:
        try:
            _mp.set_start_method('spawn')
        except RuntimeError:
            pass

    if tournament is None:
        raise ValueError("A Tournament instance is required — build it in main()")
    tournament.live_game = live_game

    psp = ParallelSelfPlay(config, num_workers=num_workers)
    psp.start()

    tournament.phase = 'running'
    game_counter = 0
    try:
        round_count = (len(tournament.schedule)
                       if tournament._swiss_pairer is None
                       else tournament.swiss_rounds)
        for rnd_idx in range(round_count):
            rnd = tournament.round_entries(rnd_idx)
            tournament.round_idx = rnd_idx
            for p_idx, entry in enumerate(rnd):
                tournament.pair_idx = p_idx
                if 'bye' in entry:
                    pl = entry['bye']
                    tournament.record_bye(pl['name'])
                    print(f"  [Tournament] {pl['name']} — bye (round {rnd_idx + 1})")
                    continue

                white, black = entry['a'], entry['b']
                tournament.current_white = white['name']
                tournament.current_black = black['name']

                for wi, (w, b) in enumerate(_color_sequence(white, black,
                                                            tournament.games_per_pair)):
                    game_counter += 1
                    tournament.game_idx = game_counter
                    label = f"Round {rnd_idx + 1} · {w['name']} vs {b['name']}"
                    print(f"\n  [Tournament] {game_counter}/{tournament.total_planned}"
                          f" — {label}")
                    result = _play_one_game(psp, w, b, live_game, game_counter, label)
                    tournament.record_result(w['name'], b['name'], result)
                    if progress_cb:
                        progress_cb(tournament, result)
        tournament.phase = 'finished'
    finally:
        print("[INFO] Stopping tournament workers…")
        psp.stop()


def print_standings(tournament):
    print("\n" + "=" * 72)
    print("  TOURNAMENT STANDINGS")
    print("=" * 72)
    rows = tournament.standings()
    if not rows:
        print("  No players.")
        return
    print(f"  {'Rank':<5}{'Model':<30}{'G':>4}{'W':>4}{'L':>4}{'D':>4}"
          f"{'Pts':>6}{'SR%':>7}{'Elo':>7}")
    for r in rows:
        print(f"  {r['rank']:<5}{r['name']:<30}{r['games']:>4}{r['wins']:>4}"
              f"{r['losses']:>4}{r['draws']:>4}{r['pts']:>6.1f}"
              f"{r['score_pct']:>7.1f}{r['elo']:>7.0f}")
    print("=" * 72)
