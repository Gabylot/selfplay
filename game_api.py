"""Game facade: uniform API over game modules.

Every AlphaZero pipeline module (network, mcts, selfplay, evaluation,
gpu_server, shared_memory_transport, ...) obtains game-specific behavior
through this module instead of reaching into ``encoding``/``chess``
directly.  Adding a new game = adding a module under ``games/`` and
registering it in ``_REGISTRY``.

The active game is chosen by ``config.game.name`` (default "chess") and
can be overridden at runtime with :func:`set_game` (used by worker
processes, which rebuild their Config from a pickled dict).
"""

import importlib

_REGISTRY = {
    "chess": "games.chess",
    "connect4": "games.connect4",
}

_override_name = None

# Memoised active game module.  Every game_api helper funnels through
# get_game(), which is on the MCTS hot path (called ~5x per leaf expansion and
# per terminal check).  Re-resolving via importlib.import_module + get_config on
# every call cost ~15% of search time; cache the module and only re-resolve when
# the active game name changes (e.g. set_game() in a worker).
_cached_game_name = None
_cached_game_mod = None


def set_game(name: str):
    """Override the active game name (used inside worker processes)."""
    global _override_name
    _override_name = name


def active_game_name() -> str:
    from config import get_config

    if _override_name is not None:
        return _override_name
    cfg = get_config()
    section = getattr(cfg, "game", None)
    return getattr(section, "name", "chess")


def list_games():
    return sorted(_REGISTRY)


def is_game(name: str) -> bool:
    return active_game_name() == name


def get_game():
    """Return the active game module (memoised on the active game name)."""
    global _cached_game_name, _cached_game_mod
    name = active_game_name()
    if name != _cached_game_name:
        mod = _REGISTRY.get(name)
        if mod is None:
            raise ValueError(
                f"Unknown game '{name}'. Available games: {sorted(_REGISTRY)}"
            )
        _cached_game_mod = importlib.import_module(mod)
        _cached_game_name = name
    return _cached_game_mod


# ── Dimension helpers (network builder + IPC plumbing) ──
def num_planes():
    return get_game().num_planes


def num_actions():
    return get_game().num_actions


def policy_channels():
    return get_game().policy_channels


def board_w():
    return get_game().board_w


def board_h():
    return get_game().board_h


# ── State lifecycle ──
def new_state(start=None):
    """Build a fresh game state (and a ``start_kind`` tag).

    ``start`` is a per-game dict (chess: fen / start_pieces /
    randomize_colors).  Returns ``(state, start_kind)``.
    """
    return get_game().new_state(start)


def apply(state, move):
    get_game().apply(state, move)


def legal_moves(state):
    return get_game().legal_moves(state)


def state_ply(state):
    return get_game().state_ply(state)


def is_terminal(state):
    return get_game().is_terminal(state)


def is_draw_claim(state):
    """Claimable draw (threefold / fifty-move) — chess only."""
    fn = getattr(get_game(), "is_draw_claim", None)
    return fn(state) if fn is not None else False


def is_fifty_moves(state):
    """Fifty-move-rule eligibility within the search tree — chess only."""
    fn = getattr(get_game(), "is_fifty_moves", None)
    return fn(state) if fn is not None else False


def terminal_value(state):
    """Exact terminal value from the side-to-move's perspective."""
    return get_game().terminal_value(state)


def adjudicate_value(state, **kwargs):
    fn = getattr(get_game(), "adjudicate_value", None)
    if fn is None:
        return 0.0
    return fn(state, **kwargs)


def encode(state):
    return get_game().encode(state)


def mask(state):
    return get_game().mask(state)


def to_index(state, move):
    return get_game().to_index(state, move)


def index_to_move(state, action_idx):
    return get_game().index_to_move(state, action_idx)


def move_key(move):
    return get_game().move_key(move)


def move_str(move):
    return get_game().move_str(move)


def describe(state):
    return get_game().describe(state)


def player_parity(state):
    """0 for the first player / white, 1 for the second / black."""
    return get_game().player_parity(state)


def player_sign(state):
    return get_game().player_sign(state)


def piece_count(state):
    """Number of pieces, or None for games without a piece notion."""
    fn = getattr(get_game(), "piece_count", None)
    return fn(state) if fn is not None else None


def final_result(state, move_count, max_game_length, **kwargs):
    """Terminal adjudication for a finished game.

    Returns a dict with ``outcome`` (first-player perspective: +1 won,
    -1 lost, 0 draw), ``termination`` (string tag) and ``result_str``.
    """
    return get_game().final_result(state, move_count, max_game_length, **kwargs)


def reference_move(state, **kwargs):
    """A fast, game-legal move for a reference/inference opponent."""
    return get_game().reference_move(state, **kwargs)