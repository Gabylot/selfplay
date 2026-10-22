"""Configuration loader for AlphaZero Chess Engine."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


def _find_config_path() -> Path:
    """Find config.yaml relative to this file."""
    return Path(__file__).parent / "config.yaml"


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from YAML file with defaults."""
    if path is None:
        path = str(_find_config_path())
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure all sections exist with defaults
    defaults = {
        "game": {
            "name": "chess",  # active game: "chess" or "connect4"
        },
        "network": {
            "num_residual_blocks": 8,
            "num_filters": 64,
            "num_policy_channels": 64,
            "num_value_channels": 1,
            "value_fc_size": 128,
        },
        "mcts": {
            "num_simulations": 400,
            "c_puct": 3.0,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.4,
            "batch_size": 8,
            "c_virtual_loss": 0.5,
            "mate_seeding": False,
            "min_fresh_sims": 64,
            # Use the Rust MCTS backend (rust_mcts.RustMCTS) for self-play.
            # Same move-selection targets as the Python engine at ~2x speed;
            # set false to force the pure-Python mcts.MCTS path.
            "use_rust_engine": True,
        },
        "selfplay": {
            "network_source": "latest",
            # FEN every self-play game starts from (None = standard start).
            "start_fen": None,
            # Start positions for self-play games: a piece set always frames a fresh
            # random legal position with exactly those pieces, e.g. "KQk,KRk"
            # (case = color). Weights per entry are allowed, e.g.
            # "standard:80,KQk:10,KRk:10" keeps 80% of games on the normal
            # start (token "standard") so the agent can't unlearn them.
            # Ignored when start_fen is set.
            "start_pieces": None,
            # For asymmetric piece specs (e.g. "KQQk") this flips the
            # colour assignment per game, so either side gets the strong
            # pieces 50% of the time.
            "start_colors_random": False,
            "max_game_length": 150,
            "adjudicate_material": True,
            "piece_values": {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9},
            "adjudicate_graded": True,
            "adjudicate_scaling": 9.0,
            "temperature_threshold": 30,
            "temperature_high": 1.0,
            "temperature_low": 0.1,
            # Endgame tuning: applied when <= endgame_piece_gate pieces stay
            # on the board (usually the "simple endgames" the net struggles
            # to convert).  Uses its own temperature and a deeper simulation
            # budget without touching the middlegame settings.
            "endgame_temperature": 0.3,
            "endgame_piece_gate": 6,
            "endgame_num_simulations": 1600,
            # Exact forced-mate override for endgame conversion
            # (see forced_mate.py).  When enabled, self-play move selection
            # is overridden by a proof search when a forced mate within
            # force_mate_max_plies exists on a board with
            # <= force_mate_gate_pieces pieces.
            "force_mate": False,
            "force_mate_max_plies": 5,
            "force_mate_gate_pieces": 8,
            # Deeper proof search for very sparse positions
            # (<= force_mate_deep_gate_pieces pieces), where the search tree
            # is small enough to stay cheap.  Converts longer endgames
            # (e.g. KR vs K mate-in-4/5) that the shallow limit misses.
            "force_mate_deep_plies": 9,
            "force_mate_deep_gate_pieces": 4,
        },
        "buffer": {
            "max_size": 100000,
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "num_batches_per_step": 10,
            "checkpoint_interval": 50,
            "training_steps_per_iteration": 5,
        },
        "evaluation": {
            "eval_interval": 50,
            "gate_games": 20,
            "gate_win_threshold": 0.55,
            "ref_opponent_games": 20,
            "elo_k_factor": 32,
        },
        "alpha_beta": {
            "depth": 2,
            "piece_values": {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9},
            "checkmate_score": 10000,
        },
        "stats": {
            "db_path": "stats.db",
            "log_interval": 1,
        },
        "tensorboard": {
            "enabled": True,
            "log_dir": "runs",
            "auto_launch": True,
            "host": "127.0.0.1",
            "port": 6006,
        },
        "gui": {
            "host": "127.0.0.1",
            "port": 5000,
            "refresh_interval": 1.0,
        },
        "tournament": {
            "system": "round_robin",
            "games_per_pair": 2,
            "swiss_rounds": 3,
            "elo_k": 32,
            "initial_elo": 1000,
        },
        "inference": {
            "use_gpu": False,
            "use_shared_memory": True,
            "state_dtype": "float32",
            "max_batch": 256,
            "max_wait_ms": 2.0,
            "prewarm_batch_sizes": [1, 8, 16, 32, 64, 128, 256],
        },
        "main": {
            "run_name": "default",
            "output_dir": "output",
        },
    }
    
    def merge(base: dict, override: dict) -> dict:
        """Deep merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result
    
    config = merge(defaults, config)
    return config


class Config:
    """Typed config wrapper with attribute access."""
    
    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> dict:
        # Deep copy: the Config may be cached and shared across callers, so
        # hand out a private copy of nested dicts.
        def _deep(d):
            return {k: _deep(v) if isinstance(v, dict) else v
                    for k, v in d.items()}
        return _deep(self._data)
    
    def get(self, key, default=None):
        """Dict-like get method."""
        return self._data.get(key, default)
    
    def __getitem__(self, key):
        """Dict-like item access."""
        return self._data[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        return f"Config({self._data})"


# Cache of Config objects keyed by resolved config path.  get_config() is on
# the hot path of every game_api dispatch (via active_game_name), and
# load_config() re-reads + YAML-parses the whole file (~15 ms).  Callers that
# pass overrides always get a fresh load (one-shot CLI entries); the bare
# no-override path is cached per path.  Use reload_config() to force a refresh.
_config_cache: Dict[str, Config] = {}

# Precompute once: Path.resolve() hits the filesystem (~100+ us), which would
# otherwise be paid on every cache hit.
_DEFAULT_CONFIG_KEY = str(_find_config_path().resolve())


def _config_cache_key(path: Optional[str]) -> str:
    if path is None:
        return _DEFAULT_CONFIG_KEY
    return str(Path(path).resolve())


def get_config(path: Optional[str] = None, overrides: Optional[Dict] = None) -> Config:
    """Load config and return as Config object.

    Results are cached per resolved path when no ``overrides`` are given
    (repeated YAML parsing was measured at ~15 ms per call on the MCTS hot
    path via game_api.active_game_name).

    Args:
        path: Path to YAML config file. None = default.
        overrides: Dict of overrides to apply (e.g. from CLI args).

    Returns:
        Config object with attribute access.
    """
    cacheable = not overrides
    key = _config_cache_key(path)
    if cacheable:
        cached = _config_cache.get(key)
        if cached is not None:
            return cached

    data = load_config(path)
    if overrides:
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        deep_update(data, overrides)
    cfg = Config(data)
    if cacheable:
        _config_cache[key] = cfg
    return cfg


def reload_config(path: Optional[str] = None) -> Config:
    """Force-reload the (no-override) config from disk and re-cache it."""
    _config_cache.pop(_config_cache_key(path), None)
    return get_config(path)