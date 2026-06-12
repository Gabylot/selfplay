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
        "network": {
            "num_residual_blocks": 4,
            "num_filters": 64,
            "num_policy_channels": 32,
            "num_value_channels": 16,
            "value_fc_size": 256,
        },
        "mcts": {
            "num_simulations": 200,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
        },
        "selfplay": {
            "network_source": "latest",
            "max_game_length": 150,
            "adjudicate_material": True,
            "piece_values": {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9},
            "temperature_threshold": 30,
            "temperature_high": 1.0,
            "temperature_low": 0.1,
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
        "gui": {
            "host": "127.0.0.1",
            "port": 5000,
            "refresh_interval": 1.0,
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
        return self._data.copy()
    
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


def get_config(path: Optional[str] = None, overrides: Optional[Dict] = None) -> Config:
    """Load config and return as Config object.
    
    Args:
        path: Path to YAML config file. None = default.
        overrides: Dict of overrides to apply (e.g. from CLI args).
    
    Returns:
        Config object with attribute access.
    """
    data = load_config(path)
    if overrides:
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        deep_update(data, overrides)
    return Config(data)