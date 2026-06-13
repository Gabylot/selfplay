"""Statistics logging and persistence for AlphaZero chess engine.

Logs to SQLite database with separate tracking for:
- Policy loss, value loss, total loss (per training step)
- Game outcomes (win/loss/draw by termination reason)
- Elo rating over time
- Promotion attempts and win rates
- MCTS statistics
- Network confidence trends
- Replay buffer composition
"""

import sqlite3
import time
import json
from typing import Optional, List, Dict, Any
from pathlib import Path


class StatsLogger:
    """SQLite-based statistics logger."""
    
    def __init__(self, db_path: str = "stats.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read/write
        self._create_tables()
    
    def _create_tables(self):
        """Create all necessary tables."""
        cursor = self.conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                policy_loss REAL NOT NULL,
                value_loss REAL NOT NULL,
                total_loss REAL NOT NULL,
                learning_rate REAL,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS game_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                step INTEGER NOT NULL,
                result REAL NOT NULL,
                result_str TEXT,
                length INTEGER NOT NULL,
                termination TEXT NOT NULL,
                avg_mcts_depth REAL,
                num_positions INTEGER,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS elo_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                elo_rating REAL NOT NULL,
                opponent_type TEXT NOT NULL,
                step INTEGER,
                games_played INTEGER,
                wins INTEGER,
                losses INTEGER,
                draws INTEGER,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS promotion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                promoted INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                games_played INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                draws INTEGER NOT NULL,
                new_elo REAL,
                old_elo REAL,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS evaluation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                opponent TEXT NOT NULL,
                games_played INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                draws INTEGER NOT NULL,
                win_rate REAL,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS mcts_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                step INTEGER NOT NULL,
                avg_tree_depth REAL,
                avg_sims_per_move REAL,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS network_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                avg_max_policy REAL,
                avg_abs_value REAL,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS buffer_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                buffer_size INTEGER,
                white_wins REAL,
                black_wins REAL,
                draws REAL,
                timestamp REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS config_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
        """)
        
        self.conn.commit()
    
    def log_training_step(self, step: int, policy_loss: float, value_loss: float,
                          total_loss: float, learning_rate: float = None):
        """Log a training step's losses (separate policy/value)."""
        self.conn.execute(
            "INSERT INTO training_log (step, policy_loss, value_loss, total_loss, learning_rate, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (step, policy_loss, value_loss, total_loss, learning_rate, time.time())
        )
        self.conn.commit()
    
    def log_game(self, game_id: int, step: int, result: float, result_str: str,
                 length: int, termination: str, avg_mcts_depth: float = 0,
                 num_positions: int = 0):
        """Log a completed self-play game."""
        self.conn.execute(
            "INSERT INTO game_log (game_id, step, result, result_str, length, termination, "
            "avg_mcts_depth, num_positions, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (game_id, step, result, result_str, length, termination, avg_mcts_depth,
             num_positions, time.time())
        )
        self.conn.commit()
    
    def log_elo(self, elo_rating: float, opponent_type: str, step: int = None,
                games_played: int = 0, wins: int = 0, losses: int = 0, draws: int = 0):
        """Log an Elo rating update."""
        self.conn.execute(
            "INSERT INTO elo_log (elo_rating, opponent_type, step, games_played, "
            "wins, losses, draws, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (elo_rating, opponent_type, step, games_played, wins, losses, draws, time.time())
        )
        self.conn.commit()
    
    def log_promotion_attempt(self, step: int, promoted: bool, win_rate: float,
                              games_played: int, wins: int, losses: int, draws: int,
                              new_elo: float = None, old_elo: float = None):
        """Log a gating/promotion attempt (critical for diagnosing gate stagnation)."""
        self.conn.execute(
            "INSERT INTO promotion_log (step, promoted, win_rate, games_played, "
            "wins, losses, draws, new_elo, old_elo, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (step, int(promoted), win_rate, games_played, wins, losses, draws,
             new_elo, old_elo, time.time())
        )
        self.conn.commit()
    
    def log_evaluation(self, step: int, opponent: str, games_played: int,
                       wins: int, losses: int, draws: int, win_rate: float = None):
        """Log evaluation match results."""
        if win_rate is None and games_played > 0:
            win_rate = wins / games_played
        self.conn.execute(
            "INSERT INTO evaluation_log (step, opponent, games_played, wins, losses, "
            "draws, win_rate, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (step, opponent, games_played, wins, losses, draws, win_rate, time.time())
        )
        self.conn.commit()
    
    def log_mcts_stats(self, game_id: int, step: int, avg_tree_depth: float,
                       avg_sims_per_move: float):
        """Log MCTS search statistics for a game."""
        self.conn.execute(
            "INSERT INTO mcts_stats (game_id, step, avg_tree_depth, avg_sims_per_move, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (game_id, step, avg_tree_depth, avg_sims_per_move, time.time())
        )
        self.conn.commit()
    
    def log_network_stats(self, step: int, avg_max_policy: float, avg_abs_value: float):
        """Log network confidence trends."""
        self.conn.execute(
            "INSERT INTO network_stats (step, avg_max_policy, avg_abs_value, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (step, avg_max_policy, avg_abs_value, time.time())
        )
        self.conn.commit()
    
    def log_buffer_stats(self, step: int, buffer_size: int, white_wins: float,
                         black_wins: float, draws: float):
        """Log replay buffer composition."""
        self.conn.execute(
            "INSERT INTO buffer_stats (step, buffer_size, white_wins, black_wins, draws, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (step, buffer_size, white_wins, black_wins, draws, time.time())
        )
        self.conn.commit()
    
    def log_config(self, step: int, config_dict: dict):
        """Log configuration snapshot."""
        self.conn.execute(
            "INSERT INTO config_log (step, config_json, timestamp) "
            "VALUES (?, ?, ?)",
            (step, json.dumps(config_dict), time.time())
        )
        self.conn.commit()
    
    # --- Query methods for GUI and analysis ---
    
    def get_training_losses(self, last_n: int = None) -> List[Dict]:
        """Get training losses over time."""
        query = "SELECT step, policy_loss, value_loss, total_loss, timestamp FROM training_log"
        if last_n:
            query += f" ORDER BY id DESC LIMIT {last_n}"
            rows = self.conn.execute(query).fetchall()
            rows.reverse()
        else:
            rows = self.conn.execute(query + " ORDER BY id ASC").fetchall()
        return [{'step': r[0], 'policy_loss': r[1], 'value_loss': r[2],
                 'total_loss': r[3], 'timestamp': r[4]} for r in rows]
    
    def get_game_outcomes(self, last_n: int = None) -> List[Dict]:
        """Get game outcomes over time."""
        query = "SELECT game_id, step, result, length, termination, avg_mcts_depth, timestamp FROM game_log"
        if last_n:
            query += f" ORDER BY id DESC LIMIT {last_n}"
            rows = self.conn.execute(query).fetchall()
            rows.reverse()
        else:
            rows = self.conn.execute(query + " ORDER BY id ASC").fetchall()
        return [{'game_id': r[0], 'step': r[1], 'result': r[2], 'length': r[3],
                 'termination': r[4], 'avg_mcts_depth': r[5], 'timestamp': r[6]} for r in rows]
    
    def get_elo_history(self) -> List[Dict]:
        """Get Elo rating history."""
        rows = self.conn.execute(
            "SELECT elo_rating, opponent_type, step, timestamp FROM elo_log ORDER BY id ASC"
        ).fetchall()
        return [{'elo': r[0], 'opponent': r[1], 'step': r[2], 'timestamp': r[3]}
                for r in rows]
    
    def get_promotion_history(self) -> List[Dict]:
        """Get promotion attempt history."""
        rows = self.conn.execute(
            "SELECT step, promoted, win_rate, games_played, wins, losses, draws, "
            "new_elo, old_elo, timestamp FROM promotion_log ORDER BY id ASC"
        ).fetchall()
        return [{'step': r[0], 'promoted': bool(r[1]), 'win_rate': r[2],
                 'games_played': r[3], 'wins': r[4], 'losses': r[5], 'draws': r[6],
                 'new_elo': r[7], 'old_elo': r[8], 'timestamp': r[9]} for r in rows]
    
    def get_evaluation_history(self) -> List[Dict]:
        """Get evaluation match history."""
        rows = self.conn.execute(
            "SELECT step, opponent, games_played, wins, losses, draws, win_rate, "
            "timestamp FROM evaluation_log ORDER BY id ASC"
        ).fetchall()
        return [{'step': r[0], 'opponent': r[1], 'games_played': r[2],
                 'wins': r[3], 'losses': r[4], 'draws': r[5], 'win_rate': r[6],
                 'timestamp': r[7]} for r in rows]
    
    def get_network_stats_history(self) -> List[Dict]:
        """Get network confidence trend data."""
        rows = self.conn.execute(
            "SELECT step, avg_max_policy, avg_abs_value, timestamp FROM network_stats "
            "ORDER BY id ASC"
        ).fetchall()
        return [{'step': r[0], 'avg_max_policy': r[1], 'avg_abs_value': r[2],
                 'timestamp': r[3]} for r in rows]
    
    def get_buffer_stats_history(self) -> List[Dict]:
        """Get replay buffer composition over time."""
        rows = self.conn.execute(
            "SELECT step, buffer_size, white_wins, black_wins, draws, timestamp "
            "FROM buffer_stats ORDER BY id ASC"
        ).fetchall()
        return [{'step': r[0], 'buffer_size': r[1], 'white_wins': r[2],
                 'black_wins': r[3], 'draws': r[4], 'timestamp': r[5]} for r in rows]
    
    def get_mcts_stats_history(self) -> List[Dict]:
        """Get MCTS statistics over time."""
        rows = self.conn.execute(
            "SELECT game_id, step, avg_tree_depth, avg_sims_per_move, timestamp "
            "FROM mcts_stats ORDER BY id ASC"
        ).fetchall()
        return [{'game_id': r[0], 'step': r[1], 'avg_depth': r[2],
                 'avg_sims': r[3], 'timestamp': r[4]} for r in rows]
    
    def get_summary(self) -> Dict:
        """Get a summary of all statistics for the GUI."""
        # Game outcomes summary
        games = self.conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN result > 0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN result < 0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN result = 0 THEN 1 ELSE 0 END), "
            "AVG(length) FROM game_log"
        ).fetchone()
        
        # Latest Elo
        elo = self.conn.execute(
            "SELECT elo_rating, step FROM elo_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        
        # Latest training step
        step = self.conn.execute(
            "SELECT MAX(step) FROM training_log"
        ).fetchone()
        
        # Latest losses
        losses = self.conn.execute(
            "SELECT policy_loss, value_loss, total_loss FROM training_log "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        
        # Promotion stats
        promos = self.conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) "
            "FROM promotion_log"
        ).fetchone()
        
        return {
            'total_games': games[0] or 0,
            'white_wins': games[1] or 0,
            'black_wins': games[2] or 0,
            'draws': games[3] or 0,
            'avg_game_length': games[4] or 0,
            'current_elo': elo[0] if elo else 1000,
            'elo_step': elo[1] if elo else 0,
            'current_step': step[0] or 0,
            'latest_policy_loss': losses[0] if losses else 0,
            'latest_value_loss': losses[1] if losses else 0,
            'latest_total_loss': losses[2] if losses else 0,
            'total_promotions_attempted': promos[0] or 0,
            'successful_promotions': promos[1] or 0,
        }
    
    def close(self):
        """Close the database connection."""
        self.conn.close()