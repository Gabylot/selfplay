"""Evaluation, gating, and Elo tracking for AlphaZero chess engine.

- Gating: periodically pit latest trained network vs current "best" network
- Reference opponent: depth-2 alpha-beta search using material evaluation
- Elo tracking with standard update formula

Design: Self-play always uses the *latest* network.
Gating and reference matches are purely monitoring signals — they do NOT
gate what generates training data. This prevents stagnation from unlucky
variance in evaluation matches early in training.
"""

import math
import chess
import numpy as np
from typing import Tuple, Optional

from network import AlphaZeroNet
from mcts import MCTS
from encoding import board_to_tensor


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 32) -> float:
    """Update rating_a based on match result. Returns new rating for A."""
    expected = elo_expected_score(rating_a, rating_b)
    return rating_a + k * (score_a - expected)


# ============================================================
# Alpha-Beta Reference Opponent (depth-2, material only)
# ============================================================

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def evaluate_material(board: chess.Board) -> float:
    """Evaluate a position based purely on material count.
    
    Returns score from white's perspective.
    """
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            score += val
        else:
            score -= val
    return score


def alpha_beta_search(board: chess.Board, depth: int, alpha: float, beta: float,
                       maximizing: bool) -> float:
    """Minimax with alpha-beta pruning.
    
    Returns evaluation from the perspective of the side to move.
    """
    if depth == 0 or board.is_game_over():
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 10000.0 if board.turn == chess.WHITE else -10000.0
            elif result == "0-1":
                return -10000.0 if board.turn == chess.WHITE else 10000.0
            else:
                return 0.0
        
        # Evaluate from side-to-move perspective
        material = evaluate_material(board)
        if board.turn == chess.BLACK:
            material = -material
        return material
    
    legal_moves = list(board.legal_moves)
    
    if maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


def alpha_beta_best_move(board: chess.Board, depth: int = 2) -> Optional[chess.Move]:
    """Find the best move using alpha-beta search.
    
    Returns the best move from the current side's perspective.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    best_move = None
    is_maximizing = (board.turn == chess.WHITE)
    
    if is_maximizing:
        best_score = -float('inf')
        for move in legal_moves:
            board.push(move)
            score = alpha_beta_search(board, depth - 1, -float('inf'), float('inf'), False)
            board.pop()
            # Negate because score is from the new position's perspective
            score = -score if board.turn == chess.WHITE else score
            if score > best_score:
                best_score = score
                best_move = move
    else:
        best_score = float('inf')
        for move in legal_moves:
            board.push(move)
            score = alpha_beta_search(board, depth - 1, -float('inf'), float('inf'), True)
            board.pop()
            score = -score if board.turn == chess.BLACK else score
            if score < best_score:
                best_score = score
                best_move = move
    
    return best_move if best_move else legal_moves[0]


def play_game_alpha_beta_vs_alpha_beta(depth: int = 2, 
                                        max_moves: int = 150) -> Tuple[str, int]:
    """Play a game between two alpha-beta agents.
    
    Returns (result_str, move_count).
    """
    board = chess.Board()
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        move = alpha_beta_best_move(board, depth)
        if move is None:
            break
        board.push(move)
        move_count += 1
    
    return board.result(), move_count


# ============================================================
# Match Playing
# ============================================================

def play_match(network_a: AlphaZeroNet, network_b: AlphaZeroNet,
               config, num_games: int = 20, 
               network_a_color: str = "alternating",
               verbose: bool = False,
               on_move=None,
               live_game=None,
               game_counter: int = 0) -> dict:
    """Play a match between two networks.
    
    Args:
        network_a: First network
        network_b: Second network
        config: Config object
        num_games: Number of games to play
        network_a_color: "alternating", "white", or "black"
        verbose: Print progress
        on_move: Optional callback(board_fen, move_uci, move_number, mcts_stats)
        live_game: Optional LiveGameState for live viewing
        game_counter: Starting game ID for live game tracking
    
    Returns:
        Dict with wins_a, wins_b, draws, results list
    """
    wins_a = 0
    wins_b = 0
    draws = 0
    results = []
    
    for game_idx in range(num_games):
        # Determine colors
        if network_a_color == "alternating":
            a_is_white = (game_idx % 2 == 0)
        elif network_a_color == "white":
            a_is_white = True
        else:
            a_is_white = False
        
        # Create MCTS engines for both networks
        mcts_a = MCTS(
            network=network_a,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=0.0,  # No Dirichlet noise in evaluation
            dirichlet_epsilon=0.0,
            batch_size=getattr(config.mcts, 'batch_size', 1),
            c_virtual_loss=getattr(config.mcts, 'c_virtual_loss', 0.5),
        )
        mcts_b = MCTS(
            network=network_b,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            batch_size=getattr(config.mcts, 'batch_size', 1),
            c_virtual_loss=getattr(config.mcts, 'c_virtual_loss', 0.5),
        )
        
        # Notify live game viewer of new game
        if live_game is not None:
            live_game.start_game(game_counter + game_idx, config.get('step', 0) if hasattr(config, 'get') else 0,
                                 game_type="gating",
                                 match_info=f"Gating Game {game_idx+1}/{num_games}")
        
        result = _play_evaluation_game(mcts_a, mcts_b, a_is_white, 
                                        max_moves=config.selfplay.max_game_length,
                                        on_move=on_move)
        
        # Notify live game viewer that game is over
        if live_game is not None:
            live_game.game_over(result, 'gating_game')
        
        if result == "1-0":
            if a_is_white:
                wins_a += 1
            else:
                wins_b += 1
        elif result == "0-1":
            if a_is_white:
                wins_b += 1
            else:
                wins_a += 1
        else:
            draws += 1
        
        results.append(result)
        
        if verbose:
            print(f"  Game {game_idx+1}/{num_games}: {result} "
                  f"({'A=white' if a_is_white else 'A=black'})")
    
    return {
        'wins_a': wins_a,
        'wins_b': wins_b,
        'draws': draws,
        'results': results,
        'win_rate_a': (wins_a + 0.5 * draws) / num_games if num_games > 0 else 0,
    }


def play_match_vs_alpha_beta(network: AlphaZeroNet, config,
                              num_games: int = 20,
                              network_color: str = "alternating",
                              verbose: bool = False,
                              on_move=None,
                              live_game=None,
                              game_counter: int = 0) -> dict:
    """Play a match between a network and the alpha-beta reference opponent.
    
    Args:
        network: The neural network
        config: Config object
        num_games: Number of games
        network_color: "alternating", "white", or "black"
        verbose: Print progress
    
    Returns:
        Dict with wins, losses, draws
    """
    wins = 0
    losses = 0
    draws = 0
    ab_depth = config.alpha_beta.depth
    
    for game_idx in range(num_games):
        if network_color == "alternating":
            network_is_white = (game_idx % 2 == 0)
        elif network_color == "white":
            network_is_white = True
        else:
            network_is_white = False
        
        # Notify live game viewer of new game
        if live_game is not None:
            live_game.start_game(game_counter + game_idx, config.get('step', 0) if hasattr(config, 'get') else 0,
                                 game_type="reference",
                                 match_info=f"Reference Game {game_idx+1}/{num_games}")
        
        board = chess.Board()
        move_count = 0
        
        mcts = MCTS(
            network=network,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            batch_size=getattr(config.mcts, 'batch_size', 1),
            c_virtual_loss=getattr(config.mcts, 'c_virtual_loss', 0.5),
        )
        
        while not board.is_game_over() and move_count < config.selfplay.max_game_length:
            is_network_turn = (board.turn == chess.WHITE) == network_is_white
            
            if is_network_turn:
                # Network's turn — use MCTS
                root = mcts.get_root(board)
                mcts.search(root)
                _, selected_move = mcts.select_move_with_temperature(root, temperature=0.1)
                if selected_move is None:
                    break
                # Build MCTS stats for callback
                mcts_stats = _build_mcts_stats(root, selected_move) if on_move else None
                board.push(selected_move)
                if on_move:
                    on_move(board.fen(), selected_move.uci(), move_count, mcts_stats=mcts_stats)
            else:
                # Alpha-beta's turn
                move = alpha_beta_best_move(board, ab_depth)
                if move is None:
                    break
                board.push(move)
                if on_move:
                    on_move(board.fen(), move.uci(), move_count)
            
            move_count += 1
        
        result = board.result()
        if result == "1-0":
            if network_is_white:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if network_is_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
        
        # Notify live game viewer that game is over
        if live_game is not None:
            live_game.game_over(result, 'reference_game')
        
        if verbose:
            print(f"  Game {game_idx+1}/{num_games}: {result} "
                  f"({'net=white' if network_is_white else 'net=black'})")
    
    win_rate = (wins + 0.5 * draws) / num_games if num_games > 0 else 0
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'games_played': num_games,
    }


# ============================================================
# Internal Helpers
# ============================================================

def _build_mcts_stats(root, selected_move):
    """Build MCTS candidate stats dict from a search root."""
    candidates = []
    if root.children:
        for move, child in sorted(root.children.items(), key=lambda x: x[1].N, reverse=True)[:10]:
            candidates.append({
                'move': move.uci(),
                'N': child.N,
                'W': float(child.W),
                'Q': float(child.Q),
                'P': float(child.P),
            })
    return {
        'selected_move': selected_move.uci(),
        'candidates': candidates,
    }


def _play_evaluation_game(mcts_a: MCTS, mcts_b: MCTS, a_is_white: bool,
                           max_moves: int = 150, on_move=None) -> str:
    """Play a single evaluation game between two MCTS agents."""
    board = chess.Board()
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        is_a_turn = (board.turn == chess.WHITE) == a_is_white
        mcts = mcts_a if is_a_turn else mcts_b
        
        root = mcts.get_root(board)
        mcts.search(root)
        _, selected_move = mcts.select_move_with_temperature(root, temperature=0.1)
        
        if selected_move is None:
            break
        
        # Build MCTS stats for callback
        mcts_stats = _build_mcts_stats(root, selected_move) if on_move else None
        
        board.push(selected_move)
        move_count += 1
        
        if on_move:
            on_move(board.fen(), selected_move.uci(), move_count, mcts_stats=mcts_stats)
    
    return board.result() if board.is_game_over() else "*"


# ============================================================
# Evaluator — orchestrates periodic evaluation
# ============================================================

class Evaluator:
    """Manages periodic evaluation: gating matches, reference matches, Elo tracking."""
    
    def __init__(self, config, stats_logger, live_game=None):
        self.config = config
        self.stats = stats_logger
        self.best_elo = 1000.0
        self.ref_elo = 800.0  # Starting Elo for the alpha-beta reference
        self.live_game = live_game  # LiveGameState for eval game viewing
    
    def run_gating_match(self, latest_network: AlphaZeroNet, 
                         best_network: AlphaZeroNet,
                         step: int, verbose: bool = False,
                         game_counter: int = 0) -> dict:
        """Run a gating match between latest and best network.
        
        Args:
            game_counter: Starting game index for numbering (for display)
        
        Returns dict with promotion result.
        """
        num_games = self.config.evaluation.gate_games
        
        # Create per-game callbacks for live board updates if live_game is available
        def _make_on_move(game_idx):
            if self.live_game is None:
                return None
            def _on_move(fen, uci, move_num, mcts_stats=None):
                self.live_game.update(fen, uci, move_num, mcts_stats=mcts_stats)
            return _on_move
        
        match_result = play_match(
            latest_network, best_network, self.config,
            num_games=num_games, verbose=verbose,
            on_move=_make_on_move(0),
            live_game=self.live_game,
            game_counter=game_counter,
        )
        
        win_rate = match_result['win_rate_a']
        promoted = win_rate > self.config.evaluation.gate_win_threshold
        
        # Log promotion attempt
        self.stats.log_promotion_attempt(
            step=step,
            promoted=promoted,
            win_rate=win_rate,
            games_played=num_games,
            wins=match_result['wins_a'],
            losses=match_result['wins_b'],
            draws=match_result['draws'],
            new_elo=self.best_elo,
            old_elo=self.best_elo,
        )
        
        # Update Elo (network A = latest, network B = best)
        score_a = match_result['win_rate_a']
        k = self.config.evaluation.elo_k_factor
        old_elo = self.best_elo
        self.best_elo = elo_update(self.best_elo, self.best_elo, score_a, k)
        
        self.stats.log_elo(
            self.best_elo, "gating", step,
            num_games, match_result['wins_a'], match_result['wins_b'], match_result['draws']
        )
        
        if verbose:
            print(f"Gating: {match_result['wins_a']}W-{match_result['wins_b']}L-{match_result['draws']}D "
                  f"(win_rate={win_rate:.2%}, promoted={promoted})")
        
        return {
            'promoted': promoted,
            'win_rate': win_rate,
            **match_result,
        }
    
    def run_reference_match(self, network: AlphaZeroNet,
                            step: int, verbose: bool = False,
                            game_counter: int = 0) -> dict:
        """Run evaluation matches against the alpha-beta reference opponent.
        
        Args:
            game_counter: Starting game index for numbering (for display)
        """
        num_games = self.config.evaluation.ref_opponent_games
        
        # Create per-game callbacks for live board updates if live_game is available
        def _make_on_move_ref(game_idx):
            if self.live_game is None:
                return None
            def _on_move_ref(fen, uci, move_num, mcts_stats=None):
                self.live_game.update(fen, uci, move_num, mcts_stats=mcts_stats)
            return _on_move_ref
        
        match_result = play_match_vs_alpha_beta(
            network, self.config,
            num_games=num_games, verbose=verbose,
            on_move=_make_on_move_ref(0),
            live_game=self.live_game,
            game_counter=game_counter,
        )
        
        # Log evaluation
        self.stats.log_evaluation(
            step=step, opponent="alpha_beta_ref",
            games_played=num_games,
            wins=match_result['wins'],
            losses=match_result['losses'],
            draws=match_result['draws'],
            win_rate=match_result['win_rate'],
        )
        
        # Update Elo for reference opponent
        # Network vs reference: score network
        k = self.config.evaluation.elo_k_factor
        net_elo = self.ref_elo + 200  # Assume network starts 200 points above ref
        new_net_elo = elo_update(net_elo, self.ref_elo, match_result['win_rate'], k)
        
        self.stats.log_elo(
            new_net_elo, "alpha_beta_ref", step,
            num_games, match_result['wins'], match_result['losses'], match_result['draws']
        )
        
        if verbose:
            print(f"vs Alpha-Beta: {match_result['wins']}W-{match_result['losses']}L"
                  f"-{match_result['draws']}D (win_rate={match_result['win_rate']:.2%})")
        
        return match_result