"""Evaluation, gating, and Elo tracking for AlphaZero.

- Gating: periodically pit latest trained network vs current "best" network
- Reference opponent: a fast game-specific reference player (chess: depth-2
  alpha-beta with material evaluation; connect4: greedy solver heuristic)
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

import game_api
from network import AlphaZeroNet
from mcts import MCTS
from selfplay import play_two_engine_match, play_engine_vs_reference


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 32) -> float:
    """Update rating_a based on match result. Returns new rating for A."""
    expected = elo_expected_score(rating_a, rating_b)
    return rating_a + k * (score_a - expected)


# ============================================================
# Alpha-Beta Reference Opponent (depth-4, material only)
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


def alpha_beta_search(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """Negamax with alpha-beta pruning.

    Returns evaluation from the perspective of the side to move.
    The score is negated at every recursion level (standard negamax),
    so callers always maximize on the value returned.
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
    best = -float('inf')
    for move in legal_moves:
        board.push(move)
        score = -alpha_beta_search(board, depth - 1, -beta, -alpha)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def alpha_beta_best_move(board: chess.Board, depth: int = 4) -> Optional[chess.Move]:
    """Find the best move using negamax with alpha-beta pruning.

    Returns the best move from the current side's perspective.  The caller
    should pass an independent copy of the board if it needs to keep its own
    history stack intact (this function pushes/pops on the shared board).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = None
    best_score = -float('inf')
    for move in legal_moves:
        board.push(move)
        score = -alpha_beta_search(board, depth - 1, -float('inf'), -best_score)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move if best_move else legal_moves[0]


def play_game_alpha_beta_vs_alpha_beta(depth: int = 4, 
                                        max_moves: int = 200) -> Tuple[str, int]:
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
            mate_seeding=getattr(config.mcts, 'mate_seeding', False),
            min_fresh_sims=getattr(config.mcts, 'min_fresh_sims', 64),
            force_mate_in_one=getattr(config.mcts, 'force_mate_in_one', False),
        )
        mcts_b = MCTS(
            network=network_b,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            batch_size=getattr(config.mcts, 'batch_size', 1),
            c_virtual_loss=getattr(config.mcts, 'c_virtual_loss', 0.5),
            mate_seeding=getattr(config.mcts, 'mate_seeding', False),
            min_fresh_sims=getattr(config.mcts, 'min_fresh_sims', 64),
        )
        
        # Notify live game viewer of new game
        if live_game is not None:
            live_game.start_game(game_counter + game_idx, config.get('step', 0) if hasattr(config, 'get') else 0,
                                 game_type="gating",
                                 match_info=f"Gating Game {game_idx+1}/{num_games}")
        
        result = _play_evaluation_game(mcts_a, mcts_b, a_is_white,
                                        max_moves=config.selfplay.max_game_length,
                                        on_move=on_move,
                                        adjudicate_material=getattr(config.selfplay, 'adjudicate_material', True),
                                        adjudicate_graded=getattr(config.selfplay, 'adjudicate_graded', True),
                                        adjudicate_scaling=getattr(config.selfplay, 'adjudicate_scaling', 9.0))
        
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
        
        board = None
        move_count = 0

        mcts = MCTS(
            network=network,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            batch_size=getattr(config.mcts, 'batch_size', 1),
            c_virtual_loss=getattr(config.mcts, 'c_virtual_loss', 0.5),
            mate_seeding=getattr(config.mcts, 'mate_seeding', False),
            min_fresh_sims=getattr(config.mcts, 'min_fresh_sims', 64),
        )

        _state, _move_count, result = play_engine_vs_reference(
            mcts, network_is_white, config.selfplay.max_game_length,
            depth=ab_depth, on_move=on_move,
            adjudicate_material=getattr(config.selfplay, 'adjudicate_material', True),
            adjudicate_graded=getattr(config.selfplay, 'adjudicate_graded', True),
            adjudicate_scaling=getattr(config.selfplay, 'adjudicate_scaling', 9.0))
        move_count = _move_count

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
                           max_moves: int = 150, on_move=None,
                           adjudicate_material=True, adjudicate_graded=True,
                           adjudicate_scaling=9.0) -> str:
    """Play a single evaluation game between two MCTS agents (any game)."""
    _state, _move_count, result = play_two_engine_match(
        mcts_a, mcts_b, a_is_white, max_moves, on_move=on_move,
        adjudicate_material=adjudicate_material,
        adjudicate_graded=adjudicate_graded,
        adjudicate_scaling=adjudicate_scaling)
    return result


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

        # Update Elo (network A = latest, network B = best) BEFORE logging the
        # promotion attempt so old_elo/new_elo reflect an actual change.
        # NOTE: gating is self-play (latest vs current best), so both ratings
        # are the same -> expected score is constant 0.5.  This is by design:
        # the "Elo" is a monitoring signal (a fixed-k walk on the gate win
        # rate), not a true two-player rating.  elo_update() with equal
        # ratings encodes exactly that.
        score_a = match_result['win_rate_a']
        k = self.config.evaluation.elo_k_factor
        old_elo = self.best_elo
        self.best_elo = elo_update(old_elo, old_elo, score_a, k)

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
            old_elo=old_elo,
        )

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