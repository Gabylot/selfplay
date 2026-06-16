"""Check the value-target sign convention.

For any game where White won (outcome=+1), the FIRST position in that
game (White to move, since White moves first) should have a stored
value of approximately +1 — because:
    value = outcome * player
    player = +1.0 if board.turn == chess.WHITE else -1.0
    => for White's first move: value = (+1) * (+1) = +1

For any game where Black won (outcome=-1), the first position
(still White to move, since it's still move 1) should have a stored
value of approximately -1:
    value = (-1) * (+1) = -1

If this doesn't hold, the sign convention is flipped somewhere and
the network is being trained on backwards value targets — which would
systematically bias it toward whichever side it's "told" is winning,
independent of actual move quality.

HOW TO RUN
==========
This script re-plays a handful of fresh self-play games using your
current network + MCTS, and checks the very first stored value against
the game's outcome. It does NOT touch your existing replay buffer or
training — it's read-only / throwaway games, just for this check.

    python tests/test_value_sign_convention.py

It will print one line per game: the outcome, the first position's
stored value, and whether they match the expected sign. At the end it
prints a summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import chess

from config import get_config
from network import AlphaZeroNet
from mcts import MCTS
from selfplay import play_one_game


def main():
    config = get_config()

    # Load whatever checkpoint your training loop currently uses.
    # Adjust this path if your checkpoints live somewhere else.
    net = AlphaZeroNet(
        num_residual_blocks=config.network.num_residual_blocks,
        num_filters=config.network.num_filters,
        num_policy_channels=config.network.num_policy_channels,
        num_value_channels=config.network.num_value_channels,
        value_fc_size=config.network.value_fc_size,
    )

    import torch
    import glob
    ckpts = sorted(glob.glob("output/*/checkpoints/*.pt")) + sorted(glob.glob("checkpoints/*.pt"))
    if ckpts:
        print(f"Loading checkpoint: {ckpts[-1]}")
        state = torch.load(ckpts[-1], map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            net.load_state_dict(state["model_state_dict"])
        else:
            net.load_state_dict(state)
    else:
        print("No checkpoint found — using randomly initialized network "
              "(still fine for this check, sign convention doesn't depend "
              "on training progress).")

    net.eval()

    mcts = MCTS(
        network=net,
        num_simulations=50,  # low — we only care about game outcomes, not strength
        c_puct=config.mcts.c_puct,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        dirichlet_epsilon=config.mcts.dirichlet_epsilon,
        batch_size=getattr(config.mcts, "batch_size", 1),
        c_virtual_loss=getattr(config.mcts, "c_virtual_loss", 0.5),
    )

    num_games = 10
    mismatches = 0
    results_seen = {"white_win": 0, "black_win": 0, "draw": 0}

    print(f"\nPlaying {num_games} games to check value sign convention...\n")

    for game_num in range(num_games):
        game_data, game_info = play_one_game(
            mcts_engine=mcts,
            max_game_length=60,  # short — we just need a result, not a full game
            adjudicate_material=True,
            piece_values=config.selfplay.piece_values,
            temp_threshold=config.selfplay.temperature_threshold,
            temp_high=config.selfplay.temperature_high,
            temp_low=config.selfplay.temperature_low,
        )

        outcome = game_info["result"]  # +1 white win, -1 black win, 0 draw

        if outcome > 0:
            results_seen["white_win"] += 1
            outcome_label = "White win"
            expected_first_value = +1.0
        elif outcome < 0:
            results_seen["black_win"] += 1
            outcome_label = "Black win"
            expected_first_value = -1.0
        else:
            results_seen["draw"] += 1
            outcome_label = "Draw"
            expected_first_value = 0.0

        if not game_data:
            print(f"Game {game_num}: {outcome_label} (outcome={outcome}) "
                  f"— no positions stored, skipping")
            continue

        first_value = game_data[0][2]  # (state, policy, value) tuple

        if outcome == 0:
            # For draws, expected is 0 — check it's close to 0
            match = abs(first_value - expected_first_value) < 0.01
        else:
            # For decisive games, check the SIGN matches
            match = (np.sign(first_value) == np.sign(expected_first_value))

        status = "OK" if match else "MISMATCH"
        if not match:
            mismatches += 1

        print(f"Game {game_num}: {outcome_label:10s} (outcome={outcome:+.1f}) | "
              f"first stored value={first_value:+.3f} | "
              f"expected sign={'+'if expected_first_value > 0 else '-' if expected_first_value < 0 else '0'} "
              f"| {status}")

    print(f"\n{'=' * 60}")
    print(f"Results seen: {results_seen}")
    print(f"Mismatches: {mismatches} / {num_games}")
    print(f"{'=' * 60}")

    if mismatches == 0:
        print("\nALL OK — value sign convention is correct.")
        print("The White/Black win-rate gap is most likely a real "
              "(if early/exaggerated) first-move effect, not a sign bug.")
    else:
        print("\nMISMATCH FOUND — the value sign convention appears flipped "
              "or inconsistent for at least one decisive game.")
        print("This means value targets may be systematically wrong, which")
        print("could explain the win-rate skew as a training artifact rather")
        print("than genuine chess understanding. Send this output back for")
        print("further debugging.")

    if results_seen["white_win"] == 0 or results_seen["black_win"] == 0:
        print("\nNOTE: this run didn't produce both White-win AND Black-win "
              "games (likely because max_game_length=60 is short and most "
              "games hit the cap as draws or one-sided material). If that's "
              "the case here, increase num_games or max_game_length and "
              "rerun until you get at least a few of each.")


if __name__ == "__main__":
    main()