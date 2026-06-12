# AlphaZero Chess Engine

A minimal AlphaZero-style chess engine that learns purely through self-play, starting from a randomly initialized network with zero human game data.

## Quick Start

### Prerequisites
- Python 3.10+
- Dependencies: `pip install python-chess torch numpy flask flask-socketio pyyaml`

### Run Sanity Check
```bash
python sanity_check.py
```
This runs a tiny configuration (2 residual blocks, 16 filters, 10 MCTS sims) through the full pipeline: encoding → network → MCTS → self-play → training → stats. Takes ~30 seconds.

### Start Training
```bash
# Basic training (no GUI)
python main.py train

# Training with GUI
python main.py train --gui

# Launch GUI only (reads existing stats DB)
python main.py gui
```

### CLI Overrides
```bash
# Use fewer simulations for faster iteration
python main.py train --sims 50

# Smaller network
python main.py train --blocks 2 --filters 32

# Custom run name (affects output directory)
python main.py train --run-name my_experiment
```

## Architecture

### Board Encoding (18 planes, 8×8×18)
| Planes | Description |
|--------|-------------|
| 0–5 | White P, N, B, R, Q, K |
| 6–11 | Black P, N, B, R, Q, K |
| 12 | Side to move |
| 13–16 | Castling rights (WK, WQ, BK, BQ) |
| 17 | En passant square |

### Move Encoding (8×8×73 = 5,851 actions)
- Planes 0–55: Queen-like moves (8 directions × 7 distances)
- Planes 56–63: Knight moves (8 offsets)
- Planes 64–72: Underpromotions (N/B/R × forward/forward-left/forward-right)
- Queen promotions are encoded as queen-like moves (forward direction)

### Network
- Small ResNet: configurable residual blocks (default 4) and filters (default 64)
- Policy head: outputs 5,851 move probabilities
- Value head: outputs scalar in [-1, 1] (expected game outcome)
- Random initialization — no pretraining

### MCTS (PUCT)
- Configurable simulations per move (default 200, tunable 50–800)
- Dirichlet noise at root for exploration (α=0.3, ε=0.25)
- Temperature-based move selection (τ=1.0 opening, τ=0.1 middlegame)

### Self-Play Design
- **Always uses the latest network** (not gated "best") — prevents stagnation from evaluation variance
- Gating/Elo are purely monitoring signals, not training gates
- Material adjudication at 150 half-move cap (configurable)
- FIFO replay buffer (default 100k positions)
- No data augmentation in v1

### Training
- Policy loss: cross-entropy against MCTS visit distribution
- Value loss: MSE against game outcome
- Both losses logged separately to SQLite and displayed on dashboard
- SGD optimizer with momentum

## Configuration

All hyperparameters are in `config.yaml`:

```yaml
network:
  num_residual_blocks: 4    # ResNet depth
  num_filters: 64           # Channel width

mcts:
  num_simulations: 200      # Per move
  c_puct: 1.5               # Exploration constant

selfplay:
  network_source: latest    # "latest" or "gated_best"
  max_game_length: 150      # Half-move cap
  temperature_threshold: 30 # Move to switch temp

training:
  batch_size: 256
  learning_rate: 0.001
```

## GUI Dashboard

Launch with `python main.py train --gui` or standalone `python main.py gui`.

The dashboard at http://127.0.0.1:5000 shows:
- **Summary stats**: total games, win/loss/draw counts, Elo, losses
- **Training loss chart**: separate policy, value, and total loss curves
- **Elo rating over time**
- **Game length trends**
- **Network confidence**: avg max policy probability and |value| predictions
- **Replay buffer composition**: rolling outcome distribution
- **Win/draw/loss rates over time** (rolling window)
- **Promotion attempts table**: all gating match results with win rates (gate stagnation diagnostics)
- **Recent games table**: result, length, termination reason, MCTS depth

Auto-refreshes every 2 seconds via polling.

## Project Structure

```
├── config.yaml        # All hyperparameters
├── config.py          # Config loading with CLI overrides
├── encoding.py        # Board encoding (18 planes) + move encoding (8×8×73)
├── network.py         # Dual-head ResNet (PyTorch)
├── mcts.py            # PUCT MCTS with Dirichlet noise
├── selfplay.py        # Self-play game generation + replay buffer
├── training.py        # Training loop (policy + value loss)
├── evaluation.py      # Gating + alpha-beta reference + Elo tracking
├── stats.py           # SQLite persistence for all metrics
├── main.py            # Entry point orchestrating everything
├── sanity_check.py    # Quick pipeline smoke test
├── gui/
│   ├── app.py         # Flask + SocketIO server
│   └── templates/
│       └── index.html # Dashboard frontend
└── README.md
```

## Output

Training creates `output/<run_name>/`:
- `checkpoints/latest.pt` — latest model checkpoint
- `checkpoints/step_N.pt` — periodic checkpoints
- `stats.db` — SQLite database with all metrics

## Expected Early Learning Trajectory

1. **Games 0–20**: Mostly draws from random play / early repetition. Network outputs near-uniform policies.
2. **Games 20–100**: Some decisive results appear as MCTS learns basic tactics. Network value predictions begin moving away from 0.
3. **Games 100–500**: Clearer patterns emerge. Network starts exploiting simple tactical oversights in MCTS-guided play. Win/draw ratios shift.
4. **Games 500+**: Should see steady Elo improvement against the depth-2 alpha-beta reference. Policy loss should decrease visibly.

With 200 MCTS sims on CPU, expect ~5–20 self-play games per hour depending on hardware. The full training loop (self-play + training + eval) is designed to run continuously.

## Key Design Decisions

1. **Latest network for self-play**: Self-play always uses the newest trained network, not the gated "best". This prevents training stagnation from unlucky variance in evaluation matches.

2. **No rotation augmentation**: Chess positions are not invariant under 90°/270° rotations (pawns would move sideways). Only horizontal mirroring is a valid augmentation, and it's omitted in v1 for simplicity.

3. **Material adjudication**: When the 150 half-move cap is hit, the result is adjudicated by material count rather than declaring a draw. This avoids degenerate repetition-draws during early random play. Can be disabled via `adjudicate_material: false`.

4. **Separate loss tracking**: Policy loss and value loss are tracked independently throughout training and on the dashboard, catching early collapse in either head.