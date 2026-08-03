"""Main entry point for the AlphaZero Chess Engine.

Training loop:
  1. Push self-play tasks to all workers.
  2. Collect finished games, add to buffer.
  3. Every N games: train.
  4. Every M games: PAUSE self-play, run eval games through the same
     worker pool, RESUME self-play.

Usage:
    python main.py train [--gui] [--workers N]
    python main.py gui
    python main.py evaluate
    python main.py sanity
    python main.py benchmark [--workers N] [--benchmark-games N] [--profile-games N]
"""

import argparse, sys, os, time, threading, signal, io
from pathlib import Path

import torch
import numpy as np

from config import get_config
from network import AlphaZeroNet, create_model_from_config, save_checkpoint, load_checkpoint
from mcts import MCTS
from selfplay import self_play_game, ReplayBuffer, ParallelSelfPlay
from training import train_one_step, create_optimizer
from evaluation import Evaluator
from stats import StatsLogger
from tb_logger import TensorBoardLogger
from gui.live_game import LiveGameState


_shutdown = False
def signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[INFO] Shutdown signal received…")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_live_message(psp, worker_live_games, result):
    """Handle a single live_start / live_move / live_end result immediately
    so the GUI stays responsive.  Returns True if a live message was
    dispatched, False otherwise (e.g. for a full selfplay result)."""
    if result is None or result.get('done'):
        return False

    rtype = result.get('type')
    if rtype == 'live_start':
        wid = result['worker_id']
        wlg = worker_live_games[wid]
        wlg.start_game(0, 0,  # game_id/step set later by caller
                       game_type=result.get('game_type', 'selfplay'),
                       match_info=result.get('match_info'))
        return True

    if rtype == 'live_move':
        wid = result['worker_id']
        wlg = worker_live_games[wid]
        wlg.update(result['fen'], result['move'], result['move_number'],
                   mcts_stats=result.get('mcts_stats'))
        return True

    if rtype == 'live_end':
        wid = result['worker_id']
        wlg = worker_live_games[wid]
        wlg.game_over(result['result'], result.get('termination', ''))
        return True

    return False


def run_training(config, gui_enabled=False, num_workers=None):
    global _shutdown
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    use_gpu = getattr(config, 'inference', None) and getattr(config.inference, 'use_gpu', False)
    if use_gpu:
        print(f"[INFO] Inference: GPU (DirectML centralized server)")

    if num_workers is None:
        num_workers = getattr(config.selfplay, 'num_workers', 8)
    print(f"[INFO] Workers: {num_workers}")

    network = create_model_from_config(config)
    network.to(device)
    print(f"[INFO] Network: {sum(p.numel() for p in network.parameters())} params")

    output_dir      = Path(config.main.output_dir) / config.main.run_name
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    stats_db_path = output_dir / config.stats.db_path
    stats = StatsLogger(str(stats_db_path))

    # TensorBoard logger — created after checkpoint load so we can pass
    # the resumed step as purge_step.  This makes TensorBoard discard any
    # events at or beyond the resumed step from previous event files,
    # preventing overlapping step ranges that make plots appear to restart.
    tb_enabled = getattr(config, 'tensorboard', None) and getattr(config.tensorboard, 'enabled', True)
    tb_log_dir = str(output_dir / getattr(config.tensorboard, 'log_dir', 'runs')) if tb_enabled else None
    tb = None  # instantiated after checkpoint restore

    # One LiveGameState per worker (worker_id set, not eval)
    worker_live_games = [
        LiveGameState(max_history=5, worker_id=i, is_eval=False)
        for i in range(num_workers)
    ]
    # Separate eval board
    eval_live_game = LiveGameState(max_history=50, worker_id=-1, is_eval=True)

    buffer    = ReplayBuffer(max_size=config.buffer.max_size)
    evaluator = Evaluator(config, stats, live_game=eval_live_game)

    optimizer = create_optimizer(
        network,
        lr=float(config.training.learning_rate),
        momentum=float(config.training.momentum),
        weight_decay=float(config.training.weight_decay),
    )

    step = game_id = eval_game_counter = 0
    best_network = create_model_from_config(config)
    best_network.load_state_dict(network.state_dict())
    best_network.to(device)

    stats.log_config(step, config.to_dict())

    # GUI
    if gui_enabled:
        from gui.app import start_gui_server
        threading.Thread(
            target=start_gui_server,
            args=(stats, config, worker_live_games, eval_live_game),
            daemon=True,
        ).start()
        print(f"[INFO] GUI → http://{config.gui.host}:{config.gui.port}")

    # Load checkpoint
    latest_ckpt = checkpoints_dir / "latest.pt"
    if latest_ckpt.exists():
        print(f"[INFO] Loading checkpoint: {latest_ckpt}")
        ckpt = load_checkpoint(str(latest_ckpt), network, optimizer)
        step             = ckpt.get('step', 0)
        game_id          = ckpt.get('game_id', 0)
        eval_game_counter= ckpt.get('eval_game_counter', 0)
        if 'best_elo' in ckpt: evaluator.best_elo = ckpt['best_elo']
        if 'ref_elo'  in ckpt: evaluator.ref_elo  = ckpt['ref_elo']
        print(f"[INFO] Resumed step={step} game={game_id}")

    # Load best network from disk (if available) so it survives restarts.
    # If no best.pt exists, copy the latest checkpoint into best_network.
    best_pt_path = output_dir / "best.pt"
    if best_pt_path.exists():
        print(f"[INFO] Loading best network: {best_pt_path}")
        load_checkpoint(str(best_pt_path), best_network)
    else:
        best_network.load_state_dict(network.state_dict())
        print(f"[INFO] No best.pt found — best_network initialised from latest checkpoint")

    # Create TensorBoard logger now that we know the resumed step.
    # purge_step tells TensorBoard to discard any events at or beyond
    # this step from previous event files, so new logs continue
    # seamlessly from where the last run left off.
    tb = TensorBoardLogger(tb_log_dir, enabled=tb_enabled, initial_step=step)
    if tb_enabled:
        print(f"[INFO] TensorBoard logging → {tb_log_dir}  (resuming from step {step})")
        print(f"[INFO] Run: tensorboard --logdir {tb_log_dir}")
    tb.log_config(step, config.to_dict())

    # Load replay buffer
    buffer_path = checkpoints_dir / "replay_buffer.npz"
    loaded_buffer = ReplayBuffer.load(str(buffer_path), max_size=config.buffer.max_size)
    if loaded_buffer is not None:
        buffer = loaded_buffer
        print(f"[INFO] Loaded replay buffer: {len(buffer)} positions from {buffer.total_games} games")

    print(f"\n{'='*60}\n  AlphaZero — step={step} games={game_id} workers={num_workers}\n{'='*60}\n")

    psp = ParallelSelfPlay(config, num_workers=num_workers)
    psp.start()
    psp.push_selfplay(network)   # kick off first round

    games_since_train = 0
    positions_since_train = 0   # track new positions added between training steps
    games_since_eval  = 0
    train_interval    = config.training.training_steps_per_iteration
    eval_interval     = config.evaluation.eval_interval

    # Per-worker live game tracking for real-time updates
    worker_live_game_ids = [0] * num_workers

    try:
        while not _shutdown:
            result = psp.collect_one(timeout=300.0)
            if result is None:
                print("[WARN] No result in 5 min — workers may be stuck")
                continue
            if result.get('done'):
                continue

            # ── Live incremental messages (real-time GUI updates) ──
            if _dispatch_live_message(psp, worker_live_games, result):
                continue

            # Only process self-play results in the main loop
            if result.get('type') != 'selfplay':
                continue

            wid = result['worker_id']

            # Deserialise
            raw       = result['game_data']
            game_data = [(np.array(s,dtype=np.float32),
                        np.array(p,dtype=np.float32),
                        float(v),
                        np.array(m,dtype=np.float32)) for s,p,v,m in raw]
            game_info = result['game_info']
            fens      = result.get('fens', [])
            moves     = result.get('moves', [])
            mcts_s    = result.get('mcts_stats', [])

            positions_since_train += game_info['num_positions']   # track new data
            buffer.add_game(game_data)
            game_id          += 1
            games_since_train += 1
            games_since_eval  += 1

            # GUI tile already updated via live_start/live_move/live_end messages.
            # No need to update LiveGameState again here since the live messages
            # already did the incremental updates.

            # Stats
            stats.log_game(game_id=game_id, step=step,
                           result=game_info['result'], result_str=game_info['result_str'],
                           length=game_info['length'], termination=game_info['termination'],
                           avg_mcts_depth=game_info['avg_mcts_depth'],
                           num_positions=game_info['num_positions'],
                           material_diff=game_info.get('material_diff', 0))
            tb.log_game(game_id=game_id, step=step,
                        result=game_info['result'], result_str=game_info['result_str'],
                        length=game_info['length'], termination=game_info['termination'],
                        avg_mcts_depth=game_info['avg_mcts_depth'],
                        num_positions=game_info['num_positions'],
                        material_diff=game_info.get('material_diff', 0))
            stats.log_mcts_stats(game_id=game_id, step=step,
                                 avg_tree_depth=game_info['avg_mcts_depth'],
                                 avg_sims_per_move=config.mcts.num_simulations)
            tb.log_mcts_stats(game_id=game_id, step=step,
                              avg_tree_depth=game_info['avg_mcts_depth'],
                              avg_sims_per_move=config.mcts.num_simulations)

            print(f"  [W{wid}] Game {game_id}: {game_info['termination']:18s} | "
                  f"{game_info['result_str']} | {game_info['length']} moves | buf={len(buffer)}")

            # ── Push next self-play task to this worker immediately ──
            import torch as _torch, io as _io
            buf2 = _io.BytesIO()
            _torch.save(network.state_dict(), buf2)
            wb = buf2.getvalue()
            try: psp._task_qs[wid].put_nowait({'type':'selfplay','weights':wb})
            except: pass

            # ── Training ──
            if (games_since_train >= train_interval
                    and len(buffer) >= int(config.training.batch_size)):
                games_since_train = 0

                # Log how many new positions arrived since the last training step
                print(f"[Cycle] Added {positions_since_train} positions, buffer size now {len(buffer)}")
                positions_since_train = 0

                # Drain live GUI messages BEFORE and BETWEEN training batches
                # so the browser stays in sync instead of showing a burst later.
                _pending_selfplay = []
                for r in psp.collect_available():
                    if _dispatch_live_message(psp, worker_live_games, r):
                        pass
                    elif r.get('type') == 'selfplay' and not r.get('done'):
                        _pending_selfplay.append(r)

                for _ in range(config.training.num_batches_per_step):
                    if _shutdown: break
                    # Drain any new live messages between batches
                    for r in psp.collect_available():
                        if _dispatch_live_message(psp, worker_live_games, r):
                            pass
                        elif r.get('type') == 'selfplay' and not r.get('done'):
                            _pending_selfplay.append(r)

                    ld = train_one_step(network, optimizer, buffer,
                                        int(config.training.batch_size), device)
                    step += 1
                    stats.log_training_step(step=step,
                                            policy_loss=ld['policy_loss'],
                                            value_loss=ld['value_loss'],
                                            total_loss=ld['total_loss'],
                                            learning_rate=config.training.learning_rate)
                    tb.log_training_step(step=step,
                                         policy_loss=ld['policy_loss'],
                                         value_loss=ld['value_loss'],
                                         total_loss=ld['total_loss'],
                                         learning_rate=config.training.learning_rate,
                                         grad_norm=ld['grad_norm'])

                # ── Push updated weights to the GPU inference server ──
                # train_one_step() calls model.train() and updates weights via
                # optimizer.step().  In GPU mode, workers use InferenceClient and
                # never load per-task weights — the GPU server must be explicitly
                # notified.  Without this, self-play between eval cycles uses
                # stale weights, producing near-zero Q values.
                network.eval()
                psp.push_weights_to_gpu(network)

                # ── Handle self-play results that arrived while training ──
                for pr in _pending_selfplay:
                    pw   = pr['worker_id']
                    praw = pr['game_data']
                    pgi  = pr['game_info']
                    pgd  = [(np.array(s,dtype=np.float32),
                            np.array(p,dtype=np.float32),
                            float(v),
                            np.array(m,dtype=np.float32)) for s,p,v,m in praw]
                    positions_since_train += pgi['num_positions']   # track pending games too
                    buffer.add_game(pgd)
                    game_id          += 1
                    games_since_train += 1
                    games_since_eval  += 1

                    stats.log_game(game_id=game_id, step=step,
                                   result=pgi['result'], result_str=pgi['result_str'],
                                   length=pgi['length'], termination=pgi['termination'],
                                   avg_mcts_depth=pgi['avg_mcts_depth'],
                                   num_positions=pgi['num_positions'],
                                   material_diff=pgi.get('material_diff', 0))
                    tb.log_game(game_id=game_id, step=step,
                                result=pgi['result'], result_str=pgi['result_str'],
                                length=pgi['length'], termination=pgi['termination'],
                                avg_mcts_depth=pgi['avg_mcts_depth'],
                                num_positions=pgi['num_positions'],
                                material_diff=pgi.get('material_diff', 0))
                    stats.log_mcts_stats(game_id=game_id, step=step,
                                         avg_tree_depth=pgi['avg_mcts_depth'],
                                         avg_sims_per_move=config.mcts.num_simulations)
                    tb.log_mcts_stats(game_id=game_id, step=step,
                                      avg_tree_depth=pgi['avg_mcts_depth'],
                                      avg_sims_per_move=config.mcts.num_simulations)

                    print(f"  [W{pw}] Game {game_id}: {pgi['termination']:18s} | "
                          f"{pgi['result_str']} | {pgi['length']} moves | buf={len(buffer)}")

                    import torch as _torch2, io as _io2
                    _b = _io2.BytesIO(); _torch2.save(network.state_dict(), _b); _wb2 = _b.getvalue()
                    try: psp._task_qs[pw].put_nowait({'type':'selfplay','weights':_wb2})
                    except: pass

                # Buffer / confidence stats
                od = buffer.get_outcome_distribution()
                stats.log_buffer_stats(step=step, buffer_size=len(buffer),
                                       white_wins=od['white_wins'],
                                       black_wins=od['black_wins'], draws=od['draws'])
                tb.log_buffer_stats(step=step, buffer_size=len(buffer),
                                    white_wins=od['white_wins'],
                                    black_wins=od['black_wins'], draws=od['draws'])
                if len(buffer) > 0:
                    n   = min(50, len(buffer))
                    ix  = np.random.choice(len(buffer), size=n, replace=False)
                    sts = np.array([buffer.buffer[i][0] for i in ix])
                    ps, vs = network.predictBatch(sts)
                    stats.log_network_stats(step,
                                            float(np.mean(np.max(ps,axis=1))),
                                            float(np.mean(np.abs(vs))))
                    tb.log_network_stats(step,
                                         float(np.mean(np.max(ps,axis=1))),
                                         float(np.mean(np.abs(vs))))

                # Checkpoint
                if step % config.training.checkpoint_interval == 0:
                    ex = {'game_id':game_id,'eval_game_counter':eval_game_counter,
                          'best_elo':evaluator.best_elo,'ref_elo':evaluator.ref_elo}
                    cp = checkpoints_dir / f"step_{step}.pt"
                    save_checkpoint(network, optimizer, str(cp), step, extra=ex)
                    save_checkpoint(network, optimizer,
                                    str(checkpoints_dir/"latest.pt"), step, extra=ex)
                    buffer.save(str(buffer_path))
                    print(f"  Checkpoint: {cp}  Buffer: {len(buffer)} positions")

                print(f"  [train] step={step} pol={ld['policy_loss']:.4f} val={ld['value_loss']:.4f}")

            # ── Eval ──────────────────────────────────────────────────────
            if games_since_eval >= eval_interval and not _shutdown:
                games_since_eval = 0
                print(f"\n[Step {step}] === EVAL (self-play paused) ===")

                # Drain any in-flight self-play results so we don't mix them
                # with eval results in the collector loop below.
                time.sleep(0.5)          # let in-flight results land
                psp.drain()              # discard them

                # Serialise both networks once
                import torch as _t, io as _i
                def _wb(net):
                    b=_i.BytesIO(); _t.save(net.state_dict(),b); return b.getvalue()
                wb_latest = _wb(network)
                wb_best   = _wb(best_network)

                # Push both networks' weights to the GPU server for eval
                if getattr(config, 'inference', None) and getattr(config.inference, 'use_gpu', False):
                    psp.push_eval_weights(wb_latest, wb_best)

                n_gate = config.evaluation.gate_games
                n_ref  = config.evaluation.ref_opponent_games
                total_eval = n_gate + n_ref

                # Build task list: half-and-half colors
                eval_tasks = []
                for gi in range(n_gate):
                    eval_tasks.append({
                        'type':'eval', 'eval_type':'gating',
                        'weights_a': wb_latest, 'weights_b': wb_best,
                        'a_is_white': (gi % 2 == 0),
                        'game_label': f"Gate {gi+1}/{n_gate}",
                    })
                for gi in range(n_ref):
                    eval_tasks.append({
                        'type':'eval', 'eval_type':'reference',
                        'weights_a': wb_latest, 'weights_b': None,
                        'a_is_white': (gi % 2 == 0),
                        'game_label': f"Ref {gi+1}/{n_ref}",
                    })

                dispatched = psp.dispatch_eval_games(eval_tasks)
                print(f"  Dispatched {dispatched} eval games to {num_workers} workers")

                # Collect eval results
                gate_wins=gate_losses=gate_draws=0
                ref_wins=ref_losses=ref_draws=0
                collected = 0
                eval_game_id_start = eval_game_counter
                eval_live_start_ids = [0] * num_workers  # per-worker eval game counter

                while collected < dispatched and not _shutdown:
                    r = psp.collect_one(timeout=300.0)
                    if r is None:
                        print("[WARN] Eval result timeout"); break
                    if r.get('done'): continue

                    rt = r.get('type')

                    # ── Live incremental messages for eval games ──
                    # Only update worker tiles in real-time (the single eval board
                    # is updated from complete results to avoid interleaving moves
                    # from parallel eval games).
                    if rt == 'live_start':
                        wid = r['worker_id']
                        eval_live_start_ids[wid] += 1
                        gid_el = eval_game_counter + eval_live_start_ids[wid]
                        gt = r.get('game_type', 'reference')
                        label = r.get('match_info', '')
                        wlg = worker_live_games[wid]
                        wlg.start_game(gid_el, step, game_type=gt, match_info=label)
                        continue

                    if rt == 'live_move':
                        wid = r['worker_id']
                        wlg = worker_live_games[wid]
                        wlg.update(r['fen'], r['move'], r['move_number'],
                                   mcts_stats=r.get('mcts_stats'))
                        continue

                    if rt == 'live_end':
                        wid = r['worker_id']
                        wlg = worker_live_games[wid]
                        wlg.game_over(r['result'], r.get('termination', ''))
                        continue

                    if rt != 'eval':
                        continue   # stray self-play result — discard

                    wid       = r['worker_id']
                    res       = r['result']
                    etype     = r['eval_type']
                    a_white   = r['a_is_white']
                    label     = r['game_label']
                    fens_e    = r.get('fens',[])
                    moves_e   = r.get('moves',[])
                    mcts_e    = r.get('mcts_stats',[])
                    collected += 1

                    # Worker tiles already updated via live_start/live_move/live_end.
                    # Update the dedicated eval board from complete results (batched)
                    # to avoid interleaving moves from parallel eval games.
                    wlg = worker_live_games[wid]
                    gt  = 'gating' if etype=='gating' else 'reference'
                    gid_e = eval_game_counter + collected
                    # Replay the full game on the eval board
                    eval_live_game.start_game(gid_e, step, game_type=gt, match_info=label)
                    for i,(fen,uci) in enumerate(zip(fens_e, moves_e)):
                        ms = mcts_e[i] if i < len(mcts_e) else None
                        eval_live_game.update(fen, uci, i+1, mcts_stats=ms)
                    eval_live_game.game_over(res, gt)

                    # Tally
                    if etype == 'gating':
                        if res=='1-0':
                            if a_white: gate_wins+=1
                            else:       gate_losses+=1
                        elif res=='0-1':
                            if a_white: gate_losses+=1
                            else:       gate_wins+=1
                        else: gate_draws+=1
                    else:
                        if res=='1-0':
                            if a_white: ref_wins+=1
                            else:       ref_losses+=1
                        elif res=='0-1':
                            if a_white: ref_losses+=1
                            else:       ref_wins+=1
                        else: ref_draws+=1

                    print(f"  [{etype[:4].upper()} W{wid}] {label}: {res}")

                eval_game_counter += collected

                # Gating decision
                total_g = gate_wins+gate_losses+gate_draws
                if total_g > 0:
                    wr = (gate_wins + 0.5*gate_draws) / total_g
                    promoted = wr > config.evaluation.gate_win_threshold
                    if promoted:
                        best_network.load_state_dict(network.state_dict())
                        save_checkpoint(best_network, None, str(output_dir / "best.pt"), step=step)
                        print(f"  [GATE] PROMOTED  win_rate={wr:.1%}  best.pt saved")
                    else:
                        print(f"  [GATE] not promoted  win_rate={wr:.1%}")
                    k = config.evaluation.elo_k_factor
                    old_elo = evaluator.best_elo
                    evaluator.best_elo = old_elo + k*(wr - 0.5)
                    stats.log_promotion_attempt(step=step, promoted=promoted,
                                                win_rate=wr, games_played=total_g,
                                                wins=gate_wins, losses=gate_losses,
                                                draws=gate_draws,
                                                old_elo=old_elo, new_elo=evaluator.best_elo)
                    tb.log_promotion_attempt(step=step, promoted=promoted,
                                             win_rate=wr, games_played=total_g,
                                             wins=gate_wins, losses=gate_losses,
                                             draws=gate_draws,
                                             old_elo=old_elo, new_elo=evaluator.best_elo)
                    stats.log_elo(evaluator.best_elo,"gating",step,
                                  total_g,gate_wins,gate_losses,gate_draws)
                    tb.log_elo(evaluator.best_elo,"gating",step,
                               total_g,gate_wins,gate_losses,gate_draws)

                # Reference stats
                total_r = ref_wins+ref_losses+ref_draws
                if total_r > 0:
                    rwr = (ref_wins + 0.5*ref_draws) / total_r
                    print(f"  [REF] win_rate={rwr:.1%}  {ref_wins}W/{ref_losses}L/{ref_draws}D")
                    stats.log_evaluation(step=step, opponent="alpha_beta_ref",
                                         games_played=total_r,
                                         wins=ref_wins, losses=ref_losses,
                                         draws=ref_draws, win_rate=rwr)
                    tb.log_evaluation(step=step, opponent="alpha_beta_ref",
                                      games_played=total_r,
                                      wins=ref_wins, losses=ref_losses,
                                      draws=ref_draws, win_rate=rwr)
                    net_elo = evaluator.ref_elo + 200
                    new_ne  = net_elo + k*(rwr - 0.5)
                    stats.log_elo(new_ne,"alpha_beta_ref",step,
                                  total_r,ref_wins,ref_losses,ref_draws)
                    tb.log_elo(new_ne,"alpha_beta_ref",step,
                               total_r,ref_wins,ref_losses,ref_draws)

                print(f"  === EVAL DONE — resuming self-play ===\n")
                # Resume self-play: push tasks to all workers
                psp.push_selfplay(network)

    finally:
        print("[INFO] Stopping workers…")
        psp.stop()
        ex = {'game_id':game_id,'eval_game_counter':eval_game_counter,
              'best_elo':evaluator.best_elo,'ref_elo':evaluator.ref_elo}
        save_checkpoint(network, optimizer,
                        str(checkpoints_dir/"latest.pt"), step, extra=ex)
        buffer.save(str(buffer_path))
        print(f"[INFO] Saved replay buffer: {len(buffer)} positions")
        stats.close()
        if tb is not None:
            tb.close()
        print("[INFO] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark mode
# ─────────────────────────────────────────────────────────────────────────────

class _BenchTimers:
    """Accumulated main-process timing categories for the benchmark report."""

    def __init__(self):
        self.startup       = 0.0
        self.wait          = 0.0
        self.recv          = 0.0
        self.deserialize   = 0.0
        self.bookkeeping   = 0.0
        self.respawn       = 0.0
        self.profile_phase = 0.0
        self.stop          = 0.0

    def total(self):
        return (self.startup + self.wait + self.recv + self.deserialize
                + self.bookkeeping + self.respawn + self.profile_phase + self.stop)


def _fmt_table(rows, title):
    """Print a left-aligned two/three-column table with percentages."""
    print(f"\n  {title}")
    print("  " + "-" * (len(title) + 2))
    name_w = max(len(r[0]) for r in rows)
    for row in rows:
        name, tsec, pct = row[0], row[1], row[2]
        print(f"  {name:<{name_w}}  {tsec:>10.3f}s  {pct:>6.1f}%")


def _print_benchmark_report(timers, total_wall, games, total_moves, total_sims,
                            gpu_stats, worker_phases):
    print("\n" + "=" * 72)
    print("  BENCHMARK REPORT")
    print("=" * 72)

    # ── Overall ──
    avg_moves = total_moves / games if games > 0 else 0
    print(f"\n  Games: {games}    Total moves: {total_moves}    "
          f"Avg moves/game: {avg_moves:.1f}")
    print(f"  Total simulations: {total_sims}    Sims/sec: "
          f"{total_sims / total_wall if total_wall > 0 else 0:.1f}")
    print(f"  Games/sec: {games / total_wall if total_wall > 0 else 0:.3f}")
    print(f"  Wall time: {total_wall:.2f}s")

    # ── Main-process breakdown ──
    labels = [
        ("Startup (net/ckpt/buffer/GPU warmup)", timers.startup),
        ("Waiting for workers",                  timers.wait),
        ("Queue receive (psp.collect_one)",      timers.recv),
        ("Deserializing game_data",              timers.deserialize),
        ("Bookkeeping",                          timers.bookkeeping),
        ("Respawning workers",                   timers.respawn),
        ("Worker CPU profiling",                 timers.profile_phase),
        ("Shutdown",                             timers.stop),
    ]
    accounted = sum(t for _, t in labels)
    other_pct = 100.0 * max(0.0, total_wall - accounted) / total_wall if total_wall > 0 else 0

    rows = [(name, t, 100.0 * t / total_wall) for name, t in labels]
    _fmt_table(rows, "Main-process time breakdown (% of wall time)")
    if other_pct > 0.5:
        print(f"  {'(unattributed)':<26} {total_wall - accounted:>10.3f}s  {other_pct:>6.1f}%")

    # Estimate worker-side game time (the dominant bucket)
    worker_time = max(0.0, total_wall - accounted)
    print(f"\n  -> Estimated worker-side game time: {worker_time:.2f}s "
          f"({100.0 * worker_time / total_wall:.1f}% of wall)")

    # ── GPU server breakdown ──
    if gpu_stats:
        print("\n" + "=" * 72)
        print("  GPU SERVER BREAKDOWN")
        print("=" * 72)
        srv_total = gpu_stats.get('server_total', 0) or 0
        s_labels = [
            ("Shader pre-warm",            gpu_stats.get('prewarm_time', 0)),
            ("Waiting for requests",       gpu_stats.get('wait_time', 0)),
            ("GPU forward + softmax",      gpu_stats.get('gpu_forward_time', 0)),
            ("Shared-mem read (states)",   gpu_stats.get('shm_read_time', 0)),
            ("Shared-mem write (results)", gpu_stats.get('shm_write_time', 0)),
            ("Weight queue drain",         gpu_stats.get('weight_drain_time', 0)),
            ("Weight load into net",       gpu_stats.get('weight_load_time', 0)),
        ]
        rows = [(name, t, 100.0 * t / srv_total if srv_total > 0 else 0)
                for name, t in s_labels]
        _fmt_table(rows, "GPU server time breakdown (% of server run)")
        print(f"  {'Server total run time':<30} {srv_total:>10.3f}s")
        print(f"\n  Total requests: {gpu_stats.get('total_requests', 0)}    "
              f"Forward passes: {gpu_stats.get('aggregated_batches', 0)}    "
              f"Aggregation cycles: {gpu_stats.get('aggregation_cycles', 0)}")
        print(f"  Samples processed: {gpu_stats.get('samples_processed', 0)}    "
              f"Avg batch size: "
              f"{gpu_stats.get('samples_processed', 0) / max(1, gpu_stats.get('aggregated_batches', 0)):.1f}")
        print(f"  Aggregation wait time: {gpu_stats.get('aggregation_wait_time', 0):.3f}s")

    # ── Worker CPU phase breakdown ──
    if worker_phases:
        print("\n" + "=" * 72)
        print("  WORKER CPU PHASE BREAKDOWN (merged across workers)")
        print("=" * 72)
        total = sum(v['total_s'] for v in worker_phases.values())
        rows = sorted(
            [(p, v['total_s'], 100.0 * v['total_s'] / total if total > 0 else 0,
              v['calls'], v['mean_ms'])
             for p, v in worker_phases.items()],
            key=lambda r: -r[1],
        )
        print(f"\n  {'Phase':<25} {'Total(s)':>10} {'%Time':>8} {'Calls':>10} {'Mean(ms)':>10}")
        print("  " + "-" * 66)
        for name, tsec, pct, calls, mean_ms in rows:
            hot = "  *** HOT" if pct > 30 else ""
            print(f"  {name:<25} {tsec:>10.4f} {pct:>7.1f}% {calls:>10} {mean_ms:>9.3f}{hot}")
        print("  " + "-" * 66)
        print(f"  {'TOTAL':<25} {total:>10.4f} {'100.0%':>8}")

    print("\n" + "=" * 72)


def _aggregate_worker_phases(summaries):
    """Merge per-worker profile phase dicts into a single aggregate."""
    merged = {}
    for s in summaries:
        if not s:
            continue
        for phase, data in s.get('phases', {}).items():
            if phase not in merged:
                merged[phase] = {'calls': 0, 'total_s': 0.0, 'mean_ms': 0.0}
            merged[phase]['calls'] += data.get('calls', 0)
            merged[phase]['total_s'] += data.get('total_s', 0.0)
    for phase, data in merged.items():
        if data['calls'] > 0:
            data['mean_ms'] = data['total_s'] / data['calls'] * 1000.0
    return merged


def run_benchmark(config, num_workers=None, num_games=10, profile_games=3):
    """Replicate the training pipeline (checkpoint load, GPU server, worker
    pool, self-play games) WITHOUT training or eval, timing every phase.

    Read-only w.r.t. the run's outputs: no stats.db writes, no TensorBoard,
    no checkpoints written, no replay-buffer writes.  The replay buffer is
    loaded (to time that I/O) but never modified or saved.
    """
    import multiprocessing as _mp
    if _mp.get_start_method(allow_none=True) is None:
        try: _mp.set_start_method('spawn')
        except RuntimeError: pass

    global _shutdown
    _shutdown = False
    wall_start = time.perf_counter()
    timers = _BenchTimers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    use_gpu = getattr(config, 'inference', None) and getattr(config.inference, 'use_gpu', False)
    if use_gpu:
        print("[INFO] Inference: GPU (DirectML centralized server)")

    if num_workers is None:
        num_workers = getattr(config.selfplay, 'num_workers', 8)
    print(f"[INFO] Workers: {num_workers}")

    # ── Startup phase ──
    t0 = time.perf_counter()

    network = create_model_from_config(config)
    network.to(device)
    print(f"[INFO] Network: {sum(p.numel() for p in network.parameters())} params")

    output_dir      = Path(config.main.output_dir) / config.main.run_name
    checkpoints_dir = output_dir / "checkpoints"

    # Checkpoint load (mirrors run_training)
    latest_ckpt = checkpoints_dir / "latest.pt"
    if latest_ckpt.exists():
        print(f"[INFO] Loading checkpoint: {latest_ckpt}")
        load_checkpoint(str(latest_ckpt), network)
    else:
        print(f"[WARN] No checkpoint found at {latest_ckpt} — using random weights")

    # Replay buffer load (read-only — never written back)
    buffer_path = checkpoints_dir / "replay_buffer.npz"
    loaded_buffer = ReplayBuffer.load(str(buffer_path), max_size=config.buffer.max_size)
    if loaded_buffer is not None:
        print(f"[INFO] Loaded replay buffer: {len(loaded_buffer)} positions "
              f"from {loaded_buffer.total_games} games (read-only)")
    else:
        print("[INFO] No replay buffer to load (read-only benchmark)")

    # GPU server timing stats queue
    stats_q = _mp.Queue() if use_gpu else None
    psp = ParallelSelfPlay(config, num_workers=num_workers, stats_queue=stats_q)
    psp.start()
    psp.push_selfplay(network)   # kick off first round

    timers.startup = time.perf_counter() - t0
    print(f"\n[Benchmark] Startup (network + checkpoint + buffer + GPU warmup): "
          f"{timers.startup:.2f}s\n")

    # ── Self-play collection phase (no training, no eval, no buffer writes) ──
    games_done   = 0
    total_moves  = 0
    total_sims   = 0
    worker_live_game_ids = [0] * num_workers  # placeholder for parity with training

    try:
        while games_done < num_games and not _shutdown:
            # Waiting for a result
            t_wait = time.perf_counter()
            result = psp.collect_one(timeout=300.0)
            timers.wait += time.perf_counter() - t_wait
            if result is None:
                print("[WARN] No result in 5 min — workers may be stuck")
                continue
            if result.get('done'):
                continue
            if result.get('type') != 'selfplay':
                continue   # ignore live messages / other result types

            wid = result['worker_id']
            t_recv = time.perf_counter()
            raw = result['game_data']
            game_info = result['game_info']
            timers.recv += time.perf_counter() - t_recv

            # Deserialise (exactly as run_training does)
            t_ds = time.perf_counter()
            game_data = [(np.array(s, dtype=np.float32),
                          np.array(p, dtype=np.float32),
                          float(v),
                          np.array(m, dtype=np.float32)) for s, p, v, m in raw]
            timers.deserialize += time.perf_counter() - t_ds

            # Bookkeeping (counts + print only — no buffer.add_game, no stats logging)
            t_bk = time.perf_counter()
            games_done += 1
            total_moves += game_info['length']
            total_sims  += game_info['length'] * config.mcts.num_simulations
            print(f"  [W{wid}] Game {games_done}/{num_games}: "
                  f"{game_info['termination']:18s} | {game_info['result_str']} | "
                  f"{game_info['length']} moves")
            timers.bookkeeping += time.perf_counter() - t_bk

            # Respawning this worker (weights serialization + task push, as in training)
            t_rp = time.perf_counter()
            import torch as _torch, io as _io
            buf2 = _io.BytesIO()
            _torch.save(network.state_dict(), buf2)
            wb = buf2.getvalue()
            try:
                psp._task_qs[wid].put_nowait({'type': 'selfplay', 'weights': wb})
            except Exception:
                pass
            timers.respawn += time.perf_counter() - t_rp

        # ── Worker CPU phase profiling (reuses the existing WorkerProfiler) ──
        if not _shutdown and profile_games > 0:
            t_pr = time.perf_counter()
            print(f"\n[Benchmark] Dispatching worker CPU profiling "
                  f"({profile_games} profiled games per worker)...")
            psp.dispatch_profile(network, num_games=profile_games)
            summaries = []
            expected = num_workers
            received = 0
            while received < expected and not _shutdown:
                r = psp.collect_one(timeout=300.0)
                if r is None:
                    print("[WARN] Profile collection timeout")
                    break
                if r.get('done'):
                    continue
                if r.get('type') == 'profile_done':
                    summaries.append(r.get('summary'))
                    received += 1
                    print(f"  [Worker Profiler] W{r.get('worker_id')} finished "
                          f"- {received}/{expected} workers done")
            timers.profile_phase = time.perf_counter() - t_pr
            worker_phases = _aggregate_worker_phases(summaries)
        else:
            worker_phases = None

    finally:
        t_stop = time.perf_counter()
        print("\n[INFO] Stopping workers…")
        psp.stop()
        timers.stop = time.perf_counter() - t_stop

    # ── Report ──
    gpu_stats = psp.get_gpu_stats() if use_gpu else None
    wall_total = time.perf_counter() - wall_start
    _print_benchmark_report(timers, wall_total, games_done, total_moves,
                            total_sims, gpu_stats, worker_phases)
    print("[Benchmark] Done. Read-only — no checkpoints, stats.db, or buffer writes occurred.")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_check(config):
    print("\n" + "="*60 + "\n  SANITY CHECK\n" + "="*60)
    from encoding import board_to_tensor, get_legal_move_mask
    from training import train_one_step, create_optimizer
    from evaluation import alpha_beta_best_move
    import chess

    ov = {'network':{'num_residual_blocks':2,'num_filters':16,
                     'num_policy_channels':8,'num_value_channels':8,'value_fc_size':32},
          'mcts':{'num_simulations':10},
          'selfplay':{'max_game_length':50,'temperature_threshold':15},
          'training':{'batch_size':8,'training_steps_per_iteration':2,
                      'checkpoint_interval':5,'num_batches_per_step':2},
          'evaluation':{'eval_interval':2,'gate_games':4,'ref_opponent_games':4},
          'buffer':{'max_size':1000}}
    lc = get_config(config_path=None, overrides=ov)
    dev = torch.device("cpu")
    net = create_model_from_config(lc)
    board = chess.Board()
    t = board_to_tensor(board)
    print(f"Tensor: {t.shape}")
    m = get_legal_move_mask(board); print(f"Mask sum: {m.sum():.0f}")
    me = MCTS(net, num_simulations=10, c_puct=1.5)
    root = me.get_root(board)
    vp,bm,st = me.search(root)
    print(f"MCTS best: {bm}  depth: {st['avg_depth']:.2f}")
    gd,gi = self_play_game(net, lc)
    print(f"Game: {gi['termination']} | {gi['result_str']} | {gi['length']} moves")
    buf = ReplayBuffer(1000); buf.add_game(gd)
    opt = create_optimizer(net)
    ld  = train_one_step(net, opt, buf, 8, dev)
    print(f"Loss: pol={ld['policy_loss']:.4f} val={ld['value_loss']:.4f}")
    mv = alpha_beta_best_move(board, 2)
    print(f"Alpha-beta: {mv}")
    print("="*60 + "\n  SANITY CHECK PASSED\n" + "="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train","gui","evaluate","sanity","benchmark"])
    parser.add_argument("--gui",      action="store_true")
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--sims",     type=int, default=None)
    parser.add_argument("--blocks",   type=int, default=None)
    parser.add_argument("--filters",  type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--workers",  type=int, default=None)
    parser.add_argument("--benchmark-games", type=int, default=10,
                        help="Self-play games to collect in benchmark mode (default: 10)")
    parser.add_argument("--profile-games", type=int, default=3,
                        help="Profiled games per worker in benchmark mode (default: 3)")
    args = parser.parse_args()

    ov = {}
    if args.sims:     ov.setdefault('mcts',{})['num_simulations']=args.sims
    if args.blocks:   ov.setdefault('network',{})['num_residual_blocks']=args.blocks
    if args.filters:  ov.setdefault('network',{})['num_filters']=args.filters
    if args.run_name: ov.setdefault('main',{})['run_name']=args.run_name
    config = get_config(path=args.config, overrides=ov if ov else None)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global output_dir
    output_dir = Path(config.main.output_dir)/config.main.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        import multiprocessing as _mp
        if _mp.get_start_method(allow_none=True) is None:
            try: _mp.set_start_method('spawn')
            except RuntimeError: pass
        run_training(config, gui_enabled=args.gui, num_workers=args.workers)

    elif args.mode == "gui":
        from gui.app import start_gui_server
        start_gui_server(stats=None, config=config, worker_live_games=[], eval_live_game=None)

    elif args.mode == "evaluate":
        use_gpu = getattr(config, 'inference', None) and getattr(config.inference, 'use_gpu', False)
        if use_gpu:
            print("[INFO] Inference: GPU (DirectML centralized server)")

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = create_model_from_config(config); net.to(dev)
        cp  = Path(config.main.output_dir)/config.main.run_name/"checkpoints"/"latest.pt"
        if cp.exists(): load_checkpoint(str(cp), net)
        else: print("[WARN] No checkpoint")
        s   = StatsLogger(str(Path(config.main.output_dir)/config.main.run_name/config.stats.db_path))
        ev  = Evaluator(config, s)

        if use_gpu:
            # Start GPU server, push weights, create InferenceClient
            import multiprocessing as _mp
            if _mp.get_start_method(allow_none=True) is None:
                try: _mp.set_start_method('spawn')
                except RuntimeError: pass

            weight_q = _mp.Queue()
            request_q = _mp.Queue()
            response_q = _mp.Queue(maxsize=256)
            gpu_ready = _mp.Event()
            gpu_shutdown = _mp.Event()

            from gpu_server import GPUInferenceServer
            server = GPUInferenceServer(
                config=config,
                request_queue=request_q,
                response_queues={0: response_q},
                weight_queue=weight_q,
                ready_event=gpu_ready,
                shutdown_event=gpu_shutdown,
            )
            gpu_proc = _mp.Process(target=server.run, daemon=True)
            gpu_proc.start()
            print("[INFO] Waiting for GPU server to warm up shaders...")
            gpu_ready.wait()
            print("[INFO] GPU server ready")

            # Push weights
            import torch as _t, io as _io
            buf = _io.BytesIO(); _t.save(net.state_dict(), buf)
            weight_q.put((0, buf.getvalue()))

            # Create InferenceClient
            from inference_client import InferenceClient
            client = InferenceClient(worker_id=0, request_queue=request_q,
                                     response_queue=response_q, network_id=0)

            r = ev.run_reference_match(client, step=0, verbose=True)

            # Shutdown GPU server
            gpu_shutdown.set()
            try: request_q.put_nowait(None)
            except: pass
            gpu_proc.join(timeout=10)
            if gpu_proc.is_alive(): gpu_proc.kill()
        else:
            r = ev.run_reference_match(net, step=0, verbose=True)

        print(f"vs Alpha-Beta: {r['win_rate']:.1%}")
        s.close()

    elif args.mode == "benchmark":
        run_benchmark(config, num_workers=args.workers,
                      num_games=args.benchmark_games,
                      profile_games=args.profile_games)

    elif args.mode == "sanity":
        run_sanity_check(config)


if __name__ == "__main__":
    main()