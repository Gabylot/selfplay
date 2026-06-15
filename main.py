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
from gui.live_game import LiveGameState


_shutdown = False
def signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[INFO] Shutdown signal received…")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_training(config, gui_enabled=False, num_workers=None):
    global _shutdown
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

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

            rtype = result.get('type')

            # ── Live incremental messages (real-time GUI updates) ──
            if rtype == 'live_start':
                wid = result['worker_id']
                wlg = worker_live_games[wid]
                worker_live_game_ids[wid] += 1
                wlg.start_game(worker_live_game_ids[wid], step,
                               game_type=result.get('game_type', 'selfplay'),
                               match_info=result.get('match_info'))
                continue

            if rtype == 'live_move':
                wid = result['worker_id']
                wlg = worker_live_games[wid]
                wlg.update(result['fen'], result['move'], result['move_number'],
                           mcts_stats=result.get('mcts_stats'))
                continue

            if rtype == 'live_end':
                wid = result['worker_id']
                wlg = worker_live_games[wid]
                wlg.game_over(result['result'], result.get('termination', ''))
                continue

            # Only process self-play results in the main loop
            if rtype != 'selfplay':
                continue

            wid = result['worker_id']

            # Deserialise
            raw       = result['game_data']
            game_data = [(np.array(s,dtype=np.float32),
                          np.array(p,dtype=np.float32),
                          float(v)) for s,p,v in raw]
            game_info = result['game_info']
            fens      = result.get('fens', [])
            moves     = result.get('moves', [])
            mcts_s    = result.get('mcts_stats', [])

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
                           num_positions=game_info['num_positions'])
            stats.log_mcts_stats(game_id=game_id, step=step,
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
                for _ in range(config.training.num_batches_per_step):
                    if _shutdown: break
                    ld = train_one_step(network, optimizer, buffer,
                                        int(config.training.batch_size), device)
                    step += 1
                    stats.log_training_step(step=step,
                                            policy_loss=ld['policy_loss'],
                                            value_loss=ld['value_loss'],
                                            total_loss=ld['total_loss'],
                                            learning_rate=config.training.learning_rate)

                # Buffer / confidence stats
                od = buffer.get_outcome_distribution()
                stats.log_buffer_stats(step=step, buffer_size=len(buffer),
                                       white_wins=od['white_wins'],
                                       black_wins=od['black_wins'], draws=od['draws'])
                if len(buffer) > 0:
                    n   = min(50, len(buffer))
                    ix  = np.random.choice(len(buffer), size=n, replace=False)
                    sts = np.array([buffer.buffer[i][0] for i in ix])
                    ps, vs = network.predict_batch(sts)
                    stats.log_network_stats(step,
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
                        print(f"  [GATE] PROMOTED  win_rate={wr:.1%}")
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
                    stats.log_elo(evaluator.best_elo,"gating",step,
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
                    net_elo = evaluator.ref_elo + 200
                    new_ne  = net_elo + k*(rwr - 0.5)
                    stats.log_elo(new_ne,"alpha_beta_ref",step,
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
        print("[INFO] Done.")


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
    parser.add_argument("mode", choices=["train","gui","evaluate","sanity"])
    parser.add_argument("--gui",      action="store_true")
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--sims",     type=int, default=None)
    parser.add_argument("--blocks",   type=int, default=None)
    parser.add_argument("--filters",  type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--workers",  type=int, default=None)
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
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = create_model_from_config(config); net.to(dev)
        cp  = Path(config.main.output_dir)/config.main.run_name/"checkpoints"/"latest.pt"
        if cp.exists(): load_checkpoint(str(cp), net)
        else: print("[WARN] No checkpoint")
        s   = StatsLogger(str(Path(config.main.output_dir)/config.main.run_name/config.stats.db_path))
        ev  = Evaluator(config, s)
        r   = ev.run_reference_match(net, step=0, verbose=True)
        print(f"vs Alpha-Beta: {r['win_rate']:.1%}")
        s.close()

    elif args.mode == "sanity":
        run_sanity_check(config)


if __name__ == "__main__":
    main()