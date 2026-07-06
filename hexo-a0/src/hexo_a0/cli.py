"""Command-line interface for HeXO AlphaZero training."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch


def main(argv=None):
    # Configure the CUDA/HIP caching allocator BEFORE anything touches the
    # device context (PYTORCH_CUDA_ALLOC_CONF is read once, at first device
    # init — setting it later is a silent no-op). Importing torch above does
    # NOT initialise the context; the first .to(cuda) does.
    from hexo_a0.gpu_memory import configure_cuda_alloc
    configure_cuda_alloc()

    parser = argparse.ArgumentParser(
        prog="hexo-a0",
        description="HeXO AlphaZero — Gumbel MCTS training on infinite hex tic-tac-toe",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Run training loop")
    train_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to TOML config file",
    )
    train_parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .pt file to resume from",
    )
    train_parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on (cpu, cuda, cuda:0, etc.)",
    )
    train_parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop training after this many gradient steps",
    )
    train_parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory to save checkpoints",
    )
    train_parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Tensorboard log directory",
    )
    train_parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable tensorboard logging",
    )
    train_parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile (use eager mode)",
    )
    train_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )

    # --- watch ---
    watch_parser = subparsers.add_parser("watch", help="Watch the AI play a game")
    watch_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to TOML config file (plain or curriculum)",
    )
    watch_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pt file",
    )
    watch_parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu, cuda)",
    )
    watch_parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between moves (default: 0.5)",
    )
    watch_parser.add_argument(
        "--stage", type=str, default=None,
        help="Curriculum stage: index (1-based) or name. If omitted, prompts interactively.",
    )
    watch_parser.add_argument(
        "--sims", type=int, default=None,
        help="Override MCTS n_simulations (for non-greedy mode)",
    )
    watch_parser.add_argument(
        "--m-actions", type=int, default=None,
        help="Override MCTS m_actions (for non-greedy mode)",
    )
    watch_parser.add_argument(
        "--c-visit", type=int, default=None,
        help="Override MCTS c_visit (sigma visit-count baseline; higher = more exploration of low-Q actions)",
    )
    watch_parser.add_argument(
        "--c-scale", type=float, default=None,
        help="Override MCTS c_scale (sigma Q scaling; higher = Q-values dominate over priors)",
    )

    # --- eval-sealbot ---
    eval_sealbot_parser = subparsers.add_parser(
        "eval-sealbot",
        help="One-shot evaluation of a checkpoint against the SealBot minimax opponent",
    )
    eval_sealbot_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to TOML config file (plain or curriculum)",
    )
    eval_sealbot_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pt file",
    )
    eval_sealbot_parser.add_argument(
        "--stage", type=str, default=None,
        help="Curriculum stage: index (1-based) or name. If omitted, prompts interactively.",
    )
    eval_sealbot_parser.add_argument(
        "--games", type=int, default=None,
        help="Number of games to play (default: cfg.eval.sealbot.games). Use 100+ for decision-quality evals.",
    )
    eval_sealbot_parser.add_argument(
        "--sims", type=int, default=None,
        help="Override MCTS n_simulations for the model (default: cfg.eval.sealbot.sims)",
    )
    eval_sealbot_parser.add_argument(
        "--m-actions", type=int, default=None,
        help="Override MCTS m_actions (default: cfg.eval.sealbot.m_actions)",
    )
    eval_sealbot_parser.add_argument(
        "--time-limit", type=float, default=None,
        help="SealBot minimax time limit per move in seconds. Defaults to cfg.eval.sealbot.time_limit; pass an explicit value (e.g. 0.05) to match a routine-eval ladder level.",
    )
    eval_sealbot_parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel game threads (default: cfg.eval.sealbot.workers)",
    )
    eval_sealbot_parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for model inference (cpu, cuda, cuda:0, ...)",
    )
    eval_sealbot_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )

    # --- curriculum ---
    curriculum_parser = subparsers.add_parser(
        "curriculum",
        help="Run automatic curriculum training through multiple stages",
    )
    curriculum_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to curriculum TOML file",
    )
    curriculum_parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: from config)",
    )
    curriculum_parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile",
    )
    curriculum_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )

    # --- head-to-head ---
    h2h_parser = subparsers.add_parser(
        "head-to-head",
        help="SPRT-bounded match between two arbitrary checkpoints (cross-config OK)",
    )
    h2h_parser.add_argument("--checkpoint-a", type=str, required=True,
                            help="Path to checkpoint A (.pt) — must carry its own model_config")
    h2h_parser.add_argument("--checkpoint-b", type=str, required=True,
                            help="Path to checkpoint B (.pt) — must carry its own model_config")
    h2h_parser.add_argument("--win-length", type=int, required=True,
                            help="Win length for the shared GameConfig")
    h2h_parser.add_argument("--radius", type=int, required=True,
                            help="Placement radius for the shared GameConfig")
    h2h_parser.add_argument("--max-moves", type=int, required=True,
                            help="Max moves for the shared GameConfig")
    h2h_parser.add_argument("--mcts-sims", type=int, default=200,
                            help="MCTS simulations per placement (default 200)")
    h2h_parser.add_argument("--mcts-m-actions", type=int, default=16,
                            help="Root candidate actions for Gumbel-Top-k (default 16)")
    h2h_parser.add_argument("--device", type=str, default="cpu",
                            help="Torch device (cpu, cuda, ...). Default cpu.")
    h2h_parser.add_argument("--sprt-s0", type=float, default=0.50,
                            help="SPRT H0 score (default 0.50, equal strength)")
    h2h_parser.add_argument("--sprt-s1", type=float, default=0.55,
                            help="SPRT H1 score (default 0.55, ~35 Elo for A)")
    h2h_parser.add_argument("--sprt-alpha", type=float, default=0.01,
                            help="SPRT type-I error rate (default 0.01)")
    h2h_parser.add_argument("--sprt-beta", type=float, default=0.05,
                            help="SPRT type-II error rate (default 0.05)")
    h2h_parser.add_argument("--window-size", type=int, default=1000,
                            help="Sliding window of recent games for SPRT state (0 = unbounded)")
    h2h_parser.add_argument("--max-games", type=int, default=1000,
                            help="Hard upper bound on games played (default 1000)")
    h2h_parser.add_argument("--seed", type=int, default=None,
                            help="RNG seed for reproducibility (default: non-deterministic)")
    h2h_parser.add_argument("--state-file", type=str, default=None,
                            help="Optional JSON file written atomically after each game for monitoring")
    h2h_parser.add_argument("--opening-plies", type=int, default=8,
                            help="Number of opening plies to play from a random opening generator (default 8, 0 disables opening generator and uses random Gumbel noise)")
    h2h_parser.add_argument("--opening-generator", type=str, default="alternate",
                            choices=["alternate", "a", "b", "champion"],
                            help="Opening generator for the first N plies (default: alternate)")
    h2h_parser.add_argument("--opening-temperature", type=float, default=0.5,
                            help="Temperature for opening generator (default: 0.5)")
    # --- serve ---
    serve_parser = subparsers.add_parser(
        "serve",
        help="Public play server + analysis tool",
    )
    serve_parser.add_argument("--config", type=str, required=True,
                              help="Curriculum TOML; the [model] section is the architecture "
                                   "fallback when the checkpoint has no embedded model_config")
    serve_parser.add_argument("--checkpoint", type=str, required=True,
                              help="Path to a single .pt checkpoint to serve")
    serve_parser.add_argument("--win-length", type=int, default=6)
    serve_parser.add_argument("--placement-radius", type=int, default=8)
    serve_parser.add_argument("--max-moves", type=int, default=400)
    serve_parser.add_argument("--mcts-sims", type=int, default=64,
                              help="MCTS simulations per bot stone (0 = raw policy argmax)")
    serve_parser.add_argument("--m-actions", type=int, default=16)
    serve_parser.add_argument("--db", type=str, default="games.sqlite")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--bind", type=str, default="127.0.0.1",
                              help="Use 0.0.0.0 to expose for Tailscale Funnel")
    serve_parser.add_argument("--url-prefix", type=str, default="",
                              help="Path prefix for proxied serving (e.g. /hexo). "
                                   "Leading slash required, no trailing slash.")
    serve_parser.add_argument("--admin-token", type=str, default=None,
                              help="Token for /admin?token=<...>. If omitted, a fresh random "
                                   "token is generated at startup. Pass an empty string to disable.")
    serve_parser.add_argument("--model-label", type=str, default=None,
                              help="Stable display label for stats continuity")
    serve_parser.add_argument("--difficulty-sims", type=str, default="16,32,64,128",
                              help="Comma-separated sim counts mapped onto "
                                   "['casual', 'easy', 'standard', 'strong']")
    serve_parser.add_argument("--default-difficulty", type=str, default="standard",
                              choices=["casual", "easy", "standard", "strong"],
                              help="Difficulty served when /new_game omits the field")
    serve_parser.add_argument("--request-timeout", type=int, default=60,
                              help="Socket/request timeout in seconds")
    serve_parser.add_argument("--inference-workers", type=int, default=2,
                              help="Concurrent inference slots shared by the bot and the "
                                   "analysis endpoints. The server runs the model on CPU, so "
                                   ">1 lets a side-line /analyze run while the bot thinks; "
                                   "torch intra-op threads are capped to cores/workers.")
    serve_parser.add_argument("--no-live-forcing", dest="live_forcing",
                              action="store_false", default=True,
                              help="Disable the live VCF forcing solver (win execution + "
                                   "pre-emptive defense) for ALL difficulty tiers; falls "
                                   "back to byte-identical pre-forcing-feature MCTS/argmax play.")

    # --- export ---
    export_parser = subparsers.add_parser(
        "export",
        help="Export checkpoint weights to safetensors (for hexo-infer / wasm)",
    )
    export_parser.add_argument("--checkpoint", type=str, required=True,
                               help="Path to a .pt checkpoint with embedded model_config")
    export_parser.add_argument("--out", type=str, required=True,
                               help="Output .safetensors path")

    # --- default-config ---
    subparsers.add_parser(
        "default-config",
        help="Print default TOML configuration to stdout",
    )

    # --- default-curriculum ---
    subparsers.add_parser(
        "default-curriculum",
        help="Print annotated curriculum TOML template to stdout",
    )

    args = parser.parse_args(argv)

    if args.command == "default-config":
        from hexo_a0.config_io import default_config_toml
        print(default_config_toml())
        return 0

    if args.command == "default-curriculum":
        from hexo_a0.config_io import default_curriculum_toml
        print(default_curriculum_toml())
        return 0

    if args.command == "train":
        return _run_train(args)

    if args.command == "watch":
        return _run_watch(args)

    if args.command == "eval-sealbot":
        return _run_eval_sealbot(args)

    if args.command == "curriculum":
        return _run_curriculum(args)

    if args.command == "head-to-head":
        return _run_head_to_head(args)

    if args.command == "serve":
        return _run_serve(args)

    if args.command == "export":
        return _run_export(args)

    return 1


def _run_train(args):
    import dataclasses

    import hexo_rs
    from hexo_a0.config_io import load_config
    from hexo_a0.model import HeXONet
    from hexo_a0.trainer import Trainer

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("hexo_a0")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        return 1
    cfg = load_config(config_path)

    # CLI args override RunConfig
    if args.device is not None:
        cfg.run = dataclasses.replace(cfg.run, device=args.device)
    if args.checkpoint_dir is not None:
        cfg.run = dataclasses.replace(cfg.run, checkpoint_dir=args.checkpoint_dir)
    if args.log_dir is not None:
        cfg.run = dataclasses.replace(cfg.run, log_dir=args.log_dir)
    if args.no_compile:
        cfg.run = dataclasses.replace(cfg.run, compile=False)
    if args.no_tensorboard:
        log_dir = None
    else:
        log_dir = cfg.run.log_dir

    device = torch.device(cfg.run.device)
    log.info("Device: %s", device)
    log.info("Game: win_length=%d, placement_radius=%d, max_moves=%d",
             cfg.game.win_length, cfg.game.placement_radius, cfg.game.max_moves)
    log.info("Model: hidden_dim=%d, num_layers=%d, num_heads=%d",
             cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.num_heads)
    log.info("Training: n_sims=%d, m_actions=%d",
             cfg.mcts.n_simulations, cfg.mcts.m_actions)

    # Create model
    model = HeXONet(cfg.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{param_count:,}")

    # Isolate this process's inductor cache from the self-play inference
    # subprocess's (a shared TORCHINDUCTOR_CACHE_DIR cross-process kernel-cache
    # race on the one APU is the leading suspect for the 2026-06-02 compile NaN).
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_hexo/train")

    # Compile the actual training entry point (the trainer calls
    # model._forward_batch_core directly, so wrapping the module was a no-op).
    # fullgraph off: the PyG Batch arg graph-breaks. NaN grads can't corrupt
    # checkpoints (_clip_and_step_if_finite skips the step).
    if cfg.run.compile and hasattr(torch, "compile"):
        try:
            model._forward_batch_core = torch.compile(
                model._forward_batch_core, dynamic=True
            )
            log.info("Compiled model._forward_batch_core (training entry point)")
        except Exception as e:
            log.warning("torch.compile failed, using eager mode: %s", e)

    # Game config (Rust struct)
    game_config = hexo_rs.GameConfig(
        win_length=cfg.game.win_length,
        placement_radius=cfg.game.placement_radius,
        max_moves=cfg.game.max_moves,
    )

    # Checkpoint directory (resolve before trainer/resume so "latest" can find it)
    ckpt_dir = Path(cfg.run.checkpoint_dir)
    if cfg.run.checkpoint_dir == "checkpoints":
        ckpt_dir = Path(log_dir or "runs") / "checkpoints"

    # Create trainer
    trainer = Trainer(model, cfg, game_config, device, log_dir=log_dir, ckpt_dir=str(ckpt_dir))

    # Resume from checkpoint
    resume = args.resume
    if resume:
        if resume == "latest":
            # Find the newest checkpoint in the checkpoint dir
            pts = sorted(p for p in ckpt_dir.glob("checkpoint_*.pt") if "_buffer" not in p.name) if ckpt_dir.exists() else []
            if not pts:
                log.error("No checkpoints found in %s", ckpt_dir)
                return 1
            resume_path = pts[-1]
        else:
            resume_path = Path(resume)
        if not resume_path.exists():
            log.error("Checkpoint not found: %s", resume_path)
            return 1
        trainer.load_checkpoint(str(resume_path))
        log.info("Resumed from checkpoint: %s (step %d)", resume_path, trainer.train_steps)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoints: %s", ckpt_dir)

    # Training loop
    max_steps = args.max_steps
    trainer.start_self_play()
    try:
        while True:
            if max_steps is not None and trainer.train_steps >= max_steps:
                log.info("Reached %d steps, stopping.", max_steps)
                break

            metrics = trainer.train()

    except KeyboardInterrupt:
        log.info("Interrupted! Saving final checkpoint...")
        ckpt_path = ckpt_dir / f"checkpoint_{trainer.train_steps:08d}.pt"
        # Skip buffer backup — live replay_buffer.db is already durable
        trainer.save_checkpoint(str(ckpt_path), save_buffer=False)
        log.info("Saved: %s", ckpt_path)
    finally:
        trainer.close()

    return 0


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------

def _run_watch(args):
    import dataclasses

    from hexo_a0.config_io import load_config
    from hexo_a0.model import HeXONet
    from hexo_a0.viewer import run_viewer

    logging.basicConfig(level=logging.WARNING)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    cfg = load_config(config_path)

    # If this is a curriculum config with stages, resolve which stage to use
    if cfg.curriculum and cfg.curriculum.stages:
        cfg = _resolve_watch_stage(cfg, args.stage)

    device = torch.device(args.device)

    # Apply MCTS CLI overrides
    mcts_overrides = {}
    if args.sims is not None:
        mcts_overrides["n_simulations"] = args.sims
    if args.m_actions is not None:
        mcts_overrides["m_actions"] = args.m_actions
    if args.c_visit is not None:
        mcts_overrides["c_visit"] = args.c_visit
    if args.c_scale is not None:
        mcts_overrides["c_scale"] = args.c_scale
    if mcts_overrides:
        cfg.mcts = dataclasses.replace(cfg.mcts, **mcts_overrides)

    gc_dict = {"win_length": cfg.game.win_length,
               "placement_radius": cfg.game.placement_radius,
               "max_moves": cfg.game.max_moves}

    model = HeXONet(cfg.model).to(device)

    # Load only model weights — skip Trainer to avoid allocating the replay
    # buffer, optimizer, etc. which can use 100+ GB of memory.
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(ckpt_sd, strict=False)
    step = checkpoint.get("train_steps", checkpoint.get("iteration", "?"))
    print(f"Loaded checkpoint: step {step}")

    print(f"Game: win_length={cfg.game.win_length}, "
          f"placement_radius={cfg.game.placement_radius}, "
          f"max_moves={cfg.game.max_moves}")
    print(f"MCTS: sims={cfg.mcts.n_simulations}, m_actions={cfg.mcts.m_actions}")

    run_viewer(gc_dict, model, cfg.mcts, device, model_config=cfg.model)
    return 0


def _run_eval_sealbot(args):
    from hexo_a0.config_io import load_config
    from hexo_a0.model import HeXONet
    from hexo_a0.sealbot_eval import evaluate_vs_sealbot

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    cfg = load_config(config_path)
    if cfg.curriculum and cfg.curriculum.stages:
        cfg = _resolve_watch_stage(cfg, args.stage)

    sb = cfg.eval.sealbot
    n_games = args.games if args.games is not None else sb.games
    sims = args.sims if args.sims is not None else sb.sims
    m_actions = args.m_actions if args.m_actions is not None else sb.m_actions
    time_limit = args.time_limit if args.time_limit is not None else sb.time_limit
    workers = args.workers if args.workers is not None else sb.workers

    if n_games <= 0:
        print(f"n_games must be > 0, got {n_games}")
        return 1

    import hexo_rs as _hr
    if sb.win_length > 0 or sb.placement_radius > 0 or sb.max_moves > 0:
        sb_game_config = _hr.GameConfig(
            sb.win_length or cfg.game.win_length,
            sb.placement_radius or cfg.game.placement_radius,
            sb.max_moves or cfg.game.max_moves,
        )
    else:
        sb_game_config = _hr.GameConfig(
            cfg.game.win_length, cfg.game.placement_radius, cfg.game.max_moves,
        )

    device = torch.device(args.device)
    model = HeXONet(cfg.model).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(ckpt_sd, strict=False)
    model.eval()
    step = checkpoint.get("train_steps", checkpoint.get("iteration", "?"))

    print(f"Checkpoint:    {ckpt_path}  (step {step})")
    print(f"Game:          win_length={sb_game_config.win_length}, "
          f"placement_radius={sb_game_config.placement_radius}, "
          f"max_moves={sb_game_config.max_moves}")
    print(f"SealBot:       time_limit={time_limit:.3f}s, games={n_games}, "
          f"workers={workers}")
    print(f"Model MCTS:    sims={sims}, m_actions={m_actions}")
    print()

    result = evaluate_vs_sealbot(
        model, sb_game_config, device,
        n_games=n_games,
        sealbot_time_limit=time_limit,
        n_simulations=sims,
        m_actions=m_actions,
        model_config=cfg.model,
        workers=workers,
    )

    decided = result["p1_wins"] + result["p2_wins"]
    print()
    print(f"Result:        W{result['wins']} / L{result['losses']} / D{result['draws']}  "
          f"({n_games} games)")
    print(f"Win rate:      {result['win_rate']:.1%}  "
          f"(95% CI {result['ci_lo']:.1%}–{result['ci_hi']:.1%})")
    print(f"Elo diff:      {result['elo_diff']:+.0f}")
    print(f"P1/P2 wins:    {result['p1_wins']} / {result['p2_wins']}  "
          f"(P1 decided rate {result['p1_decided_rate']:.0%} of {decided})")
    print(f"Mean length:   {result['mean_game_length']:.1f} placements")
    return 0


def _resolve_watch_stage(cfg, stage_arg):
    """Select a curriculum stage and build the merged config.

    If *stage_arg* is None, prompts interactively on stdin.
    Accepts a 1-based index or a stage name (case-insensitive).
    """
    from hexo_a0.curriculum import _build_stage_config

    stages = cfg.curriculum.stages

    if stage_arg is not None:
        # Try numeric index first (1-based)
        try:
            idx = int(stage_arg) - 1
            if 0 <= idx < len(stages):
                stage = stages[idx]
                name = stage.get("name", f"Stage {idx + 1}")
                print(f"Using stage {idx + 1}: {name}")
                return _build_stage_config(cfg, stage)
            else:
                print(f"Stage index {stage_arg} out of range (1-{len(stages)})")
                sys.exit(1)
        except ValueError:
            pass
        # Try name match (case-insensitive)
        for i, s in enumerate(stages):
            if s.get("name", "").lower() == stage_arg.lower():
                print(f"Using stage {i + 1}: {s.get('name')}")
                return _build_stage_config(cfg, s)
        print(f"No stage named '{stage_arg}'. Available:")
        for i, s in enumerate(stages):
            print(f"  {i + 1}. {s.get('name', '(unnamed)')}")
        sys.exit(1)

    # Interactive selection
    print("\nCurriculum stages:")
    for i, s in enumerate(stages):
        name = s.get("name", "(unnamed)")
        wl = s.get("win_length", "?")
        pr = s.get("placement_radius", "?")
        print(f"  {i + 1}. {name}  (win_length={wl}, radius={pr})")
    print()

    while True:
        try:
            choice = input("Select stage [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not choice:
            choice = "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(stages):
                stage = stages[idx]
                name = stage.get("name", f"Stage {idx + 1}")
                print(f"Using stage {idx + 1}: {name}")
                return _build_stage_config(cfg, stage)
        except ValueError:
            pass
        print(f"Please enter a number 1-{len(stages)}")


def _run_head_to_head(args):
    from hexo_a0.head_to_head import run_head_to_head

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ckpt_a = Path(args.checkpoint_a)
    ckpt_b = Path(args.checkpoint_b)
    if not ckpt_a.exists():
        print(f"Checkpoint A not found: {ckpt_a}")
        return 1
    if not ckpt_b.exists():
        print(f"Checkpoint B not found: {ckpt_b}")
        return 1

    window = None if args.window_size <= 0 else args.window_size
    state_file = Path(args.state_file) if args.state_file else None

    summary = run_head_to_head(
        checkpoint_a=ckpt_a,
        checkpoint_b=ckpt_b,
        win_length=args.win_length,
        radius=args.radius,
        max_moves=args.max_moves,
        mcts_sims=args.mcts_sims,
        mcts_m_actions=args.mcts_m_actions,
        device_str=args.device,
        sprt_s0=args.sprt_s0,
        sprt_s1=args.sprt_s1,
        sprt_alpha=args.sprt_alpha,
        sprt_beta=args.sprt_beta,
        window_size=window,
        max_games=args.max_games,
        seed=args.seed,
        state_file=state_file,
        opening_plies=args.opening_plies,
        opening_temperature=args.opening_temperature,
        opening_generator=args.opening_generator
    )
    # accept_h1 or reject_h1 → 0 (SPRT terminated); inconclusive → 2
    if summary["decision"] in ("accept_h1", "reject_h1"):
        return 0
    return 2


def _run_curriculum(args):
    import warnings
    from hexo_a0.curriculum import run_curriculum

    # Suppress known PyTorch false positive (SequentialLR triggers this on construction
    # and during checkpoint resume fast-forward).
    warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step")

    # Set root logger to DEBUG so file handler captures everything.
    # Console handler only shows warnings — the display handles CLI output.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not args.verbose:
        for handler in logging.root.handlers:
            handler.setLevel(logging.WARNING)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Curriculum config not found: {config_path}")
        return 1

    run_curriculum(
        config_path,
        device_override=args.device,
        no_compile=args.no_compile,
        verbose=args.verbose,
    )
    return 0


def _run_serve(args):
    from hexo_a0.serving.app import run

    # Without a configured handler, Python's last-resort handler drops
    # everything below WARNING — serving's INFO logs (e.g. the native
    # hexo-infer attach confirmation) would never reach docker logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    # argparse converts hyphens to underscores, so args already has the
    # attribute shape app.run() expects.
    return run(args)


def _run_export(args):
    from hexo_a0.export import export_checkpoint

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1
    meta = export_checkpoint(ckpt_path, Path(args.out))
    print(f"Exported (train_steps={meta['train_steps']}, source={meta['source_checkpoint']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
