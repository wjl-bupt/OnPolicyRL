"""CLI -- the L3 convenience layer.

    oprl train ppo --config mujoco --env HalfCheetah-v5 --lr 1e-4
    oprl configs [algo]      list available presets
    oprl sweep <name>        run a batch of experiments
    oprl components          list pluggable components

This module carries no algorithm knowledge: it never adjusts hyperparameters based on
the env name. Those come only from config/ and the command line.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import load_dict, presets


def _algo_module(name: str):
    from .algos import ppo as ppo_mod
    from .algos import vmpo as vmpo_mod

    if name == "vmpo":
        return vmpo_mod, vmpo_mod.VMPOConfig
    return ppo_mod, ppo_mod.PPOConfig


def _add_config_args(p: argparse.ArgumentParser, cfg_cls) -> None:
    """Turn dataclass fields into CLI flags, so `--help` lists every hyperparameter.

    `default=None` means "not given on the command line", letting the config layer decide
    and preserving the intended precedence order.
    """
    for f in dataclasses.fields(cfg_cls):
        if f.name.startswith("_"):
            continue
        t: type = str
        ann = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "str")
        if "bool" in ann:
            t = lambda s: s.lower() in ("1", "true", "yes", "on")  # noqa: E731
        elif "int" in ann:
            t = int
        elif "float" in ann:
            t = float
        p.add_argument(f"--{f.name}", type=t, default=None,
                       help=f"(default: {f.default})")


def _cmd_configs(args) -> int:
    """oprl configs [algo] -- list presets and their notes."""
    algos = [args.algo] if args.algo else ["ppo", "vmpo"]
    for algo in algos:
        ps = presets(algo)
        if not ps:
            print(f"config/{algo}.yaml does not exist", file=sys.stderr)
            continue
        print(f"config/{algo}.yaml:")
        for name, note in ps.items():
            keys = sorted(load_dict(name, algo))
            print(f"  {name:<10} {note.splitlines()[0] if note else ''}")
            if keys:
                print(f"             overrides: {', '.join(keys)}")
        print(f"\n  usage: oprl train {algo} --config <preset> [--any_hyperparam value]")
        print(f"  all hyperparameters: oprl train {algo} --help\n")
    return 0


def _cmd_train(args, unknown: list[str]) -> int:
    mod, cfg_cls = _algo_module(args.algo)
    algo_key = "ppo" if args.algo in ("ppo", "a2c") else args.algo

    # Precedence: dataclass defaults < YAML preset < CLI
    cfg_parser = argparse.ArgumentParser(add_help=False)
    _add_config_args(cfg_parser, cfg_cls)
    cfg_args, leftover = cfg_parser.parse_known_args(unknown)
    if leftover:
        raise SystemExit(f"unrecognized arguments: {leftover} (see --help)")

    base: dict = load_dict(args.config, algo_key)
    base.update({k: v for k, v in vars(cfg_args).items() if v is not None})

    if args.algo == "a2c":
        cfg = mod.a2c_config(**{k: v for k, v in base.items()
                                if k in cfg_cls.field_names()})
    else:
        cfg = cfg_cls.from_dict(base)   # unknown hyperparameters raise here

    if cfg.smoke:
        cfg.total_steps = min(cfg.total_steps, cfg.rollout_len * cfg.num_envs * 3)

    from .envs import make_env
    from .logger import Logger
    from .nets import ActorCritic
    from .seeding import seed_everything

    # Seed before constructing the policy, so network init is part of what cfg.seed
    # determines. train() seeds again for the update loop.
    seed_everything(cfg.seed, cfg.deterministic)

    env_spec = base.get("env") if isinstance(base.get("env"), dict) else {}
    env = make_env(
        args.env,
        num_envs=cfg.num_envs,
        device=cfg.resolve_device(),
        preset=env_spec.get("preset", "auto"),
        extra_wrappers=env_spec.get("extra_wrappers"),
    )
    # An env preset may suggest normalization; adopt it unless the config was explicit.
    p = getattr(env, "preset", None)
    if p is not None:
        if "norm_obs" not in base and p.suggest_norm_obs:
            cfg.norm_obs = True
        if "norm_reward" not in base and p.suggest_norm_reward:
            cfg.norm_reward = True

    net_spec = cfg.network
    if net_spec is None and args.hidden:
        net_spec = {"hidden": [int(x) for x in args.hidden.split(",")]}
    policy = ActorCritic.from_config(env.obs_space, env.action_space, net_spec).to(
        cfg.resolve_device()
    )

    run_dir = Path(cfg.run_dir or f"runs/{args.algo}-{args.env}-seed{cfg.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save the resolved config so it can be replayed via --config (DESIGN.md §2).
    cfg.save(run_dir / "config.yaml")
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    print(f"[oprl] {args.algo} on {args.env} | device={cfg.resolve_device()} | {run_dir}")
    if args.config:
        print(f"[oprl] config: {args.config}")
    if getattr(env, "preset", None) is not None:
        print(f"[oprl] env preset: {env.preset.name} -- {env.preset.note}")
    print(cfg.describe())

    mod.train(cfg, env, policy, Logger(run_dir=run_dir))
    env.close()
    return 0


def _cmd_sweep(args) -> int:
    from .experiment import load_sweep, run_sweep

    sweep = load_sweep(args.name)
    if args.max_parallel:
        sweep.max_parallel = args.max_parallel
    if args.serial:
        sweep.mode = "serial"
    failed = run_sweep(sweep, dry_run=args.dry_run)
    if failed:
        print(f"\n[oprl] {failed} run(s) failed", file=sys.stderr)
    return 1 if failed else 0


def _cmd_sweeps(args) -> int:
    from .experiment import list_sweeps

    sw = list_sweeps()
    if not sw:
        print("config/experiments.yaml does not exist", file=sys.stderr)
        return 1
    print("config/experiments.yaml:")
    for name, note in sw.items():
        print(f"  {name:<20} {note}")
    print("\n  usage: oprl sweep <name> [--dry-run] [--serial] [--max-parallel N]")
    return 0


def _cmd_components(args) -> int:
    """List every pluggable component -- i.e. what a config may name."""
    import oprl.advantages  # noqa: F401  triggers registration
    import oprl.envs  # noqa: F401
    import oprl.nets  # noqa: F401
    import oprl.objectives  # noqa: F401

    from .registry import describe

    print(describe())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oprl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train an algorithm")
    tr.add_argument("algo", choices=["ppo", "a2c", "vmpo"])
    tr.add_argument("--env", default="CartPole-v1")
    tr.add_argument("--config", default=None,
                    help="preset name in config/<algo>.yaml (e.g. mujoco), "
                         "or a path to a config file to replay")
    tr.add_argument("--hidden", default="64,64")

    cf = sub.add_parser("configs", help="list available presets")
    cf.add_argument("algo", nargs="?", default=None, choices=["ppo", "vmpo"])

    sw = sub.add_parser("sweep", help="run a batch of experiments")
    sw.add_argument("name")
    sw.add_argument("--dry-run", action="store_true", help="only print the commands")
    sw.add_argument("--serial", action="store_true", help="force fully sequential")
    sw.add_argument("--max-parallel", type=int, default=None)

    sub.add_parser("sweeps", help="list sweeps")
    sub.add_parser("components", help="list pluggable components")

    args, unknown = parser.parse_known_args(argv)
    if args.cmd == "configs":
        return _cmd_configs(args)
    if args.cmd == "sweep":
        return _cmd_sweep(args)
    if args.cmd == "sweeps":
        return _cmd_sweeps(args)
    if args.cmd == "components":
        return _cmd_components(args)
    return _cmd_train(args, unknown)


if __name__ == "__main__":
    sys.exit(main())
