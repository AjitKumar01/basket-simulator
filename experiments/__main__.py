"""Portable entry point. Import runners only after resolving the output directory."""

import argparse
import importlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Prepare one dataset in a new run directory"
    )
    prepare.add_argument(
        "--dataset", choices=["synthetic", "erim", "dunnhumby"], required=True
    )
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument(
        "--source", type=Path, help="ERIM canonical directory or Dunnhumby data root"
    )
    prepare.add_argument(
        "--raw-source",
        type=Path,
        help="Directory containing Dunnhumby product.csv and causal_data.csv",
    )
    prepare.add_argument(
        "--family",
        choices=[
            "latent_missions_gumbel",
            "product_ising",
            "latent_missions_gumbel_zero_price",
        ],
        default="latent_missions_gumbel",
    )
    prepare.add_argument("--seed", type=int, default=941000)
    prepare.add_argument("--train-size", type=int, default=20000)
    prepare.add_argument("--validation-size", type=int, default=5000)
    prepare.add_argument("--test-size", type=int, default=5000)
    for name in ("train", "evaluate", "monitor"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
    check = commands.add_parser(
        "check", help="Tiny mathematical equivalence checks; no dataset training"
    )
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = (args.out if args.command in ("prepare", "check") else args.run).resolve()
    os.environ["BASKET_RUN_DIR"] = str(run)
    if args.command == "prepare":
        if args.dataset != "synthetic" and args.source is None:
            parser.error("--source is required for real data")
        if min(args.train_size, args.validation_size, args.test_size) < 1:
            parser.error("Split sizes must be positive")
        run.mkdir(parents=True, exist_ok=False)
        if args.source:
            os.environ["BASKET_DATA_SOURCE"] = str(args.source.resolve())
        if args.raw_source:
            os.environ["BASKET_RAW_SOURCE"] = str(args.raw_source.resolve())
        protocol = {
            "recorded_before_preparation_and_fitting": True,
            "dataset": args.dataset,
            "source": str(args.source.resolve()) if args.source else None,
            "raw_source": str(args.raw_source.resolve()) if args.raw_source else None,
            "family": args.family if args.dataset == "synthetic" else None,
            "seed": args.seed,
            "sizes": {
                k: getattr(args, k + "_size") for k in ("train", "validation", "test")
            },
            "fit": {
                "iterations": 350,
                "calls": 500,
                "timed_seconds": 1800,
                "batch_size": 512,
                "ridge_preferences": [1, 10],
                "ridge_structure": 10,
                "threshold": 0.0002,
                "coordinates": "Fixed Fisher for preferences; both raw and scaled checks",
                "continuation": "One improving, finite final-state continuation; preserve original failure",
                "selection": "Highest validation likelihood, including initialization",
            },
            "evaluation": {
                "all_test_baskets_likelihood": True,
                "generation_contexts": 4096,
                "draws_per_replica": 128,
                "replicas": 3,
                "household_bootstraps": 1000,
                "separate_store_period_sensitivity": True,
                "price_scenario_contexts": 32,
                "discount": 0.1,
                "real_causal_truth": None,
                "hypothetical_cost_fractions": [0.3, 0.6, 0.85],
            },
            "scope": "Conditional nonempty distinct-product baskets; no visits, units, inventory or dynamic shop claim",
            "packaging": "Portable from-scratch controller adapted from current research; not a replay of historical fitted checkpoints",
            "recorded_at": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
        }
        (run / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        module = importlib.import_module(f"experiments.{args.dataset}.prepare")
        module.main()
        return
    if args.command == "check":
        run.mkdir(parents=True, exist_ok=False)
        from .preflight import main as check_math

        check_math()
        return
    if not (run / "data/config.json").exists():
        parser.error("Prepared data/config.json is missing")
    if args.command == "monitor":
        from .monitor import main as monitor

        monitor(run)
        return
    # Lock all writing workflows, including independent evaluation.
    import fcntl

    with (run / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            from .runtime import install

            install()
            if args.command == "train":
                from .preflight import main as check_math
                from .pipeline import run as train

                check_math()
                train()
            else:
                from .evaluate import main as evaluate

                evaluate()
        except Exception as exc:
            (run / "stopped.json").write_text(
                json.dumps(
                    {
                        "state": "stopped",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "time": time.time(),
                    },
                    indent=2,
                )
                + "\n"
            )
            from .common import status

            status("stopped", error=str(exc))
            raise


if __name__ == "__main__":
    main()
