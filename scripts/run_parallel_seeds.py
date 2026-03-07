#!/usr/bin/env python3
"""
scripts/run_parallel_seeds.py
==============================
Parallel Seed Launcher for 2× NVIDIA T4 (Kaggle environment)
=============================================================

Distributes N experiment seeds across 2 GPUs concurrently using
Python subprocess — NO DDP overhead, NO NCCL, NO torch.distributed.

Each seed runs as a completely independent process with its own:
    • CUDA_VISIBLE_DEVICES  (limits which physical GPU it uses)
    • GROK_SEED             (overrides cfg.seed in the experiment)
    • Hydra output_dir      (separate result directory per seed)

GPU assignment
--------------
    Seeds are round-robin assigned to GPUs:
        num_seeds=5 → seeds 1,3,5 on GPU-0 | seeds 2,4 on GPU-1
    (Sequential within each GPU; no shared memory between processes.)

    Example with default 5 seeds:
        GPU-0: seed_0, seed_2, seed_4  (3 processes, sequential)
        GPU-1: seed_1, seed_3          (2 processes, sequential)

    The two GPU queues run concurrently (in parallel via Python threads),
    so total wall-clock time ≈ max(time_on_GPU0, time_on_GPU1).

Usage
-----
    # Run all 3 experiments with default config
    python scripts/run_parallel_seeds.py

    # Run only Experiment B, 5 seeds, custom sparsities
    python scripts/run_parallel_seeds.py \\
        --experiment exp_b \\
        --num-seeds 5 \\
        --extra-args "pruning.target_sparsities=[0.0,0.5,0.9]"

    # Debug mode (fast, 2 seeds)
    python scripts/run_parallel_seeds.py --debug

    # Dry run — print commands without executing
    python scripts/run_parallel_seeds.py --dry-run

Arguments
---------
    --experiment     Which experiment to run: exp_b | exp_a | exp_c | all
    --num-seeds      Number of independent seeds (default: 5)
    --base-seed      Starting seed value (default: 0)
    --num-gpus       Number of available GPUs (default: 2)
    --extra-args     Additional Hydra overrides passed to all runs
    --debug          Enable fast debug config (overrides n_grok_steps etc.)
    --dry-run        Print subprocess commands without executing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from queue import Queue
from pathlib import Path


# ===========================================================================
# GPU Queue Worker
# ===========================================================================

def _worker(
    gpu_id:   int,
    job_queue: Queue,
    results:  list[tuple[int, int, str]],
    lock:     threading.Lock,
    dry_run:  bool,
) -> None:
    """
    Thread worker: drain job_queue, run each command with CUDA_VISIBLE_DEVICES=gpu_id.
    Appends (seed, return_code, stderr_tail) to results.
    """
    while True:
        try:
            seed, cmd_parts = job_queue.get_nowait()
        except Exception:
            break   # queue empty

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["GROK_SEED"]            = str(seed)

        cmd_str = " ".join(cmd_parts)
        print(f"  [GPU-{gpu_id}] seed={seed}  starting...\n    CMD: {cmd_str}\n")

        if dry_run:
            print(f"  [GPU-{gpu_id}] seed={seed}  DRY RUN — not executed.\n")
            with lock:
                results.append((seed, 0, "dry_run"))
            job_queue.task_done()
            continue

        try:
            t0  = time.time()
            ret = subprocess.run(
                cmd_parts,
                env=env,
                capture_output=False,   # forward stdout/stderr to terminal
                check=False,
            )
            elapsed = time.time() - t0
            status  = "✓ OK" if ret.returncode == 0 else f"✗ FAILED (rc={ret.returncode})"
            print(f"  [GPU-{gpu_id}] seed={seed}  {status}  ({elapsed:.0f}s)\n")

            with lock:
                results.append((seed, ret.returncode, status))

        except Exception as exc:
            print(f"  [GPU-{gpu_id}] seed={seed}  EXCEPTION: {exc}\n")
            with lock:
                results.append((seed, -1, str(exc)))

        finally:
            job_queue.task_done()


# ===========================================================================
# Main launcher
# ===========================================================================

EXPERIMENT_MODULES = {
    "exp_b": "experiments.exp_b_lth_then_grok",
    "exp_a": "experiments.exp_a_grok_then_prune",
    "exp_c": "experiments.exp_c_wd_ablation",
}


def build_command(
    module:      str,
    seed:        int,
    extra_args:  list[str],
    debug:       bool,
) -> list[str]:
    """
    Build the subprocess command list for one seed run.

    Uses `python -m <module>` so that Python path resolution works
    regardless of whether the package is installed or just on PYTHONPATH.

    Hydra overrides are appended as positional arguments.
    """
    cmd = [sys.executable, "-m", module]

    # Hydra output dir: separate per seed so runs don't collide
    cmd.append(f"hydra.run.dir=outputs/{module.split('.')[-1]}/seed_{seed}")

    # Pass seed as Hydra override (experiment script reads GROK_SEED env var,
    # but also accept it as a Hydra param as backup)
    cmd.append(f"seed={seed}")

    if debug:
        cmd += [
            "training.n_grok_steps=5000",
            "pruning.imp_steps_per_round=500",
            "training.log_every=50",
            "pruning.target_sparsities=[0.0,0.5,0.9]",
            "num_seeds=1",
        ]

    cmd.extend(extra_args)
    return cmd


def run_parallel(
    experiment:  str,
    seeds:       list[int],
    num_gpus:    int,
    extra_args:  list[str],
    debug:       bool,
    dry_run:     bool,
) -> None:
    """
    Distribute seeds across num_gpus GPUs and run concurrently.
    """
    modules = (
        list(EXPERIMENT_MODULES.values())
        if experiment == "all"
        else [EXPERIMENT_MODULES[experiment]]
    )

    all_results: list[tuple[int, int, str]] = []
    lock = threading.Lock()

    for module in modules:
        print(f"\n{'='*65}")
        print(f"  Module : {module}")
        print(f"  Seeds  : {seeds}")
        print(f"  GPUs   : {num_gpus}")
        print(f"{'='*65}\n")

        # Build job queue
        job_queue: Queue = Queue()
        for seed in seeds:
            cmd = build_command(module, seed, extra_args, debug)
            job_queue.put((seed, cmd))

        # Launch one worker thread per GPU
        # Each thread drains the shared queue sequentially on its GPU
        threads = []
        for gpu_id in range(num_gpus):
            t = threading.Thread(
                target=_worker,
                args=(gpu_id, job_queue, all_results, lock, dry_run),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Wait for all jobs to complete
        job_queue.join()
        for t in threads:
            t.join()

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Launcher Summary")
    print(f"{'='*65}")
    n_ok   = sum(1 for _, rc, _ in all_results if rc == 0)
    n_fail = len(all_results) - n_ok
    for seed, rc, status in sorted(all_results, key=lambda r: r[0]):
        print(f"  seed={seed:3d}  {status}")
    print(f"\n  Total: {len(all_results)} runs | {n_ok} succeeded | {n_fail} failed")

    if n_fail > 0:
        print("\n  WARNING: Some runs failed. Check logs above.")
        sys.exit(1)
    else:
        print("\n  All runs completed successfully.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel seed launcher for Grokking × LTH experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=["exp_a", "exp_b", "exp_c", "all"],
        default="exp_b",
        help="Which experiment to run (default: exp_b)",
    )
    parser.add_argument(
        "--num-seeds", "-n",
        type=int, default=5,
        help="Number of independent seeds (default: 5)",
    )
    parser.add_argument(
        "--base-seed", "-b",
        type=int, default=0,
        help="Starting seed value (default: 0). Seeds = base_seed..base_seed+num_seeds-1",
    )
    parser.add_argument(
        "--num-gpus", "-g",
        type=int, default=2,
        help="Number of available GPUs (default: 2 for Kaggle T4)",
    )
    parser.add_argument(
        "--extra-args",
        nargs="*", default=[],
        help="Additional Hydra overrides, e.g. training.weight_decay=5e-3",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable fast debug config (5k steps, 3 sparsities, 2 seeds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))

    if args.debug:
        seeds = seeds[:2]   # 2 seeds for debug
        print("DEBUG mode: using 2 seeds only.\n")

    run_parallel(
        experiment = args.experiment,
        seeds      = seeds,
        num_gpus   = args.num_gpus,
        extra_args = args.extra_args or [],
        debug      = args.debug,
        dry_run    = args.dry_run,
    )


if __name__ == "__main__":
    main()
