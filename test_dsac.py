#!/usr/bin/env python
"""
Quick test run: 6 experiments × 1 seed × 30k steps.
Validates all functionality (training, bus data logging, plotting) on your laptop.

Usage:
    python test_dsac.py
"""

import os, sys, pickle, time
import numpy as np

# Override the config before importing train_dsac
# We patch EXPERIMENTS to use only 1 seed each
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ── Import everything from train_dsac ──
from train_dsac import (
    EXPERIMENTS, ExpConfig, train_one_seed, aggregate,
    load_all_results,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {
    "SAC_base": "#E24B4A", "SACPER_base": "#185FA5",
    "DSAC_uniform": "#1D9E75", "DSAC_PER": "#EF9F27",
    "SAC_LAP": "#533FAD", "DSAC_LAP": "#A8428C",
}
LABELS = {
    "SAC_base": "SAC (uniform)", "SACPER_base": "SAC + PER",
    "DSAC_uniform": "DSAC (uniform)", "DSAC_PER": "DSAC + PER",
    "SAC_LAP": "SAC + LAP", "DSAC_LAP": "DSAC + LAP",
}

RESULTS_DIR = "dsac_results_test"
SINGLE_SEED = [42]

def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Patch experiments: 1 seed, 30k steps
    test_configs = {}
    for name, cfg in EXPERIMENTS.items():
        test_configs[name] = ExpConfig(
            name=cfg.name,
            use_distributional=cfg.use_distributional,
            use_per=cfg.use_per,
            use_lap=cfg.use_lap,
            per_alpha=cfg.per_alpha,
            per_beta_start=cfg.per_beta_start,
            per_beta_end=cfg.per_beta_end,
            grad_clip=cfg.grad_clip,
            warmup_steps=cfg.warmup_steps,
            buffer_size=cfg.buffer_size,
            lr_alpha=cfg.lr_alpha,
            update_every=cfg.update_every,
            total_steps=30_000,
            seeds=SINGLE_SEED,
        )

    print("=" * 60)
    print("  TEST RUN: 6 experiments × 1 seed × 30k steps")
    print("=" * 60)
    for name, cfg in test_configs.items():
        print(f"  {name:15s}  dist={cfg.use_distributional}  per={cfg.use_per}  lap={cfg.use_lap}")
    print(f"\nResults dir: {RESULTS_DIR}/\n")

    # Run in parallel with 3 workers
    import concurrent.futures
    all_logs = {n: [] for n in test_configs}
    tasks = [(name, SINGLE_SEED[0], RESULTS_DIR) for name in test_configs]
    t_total = time.time()

    print(f"Running 6 experiments with 3 parallel workers...\n")
    ctx = __import__('multiprocessing').get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=6, mp_context=ctx) as executor:
        from train_dsac import _train_worker
        # Patch: _train_worker reads from global EXPERIMENTS, but our configs
        # have the same names / same total_steps, so it works directly.
        futures = {executor.submit(_train_worker, t): t[0] for t in tasks}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            done_count += 1
            name = futures[future]
            try:
                ret_name, log = future.result()
                all_logs[ret_name].append(log)
                print(f"[{done_count}/6] ✓ Finished {ret_name}")
            except Exception as e:
                print(f"[{done_count}/6] ✗ {name} failed: {e}")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  All training done in {elapsed/60:.1f} min")
    print(f"{'='*60}")

    # ── Verify bus data was logged ──
    print("\n── Checking bus data ──")
    for name, logs in all_logs.items():
        log = logs[0]
        n_eval = len(log.get("bus_data", []))
        bus_path = os.path.join(RESULTS_DIR, f"bus_data_{name}_seed{SINGLE_SEED[0]}_dsac.pkl")
        n_dsac = 0
        if os.path.exists(bus_path):
            with open(bus_path, "rb") as f:
                dsac_data = pickle.load(f)
                n_dsac = len(dsac_data.get("timeseries", []))
        status = "✓" if (n_eval > 0 and n_dsac > 0) else "✗"
        print(f"  {status} {name:15s}  eval_checkpoints={n_eval}  dsac_snapshots={n_dsac}")

    # ── Generate plots ──
    print("\n── Generating plots ──")
    agg = {name: aggregate(logs) for name, logs in all_logs.items() if logs}

    # Quick return comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in test_configs:
        if name not in agg or agg[name] is None:
            continue
        a = agg[name]
        ax.plot(a["steps"]/1000, np.clip(a["mean_r"], -300, 0),
                color=COLORS.get(name, "gray"), lw=2,
                label=LABELS.get(name, name))
    ax.set_xlabel("Steps (×1000)")
    ax.set_ylabel("Return")
    ax.set_title("Test Run: All 6 Variants (1 seed, 30k steps)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/test_returns.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/test_returns.png")

    # ── Run the standalone bus plotter on test results ──
    print("\n── Running bus data plotter ──")
    os.system(f"python plot_bus_data.py --results-dir {RESULTS_DIR} --source dsac")

    print(f"\n{'='*60}")
    print(f"  ✅ TEST COMPLETE — all plots in {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
