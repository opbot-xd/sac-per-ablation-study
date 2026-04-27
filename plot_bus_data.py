#!/usr/bin/env python
"""
Standalone bus data plotter for ANM6Easy-v0 DSAC training results.

Usage:
    python plot_bus_data.py                                  # plot all experiments, full range
    python plot_bus_data.py SAC_base DSAC_PER                # specific experiments
    python plot_bus_data.py --last 50000                     # only last 50k steps
    python plot_bus_data.py --source dsac                    # use granular _dsac.pkl files
    python plot_bus_data.py --source eval                    # use eval-checkpoint data (default)
    python plot_bus_data.py --results-dir dsac_results       # custom results directory

Two data sources:
  1. "eval"  — bus data from evaluation checkpoints (inside {name}_seed{seed}.pkl)
  2. "dsac"  — granular per-step bus data (bus_data_{name}_seed{seed}_dsac.pkl)
     The dsac files have much higher resolution and are better for slicing windows.
"""

import os, sys, pickle, argparse, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Pretty defaults ──
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 0.8
rcParams['grid.linewidth'] = 0.4

EXPERIMENTS_ALL = [
    "SAC_base", "SACPER_base", "DSAC_uniform",
    "DSAC_PER", "SAC_LAP", "DSAC_LAP",
]
SEEDS = [42, 123, 456]

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
BUS_LABELS = {
    0: "Bus 0 (Slack/Grid)",
    1: "Bus 1 (Junction)",
    2: "Bus 2 (Junction)",
    3: "Bus 3 (Load+Solar)",
    4: "Bus 4 (Load+Wind)",
    5: "Bus 5 (Load+Storage)",
}
BUS_COLORS = ['#E24B4A', '#185FA5', '#1D9E75', '#EF9F27', '#533FAD', '#A8428C']


# ── Data loading ──

def load_eval_logs(results_dir, exp_names):
    """Load bus data from evaluation-checkpoint pkl files."""
    all_logs = {}
    for name in exp_names:
        logs = []
        for seed in SEEDS:
            p = os.path.join(results_dir, f"{name}_seed{seed}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    log = pickle.load(f)
                    if log.get("bus_data") and len(log["bus_data"]) > 0:
                        logs.append(log)
                    else:
                        print(f"  ⚠ {p} has no bus_data, skipping")
        if logs:
            all_logs[name] = logs
            print(f"  ✓ {name}: {len(logs)} seed(s) with eval bus data")
    return all_logs


def load_dsac_logs(results_dir, exp_names):
    """Load granular bus timeseries from _dsac.pkl files."""
    all_logs = {}
    for name in exp_names:
        logs = []
        for seed in SEEDS:
            p = os.path.join(results_dir, f"bus_data_{name}_seed{seed}_dsac.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = pickle.load(f)
                    if data.get("timeseries") and len(data["timeseries"]) > 0:
                        logs.append(data)
                    else:
                        print(f"  ⚠ {p} has no timeseries data, skipping")
        if logs:
            all_logs[name] = logs
            n_snap = len(logs[0]["timeseries"])
            print(f"  ✓ {name}: {len(logs)} seed(s), ~{n_snap} snapshots each (_dsac)")
    return all_logs


def dsac_to_eval_format(dsac_logs, last_n=None):
    """Convert _dsac timeseries to the eval-style format for uniform plotting.

    Args:
        dsac_logs: dict of {exp_name: [dsac_data, ...]}
        last_n: if set, only keep the last N steps worth of data
    Returns:
        dict in same format as load_eval_logs output
    """
    all_logs = {}
    for name, seed_data_list in dsac_logs.items():
        logs = []
        for sd in seed_data_list:
            ts = sd["timeseries"]
            if last_n is not None:
                max_step = ts[-1]["step"]
                cutoff = max_step - last_n
                ts = [t for t in ts if t["step"] >= cutoff]
            steps = [t["step"] for t in ts]
            bus_data = [t["buses"] for t in ts]
            logs.append({"steps": steps, "bus_data": bus_data})
        if logs:
            all_logs[name] = logs
    return all_logs


def get_bus_ids(all_logs):
    """Extract sorted bus IDs from the first available log."""
    for logs in all_logs.values():
        for l in logs:
            bd = l.get("bus_data", [])
            if bd and len(bd) > 0:
                return sorted(bd[0].keys())
    return []


# ── Plotting functions ──

def plot_per_experiment(all_logs, bus_ids, results_dir, suffix=""):
    """Generate per-experiment bus voltage / P / Q subplots."""
    quantities = [
        ('v_mag', 'Voltage (pu)', 'Bus Voltage Magnitude', 'voltage'),
        ('p', 'Active Power (MW)', 'Bus Active Power', 'active_power'),
        ('q', 'Reactive Power (MVAr)', 'Bus Reactive Power', 'reactive_power'),
    ]

    for exp_name, logs in all_logs.items():
        for q_key, ylabel, title_prefix, file_tag in quantities:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
            axes = axes.flatten()

            for idx, bid in enumerate(bus_ids):
                ax = axes[idx]
                for seed_i, log in enumerate(logs):
                    steps = np.array(log["steps"]) / 1000
                    vals = [bd[bid][q_key] for bd in log["bus_data"]]
                    ax.plot(steps, vals, color=BUS_COLORS[idx],
                            alpha=0.3 + 0.4 * (seed_i == 0), lw=1.5,
                            label=f'Seed {seed_i}' if len(logs) > 1 else None)

                if len(logs) > 1:
                    min_len = min(len(l["bus_data"]) for l in logs)
                    mean_vals = np.mean(
                        [[l["bus_data"][t][bid][q_key] for t in range(min_len)] for l in logs],
                        axis=0
                    )
                    steps_common = np.array(logs[0]["steps"][:min_len]) / 1000
                    ax.plot(steps_common, mean_vals, color=BUS_COLORS[idx], lw=2.5, label='Mean')

                if q_key == 'v_mag':
                    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
                    ax.axhline(0.9, color='red', ls='--', lw=1, alpha=0.4, label='V_min=0.9')
                    ax.axhline(1.1, color='red', ls='--', lw=1, alpha=0.4, label='V_max=1.1')
                else:
                    ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)

                ax.set_title(BUS_LABELS.get(bid, f'Bus {bid}'), fontsize=11, fontweight='bold')
                ax.set_xlabel('Steps (×1000)')
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=8)

            plt.suptitle(f'{title_prefix} — {LABELS.get(exp_name, exp_name)}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            out_png = os.path.join(results_dir, f"fig_bus_{file_tag}_{exp_name}{suffix}.png")
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close()
            print(f"    Saved: {out_png}")


def plot_comparison(all_logs, bus_ids, results_dir, suffix=""):
    """Generate cross-experiment comparison plots."""
    quantities = [
        ('Voltage Magnitude', 'v_mag', 'Voltage (pu)', 'voltage'),
        ('Active Power', 'p', 'Active Power (MW)', 'active_power'),
        ('Reactive Power', 'q', 'Reactive Power (MVAr)', 'reactive_power'),
    ]

    for quantity, q_key, ylabel, ylbl in quantities:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
        axes = axes.flatten()

        for idx, bid in enumerate(bus_ids):
            ax = axes[idx]
            for exp_name, logs in all_logs.items():
                min_len = min(len(l["bus_data"]) for l in logs)
                if min_len == 0:
                    continue
                mean_vals = np.mean(
                    [[l["bus_data"][t][bid][q_key] for t in range(min_len)] for l in logs],
                    axis=0
                )
                steps = np.array(logs[0]["steps"][:min_len]) / 1000
                ax.plot(steps, mean_vals, color=COLORS.get(exp_name, 'gray'),
                        lw=2, label=LABELS.get(exp_name, exp_name))

            if q_key == 'v_mag':
                ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
                ax.axhline(0.9, color='red', ls='--', lw=0.8, alpha=0.3)
                ax.axhline(1.1, color='red', ls='--', lw=0.8, alpha=0.3)
            else:
                ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)

            ax.set_title(BUS_LABELS.get(bid, f'Bus {bid}'), fontsize=11, fontweight='bold')
            ax.set_xlabel('Steps (×1000)')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7, ncol=2)

        plt.suptitle(f'Bus {quantity} — All Experiments Compared',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_png = os.path.join(results_dir, f"fig_bus_{ylbl}_comparison{suffix}.png")
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"    Saved: {out_png}")


def plot_single_bus_timeseries(all_logs, bus_ids, results_dir, bus_id=5, suffix=""):
    """Single-bus deep dive: 3-panel (V, P, Q) comparing all experiments."""
    if bus_id not in bus_ids:
        print(f"  Bus {bus_id} not found, skipping single-bus plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=120)
    panels = [
        ('v_mag', 'Voltage (pu)', 'Voltage Magnitude'),
        ('p', 'Active Power (MW)', 'Active Power'),
        ('q', 'Reactive Power (MVAr)', 'Reactive Power'),
    ]

    for ax, (q_key, ylabel, title) in zip(axes, panels):
        for exp_name, logs in all_logs.items():
            min_len = min(len(l["bus_data"]) for l in logs)
            if min_len == 0:
                continue
            mean_vals = np.mean(
                [[l["bus_data"][t][bus_id][q_key] for t in range(min_len)] for l in logs],
                axis=0
            )
            steps = np.array(logs[0]["steps"][:min_len]) / 1000
            ax.plot(steps, mean_vals, color=COLORS.get(exp_name, 'gray'),
                    lw=2, label=LABELS.get(exp_name, exp_name))

        if q_key == 'v_mag':
            ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
            ax.axhline(0.9, color='red', ls='--', lw=1, alpha=0.3)
            ax.axhline(1.1, color='red', ls='--', lw=1, alpha=0.3)
        else:
            ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Steps (×1000)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=1)

    plt.suptitle(f'{BUS_LABELS.get(bus_id, f"Bus {bus_id}")} — Deep Dive',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(results_dir, f"fig_bus{bus_id}_deep_dive{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot bus data from DSAC training results")
    parser.add_argument("experiments", nargs="*", default=None,
                        help="Experiment names to plot (default: all)")
    parser.add_argument("--results-dir", default="dsac_results",
                        help="Directory containing result files")
    parser.add_argument("--source", choices=["eval", "dsac"], default="dsac",
                        help="Data source: 'eval' (checkpoint pkl) or 'dsac' (granular _dsac.pkl)")
    parser.add_argument("--last", type=int, default=None,
                        help="Only plot the last N steps (e.g. --last 50000)")
    parser.add_argument("--bus-dive", type=int, default=5,
                        help="Bus ID for deep-dive plot (default: 5)")
    args = parser.parse_args()

    results_dir = args.results_dir
    exp_names = args.experiments if args.experiments else EXPERIMENTS_ALL
    suffix = f"_last{args.last // 1000}k" if args.last else ""
    suffix += f"_{args.source}"

    print(f"Loading results from: {results_dir}/")
    print(f"Source: {args.source}" + (f" | Last {args.last:,} steps" if args.last else " | Full range"))

    if args.source == "dsac":
        raw_logs = load_dsac_logs(results_dir, exp_names)
        if not raw_logs:
            print("\n⚠ No _dsac.pkl files found, falling back to eval data...")
            all_logs = load_eval_logs(results_dir, exp_names)
            suffix = suffix.replace("_dsac", "_eval")
        else:
            all_logs = dsac_to_eval_format(raw_logs, last_n=args.last)
    else:
        all_logs = load_eval_logs(results_dir, exp_names)

    if not all_logs:
        print("\n❌ No results with bus data found. Run train_dsac.py first.")
        sys.exit(1)

    bus_ids = get_bus_ids(all_logs)
    print(f"\nFound {len(bus_ids)} buses: {bus_ids}")
    print(f"Experiments with bus data: {list(all_logs.keys())}\n")

    print("Generating per-experiment bus plots...")
    plot_per_experiment(all_logs, bus_ids, results_dir, suffix)

    if len(all_logs) > 1:
        print("\nGenerating cross-experiment comparison plots...")
        plot_comparison(all_logs, bus_ids, results_dir, suffix)

    print(f"\nGenerating Bus {args.bus_dive} deep-dive plot...")
    plot_single_bus_timeseries(all_logs, bus_ids, results_dir,
                               bus_id=args.bus_dive, suffix=suffix)

    print(f"\n✅ All bus data plots saved to {results_dir}/")


if __name__ == "__main__":
    main()
