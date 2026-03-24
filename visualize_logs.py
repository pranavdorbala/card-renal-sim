"""
Visualize simulation log outputs from logs/simulation_runs.jsonl.

Usage:
    python visualize_logs.py
    python visualize_logs.py --log logs/simulation_runs.jsonl
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_logs(path: str = "logs/simulation_runs.jsonl"):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_runs(records, source="run_coupled_simulation"):
    """Filter to a specific source and drop NaN rows."""
    out = []
    for r in records:
        if r.get("source") != source:
            continue
        outputs = r.get("outputs", {})
        if any(v != v for v in outputs.values()):  # NaN check
            continue
        out.append(r)
    return out


def extract_trajectories(records):
    """Group consecutive multi-step runs into trajectories."""
    trajectories = []
    current = []
    prev_step = 0

    for r in records:
        step = r.get("step", 1)
        if step <= prev_step and current:
            trajectories.append(current)
            current = []
        current.append(r)
        prev_step = step

    if current:
        trajectories.append(current)
    return trajectories


def plot_disease_progression(trajectories, save_dir="plots"):
    """Plot hemodynamic trajectories as disease worsens over steps."""
    Path(save_dir).mkdir(exist_ok=True)

    # Only use multi-step trajectories (≥3 steps)
    multi = [t for t in trajectories if len(t) >= 3]
    if not multi:
        print("No multi-step trajectories found for progression plot.")
        return

    metrics = ["EF", "MAP", "CO", "GFR", "V_blood", "P_glom", "SV", "Na_excr"]
    units = {
        "EF": "%", "MAP": "mmHg", "CO": "L/min", "GFR": "mL/min",
        "V_blood": "mL", "P_glom": "mmHg", "SV": "mL", "Na_excr": "mEq/day",
    }
    # Clinical reference ranges (normal)
    ref_ranges = {
        "EF": (55, 70), "MAP": (70, 100), "CO": (4.0, 8.0),
        "GFR": (60, 120), "V_blood": (4500, 5500), "P_glom": (40, 55),
        "SV": (55, 100), "Na_excr": (1000, 4000),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, traj in enumerate(multi):
            steps = [r["step"] for r in traj]
            vals = [r["outputs"][metric] for r in traj]
            # Label with disease severity at last step
            last = traj[-1]["params"]
            label = f"Sf={last['Sf_act_scale']}, Kf={last['Kf_scale']}"
            ax.plot(steps, vals, "o-", label=label, linewidth=2, markersize=6)

        # Shade normal range
        lo, hi = ref_ranges.get(metric, (None, None))
        if lo is not None:
            ax.axhspan(lo, hi, alpha=0.1, color="green", label="Normal range")

        ax.set_xlabel("Coupling Step (month)")
        ax.set_ylabel(f"{metric} ({units.get(metric, '')})")
        ax.set_title(f"{metric} — Disease Progression")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Cardiorenal Disease Progression Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/disease_progression.png", dpi=150, bbox_inches="tight")
    print(f"Saved {save_dir}/disease_progression.png")
    plt.close()


def plot_parameter_sensitivity(records, save_dir="plots"):
    """Scatter plots: each disease parameter vs each output metric (step 1 only)."""
    Path(save_dir).mkdir(exist_ok=True)

    # Use only step-1 records to avoid temporal confounds
    step1 = [r for r in records if r.get("step", 1) == 1]
    if len(step1) < 3:
        print("Not enough step-1 records for sensitivity plot.")
        return

    params = ["Sf_act_scale", "k1_scale", "Kf_scale", "inflammation_scale", "diabetes_scale"]
    outputs = ["EF", "MAP", "CO", "GFR", "SV"]

    fig, axes = plt.subplots(len(outputs), len(params), figsize=(22, 18))

    for i, out_name in enumerate(outputs):
        for j, param_name in enumerate(params):
            ax = axes[i][j]
            x_vals, y_vals = [], []
            for r in step1:
                p = r["params"].get(param_name)
                o = r["outputs"].get(out_name)
                if p is not None and o is not None:
                    x_vals.append(p)
                    y_vals.append(o)

            ax.scatter(x_vals, y_vals, alpha=0.7, s=40, edgecolors="k", linewidth=0.5)

            if i == 0:
                ax.set_title(param_name, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(out_name, fontsize=10, fontweight="bold")
            if i == len(outputs) - 1:
                ax.set_xlabel(param_name)

            ax.grid(True, alpha=0.3)

    plt.suptitle("Parameter Sensitivity: Input Parameters vs Output Metrics (Step 1)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"Saved {save_dir}/parameter_sensitivity.png")
    plt.close()


def plot_cardiac_vs_renal(records, save_dir="plots"):
    """Show cardiorenal coupling: cardiac outputs vs renal outputs."""
    Path(save_dir).mkdir(exist_ok=True)

    pairs = [
        ("EF", "GFR", "Heart failure → kidney damage"),
        ("CO", "GFR", "Low output → low filtration"),
        ("MAP", "P_glom", "Systemic pressure → glomerular pressure"),
        ("V_blood", "Na_excr", "Volume overload → sodium excretion"),
        ("EF", "V_blood", "Pump failure → fluid retention"),
        ("SV", "GFR", "Stroke volume → filtration"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (x_metric, y_metric, title) in enumerate(pairs):
        ax = axes[idx]
        x_vals, y_vals, colors = [], [], []
        for r in records:
            x = r["outputs"].get(x_metric)
            y = r["outputs"].get(y_metric)
            if x is not None and y is not None:
                x_vals.append(x)
                y_vals.append(y)
                # Color by disease severity (sum of inflammation + diabetes)
                p = r["params"]
                severity = p.get("inflammation_scale", 0) + p.get("diabetes_scale", 0)
                colors.append(severity)

        sc = ax.scatter(x_vals, y_vals, c=colors, cmap="RdYlGn_r", alpha=0.8,
                        s=50, edgecolors="k", linewidth=0.5, vmin=0, vmax=2)
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Inflammation + Diabetes")

    plt.suptitle("Cardiorenal Coupling: Heart vs Kidney Outputs",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cardiac_vs_renal.png", dpi=150, bbox_inches="tight")
    print(f"Saved {save_dir}/cardiac_vs_renal.png")
    plt.close()


def plot_healthy_vs_extreme(records, save_dir="plots"):
    """Bar chart comparing healthy baseline vs extreme disease."""
    Path(save_dir).mkdir(exist_ok=True)

    # Find a healthy step-1 and the most extreme step-1
    step1 = [r for r in records if r.get("step", 1) == 1]
    healthy = None
    extreme = None

    for r in step1:
        p = r["params"]
        if (p.get("Sf_act_scale", 1) == 1.0 and p.get("Kf_scale", 1) == 1.0
                and p.get("inflammation_scale", 0) == 0.0):
            healthy = r
        severity = (1 - p.get("Sf_act_scale", 1)) + (1 - p.get("Kf_scale", 1)) + \
                   p.get("inflammation_scale", 0) + p.get("diabetes_scale", 0)
        if extreme is None or severity > extreme["_severity"]:
            r["_severity"] = severity
            extreme = r

    if not healthy or not extreme:
        print("Need both healthy and extreme records for comparison.")
        return

    metrics = ["EF", "MAP", "CO", "GFR", "SV"]
    units = {"EF": "%", "MAP": "mmHg", "CO": "L/min", "GFR": "mL/min", "SV": "mL"}

    h_vals = [healthy["outputs"][m] for m in metrics]
    e_vals = [extreme["outputs"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, h_vals, width, label="Healthy", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x + width/2, e_vals, width, label="Severe CRS", color="#F44336", alpha=0.8)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({units[m]})" for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ep = extreme["params"]
    ax.set_title(
        f"Healthy vs Severe Cardiorenal Syndrome\n"
        f"(Sf={ep['Sf_act_scale']}, k1={ep['k1_scale']}, Kf={ep['Kf_scale']}, "
        f"infl={ep['inflammation_scale']}, diab={ep['diabetes_scale']})",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/healthy_vs_extreme.png", dpi=150, bbox_inches="tight")
    print(f"Saved {save_dir}/healthy_vs_extreme.png")
    plt.close()


def print_sanity_check(records):
    """Print physiological sanity checks on the results."""
    print("\n" + "=" * 60)
    print("PHYSIOLOGICAL SANITY CHECK")
    print("=" * 60)

    issues = []
    for r in records:
        o = r["outputs"]
        p = r["params"]
        step = r.get("step", "?")
        tag = f"step={step}, Sf={p.get('Sf_act_scale','?')}, Kf={p.get('Kf_scale','?')}"

        # Check ranges
        if o["EF"] < 10:
            issues.append(f"  EF={o['EF']:.1f}% (dangerously low) — {tag}")
        if o["EF"] > 75:
            issues.append(f"  EF={o['EF']:.1f}% (supraphysiological) — {tag}")
        if o["MAP"] < 50:
            issues.append(f"  MAP={o['MAP']:.1f} mmHg (shock) — {tag}")
        if o["MAP"] > 130:
            issues.append(f"  MAP={o['MAP']:.1f} mmHg (hypertensive crisis) — {tag}")
        if o["GFR"] < 5:
            issues.append(f"  GFR={o['GFR']:.1f} mL/min (dialysis territory) — {tag}")
        if o["GFR"] > 150:
            issues.append(f"  GFR={o['GFR']:.1f} mL/min (hyperfiltration) — {tag}")
        if o["V_blood"] > 7000:
            issues.append(f"  V_blood={o['V_blood']:.0f} mL (massive overload) — {tag}")
        if o["CO"] < 1.0:
            issues.append(f"  CO={o['CO']:.2f} L/min (cardiogenic shock) — {tag}")

    if issues:
        print(f"\nFound {len(issues)} flagged values:")
        for issue in issues:
            print(issue)
    else:
        print("\nAll outputs within expected physiological ranges.")

    # Check expected directions
    print("\n--- Directional checks ---")
    step1 = [r for r in records if r.get("step") == 1]
    healthy = [r for r in step1
               if r["params"].get("Sf_act_scale", 1) == 1.0
               and r["params"].get("Kf_scale", 1) == 1.0
               and r["params"].get("inflammation_scale", 0) == 0.0]
    diseased = [r for r in step1
                if r["params"].get("Sf_act_scale", 1) < 0.5
                or r["params"].get("Kf_scale", 1) < 0.5]

    if healthy and diseased:
        h = healthy[0]["outputs"]
        d = diseased[0]["outputs"]
        checks = [
            ("EF",      "decreases with disease", d["EF"] < h["EF"]),
            ("GFR",     "decreases with disease", d["GFR"] < h["GFR"]),
            ("V_blood", "increases with disease",  d["V_blood"] > h["V_blood"]),
            ("P_glom",  "increases with disease",  d["P_glom"] > h["P_glom"]),
        ]
        for metric, expected, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {metric}: {expected} (healthy={h[metric]:.1f}, diseased={d[metric]:.1f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize cardiorenal simulation logs")
    parser.add_argument("--log", default="logs/simulation_runs.jsonl", help="Path to JSONL log file")
    parser.add_argument("--save-dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()

    records = load_logs(args.log)
    print(f"Loaded {len(records)} records from {args.log}")

    # Filter to successful non-NaN coupled simulation runs
    coupled = filter_runs(records, source="run_coupled_simulation")
    rl_runs = filter_runs(records, source="run_coupled_simulation_rl")
    print(f"  {len(coupled)} coupled simulation records (non-NaN)")
    print(f"  {len(rl_runs)} RL simulation records (non-NaN)")

    # Sanity checks
    print_sanity_check(coupled)

    # Trajectories
    trajectories = extract_trajectories(coupled)
    print(f"Found {len(trajectories)} trajectories ({sum(1 for t in trajectories if len(t) >= 3)} with ≥3 steps)")

    # Generate plots
    plot_disease_progression(trajectories, save_dir=args.save_dir)
    plot_parameter_sensitivity(coupled, save_dir=args.save_dir)
    plot_cardiac_vs_renal(coupled, save_dir=args.save_dir)
    plot_healthy_vs_extreme(coupled, save_dir=args.save_dir)

    print(f"\nAll plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
