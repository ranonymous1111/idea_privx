#!/usr/bin/env python3
"""
plot_results.py — Generate paper-quality plots for PrivX / PrivF experiments

Reads ablation CSVs from results/ and generates:
  1. Main comparison bar chart: PrivF-Cosine vs PrivF-Diff vs PrivX per dataset
  2. Window size ablation line plot (AP vs window_size)
  3. Train/Test split ablation line plot (AP vs ρ)
  4. DP mechanism comparison (AP vs epsilon, faceted by noise type)
  5. Adaptive attacker heatmap (AP vs κ × ρ)
  6. PrivX vs PrivF marginal leakage bar chart (by explainer)

All figures saved to results/figures/ as PDF + PNG.

Usage:
  python plot_results.py
  python plot_results.py --results-dir ./results --output-dir ./results/figures
"""

import os
import argparse
import glob
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  2,
    "lines.markersize": 6,
})

COLORS = {
    "PrivF-Cosine":    "#1f77b4",
    "PrivF-Diff":      "#ff7f0e",
    "Grad":            "#2ca02c",
    "GradInput":       "#d62728",
    "GNNExplainer":    "#9467bd",
    "GraphLime":       "#8c564b",
    "gaussian":        "#1f77b4",
    "laplacian":       "#ff7f0e",
    "renyi":           "#2ca02c",
}
MARKERS = {
    "PrivF-Cosine":    "o",
    "PrivF-Diff":      "s",
    "Grad":            "^",
    "GradInput":       "D",
    "GNNExplainer":    "v",
    "GraphLime":       "P",
    "gaussian":        "o",
    "laplacian":       "s",
    "renyi":           "^",
}


# ──────────────────────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_master(results_dir):
    """Load the latest master_results_*.csv, or merge on-the-fly."""
    masters = sorted(glob.glob(os.path.join(results_dir, "master_results_*.csv")))
    if masters:
        df = pd.read_csv(masters[-1])
        return df

    # Fallback: run collect on-the-fly
    collect_path = os.path.join(results_dir, "collect_results.py")
    if os.path.exists(collect_path):
        import subprocess
        subprocess.run(["python", collect_path, "--results-dir", results_dir], check=False)
        masters = sorted(glob.glob(os.path.join(results_dir, "master_results_*.csv")))
        if masters:
            return pd.read_csv(masters[-1])

    return pd.DataFrame()


def best_scale(df):
    """Pick the best guidance scale per (dataset, explainer, epsilon) combo."""
    if "scale" not in df.columns:
        return df
    idx = df.groupby(["dataset", "explainer", "epsilon"])["AP_mean"].idxmax()
    return df.loc[idx].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Main comparison (AP vs epsilon, per dataset)
# ──────────────────────────────────────────────────────────────────────────────
def plot_main_comparison(df, output_dir):
    if df.empty or "dataset" not in df.columns:
        return

    datasets = sorted(df["dataset"].dropna().unique())
    n_ds = len(datasets)
    if n_ds == 0:
        return

    ncols = min(4, n_ds)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    methods = sorted(df["explainer"].dropna().unique()) if "explainer" in df.columns else ["PrivF"]

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        for method in methods:
            m_sub = sub[sub["explainer"] == method] if "explainer" in sub.columns else sub
            if m_sub.empty:
                continue
            m_sub = m_sub.sort_values("epsilon")
            eps = m_sub["epsilon"].astype(float)
            ap  = m_sub["AP_mean"]
            ap_std = m_sub.get("AP_std", pd.Series([0] * len(ap)))
            c = COLORS.get(method, None)
            mk = MARKERS.get(method, "o")
            ax.plot(eps, ap, marker=mk, color=c, label=method)
            ax.fill_between(eps, ap - ap_std, ap + ap_std, alpha=0.15, color=c)

        ax.set_title(ds, fontweight="bold")
        ax.set_xlabel("ε (DP budget)")
        ax.set_ylabel("AP (↑)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               bbox_to_anchor=(0.5, -0.02))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("PrivX vs PrivF: AP Score vs ε", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "main_comparison")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Window size ablation
# ──────────────────────────────────────────────────────────────────────────────
def plot_window_ablation(df, output_dir):
    tbl_path = os.path.join(os.path.dirname(output_dir), "tables", "ablation_window.csv")
    if not os.path.exists(tbl_path):
        if df.empty or "window_size" not in df.columns:
            return
        tbl = df[df["window_size"].notna()]
    else:
        tbl = pd.read_csv(tbl_path)
    if tbl.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ds in zip(axes, ["Cora", "Texas"]):
        sub = tbl[tbl["dataset"] == ds] if "dataset" in tbl.columns else tbl
        if sub.empty:
            continue
        # Best epsilon slice
        for ws, grp in sub.groupby("window_size"):
            ep_sub = grp.sort_values("epsilon")
            ax.plot(ep_sub["epsilon"].astype(float), ep_sub["AP_mean"],
                    marker="o", label=f"ws={ws}")
        ax.set_title(f"{ds} — Window Size Ablation")
        ax.set_xlabel("ε")
        ax.set_ylabel("AP")
        ax.set_xscale("log")
        ax.legend(title="window_size")

    plt.tight_layout()
    _save(fig, output_dir, "ablation_window")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Train/Test split ablation
# ──────────────────────────────────────────────────────────────────────────────
def plot_split_ablation(df, output_dir):
    tbl_path = os.path.join(os.path.dirname(output_dir), "tables", "ablation_split.csv")
    if not os.path.exists(tbl_path):
        if df.empty or "train_pct" not in df.columns:
            return
        tbl = df[df["train_pct"].notna() & (df["dataset"] == "CiteSeer")]
    else:
        tbl = pd.read_csv(tbl_path)
    if tbl.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for split, grp in tbl.groupby("train_pct"):
        ep_sub = grp.sort_values("epsilon")
        ax.plot(ep_sub["epsilon"].astype(float), ep_sub["AP_mean"],
                marker="o", label=f"ρ={int(split)}%")
    ax.set_title("CiteSeer — Train/Test Split Ablation (ρ)")
    ax.set_xlabel("ε")
    ax.set_ylabel("AP")
    ax.set_xscale("log")
    ax.legend(title="train %")
    plt.tight_layout()
    _save(fig, output_dir, "ablation_split")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: DP mechanism ablation
# ──────────────────────────────────────────────────────────────────────────────
def plot_noise_ablation(df, output_dir):
    tbl_path = os.path.join(os.path.dirname(output_dir), "tables", "ablation_noise.csv")
    if not os.path.exists(tbl_path):
        if df.empty or "noise_type" not in df.columns:
            return
        tbl = df[df["noise_type"].notna() & (df["dataset"] == "Cora")]
    else:
        tbl = pd.read_csv(tbl_path)
    if tbl.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for ntype, grp in tbl.groupby("noise_type"):
        ep_sub = grp.sort_values("epsilon")
        c = COLORS.get(ntype)
        ax.plot(ep_sub["epsilon"].astype(float), ep_sub["AP_mean"],
                marker=MARKERS.get(ntype, "o"), color=c, label=ntype.capitalize())
    ax.set_title("Cora — DP Mechanism Ablation")
    ax.set_xlabel("ε")
    ax.set_ylabel("AP")
    ax.set_xscale("log")
    ax.legend(title="Mechanism")
    plt.tight_layout()
    _save(fig, output_dir, "ablation_noise")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Adaptive attacker κ × ρ heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_attacker_heatmap(df, output_dir):
    tbl_path = os.path.join(os.path.dirname(output_dir), "tables", "ablation_kappa.csv")
    if not os.path.exists(tbl_path):
        if df.empty or "kappa" not in df.columns:
            return
        tbl = df[df["kappa"].notna() & (df["dataset"] == "Cora")]
    else:
        tbl = pd.read_csv(tbl_path)
    if tbl.empty:
        return

    if "kappa" not in tbl.columns or "train_pct" not in tbl.columns:
        return

    # Average AP over epsilon values
    pivot = tbl.groupby(["kappa", "train_pct"])["AP_mean"].mean().unstack(fill_value=0.0)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"ρ={v}%" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"κ={v}" for v in pivot.index])
    ax.set_xlabel("Train split (ρ)")
    ax.set_ylabel("Estimation error (κ)")
    ax.set_title("Cora — Adaptive Attacker: AP vs κ × ρ")
    plt.colorbar(im, ax=ax, label="Mean AP")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.tight_layout()
    _save(fig, output_dir, "ablation_attacker_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: PrivX vs PrivF marginal leakage bar chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_privx_vs_privf(df, output_dir):
    if df.empty or "explainer" not in df.columns:
        return

    # For each dataset, compare PrivF (explainer='PrivF') vs each PrivX explainer
    # Use a specific epsilon (e.g., 5.0)
    target_eps = 5.0
    sub = df[abs(df["epsilon"].astype(float) - target_eps) < 0.1]
    if sub.empty:
        return

    datasets = sorted(sub["dataset"].dropna().unique())
    explainers = [e for e in ["Grad", "GradInput", "GNNExplainer", "GraphLime"]
                  if e in sub["explainer"].unique()]
    if not explainers:
        return

    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(datasets)), 4))

    # PrivF baseline
    privf = sub[sub["explainer"] == "PrivF"].groupby("dataset")["AP_mean"].mean().reindex(datasets)
    ax.bar(x, privf.fillna(0), width, label="PrivF-Diff", color=COLORS["PrivF-Diff"])

    for i, exp in enumerate(explainers):
        vals = sub[sub["explainer"] == exp].groupby("dataset")["AP_mean"].mean().reindex(datasets)
        ax.bar(x + (i + 1) * width, vals.fillna(0), width,
               label=exp, color=COLORS.get(exp))

    ax.set_xticks(x + width * (len(explainers) / 2))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("AP score")
    ax.set_title(f"PrivX vs PrivF marginal leakage (ε={target_eps})")
    ax.legend()
    plt.tight_layout()
    _save(fig, output_dir, "privx_vs_privf")


# ──────────────────────────────────────────────────────────────────────────────
# Save helper
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Plot PrivX/PrivF experiment results")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output-dir",  type=str, default="./results/figures")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir} ...")
    df = load_master(args.results_dir)

    if df.empty:
        print("No master results found. Run collect_results.py first.")
        return

    print(f"Loaded {len(df)} rows. Generating plots ...")

    df = best_scale(df)

    plot_main_comparison(df, args.output_dir)
    plot_window_ablation(df, args.output_dir)
    plot_split_ablation(df, args.output_dir)
    plot_noise_ablation(df, args.output_dir)
    plot_attacker_heatmap(df, args.output_dir)
    plot_privx_vs_privf(df, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
