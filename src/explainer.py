#!/usr/bin/env python3
"""
explainer.py — Batch wrapper to compute all missing GNN explanations

Checks which (dataset, explainer) combinations are missing or incomplete
in saved_explanations/ and submits them across available GPUs using subprocesses.

Usage:
  # Compute all missing explanations (auto-detect)
  python explainer.py --auto

  # Compute specific explainer for specific datasets
  python explainer.py --datasets Cornell Wisconsin Squirrel --explainers Grad GNNExplainer

  # Dry run to see what would be computed
  python explainer.py --auto --dry-run

IMPORTANT: Heterophilic datasets (Texas, Cornell, Wisconsin, Squirrel, Chameleon,
           Amazon-ratings) are processed with their ORIGINAL graph structure.
           The graph is never converted to a homophilic version.
"""

import os
import sys
import argparse
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
EXPLANATION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../saved_explanations")
)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
PHASE_01B = os.path.join(os.path.dirname(__file__), "phase_01b_explain.py")

# All datasets × expected node counts (approximate upper bound for completeness check)
DATASET_NODE_COUNTS = {
    "Cora":           2708,
    "CiteSeer":       3327,
    "PubMed":         19717,
    "Texas":          183,
    "Cornell":        183,
    "Wisconsin":      251,
    "Chameleon":      2277,
    "Squirrel":       5201,
    "Amazon-ratings": 24492,
    "AmazonBook":     None,   # heterogeneous — skip
    "IMDB":           11616,
    "ogbn-arxiv":     169343,
    "Reddit":         None,   # very large — skip by default
    "Bitcoinalpha":   3783,
}

HETEROPHILIC_DATASETS = [
    "Texas", "Cornell", "Wisconsin", "Squirrel", "Chameleon", "Amazon-ratings"
]

ALL_EXPLAINERS = ["Grad", "GradInput", "GNNExplainer", "GraphLime"]
BACKBONE = "GCN"

# Datasets that are skippable (heterogeneous / no node features)
SKIP_DATASETS = {"AmazonBook", "Reddit"}


# ──────────────────────────────────────────────────────────────────────────────
# Check completeness
# ──────────────────────────────────────────────────────────────────────────────
def count_existing_explanations(dataset_name, explainer, backbone=BACKBONE):
    """Count how many per-node explanation files exist."""
    exp_dir = os.path.join(EXPLANATION_DIR, explainer, backbone, dataset_name)
    if not os.path.exists(exp_dir):
        return 0
    files = glob.glob(os.path.join(exp_dir, "feature_masks_node=*.pt"))
    return len(files)


def get_missing_jobs(datasets=None, explainers=None):
    """
    Return list of (dataset, explainer) pairs that are incomplete.
    'Incomplete' = fewer files than expected node count.
    """
    if datasets is None:
        datasets = [d for d in DATASET_NODE_COUNTS if d not in SKIP_DATASETS]
    if explainers is None:
        explainers = ALL_EXPLAINERS

    missing = []
    for ds in datasets:
        if ds in SKIP_DATASETS:
            continue
        expected = DATASET_NODE_COUNTS.get(ds)
        for exp in explainers:
            existing = count_existing_explanations(ds, exp)
            if expected is None or existing < expected:
                missing.append((ds, exp, existing, expected))

    return missing


def print_status(datasets=None, explainers=None):
    """Print a table of explanation completion status."""
    if datasets is None:
        datasets = [d for d in DATASET_NODE_COUNTS if d not in SKIP_DATASETS]
    if explainers is None:
        explainers = ALL_EXPLAINERS

    print(f"\n{'Dataset':<20} " + " ".join(f"{e:<15}" for e in explainers))
    print("-" * (20 + 16 * len(explainers)))
    for ds in datasets:
        expected = DATASET_NODE_COUNTS.get(ds, "?")
        row = f"{ds:<20} "
        for exp in explainers:
            n = count_existing_explanations(ds, exp)
            if expected is None:
                status = f"{n}/?"
            else:
                pct = 100 * n / expected if expected > 0 else 0
                status = f"{n}/{expected}({pct:.0f}%)"
            row += f"{status:<15} "
        print(row)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Run a single (dataset, explainer) job
# ──────────────────────────────────────────────────────────────────────────────
def run_job(dataset, explainer, gpu_id, data_dir=DATA_DIR, explanation_dir=EXPLANATION_DIR,
            conda_env="ddpy"):
    """Launch phase_01b_explain.py for one (dataset, explainer) pair on a given GPU."""
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", PHASE_01B,
        "--dataset", dataset,
        "--explainer", explainer,
        "--data-dir", data_dir,
        "--explanation-dir", explanation_dir,
        "--device", f"cuda:{gpu_id}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True
    )
    return dataset, explainer, result.returncode, result.stdout[-2000:], result.stderr[-2000:]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Batch explanation computation for PrivX")
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-detect and compute all missing explanations.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=list(DATASET_NODE_COUNTS.keys()),
        help="Datasets to process (default: all non-skippable).",
    )
    parser.add_argument(
        "--explainers", nargs="+", default=None,
        choices=ALL_EXPLAINERS,
        help="Explainers to use (default: all four).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=4,
        help="Number of GPUs to distribute work across.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without actually running.",
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--explanation-dir", type=str, default=EXPLANATION_DIR)
    parser.add_argument("--conda-env", type=str, default="ddpy")
    parser.add_argument(
        "--status", action="store_true",
        help="Print completion status table and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.status:
        print_status(args.datasets, args.explainers)
        return

    datasets  = args.datasets
    explainers = args.explainers

    if args.auto or (datasets is None and explainers is None):
        missing = get_missing_jobs(datasets, explainers)
    else:
        if datasets is None:
            datasets = [d for d in DATASET_NODE_COUNTS if d not in SKIP_DATASETS]
        if explainers is None:
            explainers = ALL_EXPLAINERS
        missing = get_missing_jobs(datasets, explainers)

    if not missing:
        print("All explanations are complete — nothing to do.")
        print_status(datasets, explainers)
        return

    print(f"\nFound {len(missing)} incomplete (dataset, explainer) pairs:")
    for ds, exp, existing, expected in missing:
        print(f"  {ds:<20} {exp:<15} {existing}/{expected if expected else '?'}")

    if args.dry_run:
        print("\n[dry-run] Not launching any jobs.")
        return

    # Distribute jobs across GPUs
    num_gpus = args.num_gpus
    jobs = [(ds, exp) for ds, exp, _, _ in missing]

    print(f"\nLaunching {len(jobs)} jobs across {num_gpus} GPUs...")

    failed = []
    with tqdm(total=len(jobs), desc="Explanation jobs") as pbar:
        # Round-robin GPU assignment (sequential, not parallel to avoid OOM)
        for i, (ds, exp) in enumerate(jobs):
            gpu_id = i % num_gpus
            print(f"\n  [{i+1}/{len(jobs)}] {ds} / {exp} → GPU {gpu_id}")
            ds_ret, exp_ret, rc, stdout, stderr = run_job(
                ds, exp, gpu_id,
                data_dir=args.data_dir,
                explanation_dir=args.explanation_dir,
                conda_env=args.conda_env,
            )
            if rc != 0:
                print(f"  ERROR (rc={rc}):\n{stderr[-500:]}")
                failed.append((ds, exp))
            else:
                print(f"  OK: {ds}/{exp}")
            pbar.update(1)

    print("\n=== Summary ===")
    print(f"Completed: {len(jobs) - len(failed)}/{len(jobs)}")
    if failed:
        print(f"Failed: {failed}")
    print_status(args.datasets, args.explainers)


if __name__ == "__main__":
    main()
