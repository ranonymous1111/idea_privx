#!/usr/bin/env python3
"""
Table 3: Adaptive Attacker Ablation — κ × ρ sweep.

Sweeps kappa (epsilon estimation error) and rho (partial observation fraction)
on a single pre-trained model to produce the AP pivot table for the paper.

Paper target:
    Dataset:  CiteSeer
    Explainer: GraphLime
    ε = 1.0, Gaussian DP
    κ ∈ {0.0, 0.1, 0.3, 1.0}
    ρ ∈ {0.25, 0.50, 0.75, 1.00}

Usage:
    cd src
    python run_ablation_table3.py \\
        --dataset CiteSeer --explainer GraphLime \\
        --gnn-type sage --noise-type gaussian \\
        --train-epsilon 1.0 --window-size 128 --train-pct 20 \\
        --num-test-samples 50 \\
        --output-dir ../result_exp/CiteSeer/GraphLime/GCN/ \\
        --device cuda:0

    # Custom kappa/rho grids:
    python run_ablation_table3.py --kappas 0.0,0.1,0.3,1.0 --rhos 0.25,0.5,0.75,1.0
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALL_DATASETS = [
    "Cora", "CiteSeer", "PubMed",
    "Amazon-Computers", "Amazon-Photo",
    "AmazonBook", "AmazonProducts",
    "Reddit", "Amazon-ratings", "ogbn-arxiv", "IMDB",
    "Texas", "Cornell", "Wisconsin", "Chameleon", "Squirrel",
]

EXPLAINER_CHOICES = ["Grad", "GradInput", "GNNExplainer", "GraphLime"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Table 3: Adaptive Attacker Ablation (kappa x rho sweep)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="CiteSeer", choices=ALL_DATASETS)
    parser.add_argument(
        "--explainer", type=str, default="GraphLime", choices=EXPLAINER_CHOICES,
        help="Explainer used for PrivX. Set to 'none' for PrivF mode.",
    )
    parser.add_argument(
        "--use-explanations", action="store_true", default=True,
        help="Use PrivX (explanation features). Disable for PrivF.",
    )
    parser.add_argument("--gnn-type", type=str, default="sage", choices=["gin", "gcn", "sage"])
    parser.add_argument(
        "--noise-type", type=str, default="gaussian",
        choices=["gaussian", "laplacian", "renyi"],
    )
    parser.add_argument(
        "--train-epsilon", type=float, default=1.0,
        help="Epsilon used during training (model to load).",
    )
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--train-pct", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-test-samples", type=int, default=50)
    parser.add_argument(
        "--kappas", type=str, default="0.0,0.1,0.3,1.0",
        help="Comma-separated kappa values to sweep.",
    )
    parser.add_argument(
        "--rhos", type=str, default="0.25,0.5,0.75,1.0",
        help="Comma-separated rho values to sweep.",
    )
    parser.add_argument(
        "--eval-epsilon", type=float, default=1.0,
        help="Epsilon used at evaluation time (fixed for Table 3).",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=0.0,
        help="Guidance scale for reconstruction.",
    )
    parser.add_argument(
        "--explanation-backbone", type=str, default="GCN",
        choices=["GCN", "GIN", "GraphSAGE"],
        help="Backbone used when generating explanations (sets data subfolder).",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data_exp",
        help="Directory for explanation-based train/test data.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results. Defaults to ../result_exp/{dataset}/{explainer}/{backbone}/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    kappas = [float(k.strip()) for k in args.kappas.split(",")]
    rhos = [float(r.strip()) for r in args.rhos.split(",")]

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "../result_exp", args.dataset, args.explainer, args.explanation_backbone
        )
    os.makedirs(args.output_dir, exist_ok=True)

    from phase_05_ablation import run_study

    print(f"\n{'#' * 60}")
    print("TABLE 3: ADAPTIVE ATTACKER ABLATION")
    print(f"{'#' * 60}")
    print(f"Dataset:      {args.dataset}")
    print(f"Explainer:    {args.explainer}")
    print(f"Train ε:      {args.train_epsilon}  |  Eval ε: {args.eval_epsilon}")
    print(f"Noise type:   {args.noise_type}")
    print(f"kappa values: {kappas}")
    print(f"rho values:   {rhos}")
    print(f"Test samples: {args.num_test_samples}")
    print(f"Output dir:   {args.output_dir}")
    print()

    all_rows = []

    for rho in rhos:
        for kappa in kappas:
            print(f"\n--- rho={rho}, kappa={kappa} ---")
            results_path = run_study(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                num_test_samples=args.num_test_samples,
                epsilons=[args.eval_epsilon],
                guidance_scales=[args.guidance_scale],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                diffusion_steps=args.diffusion_steps,
                gnn_type=args.gnn_type,
                train_epsilon=args.train_epsilon,
                noise_type=args.noise_type,
                delta=args.delta,
                alpha=args.alpha,
                seed=args.seed,
                device=args.device,
                window_size=args.window_size,
                train_pct=args.train_pct,
                use_explanations=args.use_explanations,
                explainer=args.explainer,
                kappa=kappa,
                rho=rho,
                explanation_backbone=args.explanation_backbone,
            )
            if results_path is not None:
                df = pd.read_csv(results_path)
                # Take the first row (single eval-epsilon run)
                row = df.iloc[0].to_dict()
                row["rho"] = rho
                row["kappa"] = kappa
                all_rows.append(row)

    if not all_rows:
        print("No results collected — check that models are trained.")
        return

    combined = pd.DataFrame(all_rows)

    # --- AP Pivot: rows=rho, cols=kappa ---
    pivot_ap = combined.pivot(index="rho", columns="kappa", values="AP_mean")
    pivot_ap.index.name = "rho \\ kappa"
    print("\n" + "=" * 60)
    print("TABLE 3 — AP (mean) pivot  [rows=ρ, cols=κ]")
    print("=" * 60)
    print(pivot_ap.to_string())

    # Save combined CSV and pivot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(
        args.output_dir,
        f"table3_adaptive_{args.dataset}_{args.explainer}_{args.noise_type}"
        f"_eps{args.eval_epsilon}_{timestamp}.csv",
    )
    pivot_path = combined_path.replace(".csv", "_pivot.csv")

    combined.to_csv(combined_path, index=False)
    pivot_ap.to_csv(pivot_path)

    print(f"\nFull results saved to: {combined_path}")
    print(f"AP pivot saved to:     {pivot_path}")


if __name__ == "__main__":
    main()
