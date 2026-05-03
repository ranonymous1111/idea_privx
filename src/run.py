#!/usr/bin/env python3
"""
Unified runner script for D3PM Graph Reconstruction Attack pipeline.

Usage:
    # Run full pipeline with defaults
    python run.py --dataset CiteSeer

    # Run only data preparation
    python run.py --dataset Cora --stage data --num-subgraphs 10000

    # Run only training
    python run.py --dataset CiteSeer --stage train --epochs 5000 --batch-size 128

    # Run only evaluation/ablation
    python run.py --dataset CiteSeer --stage eval --num-test-samples 5_000

    # Run full pipeline with custom params
    python run.py --dataset PubMed --epochs 2000 --batch-size 32 --lr 5e-4
"""

import argparse
import os
import sys

# Ensure local imports work (src folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Dataset type mapping
PLANETOID_DATASETS = ["Cora", "CiteSeer", "PubMed"]
AMAZON_DATASETS = ["Amazon-Computers", "Amazon-Photo"]
AMAZON_HETERO_DATASETS = ["AmazonBook", "AmazonProducts"]
REDDIT_DATASETS = ["Reddit"]
HETEROPHILOUS_DATASETS = ["Amazon-ratings"]  # From HeterophilousGraphDataset
OGB_DATASETS = ["ogbn-arxiv"]  # Requires ogb package
OTHER_HETERO_DATASETS = ["IMDB"]  # Other heterogeneous datasets
WEBKB_DATASETS = ["Texas", "Cornell", "Wisconsin"]  # WebKB datasets
WIKIPEDIA_DATASETS = ["Chameleon", "Squirrel"]  # Wikipedia network datasets
ALL_DATASETS = (
    PLANETOID_DATASETS
    + AMAZON_DATASETS
    + AMAZON_HETERO_DATASETS
    + REDDIT_DATASETS
    + HETEROPHILOUS_DATASETS
    + OGB_DATASETS
    + OTHER_HETERO_DATASETS
    + WEBKB_DATASETS
    + WIKIPEDIA_DATASETS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="D3PM Graph Reconstruction Attack Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "data", "train", "eval"],
        help="Which stage(s) to run: data, train, eval, or all",
    )

    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        default="CiteSeer",
        choices=ALL_DATASETS,
        help=f"Dataset to use. Options: {ALL_DATASETS}",
    )
    parser.add_argument(
        "--num-subgraphs",
        type=int,
        default=5000,
        help="Number of subgraphs to generate",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Size of subgraph windows (number of nodes)",
    )
    parser.add_argument(
        "--train-pct",
        type=int,
        default=20,
        help="Percentage of data for training (default 20%%, rest for testing)",
    )

    # Training args
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    parser.add_argument(
        "--warmup-pct",
        type=float,
        default=0.1,
        help="Percentage of training for LR warmup",
    )
    parser.add_argument(
        "--sparsity-weight",
        type=float,
        default=0.1,
        help="Weight for sparsity regularization loss",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension for model"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of GNN layers"
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=100, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gin",
        choices=["gin", "gcn", "sage"],
        help="GNN type: 'gin' (Graph Isomorphism Network), 'gcn' (Graph Convolutional Network), or 'sage' (GraphSAGE)",
    )
    parser.add_argument(
        "--train-epsilon",
        type=float,
        default=5.0,
        help="Epsilon value for DP noise during training",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplacian", "renyi"],
        help="DP noise type: 'gaussian', 'laplacian', or 'renyi'",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Delta parameter for (epsilon,delta)-DP (gaussian/renyi)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Alpha parameter for Renyi DP (order of Renyi divergence)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Epochs between logging"
    )

    # Evaluation args
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=50,
        help="Number of test samples for evaluation",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0,8.0,16.0",
        help="Comma-separated epsilon values for evaluation",
    )
    parser.add_argument(
        "--guidance-scales",
        type=str,
        default="0.0,",
        help="Comma-separated guidance scale values",
    )
    parser.add_argument(
        "--kappa", type=float, default=0.0,
        help="Adaptive attacker epsilon estimation error (0=perfect, 1=100%% error).",
    )
    parser.add_argument(
        "--rho", type=float, default=1.0,
        help="Partial observation fraction: attacker observes features for rho fraction "
             "of nodes (1.0=full observation, 0.25=only 25%% of nodes visible).",
    )

    # Output args
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory for train/test data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models/results (default: ./results/{dataset})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of dataset even if it exists",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (default: auto-generated with timestamp)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", args.dataset)

    print(f"\n{'#' * 60}")
    print("D3PM GRAPH RECONSTRUCTION ATTACK PIPELINE")
    print(f"{'#' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Stage: {args.stage}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # ==================== DATA STAGE ====================
    if args.stage in ["all", "data"]:
        print(f"\n{'=' * 60}")
        print("STAGE 1: DATA PREPARATION")
        print(f"{'=' * 60}")

        # Import here to avoid circular imports and allow lazy loading
        from phase_01_data import create_split_dataset

        create_split_dataset(
            dataset_name=args.dataset,
            num_subgraphs=args.num_subgraphs,
            window_size=args.window_size,
            data_dir=args.data_dir,
            seed=args.seed,
            force_recreate=args.force_recreate,
            train_pct=args.train_pct,
        )

    # ==================== TRAINING STAGE ====================
    if args.stage in ["all", "train"]:
        print(f"\n{'=' * 60}")
        print("STAGE 2: TRAINING")
        print(f"{'=' * 60}")

        from phase_03_train import train

        model_path, actual_run_name = train(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_pct=args.warmup_pct,
            sparsity_weight=args.sparsity_weight,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            diffusion_steps=args.diffusion_steps,
            gnn_type=args.gnn_type,
            train_epsilon=args.train_epsilon,
            noise_type=args.noise_type,
            delta=args.delta,
            alpha=args.alpha,
            log_interval=args.log_interval,
            seed=args.seed,
            device=args.device,
            run_name=args.run_name,
            window_size=args.window_size,
            train_pct=args.train_pct,
        )

    # ==================== EVALUATION STAGE ====================
    if args.stage in ["all", "eval"]:
        print(f"\n{'=' * 60}")
        print("STAGE 3: EVALUATION / ABLATION STUDY")
        print(f"{'=' * 60}")

        from phase_05_ablation import run_study

        # Parse epsilon values
        epsilons = []
        for e in args.epsilons.split(","):
            e = e.strip()
            if e:  # Skip empty strings
                epsilons.append(float("inf") if e.lower() == "inf" else float(e))

        # Parse guidance scales
        scales = [
            float(s.strip()) for s in args.guidance_scales.split(",") if s.strip()
        ]

        run_study(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_test_samples=args.num_test_samples,
            epsilons=epsilons,
            guidance_scales=scales,
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
            run_name=args.run_name,
            window_size=args.window_size,
            train_pct=args.train_pct,
            kappa=args.kappa,
            rho=args.rho,
        )

    print(f"\n{'#' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
