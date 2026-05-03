import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Local imports (when running from src folder)
try:
    from phase_04_attack import reconstruct_with_model, load_model
except ImportError:
    from .phase_04_attack import reconstruct_with_model, load_model


ALL_DATASETS = [
    "Cora", "CiteSeer", "PubMed",
    "Amazon-Computers", "Amazon-Photo",
    "AmazonBook", "AmazonProducts",
    "Reddit",
    "Amazon-ratings",
    "ogbn-arxiv",
    "IMDB",
    "Texas", "Cornell", "Wisconsin",
    "Chameleon", "Squirrel",
    "Bitcoinalpha",
]

HETEROPHILIC_DATASETS = ["Texas", "Cornell", "Wisconsin", "Squirrel", "Chameleon", "Amazon-ratings"]


def run_study(
    dataset_name="CiteSeer",
    data_dir="../data",
    output_dir="./results/CiteSeer",
    num_test_samples=20,
    epsilons=None,
    guidance_scales=None,
    hidden_dim=128,
    num_layers=4,
    diffusion_steps=100,
    gnn_type="gin",
    train_epsilon=5.0,
    noise_type="gaussian",
    delta=1e-5,
    alpha=10.0,
    seed=42,
    device="auto",
    run_name=None,
    window_size=32,
    train_pct=20,
    use_explanations=False,
    explainer="none",
    kappa=0.0,
    temperature=1.0,
    rho=1.0,
    explanation_backbone="GCN",
):
    """Run ablation study on test set."""
    print("Running Rigorous Ablation Study (Test Set Evaluation)...")

    # Default values
    if epsilons is None:
        # epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
        epsilons = [
            8.0,
        ]
    if guidance_scales is None:
        guidance_scales = [0.0, 5.0, 10.0]

    print(f"Dataset: {dataset_name}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Epsilons: {epsilons}")
    print(f"Guidance scales: {guidance_scales}")
    print(f"Test samples: {num_test_samples}")

    results = []

    # # ============ RANDOM BASELINE ============
    # print("\nEvaluating Random Baseline...")
    # baseline_aps = []
    # baseline_aucs = []
    # baseline_deg_corrs = []
    # baseline_micro_f1s = []

    # for sample_idx in range(num_test_samples):
    #     s = seed + sample_idx
    #     ap, auc, deg_corr, micro_f1 = random_baseline(
    #         seed=s,
    #         sample_idx=sample_idx,
    #         dataset_name=dataset_name,
    #         data_dir=data_dir,
    #         device=device,
    #     )
    #     baseline_aps.append(ap)
    #     baseline_aucs.append(auc)
    #     baseline_deg_corrs.append(deg_corr)
    #     baseline_micro_f1s.append(micro_f1)

    # baseline_ap_mean = np.mean(baseline_aps)
    # baseline_ap_std = np.std(baseline_aps)
    # baseline_auc_mean = np.mean(baseline_aucs)
    # baseline_deg_corr_mean = np.mean(baseline_deg_corrs)
    # baseline_micro_f1_mean = np.mean(baseline_micro_f1s)

    # print(
    #     f"  -> Random Baseline: AP {baseline_ap_mean:.4f} ± {baseline_ap_std:.4f} | AUC {baseline_auc_mean:.4f} | DegCorr {baseline_deg_corr_mean:.4f} | MicroF1 {baseline_micro_f1_mean:.4f}"
    # )

    # results.append(
    #     {
    #         "epsilon": "Random",
    #         "scale": "N/A",
    #         "AP_mean": baseline_ap_mean,
    #         "AP_std": baseline_ap_std,
    #         "AUC_mean": baseline_auc_mean,
    #         "DegCorr_mean": baseline_deg_corr_mean,
    #         "MicroF1_mean": baseline_micro_f1_mean,
    #     }
    # )

    # Auto-detect heterophilic datasets for guidance direction
    heterophilic = dataset_name in HETEROPHILIC_DATASETS

    # ============ LOAD MODEL ONCE ============
    print("\nLoading model for evaluation...")
    load_result = load_model(
        dataset_name=dataset_name,
        data_dir=data_dir,
        output_dir=output_dir,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        diffusion_steps=diffusion_steps,
        gnn_type=gnn_type,
        noise_type=noise_type,
        train_epsilon=train_epsilon,
        device=device,
        window_size=window_size,
        train_pct=train_pct,
        use_explanations=use_explanations,
        explainer=explainer,
        explanation_backbone=explanation_backbone,
    )
    model, diffusion, data, norm_stats, device, feat_key = load_result

    if model is None:
        print("Failed to load model. Exiting...")
        return None

    # ============ MODEL EVALUATION ============
    total_runs = len(epsilons) * len(guidance_scales) * num_test_samples
    start_time = time.time()
    count = 0

    for eps in epsilons:
        for scale in guidance_scales:
            batch_aps = []
            batch_aucs = []
            batch_deg_corrs = []
            batch_micro_f1s = []

            print(f"\nEvaluating Eps: {eps} | Scale: {scale} ...")

            for sample_idx in range(num_test_samples):
                s = seed + sample_idx

                ap, auc, deg_corr, micro_f1 = reconstruct_with_model(
                    model=model,
                    diffusion=diffusion,
                    data=data,
                    norm_stats=norm_stats,
                    epsilon=eps,
                    guidance_scale=scale,
                    seed=s,
                    sample_idx=sample_idx,
                    noise_type=noise_type,
                    delta=delta,
                    alpha=alpha,
                    device=device,
                    feat_key=feat_key,
                    kappa=kappa,
                    temperature=temperature,
                    heterophilic=heterophilic,
                    rho=rho,
                )

                batch_aps.append(ap)
                batch_aucs.append(auc)
                batch_deg_corrs.append(deg_corr)
                batch_micro_f1s.append(micro_f1)

                count += 1
                if count % 10 == 0:
                    print(f"  Progress: {count}/{total_runs} runs completed.")

            mean_ap = np.mean(batch_aps)
            std_ap = np.std(batch_aps)
            mean_auc = np.mean(batch_aucs)
            mean_deg_corr = np.mean(batch_deg_corrs)
            mean_micro_f1 = np.mean(batch_micro_f1s)

            print(
                f"  -> Result: AP {mean_ap:.4f} ± {std_ap:.4f} | AUC {mean_auc:.4f} | DegCorr {mean_deg_corr:.4f} | MicroF1 {mean_micro_f1:.4f}"
            )

            results.append(
                {
                    "epsilon": str(eps),
                    "scale": scale,
                    "kappa": kappa,
                    "rho": rho,
                    "AP_mean": mean_ap,
                    "AP_std": std_ap,
                    "AUC_mean": mean_auc,
                    "DegCorr_mean": mean_deg_corr,
                    "MicroF1_mean": mean_micro_f1,
                }
            )  # 4. Save and Format
    df = pd.DataFrame(results)

    # Generate unique results name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_tag = f"_{explainer}" if use_explanations and explainer != "none" else "_PrivF"
        kappa_tag = f"_kappa{kappa}" if kappa > 0 else ""
        rho_tag = f"_rho{rho}" if rho < 1.0 else ""
        run_name = (
            f"{dataset_name}_eps{train_epsilon}_model{gnn_type}_{noise_type}"
            f"_ws{window_size}_split{train_pct}_{100 - train_pct}"
            f"{exp_tag}{kappa_tag}{rho_tag}_n{num_test_samples}_{timestamp}"
        )

    results_path = os.path.join(output_dir, f"ablation_{run_name}.csv")
    df.to_csv(results_path, index=False)
    print(f"\nStudy Complete in {(time.time() - start_time) / 60:.1f} mins.")
    print(f"Results saved to '{results_path}'")

    # Print Random Baseline for reference
    # print("\n--- Random Baseline ---")
    # print(
    #     f"AP: {baseline_ap_mean:.4f} ± {baseline_ap_std:.4f} | AUC: {baseline_auc_mean:.4f} | DegCorr: {baseline_deg_corr_mean:.4f} | MicroF1: {baseline_micro_f1_mean:.4f}"
    # )

    # Create Pivot Table for easy copy-pasting into LaTeX/Papers
    model_df = df[df["epsilon"] != "Random"]
    pivot = model_df.pivot(index="epsilon", columns="scale", values="AP_mean")
    print("\n--- Mean AP Scores (Pivot) ---")
    print(pivot)

    # # Print improvement over baseline
    # print("\n--- Improvement over Random Baseline ---")
    # for _, row in model_df.iterrows():
    #     ap_improvement = row["AP_mean"] - baseline_ap_mean
    #     auc_improvement = row["AUC_mean"] - baseline_auc_mean
    #     deg_improvement = row["DegCorr_mean"] - baseline_deg_corr_mean
    #     f1_improvement = row["MicroF1_mean"] - baseline_micro_f1_mean
    #     print(
    #         f"Eps={row['epsilon']}, Scale={row['scale']}: +{ap_improvement:.4f} AP, +{auc_improvement:.4f} AUC, +{deg_improvement:.4f} DegCorr, +{f1_improvement:.4f} MicroF1"
    #     )

    # Create AUC Pivot Table
    pivot_auc = model_df.pivot(index="epsilon", columns="scale", values="AUC_mean")
    print("\n--- Mean AUC Scores (Pivot) ---")
    print(pivot_auc)

    # Create Degree Correlation Pivot Table
    pivot_deg = model_df.pivot(index="epsilon", columns="scale", values="DegCorr_mean")
    print("\n--- Mean Degree Correlation (Pivot) ---")
    print(pivot_deg)

    # Create Micro F1 Pivot Table
    pivot_f1 = model_df.pivot(index="epsilon", columns="scale", values="MicroF1_mean")
    print("\n--- Mean Micro F1 Score (Pivot) ---")
    print(pivot_f1)

    return results_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--dataset", type=str, default="CiteSeer", choices=ALL_DATASETS)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--output-dir", type=str, default="./results/CiteSeer")
    parser.add_argument("--num-test-samples", type=int, default=5_000)
    parser.add_argument("--epsilons", type=str, default="0.1,0.5,1.0,2.0,5.0,8.0,16.0")
    parser.add_argument("--guidance-scales", type=str, default="0.0,5.0,10.0")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-type", type=str, default="gaussian",
        choices=["gaussian", "laplacian", "renyi"],
    )
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--gnn-type", type=str, default="gin", choices=["gin", "gcn", "sage"],
    )
    parser.add_argument(
        "--train-epsilon", type=float, default=5.0,
        help="Epsilon value used during training",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--train-pct", type=int, default=20)
    parser.add_argument(
        "--use-explanations", action="store_true",
        help="PrivX mode: use explanation features (phi) instead of raw features.",
    )
    parser.add_argument(
        "--explainer", type=str, default="none",
        choices=["none", "Grad", "GradInput", "GNNExplainer", "GraphLime"],
    )
    parser.add_argument(
        "--explanation-backbone", type=str, default="GCN",
        choices=["GCN", "GIN", "GraphSAGE"],
        help="Backbone used when generating explanations (sets the subfolder name).",
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
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature scaling for logits before evaluation.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse epsilon values
    epsilons = []
    for e in args.epsilons.split(","):
        e = e.strip()
        epsilons.append(float("inf") if e.lower() == "inf" else float(e))

    # Parse guidance scales
    scales = [float(s.strip()) for s in args.guidance_scales.split(",")]

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
        window_size=args.window_size,
        train_pct=args.train_pct,
        use_explanations=args.use_explanations,
        explainer=args.explainer,
        kappa=args.kappa,
        temperature=args.temperature,
        rho=args.rho,
        explanation_backbone=args.explanation_backbone,
        run_name=args.run_name,
    )
