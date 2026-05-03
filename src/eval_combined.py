"""
Evaluate a combined model on individual datasets.
The combined model expects features padded to max_feat_dim (e.g., 3703 from CiteSeer).
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from datetime import datetime
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
from phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN


class GaussianDPSanitizer:
    """Gaussian mechanism for differential privacy."""

    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5):
        self.sigma = (
            0.0
            if epsilon == float("inf")
            else (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
        )

    def sanitize(self, x):
        return x + torch.randn_like(x) * self.sigma if self.sigma > 0 else x


class LaplacianDPSanitizer:
    """Laplacian mechanism for differential privacy."""

    def __init__(self, epsilon, sensitivity=1.0):
        self.scale = 0.0 if epsilon == float("inf") else sensitivity / epsilon

    def sanitize(self, x):
        if self.scale > 0:
            laplace_noise = (
                torch.distributions.Laplace(0, self.scale).sample(x.shape).to(x.device)
            )
            return x + laplace_noise
        return x


def get_sanitizer(noise_type, epsilon, sensitivity=1.0):
    if noise_type == "gaussian":
        return GaussianDPSanitizer(epsilon, sensitivity)
    elif noise_type == "laplacian":
        return LaplacianDPSanitizer(epsilon, sensitivity)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def load_combined_model(
    model_dir="./results/combined",
    gnn_type="gcn",
    train_epsilon=5.0,
    noise_type="gaussian",
    window_size=32,
    device="cuda",
):
    """Load the combined model and its metadata."""
    model_path = os.path.join(
        model_dir,
        f"model_combined_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}.pth",
    )
    metadata_path = os.path.join(
        model_dir,
        f"model_combined_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}_metadata.pt",
    )

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None, None

    # Load metadata
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
        print(f"Loaded metadata: {metadata}")
    else:
        # Default metadata if not found
        metadata = {
            "feat_dim": 3703,  # Default to CiteSeer (largest)
            "window_size": 32,
            "hidden_dim": 128,
            "num_layers": 4,
            "diffusion_steps": 100,
        }
        print(f"Metadata not found, using defaults: {metadata}")

    # Create model
    model = ConditionalDenseGNN(
        num_nodes=metadata["window_size"],
        feature_dim=metadata["feat_dim"],
        hidden_dim=metadata["hidden_dim"],
        num_layers=metadata["num_layers"],
        gnn_type=gnn_type,
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    diffusion = DiscreteDiffusionBase(
        num_steps=metadata["diffusion_steps"], device=device
    )

    return model, diffusion, metadata


def evaluate_on_dataset(
    model,
    diffusion,
    metadata,
    dataset_name="Cora",
    data_dir="../data",
    num_test_samples=500,
    epsilons=None,
    noise_type="gaussian",
    seed=42,
    device="cuda",
):
    """Evaluate combined model on a specific dataset."""
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 16.0]

    # Load test data
    test_path = os.path.join(data_dir, f"{dataset_name}_test_data.pt")
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found")
        return None

    data = torch.load(test_path)
    print(f"\nLoaded {dataset_name}: {len(data)} test samples")

    # Get feature dimension of this dataset
    dataset_feat_dim = data[0]["x"].shape[1]
    model_feat_dim = metadata["feat_dim"]
    window_size = metadata["window_size"]

    print(f"Dataset feature dim: {dataset_feat_dim}, Model expects: {model_feat_dim}")

    # Pre-compute normalization stats
    all_feats = torch.stack([d["x"] for d in data])
    feat_mean = all_feats.mean(dim=(0, 1), keepdim=True)
    feat_std = all_feats.std(dim=(0, 1), keepdim=True) + 1e-8

    num_samples = min(num_test_samples, len(data))

    results = []

    # Random baseline
    print(f"\nEvaluating Random Baseline on {dataset_name}...")
    baseline_aps, baseline_aucs = [], []
    for sample_idx in range(num_samples):
        sample = data[sample_idx]
        true_adj = sample["adj"].to(device)
        N = true_adj.shape[0]

        torch.manual_seed(seed + sample_idx)
        random_probs = torch.rand(N, N, device=device)
        random_probs = (random_probs + random_probs.T) / 2

        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        y_true = true_adj[mask].cpu().numpy()
        y_score = random_probs[mask].cpu().numpy()

        if len(np.unique(y_true)) >= 2:
            baseline_aps.append(average_precision_score(y_true, y_score))
            baseline_aucs.append(roc_auc_score(y_true, y_score))

    baseline_ap = np.mean(baseline_aps)
    baseline_auc = np.mean(baseline_aucs)
    print(f"  Random Baseline: AP={baseline_ap:.4f}, AUC={baseline_auc:.4f}")

    results.append(
        {
            "dataset": dataset_name,
            "epsilon": "Random",
            "AP_mean": baseline_ap,
            "AUC_mean": baseline_auc,
        }
    )

    # Model evaluation
    for eps in epsilons:
        print(f"\nEvaluating {dataset_name} @ eps={eps}...")
        batch_aps, batch_aucs = [], []

        sanitizer = get_sanitizer(noise_type, epsilon=eps)

        for sample_idx in range(num_samples):
            sample = data[sample_idx]

            true_adj = sample["adj"].to(device)
            true_feat = sample["x"].to(device)

            # Normalize features
            true_feat = (true_feat - feat_mean[0, 0].to(device)) / feat_std[0, 0].to(
                device
            )

            N = true_adj.shape[0]

            # Pad features to match model's expected dimension
            if dataset_feat_dim < model_feat_dim:
                padding = torch.zeros(
                    N, model_feat_dim - dataset_feat_dim, device=device
                )
                true_feat = torch.cat([true_feat, padding], dim=1)

            # Apply DP noise
            torch.manual_seed(seed + sample_idx)
            dp_feats = sanitizer.sanitize(true_feat).unsqueeze(0)  # [1, N, F]

            # Sampling
            x_t = torch.randint(0, 2, (1, N, N), device=device)

            with torch.no_grad():
                for t_idx in reversed(range(diffusion.num_steps)):
                    t = torch.full((1,), t_idx, device=device, dtype=torch.long)
                    logits = model(x_t, dp_feats, t)
                    log_post = diffusion.compute_posterior_logits(x_t, logits, t)
                    x_t = torch.distributions.Categorical(torch.exp(log_post)).sample()

            # Evaluate
            probs = F.softmax(logits, dim=-1)[0, :, :, 1]
            mask = ~torch.eye(N, dtype=torch.bool, device=device)

            y_true = true_adj[mask].cpu().numpy()
            y_score = probs[mask].cpu().numpy()

            if len(np.unique(y_true)) >= 2:
                batch_aps.append(average_precision_score(y_true, y_score))
                batch_aucs.append(roc_auc_score(y_true, y_score))

        mean_ap = np.mean(batch_aps)
        mean_auc = np.mean(batch_aucs)
        print(f"  -> AP={mean_ap:.4f}, AUC={mean_auc:.4f}")

        results.append(
            {
                "dataset": dataset_name,
                "epsilon": eps,
                "AP_mean": mean_ap,
                "AUC_mean": mean_auc,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate combined model on individual datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=["Cora", "CiteSeer", "PubMed", "all"],
    )
    parser.add_argument("--model-dir", type=str, default="./results/combined")
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument(
        "--gnn-type", type=str, default="gcn", choices=["gin", "gcn", "sage"]
    )
    parser.add_argument("--train-epsilon", type=float, default=5.0)
    parser.add_argument(
        "--noise-type", type=str, default="gaussian", choices=["gaussian", "laplacian"]
    )
    parser.add_argument("--num-test-samples", type=int, default=500)
    parser.add_argument("--epsilons", type=str, default="0.1,0.5,1.0,2.0,5.0,8.0,16.0")
    parser.add_argument(
        "--window-size", type=int, default=32, help="Window size for model"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Parse epsilons
    epsilons = [
        float("inf") if e.strip().lower() == "inf" else float(e.strip())
        for e in args.epsilons.split(",")
        if e.strip()
    ]

    # Load combined model
    model, diffusion, metadata = load_combined_model(
        model_dir=args.model_dir,
        gnn_type=args.gnn_type,
        train_epsilon=args.train_epsilon,
        noise_type=args.noise_type,
        window_size=args.window_size,
        device=device,
    )

    if model is None:
        return

    # Datasets to evaluate
    if args.dataset == "all":
        datasets = ["Cora", "CiteSeer", "PubMed"]
    else:
        datasets = [args.dataset]

    all_results = []

    for dataset_name in datasets:
        results = evaluate_on_dataset(
            model=model,
            diffusion=diffusion,
            metadata=metadata,
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            num_test_samples=args.num_test_samples,
            epsilons=epsilons,
            noise_type=args.noise_type,
            seed=args.seed,
            device=device,
        )
        if results:
            all_results.extend(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Combined Model Evaluation")
    print("=" * 70)

    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        args.model_dir,
        f"eval_combined_{args.gnn_type}_{args.train_epsilon:.1f}_ws{args.window_size}_{timestamp}.csv",
    )
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
