"""
Cosine Similarity Baseline for Graph Reconstruction from DP-Noised Features.

This script:
1. Loads test features
2. Adds DP noise at various epsilon levels
3. Computes cosine similarity to reconstruct adjacency
4. Compares reconstructed adjacency with original
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

# Fix for PyTorch 2.6+ weights_only issue with PyG/OGB datasets
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass  # Older versions don't need this

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc as sk_auc,
)
from scipy.stats import spearmanr


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
        # For Laplace mechanism: scale = sensitivity / epsilon
        self.scale = 0.0 if epsilon == float("inf") else sensitivity / epsilon

    def sanitize(self, x):
        if self.scale > 0:
            # Laplace noise: sample from Laplace(0, scale)
            laplace_noise = (
                torch.distributions.Laplace(0, self.scale).sample(x.shape).to(x.device)
            )
            return x + laplace_noise
        return x


class RenyiDPSanitizer:
    """
    Renyi Differential Privacy (RDP) Sanitizer.

    Renyi DP provides tighter composition bounds than standard (epsilon,delta)-DP.
    It uses alpha-Renyi divergence to measure privacy loss.

    For Gaussian mechanism under RDP:
    - A mechanism satisfies (alpha, epsilon_rdp)-RDP if sigma >= sqrt(alpha / (2 * epsilon_rdp))
    - Can be converted to (epsilon, delta)-DP via:
      epsilon = epsilon_rdp + log(1/delta) / (alpha - 1)
    """

    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5, alpha=10.0):
        """
        Args:
            epsilon: Target epsilon for (epsilon, delta)-DP
            sensitivity: L2 sensitivity of the query
            delta: Target delta for (epsilon, delta)-DP
            alpha: Order of Renyi divergence (alpha > 1)
        """
        self.alpha = alpha
        if epsilon == float("inf"):
            self.sigma = 0.0
        else:
            # Convert (epsilon, delta)-DP to RDP epsilon
            # epsilon_rdp = epsilon - log(1/delta) / (alpha - 1)
            epsilon_rdp = max(epsilon - np.log(1 / delta) / (alpha - 1), 0.01)
            # Compute sigma for RDP: sigma = sqrt(alpha * sensitivity^2 / (2 * epsilon_rdp))
            self.sigma = np.sqrt(alpha * (sensitivity**2) / (2 * epsilon_rdp))

    def sanitize(self, x):
        if self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x


def get_sanitizer(noise_type, epsilon, sensitivity=1.0, delta=1e-5, alpha=10.0):
    """Factory function to get the appropriate sanitizer."""
    if noise_type == "gaussian":
        return GaussianDPSanitizer(epsilon, sensitivity, delta)
    elif noise_type == "laplacian":
        return LaplacianDPSanitizer(epsilon, sensitivity)
    elif noise_type == "renyi":
        return RenyiDPSanitizer(epsilon, sensitivity, delta, alpha)
    else:
        raise ValueError(
            f"Unknown noise type: {noise_type}. Choose 'gaussian', 'laplacian', or 'renyi'."
        )


def cosine_similarity_reconstruction(features):
    """
    Reconstruct adjacency using cosine similarity of features.
    Higher similarity = more likely to be connected (homophily assumption).
    """
    # Normalize features for cosine similarity
    feat_norm = F.normalize(features, p=2, dim=-1)
    # Cosine similarity matrix: [N, N], range [-1, 1]
    similarity = torch.mm(feat_norm, feat_norm.T)
    # Convert to [0, 1] range for probability interpretation
    probs = (similarity + 1) / 2
    return probs


def evaluate_reconstruction(
    true_adj,
    pred_probs,
    device,
    balanced_eval=True,
    seed=42,
    sample_idx=0,
):
    """Evaluate reconstruction quality."""
    N = true_adj.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=device)

    y_true = true_adj[mask].cpu().numpy()
    y_score = pred_probs[mask].cpu().numpy()

    if balanced_eval:
        # Match run_exp/phase_04_attack balanced evaluation protocol.
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > len(pos_idx):
            rng = np.random.default_rng(seed + sample_idx)
            neg_keep = rng.choice(neg_idx, size=len(pos_idx), replace=False)
            keep_idx = np.concatenate([pos_idx, neg_keep])
            rng.shuffle(keep_idx)
            y_true = y_true[keep_idx]
            y_score = y_score[keep_idx]

    # Safety check for flat graphs
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.0, 0.0

    # Match run_exp AUROC/AP computation.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = sk_auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)

    # Use PR-optimal threshold for sparse/imbalanced graphs.
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    if len(thresholds) > 0:
        optimal_threshold = thresholds[np.argmax(f1_scores[:-1])]
    else:
        optimal_threshold = 0.5
    y_pred = (y_score >= optimal_threshold).astype(int)
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    # Degree correlation
    true_degrees = true_adj.sum(dim=1).cpu().numpy()
    pred_degrees = pred_probs.sum(dim=1).cpu().numpy()
    degree_corr, _ = spearmanr(true_degrees, pred_degrees)

    return ap, auc, degree_corr, micro_f1


def run_cosine_baseline(
    dataset_name="Cora",
    data_dir="../data",
    output_dir=None,
    num_test_samples=100,
    epsilons=None,
    noise_type="gaussian",
    seed=42,
    device="auto",
    delta=1e-5,
    alpha=10.0,
    window_size=64,
    train_pct=20,
    balanced_eval=True,
):
    """Run cosine similarity baseline for all epsilon values."""
    import pandas as pd
    from datetime import datetime

    # Device selection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # Default epsilons
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 16.0]

    # Load test data — try new naming convention first, fall back to legacy
    test_pct = 100 - train_pct
    test_path_new = os.path.join(
        data_dir, f"{dataset_name}_test_data_{window_size}_{test_pct}.pt"
    )
    test_path_old = os.path.join(data_dir, f"{dataset_name}_test_data.pt")
    if os.path.exists(test_path_new):
        test_path = test_path_new
    elif os.path.exists(test_path_old):
        test_path = test_path_old
    else:
        print(
            f"ERROR: Neither {test_path_new} nor {test_path_old} found. Run phase_01_data.py first."
        )
        return

    data = torch.load(test_path)
    print(f"Loaded {len(data)} test samples from {test_path}")

    # Limit samples
    num_samples = min(num_test_samples, len(data))
    print(f"Evaluating on {num_samples} samples")

    # Pre-compute normalization stats (same as training)
    all_feats = torch.stack([d["x"] for d in data])
    feat_mean = all_feats.mean(dim=(0, 1), keepdim=True)
    feat_std = all_feats.std(dim=(0, 1), keepdim=True) + 1e-8

    print("\n" + "=" * 70)
    print(f"COSINE SIMILARITY BASELINE RESULTS (Noise: {noise_type.upper()})")
    print("=" * 70)
    print(
        f"{'Epsilon':<12} {'AP Mean':<12} {'AP Std':<12} {'AUC Mean':<12} {'DegCorr':<12} {'MicroF1':<12}"
    )
    print("-" * 70)

    results = []

    for eps in epsilons:
        batch_aps = []
        batch_aucs = []
        batch_deg_corrs = []
        batch_micro_f1s = []

        for sample_idx in range(num_samples):
            sample = data[sample_idx]

            true_adj = sample["adj"].to(device)
            true_feat = sample["x"].to(device)

            # Normalize features
            true_feat = (true_feat - feat_mean[0, 0].to(device)) / feat_std[0, 0].to(
                device
            )

            # Apply DP noise
            torch.manual_seed(seed + sample_idx)
            sanitizer = get_sanitizer(noise_type, epsilon=eps, delta=delta, alpha=alpha)
            dp_feats = sanitizer.sanitize(true_feat)

            # # Reconstruct using cosine similarity
            pred_probs = cosine_similarity_reconstruction(dp_feats)
            # pred_probs = cosine_similarity_reconstruction(true_feat)  ## TODO: Used for testing without noise
            # Evaluate
            ap, auc, deg_corr, micro_f1 = evaluate_reconstruction(
                true_adj,
                pred_probs,
                device,
                balanced_eval=balanced_eval,
                seed=seed,
                sample_idx=sample_idx,
            )

            batch_aps.append(ap)
            batch_aucs.append(auc)
            batch_deg_corrs.append(deg_corr)
            batch_micro_f1s.append(micro_f1)

        mean_ap = np.mean(batch_aps)
        std_ap = np.std(batch_aps)
        mean_auc = np.mean(batch_aucs)
        mean_deg_corr = np.mean(batch_deg_corrs)
        mean_micro_f1 = np.mean(batch_micro_f1s)

        eps_str = "inf" if eps == float("inf") else f"{eps:.1f}"
        print(
            f"{eps_str:<12} {mean_ap:<12.4f} {std_ap:<12.4f} {mean_auc:<12.4f} {mean_deg_corr:<12.4f} {mean_micro_f1:<12.4f}"
        )

        results.append(
            {
                "epsilon": eps_str,
                "AP_mean": mean_ap,
                "AP_std": std_ap,
                "AUC_mean": mean_auc,
                "DegCorr_mean": mean_deg_corr,
                "MicroF1_mean": mean_micro_f1,
            }
        )

    print("=" * 70)

    # Summary
    print("\n--- Summary ---")
    print(
        "This baseline uses cosine similarity on DP-noised features to reconstruct edges."
    )
    print(
        "Higher epsilon = less noise = better reconstruction (upper bound: eps=inf means no noise)"
    )
    print("Compare these results with your diffusion model to see the improvement.")

    # Save results to CSV
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = (
            f"ablation_{dataset_name}_cosine_{noise_type}"
            f"_ws{window_size}_split{train_pct}_{test_pct}"
            f"_n{num_samples}_{timestamp}.csv"
        )
        csv_path = os.path.join(output_dir, csv_name)
        df = pd.DataFrame(results)
        df["dataset"] = dataset_name
        df["explainer"] = "PrivF-Cosine"
        df["noise_type"] = noise_type
        df["window_size"] = window_size
        df["train_pct"] = train_pct
        df["scale"] = 0.0
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Cosine Similarity Baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=ALL_DATASETS,
        help=f"Dataset to use. Options: {ALL_DATASETS}",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save result CSV (default: no save)",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--train-pct", type=int, default=20)
    parser.add_argument("--num-test-samples", type=int, default=100)
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0,10.0,inf",
        help="Comma-separated epsilon values",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
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
        "--no-balanced-eval",
        action="store_true",
        help="Disable balanced edge evaluation (uses all off-diagonal edges)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse epsilons
    epsilons = []
    for e in args.epsilons.split(","):
        e = e.strip()
        if e:
            epsilons.append(float("inf") if e.lower() == "inf" else float(e))

    run_cosine_baseline(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_test_samples=args.num_test_samples,
        epsilons=epsilons,
        noise_type=args.noise_type,
        seed=args.seed,
        device=args.device,
        delta=args.delta,
        alpha=args.alpha,
        window_size=args.window_size,
        train_pct=args.train_pct,
        balanced_eval=not args.no_balanced_eval,
    )
