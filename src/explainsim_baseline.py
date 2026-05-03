"""
ExplainSim Baseline for Graph Reconstruction from DP-Noised Explanation Vectors.

Mirrors cosine_baseline_subgraph.py but loads explanation-based subgraph data
from data_exp/ (created by run_exp.py --stage data) instead of raw features.

Pipeline:
  1. Load explanation subgraph test data  ({dataset}_{explainer}_test_data_{ws}_{pct}.pt)
  2. Apply DP noise to explanation vectors φ̃ = φ + noise(ε)
  3. Compute cosine similarity between noised explanations
  4. Evaluate edge reconstruction vs ground-truth adjacency

Reference: "Private Graph Extraction via Feature Explanation" (Olatunji et al.)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc as sk_auc,
)
from scipy.stats import spearmanr

PLANETOID_DATASETS = ["Cora", "CiteSeer", "PubMed"]
AMAZON_DATASETS = ["Amazon-Computers", "Amazon-Photo"]
AMAZON_HETERO_DATASETS = ["AmazonBook", "AmazonProducts"]
REDDIT_DATASETS = ["Reddit"]
HETEROPHILOUS_DATASETS = ["Amazon-ratings"]
OGB_DATASETS = ["ogbn-arxiv"]
OTHER_HETERO_DATASETS = ["IMDB"]
WEBKB_DATASETS = ["Texas", "Cornell", "Wisconsin"]
WIKIPEDIA_DATASETS = ["Chameleon", "Squirrel"]
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
EXPLAINER_CHOICES = ["Grad", "GradInput", "GNNExplainer", "GraphLime"]


# ── DP sanitizers (identical to cosine_baseline_subgraph.py) ─────────────────


class GaussianDPSanitizer:
    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5):
        self.sigma = (
            0.0
            if epsilon == float("inf")
            else (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
        )

    def sanitize(self, x):
        return x + torch.randn_like(x) * self.sigma if self.sigma > 0 else x


class LaplacianDPSanitizer:
    def __init__(self, epsilon, sensitivity=1.0):
        self.scale = 0.0 if epsilon == float("inf") else sensitivity / epsilon

    def sanitize(self, x):
        if self.scale > 0:
            return x + torch.distributions.Laplace(0, self.scale).sample(x.shape).to(
                x.device
            )
        return x


class RenyiDPSanitizer:
    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5, alpha=10.0):
        self.alpha = alpha
        if epsilon == float("inf"):
            self.sigma = 0.0
        else:
            epsilon_rdp = max(epsilon - np.log(1 / delta) / (alpha - 1), 0.01)
            self.sigma = np.sqrt(alpha * (sensitivity**2) / (2 * epsilon_rdp))

    def sanitize(self, x):
        return x + torch.randn_like(x) * self.sigma if self.sigma > 0 else x


def get_sanitizer(noise_type, epsilon, sensitivity=1.0, delta=1e-5, alpha=10.0):
    if noise_type == "gaussian":
        return GaussianDPSanitizer(epsilon, sensitivity, delta)
    elif noise_type == "laplacian":
        return LaplacianDPSanitizer(epsilon, sensitivity)
    elif noise_type == "renyi":
        return RenyiDPSanitizer(epsilon, sensitivity, delta, alpha)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def cosine_similarity_reconstruction(features):
    feat_norm = F.normalize(features, p=2, dim=-1)
    similarity = torch.mm(feat_norm, feat_norm.T)
    return (similarity + 1) / 2


def evaluate_reconstruction(
    true_adj,
    pred_probs,
    device,
    balanced_eval=True,
    seed=42,
    sample_idx=0,
):
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
    true_deg = true_adj.sum(dim=1).cpu().numpy()
    pred_deg = pred_probs.sum(dim=1).cpu().numpy()
    deg_corr, _ = spearmanr(true_deg, pred_deg)
    return ap, auc, deg_corr, micro_f1


# ── Main run function ─────────────────────────────────────────────────────────


def run_explainsim_baseline(
    dataset_name="Cora",
    explainer="Grad",
    exp_data_dir="../data_exp",
    output_dir=None,
    num_test_samples=500,
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
    import pandas as pd
    from datetime import datetime

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 16.0]

    # Load explanation-based subgraph test data
    test_pct = 100 - train_pct
    test_path = os.path.join(
        exp_data_dir,
        f"{dataset_name}_{explainer}_test_data_{window_size}_{test_pct}.pt",
    )
    if not os.path.exists(test_path):
        print(f"ERROR: Test data not found: {test_path}")
        print(
            f"  Run first: python run_exp.py --dataset {dataset_name} "
            f"--explainer {explainer} --stage data"
        )
        return

    data = torch.load(test_path, weights_only=False)
    print(f"Loaded {len(data)} test samples from {test_path}")

    num_samples = min(num_test_samples, len(data))
    print(f"Evaluating on {num_samples} samples")

    # Normalise explanation vectors (same as training)
    all_feats = torch.stack([d["x"] for d in data])
    feat_mean = all_feats.mean(dim=(0, 1), keepdim=True)
    feat_std = all_feats.std(dim=(0, 1), keepdim=True) + 1e-8

    print("\n" + "=" * 70)
    print(
        f"ExplainSim BASELINE  |  Explainer: {explainer}  |  Noise: {noise_type.upper()}"
    )
    print("=" * 70)
    print(
        f"{'Epsilon':<12} {'AP Mean':<12} {'AP Std':<12} {'AUC Mean':<12} "
        f"{'DegCorr':<12} {'MicroF1':<12}"
    )
    print("-" * 70)

    results = []
    for eps in epsilons:
        batch_aps, batch_aucs, batch_dcorrs, batch_f1s = [], [], [], []

        for sample_idx in range(num_samples):
            sample = data[sample_idx]
            true_adj = sample["adj"].to(device)
            expl_vec = sample["x"].to(device)

            # Normalise
            expl_vec = (expl_vec - feat_mean[0, 0].to(device)) / feat_std[0, 0].to(
                device
            )

            # Apply DP noise to explanation vectors
            torch.manual_seed(seed + sample_idx)
            sanitizer = get_sanitizer(noise_type, epsilon=eps, delta=delta, alpha=alpha)
            noisy_expl = sanitizer.sanitize(expl_vec)

            # Reconstruct via cosine similarity on noised explanations
            pred_probs = cosine_similarity_reconstruction(noisy_expl)

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
            batch_dcorrs.append(deg_corr)
            batch_f1s.append(micro_f1)

        mean_ap = np.mean(batch_aps)
        std_ap = np.std(batch_aps)
        mean_auc = np.mean(batch_aucs)
        mean_dc = np.mean(batch_dcorrs)
        mean_f1 = np.mean(batch_f1s)

        eps_str = "inf" if eps == float("inf") else f"{eps:.1f}"
        print(
            f"{eps_str:<12} {mean_ap:<12.4f} {std_ap:<12.4f} {mean_auc:<12.4f} "
            f"{mean_dc:<12.4f} {mean_f1:<12.4f}"
        )

        results.append(
            {
                "epsilon": eps_str,
                "AP_mean": mean_ap,
                "AP_std": std_ap,
                "AUC_mean": mean_auc,
                "DegCorr_mean": mean_dc,
                "MicroF1_mean": mean_f1,
            }
        )

    print("=" * 70)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = (
            f"ablation_{dataset_name}_ExplainSim_{explainer}_{noise_type}"
            f"_ws{window_size}_split{train_pct}_{test_pct}"
            f"_n{num_samples}_{timestamp}.csv"
        )
        csv_path = os.path.join(output_dir, csv_name)
        df = pd.DataFrame(results)
        df["dataset"] = dataset_name
        df["explainer"] = f"ExplainSim-{explainer}"
        df["noise_type"] = noise_type
        df["window_size"] = window_size
        df["train_pct"] = train_pct
        df["scale"] = 0.0
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="ExplainSim Baseline")
    parser.add_argument("--dataset", type=str, required=True, choices=ALL_DATASETS)
    parser.add_argument(
        "--explainer", type=str, required=True, choices=EXPLAINER_CHOICES
    )
    parser.add_argument(
        "--exp-data-dir",
        type=str,
        default="../data_exp",
        help="Directory with explanation subgraph data from run_exp.py --stage data",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--train-pct", type=int, default=20)
    parser.add_argument("--num-test-samples", type=int, default=500)
    parser.add_argument("--epsilons", type=str, default="0.1,0.5,1.0,2.0,5.0,8.0,16.0")
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplacian", "renyi"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--no-balanced-eval",
        action="store_true",
        help="Disable balanced edge evaluation (uses all off-diagonal edges)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    epsilons = []
    for e in args.epsilons.split(","):
        e = e.strip()
        if e:
            epsilons.append(float("inf") if e.lower() == "inf" else float(e))

    run_explainsim_baseline(
        dataset_name=args.dataset,
        explainer=args.explainer,
        exp_data_dir=args.exp_data_dir,
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
