"""
Paper Baselines: GSEF, SLaPS, GSE, GSEF-concat, GSEF-multi
Graph Reconstruction from DP-Noised Features and/or Explanations.

Methods:
  - SLaPS      : DP-noised features only  → cosine similarity
  - GSE        : DP-noised explanations only → cosine similarity
  - GSEF       : DP-noised features ⊙ explanations (element-wise) → cosine similarity
  - GSEF-concat: DP-noised [features ∥ explanations] (concatenate) → cosine similarity
  - GSEF-multi : DP-noised features ⊙ explanations (element-wise) → cosine similarity

DP noise is applied INDEPENDENTLY to features and explanations before combination,
mirroring the approach in phase_04_attack.py:
  normalize → DP noise → combine → cosine similarity → evaluate

Reference: "Private Graph Extraction via Feature Explanation" (Olatunji et al.)
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_curve, auc as sk_auc
from scipy.stats import spearmanr

# PyTorch 2.6+ / PyG serialization fix
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass


# ── Constants ─────────────────────────────────────────────────────────────────

METHODS_FEATURE_ONLY = ("SLaPS",)
METHODS_EXPL_ONLY = ("GSE",)
METHODS_BOTH = ("GSEF", "GSEF-concat", "GSEF-multi")
ALL_METHODS = METHODS_FEATURE_ONLY + METHODS_EXPL_ONLY + METHODS_BOTH

EXPLAINER_CHOICES = ["Grad", "GradInput", "GNNExplainer", "GraphLime"]

ALL_DATASETS = [
    "Cora",
    "CiteSeer",
    "PubMed",
    "Amazon-Computers",
    "Amazon-Photo",
    "AmazonBook",
    "AmazonProducts",
    "Reddit",
    "Amazon-ratings",
    "ogbn-arxiv",
    "IMDB",
    "Texas",
    "Cornell",
    "Wisconsin",
    "Chameleon",
    "Squirrel",
]


# ── DP Sanitizers (identical to cosine_baseline_subgraph.py / phase_04_attack.py) ─


class GaussianDPSanitizer:
    """Gaussian mechanism: (epsilon, delta)-DP."""

    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5):
        self.sigma = (
            0.0
            if epsilon == float("inf")
            else (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
        )

    def sanitize(self, x):
        return x + torch.randn_like(x) * self.sigma if self.sigma > 0 else x


class LaplacianDPSanitizer:
    """Laplace mechanism: pure epsilon-DP (delta=0)."""

    def __init__(self, epsilon, sensitivity=1.0):
        self.scale = 0.0 if epsilon == float("inf") else sensitivity / epsilon

    def sanitize(self, x):
        if self.scale > 0:
            return x + torch.distributions.Laplace(0, self.scale).sample(x.shape).to(
                x.device
            )
        return x


class RenyiDPSanitizer:
    """Renyi DP (alpha, epsilon_rdp): tighter composition via Gaussian noise."""

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
        raise ValueError(
            f"Unknown noise type: {noise_type}. Choose gaussian/laplacian/renyi."
        )


# ── Core helpers ──────────────────────────────────────────────────────────────


def cosine_similarity_reconstruction(features):
    """Predict edge probabilities via cosine similarity (homophily assumption)."""
    feat_norm = F.normalize(features, p=2, dim=-1)
    similarity = torch.mm(feat_norm, feat_norm.T)  # [N, N], range [-1, 1]
    return (similarity + 1) / 2  # map to [0, 1]


def evaluate_reconstruction(
    true_adj,
    pred_probs,
    device,
    balanced_eval=True,
    seed=42,
    sample_idx=0,
):
    """Return (AP, AUC, DegCorr, MicroF1) for one subgraph sample."""
    N = true_adj.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    y_true = true_adj[mask].cpu().numpy()
    y_score = pred_probs[mask].cpu().numpy()

    if balanced_eval:
        # Baseline-style balancing: keep all positives and downsample negatives.
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

    # Baseline-style AUROC/AP.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = sk_auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)

    # Keep MicroF1 as an extra metric (with PR-optimal threshold).
    from sklearn.metrics import precision_recall_curve

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


def combine_features(feat_noised, expl_noised, method):
    """Combine DP-noised features and explanations according to method."""
    if method == "SLaPS":
        return feat_noised
    elif method == "GSE":
        return expl_noised
    elif method in ("GSEF", "GSEF-multi"):
        # Element-wise product; truncate to min dim if shapes differ
        if feat_noised.shape[-1] != expl_noised.shape[-1]:
            d = min(feat_noised.shape[-1], expl_noised.shape[-1])
            return feat_noised[..., :d] * expl_noised[..., :d]
        return feat_noised * expl_noised
    elif method == "GSEF-concat":
        return torch.cat([feat_noised, expl_noised], dim=-1)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Main run function ─────────────────────────────────────────────────────────


def run_paper_baseline(
    method,
    dataset_name,
    explainer=None,
    data_dir="../data",
    explanation_dir="../data_exp",
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
    """
    Run the chosen paper baseline for all epsilon values.

    Args:
        method        : One of ALL_METHODS
        dataset_name  : Dataset name (must exist in data_dir / explanation_dir)
        explainer     : Required for methods using explanations (GSE/GSEF/variants)
        data_dir      : Directory with feature subgraph data (phase_01_data.py output)
        explanation_dir: Directory with explanation subgraph data (run_exp.py output)
        output_dir    : Where to save result CSV (None = no save)
        num_test_samples: How many test samples to evaluate
        epsilons      : List of epsilon values to sweep
        noise_type    : 'gaussian', 'laplacian', or 'renyi'
        seed          : Base random seed (features use seed+i, explanations seed+i+1M)
        delta         : DP delta for gaussian/renyi
        alpha         : Renyi order for renyi
        window_size   : Subgraph size (must match how data was created)
        train_pct     : Train split percentage (test = 100 - train_pct)
    """
    import pandas as pd

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 16.0]

    uses_features = method in METHODS_FEATURE_ONLY or method in METHODS_BOTH
    uses_explanations = method in METHODS_EXPL_ONLY or method in METHODS_BOTH

    if uses_explanations and (explainer is None or explainer.lower() == "none"):
        print(
            f"ERROR: Method '{method}' requires --explainer (e.g., Grad, GradInput, etc.)"
        )
        return

    test_pct = 100 - train_pct

    # ── Load feature subgraph data ────────────────────────────────────────────
    feat_data = feat_mean = feat_std = None
    if uses_features:
        feat_path = os.path.join(
            data_dir, f"{dataset_name}_test_data_{window_size}_{test_pct}.pt"
        )
        if not os.path.exists(feat_path):
            print(f"ERROR: Feature data not found: {feat_path}")
            print(f"  Run: python phase_01_data.py --dataset {dataset_name}")
            return
        feat_data = torch.load(feat_path, weights_only=False)
        all_feats = torch.stack([d["x"] for d in feat_data])
        feat_mean = all_feats.mean(dim=(0, 1), keepdim=True)
        feat_std = all_feats.std(dim=(0, 1), keepdim=True) + 1e-8
        print(f"Loaded {len(feat_data)} feature samples  ({feat_path})")

    # ── Load explanation subgraph data ────────────────────────────────────────
    expl_data = expl_mean = expl_std = None
    if uses_explanations:
        expl_path = os.path.join(
            explanation_dir,
            f"{dataset_name}_{explainer}_test_data_{window_size}_{test_pct}.pt",
        )
        if not os.path.exists(expl_path):
            print(f"ERROR: Explanation data not found: {expl_path}")
            print(
                f"  Run: python run_exp.py --dataset {dataset_name} "
                f"--explainer {explainer} --stage data"
            )
            return
        expl_data = torch.load(expl_path, weights_only=False)
        all_expls = torch.stack([d["x"] for d in expl_data])
        expl_mean = all_expls.mean(dim=(0, 1), keepdim=True)
        expl_std = all_expls.std(dim=(0, 1), keepdim=True) + 1e-8
        print(f"Loaded {len(expl_data)} explanation samples ({expl_path})")

    # Number of samples to evaluate
    counts = [len(feat_data)] if feat_data else []
    counts += [len(expl_data)] if expl_data else []
    num_samples = min(num_test_samples, *counts)
    print(f"Evaluating on {num_samples} samples")

    # ── Header ─────────────────────────────────────────────────────────────────
    expl_tag = f"-{explainer}" if uses_explanations else ""
    print("\n" + "=" * 70)
    print(
        f"{method}{expl_tag} BASELINE  |  {dataset_name}  |  Noise: {noise_type.upper()}"
    )
    print("=" * 70)
    print(
        f"{'Epsilon':<12} {'AP Mean':<12} {'AP Std':<12} "
        f"{'AUC Mean':<12} {'DegCorr':<12} {'MicroF1':<12}"
    )
    print("-" * 70)

    results = []

    for eps in epsilons:
        batch_aps, batch_aucs, batch_dcorrs, batch_f1s = [], [], [], []

        for sample_idx in range(num_samples):
            # Reference adjacency (features and explanations share the same adj)
            ref_data = feat_data if feat_data is not None else expl_data
            true_adj = ref_data[sample_idx]["adj"].to(device)

            # ── Apply DP noise to features ────────────────────────────────────
            feat_noised = None
            if feat_data is not None:
                torch.manual_seed(seed + sample_idx)
                sanitizer_f = get_sanitizer(
                    noise_type, epsilon=eps, delta=delta, alpha=alpha
                )
                feat = feat_data[sample_idx]["x"].to(device)
                # Normalize (same as phase_04_attack.py)
                feat = (feat - feat_mean[0, 0].to(device)) / feat_std[0, 0].to(device)
                feat_noised = sanitizer_f.sanitize(feat)

            # ── Apply DP noise to explanations (independent seed) ─────────────
            expl_noised = None
            if expl_data is not None:
                torch.manual_seed(seed + sample_idx + 1_000_000)  # independent seed
                sanitizer_e = get_sanitizer(
                    noise_type, epsilon=eps, delta=delta, alpha=alpha
                )
                expl = expl_data[sample_idx]["x"].to(device)
                # Normalize
                expl = (expl - expl_mean[0, 0].to(device)) / expl_std[0, 0].to(device)
                expl_noised = sanitizer_e.sanitize(expl)

            # ── Combine and reconstruct ───────────────────────────────────────
            combined = combine_features(feat_noised, expl_noised, method)
            pred_probs = cosine_similarity_reconstruction(combined)

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
            f"{eps_str:<12} {mean_ap:<12.4f} {std_ap:<12.4f} "
            f"{mean_auc:<12.4f} {mean_dc:<12.4f} {mean_f1:<12.4f}"
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

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        expl_suffix = f"_{explainer}" if uses_explanations and explainer else ""
        csv_name = (
            f"ablation_{dataset_name}_{method}{expl_suffix}_{noise_type}"
            f"_ws{window_size}_split{train_pct}_{test_pct}"
            f"_n{num_samples}_{timestamp}.csv"
        )
        csv_path = os.path.join(output_dir, csv_name)
        import pandas as pd

        df = pd.DataFrame(results)
        df["dataset"] = dataset_name
        df["explainer"] = f"{method}{expl_tag}"
        df["noise_type"] = noise_type
        df["window_size"] = window_size
        df["train_pct"] = train_pct
        df["scale"] = 0.0
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paper Baselines: GSEF / SLaPS / GSE / GSEF-concat / GSEF-multi\n"
            "DP noise applied independently to features and explanations."
        )
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(ALL_METHODS),
        help="Attack method",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=ALL_DATASETS)
    parser.add_argument(
        "--explainer",
        type=str,
        default=None,
        choices=EXPLAINER_CHOICES,
        help="Required for GSE/GSEF/GSEF-concat/GSEF-multi",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Feature subgraph data dir (phase_01_data.py output)",
    )
    parser.add_argument(
        "--explanation-dir",
        type=str,
        default="../data_exp",
        help="Explanation subgraph data dir (run_exp.py --stage data output)",
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

    run_paper_baseline(
        method=args.method,
        dataset_name=args.dataset,
        explainer=args.explainer,
        data_dir=args.data_dir,
        explanation_dir=args.explanation_dir,
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
