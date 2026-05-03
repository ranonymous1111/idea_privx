"""
Cosine Similarity Baseline for Graph Reconstruction from DP-Noised Features.

This script:
1. Loads the FULL raw graph (Cora, CiteSeer, PubMed, Amazon, Reddit, etc.)
2. Adds DP noise to the entire feature matrix at various epsilon levels
3. Computes cosine similarity to reconstruct the full adjacency matrix
4. Compares reconstructed adjacency with original using reconstruction metrics
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
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Reddit,
    AmazonBook,
    AmazonProducts,
    HeterophilousGraphDataset,
    IMDB,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.utils import degree
import scipy.sparse as sp


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

# Datasets that are heterogeneous or don't have features
HETERO_DATASETS = ["AmazonBook", "IMDB"]
NO_FEATURE_DATASETS = [
    "AmazonBook"
]  # Datasets without node features (IMDB has features)


def generate_positional_features(num_nodes, edge_index, feature_dim=128):
    """
    Generate synthetic features for graphs without node features.
    Uses degree-based features + random positional encoding.
    """
    # Degree features
    deg = degree(edge_index[0], num_nodes=num_nodes).float()
    deg_normalized = deg / (deg.max() + 1e-8)

    # Random positional encoding (fixed by seed for reproducibility)
    pos_enc = torch.randn(num_nodes, feature_dim - 1)

    # Combine: [degree, positional_encoding]
    features = torch.cat([deg_normalized.unsqueeze(1), pos_enc], dim=1)
    return features


def load_dataset(dataset_name, data_dir, feature_dim=128):
    """
    Load dataset based on name. Handles Planetoid, Amazon, Reddit, and heterogeneous datasets.
    For datasets without features, generates synthetic features.
    """
    data = None
    is_hetero = dataset_name in HETERO_DATASETS

    if dataset_name in PLANETOID_DATASETS:
        dataset = Planetoid(
            root=os.path.join(data_dir, dataset_name), name=dataset_name
        )
        data = dataset[0]
    elif dataset_name == "Amazon-Computers":
        dataset = Amazon(root=os.path.join(data_dir, "Amazon"), name="Computers")
        data = dataset[0]
    elif dataset_name == "Amazon-Photo":
        dataset = Amazon(root=os.path.join(data_dir, "Amazon"), name="Photo")
        data = dataset[0]
    elif dataset_name == "AmazonBook":
        dataset = AmazonBook(root=os.path.join(data_dir, "AmazonBook"))
        hetero_data = dataset[0]
        # Convert heterogeneous to homogeneous: use user-book bipartite edges
        edge_index = hetero_data["user", "rates", "book"].edge_index
        num_users = hetero_data["user"].num_nodes
        num_books = hetero_data["book"].num_nodes
        num_nodes = num_users + num_books

        # Shift book indices to create unified node space
        edge_index_shifted = edge_index.clone()
        edge_index_shifted[1] = edge_index[1] + num_users

        # Make undirected
        edge_index_full = torch.cat(
            [edge_index_shifted, edge_index_shifted.flip(0)], dim=1
        )

        # Generate synthetic features
        features = generate_positional_features(num_nodes, edge_index_full, feature_dim)

        # Create a simple Data object
        from torch_geometric.data import Data

        data = Data(x=features, edge_index=edge_index_full, num_nodes=num_nodes)

    elif dataset_name == "AmazonProducts":
        dataset = AmazonProducts(root=os.path.join(data_dir, "AmazonProducts"))
        data = dataset[0]
    elif dataset_name == "Reddit":
        dataset = Reddit(root=os.path.join(data_dir, "Reddit"))
        data = dataset[0]
    elif dataset_name == "Amazon-ratings":
        # Heterophilous dataset from "A Critical Look at GNN Evaluation under Heterophily"
        dataset = HeterophilousGraphDataset(
            root=os.path.join(data_dir, "HeterophilousGraph"), name="Amazon-ratings"
        )
        data = dataset[0]
    elif dataset_name == "ogbn-arxiv":
        # OGB arxiv citation network - requires ogb package
        try:
            from ogb.nodeproppred import PygNodePropPredDataset

            dataset = PygNodePropPredDataset(
                name="ogbn-arxiv", root=os.path.join(data_dir, "OGB")
            )
            data = dataset[0]
            # OGB datasets store features in data.x and edge_index in data.edge_index
            # Convert to undirected if needed
            from torch_geometric.utils import to_undirected

            data.edge_index = to_undirected(data.edge_index)
        except ImportError:
            raise ImportError(
                "ogbn-arxiv requires the 'ogb' package. Install with: pip install ogb"
            )
    elif dataset_name == "IMDB":
        # IMDB heterogeneous dataset (movie/director/actor)
        dataset = IMDB(root=os.path.join(data_dir, "IMDB"))
        hetero_data = dataset[0]

        # Convert heterogeneous to homogeneous
        num_movies = hetero_data["movie"].num_nodes
        num_directors = hetero_data["director"].num_nodes
        num_actors = hetero_data["actor"].num_nodes
        num_nodes = num_movies + num_directors + num_actors

        edge_lists = []

        # Movie-Director edges
        if ("movie", "to", "director") in hetero_data.edge_types:
            md_edges = hetero_data["movie", "to", "director"].edge_index.clone()
            md_edges[1] = md_edges[1] + num_movies
            edge_lists.append(md_edges)
            edge_lists.append(md_edges.flip(0))

        # Movie-Actor edges
        if ("movie", "to", "actor") in hetero_data.edge_types:
            ma_edges = hetero_data["movie", "to", "actor"].edge_index.clone()
            ma_edges[1] = ma_edges[1] + num_movies + num_directors
            edge_lists.append(ma_edges)
            edge_lists.append(ma_edges.flip(0))

        edge_index_full = (
            torch.cat(edge_lists, dim=1)
            if edge_lists
            else torch.zeros((2, 0), dtype=torch.long)
        )

        # Combine features from all node types
        movie_x = (
            hetero_data["movie"].x
            if hetero_data["movie"].x is not None
            else torch.zeros(num_movies, feature_dim)
        )
        director_x = torch.zeros(num_directors, movie_x.shape[1])
        actor_x = torch.zeros(num_actors, movie_x.shape[1])

        features = torch.cat([movie_x, director_x, actor_x], dim=0)

        from torch_geometric.data import Data

        data = Data(x=features, edge_index=edge_index_full, num_nodes=num_nodes)
    elif dataset_name in WEBKB_DATASETS:
        # WebKB datasets: Texas, Cornell, Wisconsin
        # Small heterophilous graphs from university web pages
        dataset = WebKB(root=os.path.join(data_dir, "WebKB"), name=dataset_name)
        data = dataset[0]
    elif dataset_name in WIKIPEDIA_DATASETS:
        # Wikipedia network datasets: Chameleon, Squirrel
        # Heterophilous graphs from Wikipedia page-page networks
        dataset = WikipediaNetwork(
            root=os.path.join(data_dir, "WikipediaNetwork"), name=dataset_name.lower()
        )
        data = dataset[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {ALL_DATASETS}")

    # Generate features if dataset doesn't have them
    if data.x is None:
        print(
            f"  Dataset {dataset_name} has no features, generating synthetic features..."
        )
        data.x = generate_positional_features(
            data.num_nodes, data.edge_index, feature_dim
        )

    return data


class GaussianDPSanitizer:
    """Gaussian mechanism for differential privacy.

    Provides (epsilon, delta)-differential privacy.
    Noise is calibrated using the analytic Gaussian mechanism.
    """

    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = (
            0.0
            if epsilon == float("inf")
            else (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
        )

    def sanitize(self, x):
        return x + torch.randn_like(x) * self.sigma if self.sigma > 0 else x


class LaplacianDPSanitizer:
    """Laplacian mechanism for differential privacy.

    Provides pure epsilon-differential privacy (delta=0).
    Noise scale is sensitivity/epsilon.
    """

    def __init__(self, epsilon, sensitivity=1.0, delta=None):
        # delta is ignored for Laplacian mechanism (pure DP)
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = 0.0 if epsilon == float("inf") else sensitivity / epsilon

    def sanitize(self, x):
        if self.scale > 0:
            laplace_noise = (
                torch.distributions.Laplace(0, self.scale).sample(x.shape).to(x.device)
            )
            return x + laplace_noise
        return x


class RenyiDPSanitizer:
    """Renyi Differential Privacy (RDP) mechanism using Gaussian noise.

    Renyi DP provides tighter privacy accounting through composition.
    Uses Gaussian noise calibrated for (alpha, epsilon)-RDP.
    """

    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5, alpha=10.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.alpha = alpha

        if epsilon == float("inf"):
            self.sigma = 0.0
        else:
            epsilon_rdp = max(epsilon - np.log(1 / delta) / (alpha - 1), 0.01)
            self.sigma = np.sqrt(alpha * (sensitivity**2) / (2 * epsilon_rdp))

    def sanitize(self, x):
        return x + torch.randn_like(x) * self.sigma if self.sigma > 0 else x


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


def compute_cosine_similarity_for_pairs(features, pairs):
    """
    Compute cosine similarity only for specified node pairs.
    Memory efficient - doesn't create full N x N matrix.

    Args:
        features: [N, F] node features
        pairs: [2, M] node pairs to compute similarity for

    Returns:
        similarities: [M] cosine similarities in [0, 1] range
    """
    # Normalize features for cosine similarity
    feat_norm = F.normalize(features, p=2, dim=-1)

    # Get features for source and target nodes
    src_feats = feat_norm[pairs[0]]  # [M, F]
    tgt_feats = feat_norm[pairs[1]]  # [M, F]

    # Compute cosine similarity for each pair
    similarity = (src_feats * tgt_feats).sum(dim=1)  # [M]

    # Convert to [0, 1] range
    probs = (similarity + 1) / 2
    return probs


def sample_negative_edges(edge_index, num_nodes, num_neg_samples, seed=None):
    """
    Sample negative edges (non-existing edges) efficiently.

    Args:
        edge_index: [2, E] existing edges
        num_nodes: total number of nodes
        num_neg_samples: number of negative samples to generate
        seed: random seed for reproducibility

    Returns:
        neg_edges: [2, num_neg_samples] negative edge pairs
    """
    if seed is not None:
        np.random.seed(seed)

    # Create set of existing edges for fast lookup
    edge_set = set()
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, tgt = edge_index_np[0, i], edge_index_np[1, i]
        if src != tgt:  # Skip self-loops
            edge_set.add((src, tgt))

    # Sample random pairs until we have enough negatives
    neg_edges = []
    max_attempts = num_neg_samples * 10
    attempts = 0

    while len(neg_edges) < num_neg_samples and attempts < max_attempts:
        # Sample batch of random pairs
        batch_size = min(num_neg_samples * 2, num_neg_samples - len(neg_edges) + 1000)
        src = np.random.randint(0, num_nodes, size=batch_size)
        tgt = np.random.randint(0, num_nodes, size=batch_size)

        for s, t in zip(src, tgt):
            if s != t and (s, t) not in edge_set and (t, s) not in edge_set:
                neg_edges.append((s, t))
                if len(neg_edges) >= num_neg_samples:
                    break
        attempts += batch_size

    if len(neg_edges) < num_neg_samples:
        print(f"Warning: Could only sample {len(neg_edges)} negative edges")

    neg_edges = np.array(neg_edges[:num_neg_samples]).T  # [2, num_neg_samples]
    return torch.tensor(neg_edges, dtype=torch.long)


def evaluate_reconstruction_sparse(
    edge_index, features, num_nodes, device, num_neg_samples=None, seed=None
):
    """
    Evaluate reconstruction quality using sparse edge sampling.
    Memory efficient - doesn't create dense N x N matrices.

    Args:
        edge_index: [2, E] true edges (sparse)
        features: [N, F] node features (for computing predicted similarities)
        num_nodes: total number of nodes
        device: torch device
        num_neg_samples: number of negative samples (default: same as positive)
        seed: random seed for negative sampling

    Returns:
        ap, auc, degree_corr, micro_f1
    """
    # Remove self-loops and get unique edges
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    # For undirected graphs, keep only one direction (src < tgt)
    mask = edge_index[0] < edge_index[1]
    pos_edges = edge_index[:, mask]  # [2, E/2]
    num_pos = pos_edges.shape[1]

    if num_neg_samples is None:
        num_neg_samples = num_pos  # Balance positive and negative

    # Sample negative edges
    neg_edges = sample_negative_edges(edge_index, num_nodes, num_neg_samples, seed)
    neg_edges = neg_edges.to(device)

    # Compute predicted similarities for positive edges
    pos_scores = compute_cosine_similarity_for_pairs(features, pos_edges)

    # Compute predicted similarities for negative edges
    neg_scores = compute_cosine_similarity_for_pairs(features, neg_edges)

    # Combine scores and labels
    y_score = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    y_true = np.concatenate([np.ones(num_pos), np.zeros(len(neg_scores))])

    # Safety check
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.0, 0.0

    # Compute metrics (same style as run_exp/phase_04_attack)
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

    # Degree correlation (using sparse computation)
    # True degrees from edge_index
    true_degrees = degree(edge_index[0], num_nodes=num_nodes).cpu().numpy()

    # Predicted degrees: sum of similarities to all other nodes (sampled approximation)
    # For efficiency, we sample a subset of nodes to estimate degree
    sample_size = min(1000, num_nodes)
    sample_idx = np.random.choice(num_nodes, sample_size, replace=False)

    feat_norm = F.normalize(features, p=2, dim=-1)
    sample_feats = feat_norm[sample_idx]  # [sample_size, F]

    # Approximate predicted degrees by similarity to sampled nodes
    pred_degrees_approx = []
    batch_size = 1000
    for i in range(0, num_nodes, batch_size):
        end_i = min(i + batch_size, num_nodes)
        batch_feats = feat_norm[i:end_i]  # [batch, F]
        # Similarity to sampled nodes
        sim = torch.mm(batch_feats, sample_feats.T)  # [batch, sample_size]
        sim = (sim + 1) / 2  # Convert to [0, 1]
        pred_degrees_approx.append(
            sim.sum(dim=1).cpu().numpy() * (num_nodes / sample_size)
        )

    pred_degrees = np.concatenate(pred_degrees_approx)
    degree_corr, _ = spearmanr(true_degrees, pred_degrees)

    return ap, auc, degree_corr, micro_f1


def run_cosine_baseline(
    dataset_name="Cora",
    data_dir="./data",
    num_runs=10,
    epsilons=None,
    noise_type="gaussian",
    seed=42,
    device="auto",
    delta=1e-5,
    alpha=10.0,
):
    """Run cosine similarity baseline on FULL graph for all epsilon values."""

    # Device selection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # Default epsilons
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]

    # Load FULL raw dataset
    print(f"Loading full {dataset_name} dataset...")
    data = load_dataset(dataset_name, data_dir)

    # Extract features and adjacency (keep sparse)
    features = data.x.to(device)  # [N, F]
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes
    num_edges = data.num_edges // 2  # Undirected

    # Compute edge density without dense matrix
    edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2)

    print(f"  Nodes: {num_nodes}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Edges: {num_edges} (undirected)")
    print(f"  Edge density: {edge_density:.6f}")
    print(f"  Memory mode: SPARSE (sampling-based evaluation)")

    # Normalize features (zero mean, unit variance)
    feat_mean = features.mean(dim=0, keepdim=True)
    feat_std = features.std(dim=0, keepdim=True) + 1e-8
    features_norm = (features - feat_mean) / feat_std

    print(f"\nRunning {num_runs} trials per epsilon...")

    print("\n" + "=" * 85)
    print(f"COSINE SIMILARITY BASELINE - FULL GRAPH (Noise: {noise_type.upper()})")
    print("=" * 85)
    print(
        f"{'Epsilon':<12} {'AP Mean':<12} {'AP Std':<12} {'AUC Mean':<12} {'DegCorr':<12} {'MicroF1':<12}"
    )
    print("-" * 85)

    results = []

    for eps in epsilons:
        batch_aps = []
        batch_aucs = []
        batch_deg_corrs = []
        batch_micro_f1s = []

        for run_idx in range(num_runs):
            # Apply DP noise to the FULL feature matrix
            torch.manual_seed(seed + run_idx)
            sanitizer = get_sanitizer(noise_type, epsilon=eps, delta=delta, alpha=alpha)
            dp_feats = sanitizer.sanitize(features_norm)

            # Evaluate using sparse sampling-based method
            ap, auc, deg_corr, micro_f1 = evaluate_reconstruction_sparse(
                edge_index,
                dp_feats,
                num_nodes,
                device,
                num_neg_samples=None,  # Same as positive
                seed=seed + run_idx,
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

    print("=" * 85)

    # Summary
    print("\n--- Summary ---")
    print(f"Dataset: {dataset_name} (Full graph: {num_nodes} nodes)")
    print(
        "This baseline uses cosine similarity on DP-noised features to reconstruct edges."
    )
    print(
        "Higher epsilon = less noise = better reconstruction (upper bound: eps=inf means no noise)"
    )
    print("Compare these results with your diffusion model to see the improvement.")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cosine Similarity Baseline (Full Graph)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=ALL_DATASETS,
        help=f"Dataset to use. Options: {ALL_DATASETS}",
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of trials per epsilon (for std calculation)",
    )
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
        num_runs=args.num_runs,
        epsilons=epsilons,
        noise_type=args.noise_type,
        seed=args.seed,
        device=args.device,
        delta=args.delta,
        alpha=args.alpha,
    )
