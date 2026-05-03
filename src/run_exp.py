#!/usr/bin/env python3
"""
Unified runner script for D3PM Graph Reconstruction Attack pipeline
using feature explanations instead of raw features.

This script mirrors run.py but uses explanations (from saved_explanations)
in place of the original node features.

Usage:
    # Run full pipeline with Grad explanations
    python run_exp.py --dataset Cora --explainer Grad

    # Run training with GNNExplainer explanations
    python run_exp.py --dataset CiteSeer --stage train --explainer GNNExplainer

    # Run evaluation with GraphLime explanations
    python run_exp.py --dataset PubMed --stage eval --explainer GraphLime

    Note: Explanations must already exist in saved_explanations/ directory
"""

import argparse
import os
import sys
import torch
import numpy as np
from datetime import datetime
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import APPNP

# Fix for PyTorch 2.6+ weights_only issue with PyG/OGB datasets
# Add safe globals for torch_geometric classes
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass  # Older versions don't need this

# Ensure local imports work (src folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Dataset type mapping
PLANETOID_DATASETS = ["Cora", "CiteSeer", "PubMed"]
AMAZON_DATASETS = ["Amazon-Computers", "Amazon-Photo"]
AMAZON_HETERO_DATASETS = ["AmazonBook", "AmazonProducts"]
REDDIT_DATASETS = ["Reddit"]
HETEROPHILOUS_DATASETS = ["Amazon-ratings"]
OGB_DATASETS = ["ogbn-arxiv"]
OTHER_HETERO_DATASETS = ["IMDB"]
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

EXPLAINER_CHOICES = ["Grad", "GradInput", "GNNExplainer", "GraphLime"]


def normalize_backbone_name(backbone):
    """Normalize explanation backbone names to folder names under saved_explanations/."""
    key = str(backbone).strip().lower()
    mapping = {
        "gcn": "GCN",
        "gin": "GIN",
        "sage": "GraphSAGE",
        "graphsage": "GraphSAGE",
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported explanation backbone '{backbone}'. Use one of: GCN, GIN, GraphSAGE"
        )
    return mapping[key]


def subgraph(model, node_idx, x, edge_index, **kwargs):
    """Extract k-hop subgraph for a node."""
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    flow = "source_to_target"
    for module in model.modules():
        if isinstance(module, MessagePassing):
            flow = module.flow
            break

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, APPNP):
                num_hops += module.K
            else:
                num_hops += 1

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx,
        num_hops,
        edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
        flow=flow,
    )

    x = x[subset]
    for key, item in kwargs:
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]
        kwargs[key] = item

    return x, edge_index, mapping, edge_mask, kwargs


def fidelity(
    model,
    node_idx,
    full_feature_matrix,
    edge_index=None,
    node_mask=None,
    feature_mask=None,
    edge_mask=None,
    samples=100,
    random_seed=12345,
    device="cpu",
    validity=False,
):
    """
    Distortion/Fidelity (for Node Classification)
    :param model: GNN model which is explained
    :param node_idx: The node which is explained
    :param full_feature_matrix: The feature matrix from the Graph (X)
    :param edge_index: All edges
    :param node_mask: Binary tensor with 1/0 for each node in the computational graph
    :param feature_mask: Binary tensor with 1/0 for each feature
    :param edge_mask: Edge mask
    :param samples: Number of random samples
    :param random_seed: Random seed
    :param device: Device
    :param validity: If True, use zero features instead of random
    :return: Fidelity score (fraction of correct predictions)
    """
    if edge_mask is None and feature_mask is None and node_mask is None:
        raise ValueError("At least supply one mask")

    (
        computation_graph_feature_matrix,
        computation_graph_edge_index,
        mapping,
        hard_edge_mask,
        kwargs,
    ) = subgraph(model, node_idx, full_feature_matrix, edge_index)

    # Get predicted label
    log_logits = model(computation_graph_feature_matrix, computation_graph_edge_index)
    predicted_labels = log_logits.argmax(dim=-1)
    predicted_label = predicted_labels[mapping]

    # Fill missing masks
    if feature_mask is None:
        (num_nodes, num_features) = full_feature_matrix.size()
        feature_mask = torch.ones((1, num_features), device=device)

    num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
    if node_mask is None:
        node_mask = torch.ones((1, num_computation_graph_nodes), device=device)

    # Set edge mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = edge_mask

    (num_nodes, num_features) = full_feature_matrix.size()
    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # Retrieve complete mask as matrix
    mask = node_mask.T.matmul(feature_mask)

    if validity:
        samples = 1
        full_feature_matrix = torch.zeros_like(full_feature_matrix)

    correct = 0.0

    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)
    random_indices = torch.randint(
        num_nodes,
        (samples, num_nodes_computation_graph, num_features),
        generator=rng,
        device=device,
    )
    random_indices = random_indices.type(torch.int64)

    for i in range(samples):
        random_features = torch.gather(
            full_feature_matrix, dim=0, index=random_indices[i, :, :]
        )

        randomized_features = (
            mask * computation_graph_feature_matrix + (1 - mask) * random_features
        )

        log_logits = model(randomized_features, computation_graph_edge_index)
        distorted_labels = log_logits.argmax(dim=-1)

        if distorted_labels[mapping] == predicted_label:
            correct += 1

    # Reset mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    return correct / samples


def compute_explanation_fidelity(original_explanations, noisy_explanations):
    """
    Compute fidelity metrics between original and deprivatized explanations.

    Args:
        original_explanations: [B, N, F] or [N, F] tensor of original explanation features
        noisy_explanations: [B, N, F] or [N, F] tensor of noisy (deprivatized) explanation features

    Returns:
        dict with fidelity metrics: cosine_sim, pearson_corr, mse, mae
    """
    import torch.nn.functional as F
    from scipy.stats import pearsonr

    # Flatten to compute global metrics
    orig_flat = original_explanations.reshape(-1).cpu().numpy()
    noisy_flat = noisy_explanations.reshape(-1).cpu().numpy()

    # Cosine similarity
    orig_norm = F.normalize(original_explanations.reshape(-1, 1), p=2, dim=0)
    noisy_norm = F.normalize(noisy_explanations.reshape(-1, 1), p=2, dim=0)
    cosine_sim = (orig_norm * noisy_norm).sum().item()

    # Pearson correlation
    pearson_corr, _ = pearsonr(orig_flat, noisy_flat)

    # MSE and MAE
    mse = ((original_explanations - noisy_explanations) ** 2).mean().item()
    mae = (original_explanations - noisy_explanations).abs().mean().item()

    return {
        "cosine_similarity": cosine_sim,
        "pearson_correlation": pearson_corr,
        "mse": mse,
        "mae": mae,
    }


def load_explanation_features(
    dataset_name,
    explainer,
    explanation_dir="../saved_explanations",
    explanation_backbone="GCN",
):
    """
    Load explanation features for a dataset from per-node files.

    Args:
        dataset_name: Name of the dataset
        explainer: Which explainer to use ('Grad', 'GradInput', 'GNNExplainer', 'GraphLime')
        explanation_dir: Directory where explanations are saved

    Returns:
        explanation_features: [N, F] tensor
        original_features: [N, F] tensor
        edge_index: [2, E] tensor
    """
    from torch_geometric.datasets import (
        Planetoid,
        Amazon,
        WebKB,
        WikipediaNetwork,
        Reddit,
        AmazonBook,
        AmazonProducts,
        HeterophilousGraphDataset,
        IMDB,
    )
    from torch_geometric.data import Data
    import glob

    backbone_tag = normalize_backbone_name(explanation_backbone)

    # Build path to explainer directory
    explainer_dir = os.path.join(explanation_dir, explainer, backbone_tag, dataset_name)

    if not os.path.exists(explainer_dir):
        raise FileNotFoundError(
            f"Explanations directory not found: {explainer_dir}\n"
            f"Expected structure: saved_explanations/{explainer}/{backbone_tag}/{dataset_name}/"
        )

    # Load original dataset to get graph structure and original features
    print(f"Loading original {dataset_name} dataset...")
    if dataset_name in PLANETOID_DATASETS:
        dataset = Planetoid(root="../data", name=dataset_name)
    elif dataset_name in AMAZON_DATASETS:
        dataset = Amazon(root="../data", name=dataset_name.split("-")[1])
    elif dataset_name in WEBKB_DATASETS:
        dataset = WebKB(root="../data", name=dataset_name)
    elif dataset_name in WIKIPEDIA_DATASETS:
        dataset = WikipediaNetwork(root="../data", name=dataset_name)
    elif dataset_name == "AmazonBook":
        dataset = AmazonBook(root="../data/AmazonBook")
    elif dataset_name == "AmazonProducts":
        dataset = AmazonProducts(root="../data/AmazonProducts")
    elif dataset_name == "Reddit":
        dataset = Reddit(root="../data/Reddit")
    elif dataset_name == "Amazon-ratings":
        dataset = HeterophilousGraphDataset(
            root="../data/HeterophilousGraph", name="Amazon-ratings"
        )
    elif dataset_name == "IMDB":
        dataset = IMDB(root="../data/IMDB")
    elif dataset_name == "ogbn-arxiv":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset

            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="../data/OGB")
        except ImportError:
            raise ImportError(
                "ogbn-arxiv requires 'ogb' package. Install with: pip install ogb"
            )
    else:
        raise ValueError(f"Dataset {dataset_name} loading not implemented yet")

    data = dataset[0]

    # Check if data is heterogeneous (like phase_01_data.py does)
    from torch_geometric.data import HeteroData

    is_hetero = isinstance(data, HeteroData)

    if is_hetero:
        # Handle heterogeneous graphs - convert to homogeneous like phase_01_data.py
        print(f"  Heterogeneous graph detected")

        node_types = data.node_types
        edge_types = data.edge_types

        print(f"    Node types: {node_types}")
        print(f"    Edge types: {edge_types}")

        # Calculate total nodes
        total_nodes = sum(data[nt].num_nodes for nt in node_types)
        print(f"    Total nodes: {total_nodes}")

        # Create unified node index mapping
        node_offset = {}
        current_offset = 0
        for nt in node_types:
            node_offset[nt] = current_offset
            current_offset += data[nt].num_nodes

        # Build unified edge index
        all_edges_tensors = []
        for et in edge_types:
            src_type, _, dst_type = et
            edge_idx = data[et].edge_index
            # Convert to global indices
            src_global = edge_idx[0] + node_offset[src_type]
            dst_global = edge_idx[1] + node_offset[dst_type]
            all_edges_tensors.append(torch.stack([src_global, dst_global], dim=0))

        edge_index = (
            torch.cat(all_edges_tensors, dim=1)
            if all_edges_tensors
            else torch.zeros((2, 0), dtype=torch.long)
        )

        # Prepare unified features
        feature_list = []
        max_feat_dim = 0

        # Find max feature dimension
        for nt in node_types:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                max_feat_dim = max(max_feat_dim, data[nt].x.shape[1])

        if max_feat_dim == 0:
            max_feat_dim = 128

        # Concatenate or generate features for each node type
        for nt in node_types:
            num_nodes_nt = data[nt].num_nodes
            if hasattr(data[nt], "x") and data[nt].x is not None:
                feat = data[nt].x
                # Pad if necessary
                if feat.shape[1] < max_feat_dim:
                    padding = torch.zeros(num_nodes_nt, max_feat_dim - feat.shape[1])
                    feat = torch.cat([feat, padding], dim=1)
            else:
                # Generate synthetic features for this node type
                feat = torch.randn(num_nodes_nt, max_feat_dim)
            feature_list.append(feat)

        original_features = torch.cat(feature_list, dim=0)
        num_nodes = total_nodes
        num_features = max_feat_dim

    else:
        # Regular homogeneous graph
        num_nodes = data.num_nodes
        num_features = data.num_features
        edge_index = data.edge_index
        original_features = data.x

    print(f"Dataset info: {num_nodes} nodes, {num_features} features")

    # Load per-node explanation files
    print(f"Loading {explainer} explanations from {explainer_dir}...")

    # Get all feature_masks_node files
    mask_files = sorted(
        glob.glob(os.path.join(explainer_dir, "feature_masks_node=*.pt"))
    )

    if len(mask_files) == 0:
        raise FileNotFoundError(
            f"No feature mask files found in {explainer_dir}\n"
            f"Expected files: feature_masks_node=0.pt, feature_masks_node=1.pt, ..."
        )

    print(f"Found {len(mask_files)} explanation files")

    # First, load one explanation to get the actual feature dimension
    first_mask_file = os.path.join(explainer_dir, "feature_masks_node=0.pt")
    if os.path.exists(first_mask_file):
        first_mask = torch.load(first_mask_file)
        if first_mask.dim() == 2 and first_mask.size(0) == 1:
            first_mask = first_mask.squeeze(0)
        actual_feat_dim = first_mask.shape[0]
        print(f"  Detected explanation feature dimension: {actual_feat_dim}")

        # Update num_features to match explanations
        if actual_feat_dim != num_features:
            print(
                f"  Adjusting feature dimension from {num_features} to {actual_feat_dim} to match explanations"
            )
            num_features = actual_feat_dim

            # Regenerate original_features with correct dimension if needed
            if is_hetero:
                # For heterogeneous graphs, regenerate features with correct dim
                feature_list = []
                max_feat_dim = actual_feat_dim

                for nt in data.node_types:
                    num_nodes_nt = data[nt].num_nodes
                    if hasattr(data[nt], "x") and data[nt].x is not None:
                        feat = data[nt].x
                        # Pad or truncate to match
                        if feat.shape[1] < max_feat_dim:
                            padding = torch.zeros(
                                num_nodes_nt, max_feat_dim - feat.shape[1]
                            )
                            feat = torch.cat([feat, padding], dim=1)
                        elif feat.shape[1] > max_feat_dim:
                            feat = feat[:, :max_feat_dim]
                    else:
                        feat = torch.randn(num_nodes_nt, max_feat_dim)
                    feature_list.append(feat)

                original_features = torch.cat(feature_list, dim=0)

    # Initialize explanation tensor with correct dimension
    explanation_features = torch.zeros(num_nodes, num_features, dtype=torch.float32)

    # Load each node's explanation
    loaded_count = 0
    for node_idx in range(num_nodes):
        mask_file = os.path.join(explainer_dir, f"feature_masks_node={node_idx}.pt")
        if os.path.exists(mask_file):
            mask = torch.load(mask_file)
            # Mask is [1, F], squeeze to [F]
            if mask.dim() == 2 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            explanation_features[node_idx] = mask
            loaded_count += 1

    print(f"Loaded explanations for {loaded_count}/{num_nodes} nodes")
    print(f"  Explanation shape: {explanation_features.shape}")
    print(f"  Original features shape: {original_features.shape}")

    if loaded_count != num_nodes:
        print(f"WARNING: Missing explanations for {num_nodes - loaded_count} nodes")

    return explanation_features, original_features, edge_index


def create_explanation_dataset(
    dataset_name,
    explainer,
    num_subgraphs=5000,
    window_size=32,
    data_dir="../data",
    explanation_dir="../saved_explanations",
    output_dir="../data_exp",
    seed=42,
    force_recreate=False,
    use_combined=False,
    train_pct=20,
    explanation_backbone="GCN",
):
    """
    Create train/test dataset using explanation features.

    Args:
        dataset_name: Name of the dataset
        explainer: Which explainer to use
        num_subgraphs: Number of subgraphs to generate
        window_size: Size of subgraph windows
        data_dir: Directory for original data
        explanation_dir: Directory with saved explanations
        output_dir: Directory to save explanation-based data
        seed: Random seed
        force_recreate: Force recreation even if exists
        use_combined: If True, concatenate original features with explanations

    Returns:
        train_path, test_path
    """
    from torch_geometric.utils import to_networkx
    import networkx as nx
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    backbone_tag = normalize_backbone_name(explanation_backbone)
    suffix = "_combined" if use_combined else ""
    test_pct = 100 - train_pct
    train_path = os.path.join(
        output_dir,
        f"{dataset_name}_{explainer}_{backbone_tag}{suffix}_train_data_{window_size}_{train_pct}.pt",
    )
    test_path = os.path.join(
        output_dir,
        f"{dataset_name}_{explainer}_{backbone_tag}{suffix}_test_data_{window_size}_{test_pct}.pt",
    )

    # Check if dataset already exists
    if not force_recreate and os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Explanation dataset already exists in {output_dir}")
        print(f"  -> {train_path}")
        print(f"  -> {test_path}")
        print("Use --force-recreate to regenerate.")
        return train_path, test_path

    # Load explanations
    explanation_features, original_features, edge_index = load_explanation_features(
        dataset_name, explainer, explanation_dir, explanation_backbone=backbone_tag
    )

    # Determine which features to use
    if use_combined:
        # Concatenate original features with explanation features
        features = torch.cat([original_features, explanation_features], dim=1)
        print(
            f"Using combined features: {original_features.shape[1]} (original) + {explanation_features.shape[1]} (explanation) = {features.shape[1]}"
        )
    else:
        # Use only explanation features
        features = explanation_features
        print(f"Using explanation features only: {features.shape}")

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create NetworkX graph for subgraph sampling
    num_nodes = features.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    node_indices = list(range(num_nodes))

    processed_data = []
    seen_hashes = set()
    attempts = 0
    max_attempts = num_subgraphs * 20

    print(
        f"Generating {num_subgraphs} UNIQUE subgraphs with {explainer} explanations (N={window_size})..."
    )

    pbar = tqdm(total=num_subgraphs)
    while len(processed_data) < num_subgraphs and attempts < max_attempts:
        attempts += 1

        # Ego-net sampling
        center = np.random.choice(node_indices)
        bfs_tree = nx.bfs_tree(G, source=center)
        neighbors = list(bfs_tree.nodes())

        # Padding or Cutting
        if len(neighbors) < window_size:
            needed = window_size - len(neighbors)
            others = list(set(node_indices) - set(neighbors))
            if len(others) < needed:
                continue
            neighbors.extend(np.random.choice(others, needed, replace=False))

        selected_nodes = neighbors[:window_size]
        mapping = {node: idx for idx, node in enumerate(selected_nodes)}

        sub_adj = torch.zeros((window_size, window_size), dtype=torch.float)

        induced = G.subgraph(selected_nodes)
        for u, v in induced.edges():
            sub_adj[mapping[u], mapping[v]] = 1.0
            sub_adj[mapping[v], mapping[u]] = 1.0

        # Check for duplicate
        adj_hash = hash(sub_adj.numpy().tobytes())
        if adj_hash in seen_hashes:
            continue
        seen_hashes.add(adj_hash)

        # Extract features (explanations)
        mask = torch.tensor(selected_nodes, dtype=torch.long)
        sub_x = features[mask]

        processed_data.append({"adj": sub_adj, "x": sub_x})
        pbar.update(1)

    pbar.close()

    if len(processed_data) < num_subgraphs:
        print(f"WARNING: Only generated {len(processed_data)} unique subgraphs")

    # Split by train_pct
    print(f"Splitting Data ({train_pct}% Train / {100 - train_pct}% Test)...")
    np.random.shuffle(processed_data)
    split_idx = int(len(processed_data) * (train_pct / 100.0))

    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    torch.save(train_data, train_path)
    torch.save(test_data, test_path)

    print(f"Saved: {len(train_data)} Training graphs, {len(test_data)} Test graphs.")
    print(f"  -> {train_path}")
    print(f"  -> {test_path}")

    return train_path, test_path


def train_with_explanations(
    dataset_name="CiteSeer",
    explainer="grad",
    data_dir="../data_exp",
    output_dir=None,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    warmup_pct=0.1,
    sparsity_weight=0.1,
    hidden_dim=128,
    num_layers=4,
    diffusion_steps=100,
    gnn_type="gin",
    train_epsilon=5.0,
    noise_type="gaussian",
    delta=1e-5,
    alpha=10.0,
    log_interval=10,
    seed=42,
    device="auto",
    run_name=None,
    use_combined=False,
    window_size=32,
    train_pct=20,
    explanation_backbone="GCN",
):
    """
    Train the model using explanation features.
    Wraps phase_03_train.train() with explanation-specific data paths.
    """
    try:
        from phase_03_train import train
    except ImportError:
        from .phase_03_train import train

    backbone_tag = normalize_backbone_name(explanation_backbone)
    suffix = "_combined" if use_combined else ""

    # Override data paths to use explanation data - with new naming convention
    train_path = os.path.join(
        data_dir,
        f"{dataset_name}_{explainer}_{backbone_tag}{suffix}_train_data_{window_size}_{train_pct}.pt",
    )

    legacy_train_path = os.path.join(
        data_dir,
        f"{dataset_name}_{explainer}{suffix}_train_data_{window_size}_{train_pct}.pt",
    )
    if not os.path.exists(train_path) and os.path.exists(legacy_train_path):
        train_path = legacy_train_path
        print(f"Using legacy explanation dataset path: {legacy_train_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}.\n"
            f"Run data stage first: python run_exp.py --dataset {dataset_name} --explainer {explainer} --stage data"
        )

    if output_dir is None:
        output_dir = os.path.join("../result_exp", dataset_name, explainer, backbone_tag)
    os.makedirs(output_dir, exist_ok=True)

    # Custom training that loads from explanation data path
    import torch
    import torch.nn.functional as F
    import math
    import time

    try:
        from phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN
        from phase_03_train import get_sanitizer
    except ImportError:
        from .phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN
        from .phase_03_train import get_sanitizer

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # Load explanation-based training data
    data = torch.load(train_path)
    print(f"Loaded training data from {train_path}")

    adj_tensor = torch.stack([d["adj"] for d in data]).to(device)
    feat_tensor = torch.stack([d["x"] for d in data]).to(device)

    # Normalize features
    feat_mean = feat_tensor.mean(dim=(0, 1), keepdim=True)
    feat_std = feat_tensor.std(dim=(0, 1), keepdim=True) + 1e-8
    feat_tensor = (feat_tensor - feat_mean) / feat_std
    print(
        f"Features normalized: mean={feat_tensor.mean().item():.4f}, std={feat_tensor.std().item():.4f}"
    )

    N = adj_tensor.shape[1]
    F_dim = feat_tensor.shape[2]

    print(f"Loaded {len(adj_tensor)} graphs, N={N}, F={F_dim}")
    print(f"Using {explainer.upper()} explanation features")

    diffusion = DiscreteDiffusionBase(num_steps=diffusion_steps, device=device)
    model = ConditionalDenseGNN(
        num_nodes=N,
        feature_dim=F_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, (len(adj_tensor) + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(warmup_pct * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"Training on {len(adj_tensor)} graphs with {explainer} explanations...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    model.train()
    start_time = time.time()

    deprivatized_dir = os.path.join(output_dir, "deprivatized_explanations")
    os.makedirs(deprivatized_dir, exist_ok=True)

    for epoch in range(epochs):
        perm = torch.randperm(len(adj_tensor))
        total_loss = 0

        for i in range(0, len(adj_tensor), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) == 0:
                continue

            x_0 = adj_tensor[idx]
            batch_feats = feat_tensor[idx]
            B = x_0.size(0)

            sanitizer = get_sanitizer(
                noise_type, epsilon=train_epsilon, delta=delta, alpha=alpha
            )
            noisy_feats = sanitizer.sanitize(batch_feats)

            t = torch.randint(0, diffusion.num_steps, (B,), device=device)
            x_t = diffusion.q_sample(x_0, t)

            logits = model(x_t, noisy_feats, t)
            loss = F.cross_entropy(logits.view(-1, 2), x_0.long().view(-1))

            edge_probs = F.softmax(logits, dim=-1)[:, :, :, 1]
            target_density = x_0.float().mean()
            pred_density = edge_probs.mean()
            sparsity_loss = (pred_density - target_density).abs()

            loss = loss + sparsity_weight * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{epochs}: Loss {total_loss:.4f} | LR {current_lr:.6f} | Time {elapsed / 60:.1f}m"
            )

            # Save deprivatized explanations at regular intervals
            if (epoch + 1) % (log_interval * 10) == 0 or (epoch + 1) == epochs:
                # Sample a batch to save
                sample_idx = torch.randperm(len(feat_tensor))[
                    : min(100, len(feat_tensor))
                ]
                sample_feats = feat_tensor[sample_idx]
                sanitizer = get_sanitizer(
                    noise_type, epsilon=train_epsilon, delta=delta, alpha=alpha
                )
                sample_noisy = sanitizer.sanitize(sample_feats)

                save_path = os.path.join(
                    deprivatized_dir, f"epoch_{epoch + 1}_explanations.pt"
                )
                torch.save(
                    {
                        "original": sample_feats.cpu(),
                        "deprivatized": sample_noisy.cpu(),
                        "epsilon": train_epsilon,
                        "noise_type": noise_type,
                        "epoch": epoch + 1,
                    },
                    save_path,
                )
                print(f"  -> Saved deprivatized explanations to {save_path}")

    os.makedirs(output_dir, exist_ok=True)

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_samples = len(adj_tensor)
        run_name = f"{dataset_name}_{explainer}_{backbone_tag}_{gnn_type}_eps{train_epsilon:.1f}_n{num_samples}_{timestamp}"

    model_path = os.path.join(
        output_dir,
        f"model_{explainer}_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}.pth",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print(f"Total training time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Deprivatized explanations saved in: {deprivatized_dir}")

    return model_path, run_name


def evaluate_with_explanations(
    dataset_name="CiteSeer",
    explainer="grad",
    data_dir="../data_exp",
    output_dir=None,
    num_test_samples=500,
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
    use_combined=False,
    window_size=32,
    train_pct=20,
    balanced_eval=True,
    explanation_backbone="GCN",
    explanation_dir="../saved_explanations",
    kappa=0.0,
    rho=1.0,
):
    """
    Evaluate the model using explanation features.
    """
    import pandas as pd
    import time

    try:
        from phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN
        from phase_04_attack import (
            reconstruct_with_model,
        )
    except ImportError:
        from .phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN
        from .phase_04_attack import (
            reconstruct_with_model,
        )

    suffix = "_combined" if use_combined else ""
    backbone_tag = normalize_backbone_name(explanation_backbone)

    if output_dir is None:
        output_dir = os.path.join("../result_exp", dataset_name, explainer, backbone_tag)

    # Use new naming convention with window_size and train_pct
    test_pct = 100 - train_pct
    test_path = os.path.join(
        data_dir,
        f"{dataset_name}_{explainer}_{backbone_tag}{suffix}_test_data_{window_size}_{test_pct}.pt",
    )

    legacy_test_path = os.path.join(
        data_dir,
        f"{dataset_name}_{explainer}{suffix}_test_data_{window_size}_{test_pct}.pt",
    )
    if not os.path.exists(test_path) and os.path.exists(legacy_test_path):
        test_path = legacy_test_path
        print(f"Using legacy explanation dataset path: {legacy_test_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test data not found at {test_path}.\nRun data stage first."
        )

    if epsilons is None:
        epsilons = [8.0]
    if guidance_scales is None:
        guidance_scales = [0.0]

    print(f"Running Evaluation with {explainer.upper()} explanations...")
    print(f"Explanation backbone folder: {backbone_tag}")
    print(f"Test samples: {num_test_samples}")

    # Device selection
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    # Load test data
    data = torch.load(test_path)
    sample = data[0]
    N = sample["adj"].shape[0]
    F_dim = sample["x"].shape[1]

    # Pre-compute normalization stats
    all_feats = torch.stack([d["x"] for d in data])
    feat_mean = all_feats.mean(dim=(0, 1), keepdim=True)
    feat_std = all_feats.std(dim=(0, 1), keepdim=True) + 1e-8

    # Create model
    diffusion = DiscreteDiffusionBase(num_steps=diffusion_steps, device=device_obj)
    model = ConditionalDenseGNN(
        num_nodes=N,
        feature_dim=F_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    ).to(device_obj)

    # Load model weights
    model_path = os.path.join(
        output_dir,
        f"model_{explainer}_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}.pth",
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device_obj))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Model not found at {model_path}: {e}")
        return None

    model.eval()
    norm_stats = (feat_mean, feat_std)

    results = []

    # # Random baseline
    # print("\nEvaluating Random Baseline...")
    # baseline_aps = []
    # baseline_aucs = []

    # for sample_idx in range(min(num_test_samples, len(data))):
    #     s = seed + sample_idx
    #     ap, auc, _, _ = random_baseline(
    #         seed=s,
    #         sample_idx=sample_idx,
    #         dataset_name=dataset_name,
    #         data_dir="./data",  # Use original data dir for random baseline
    #         device=device,
    #     )
    #     baseline_aps.append(ap)
    #     baseline_aucs.append(auc)

    # baseline_ap_mean = np.mean(baseline_aps)
    # baseline_auc_mean = np.mean(baseline_aucs)
    # print(
    #     f"  -> Random Baseline: AP {baseline_ap_mean:.4f} | AUC {baseline_auc_mean:.4f}"
    # )

    # results.append(
    #     {
    #         "epsilon": "Random",
    #         "scale": "N/A",
    #         "AP_mean": baseline_ap_mean,
    #         "AUC_mean": baseline_auc_mean,
    #     }
    # )

    # Load original dataset for fidelity calculation
    print(f"Loading original {dataset_name} dataset for fidelity calculation...")
    explanation_features, original_features, full_edge_index = (
        load_explanation_features(
            dataset_name,
            explainer,
            explanation_dir,
            explanation_backbone=backbone_tag,
        )
    )

    # Model evaluation
    total_runs = len(epsilons) * len(guidance_scales) * min(num_test_samples, len(data))
    start_time = time.time()
    count = 0

    # Create a simple GNN model for fidelity calculation
    try:
        from torch_geometric.nn import GCN

        # Simple GCN for fidelity calculation
        num_classes = 10  # Assume up to 10 classes for node classification
        gnn_model = GCN(
            in_channels=original_features.shape[1],
            hidden_channels=128,
            num_layers=2,
            out_channels=num_classes,
        ).to(device_obj)
        gnn_model.eval()
        use_fidelity = True
        print(f"Created simple GNN model for fidelity calculation")
    except Exception as e:
        print(f"Warning: Could not create GNN model for fidelity: {e}")
        use_fidelity = False
        gnn_model = None

    for eps in epsilons:
        for scale in guidance_scales:
            batch_aps = []
            batch_aucs = []
            batch_deg_corrs = []
            batch_micro_f1s = []
            batch_fidelities = []

            print(f"\nEvaluating Eps: {eps} | Scale: {scale} ...")

            for sample_idx in range(min(num_test_samples, len(data))):
                s = seed + sample_idx

                # Get reconstruction metrics
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
                    device=device_obj,
                    balanced_eval=balanced_eval,
                    kappa=kappa,
                    rho=rho,
                )

                batch_aps.append(ap)
                batch_aucs.append(auc)
                batch_deg_corrs.append(deg_corr)
                batch_micro_f1s.append(micro_f1)

                # Calculate fidelity for this sample using the provided fidelity function
                if use_fidelity and gnn_model is not None:
                    try:
                        # Get the explanation features for this sample
                        sample_feats = data[sample_idx]["x"].to(device_obj)

                        # Ensure feature mask matches the original feature dimension
                        num_orig_features = original_features.shape[1]

                        # Create feature mask from explanation (normalized)
                        # Average across all nodes to get a single feature importance vector
                        avg_explanation = sample_feats.mean(dim=0)  # [F_sample]

                        # If dimensions don't match, pad or truncate
                        if avg_explanation.shape[0] < num_orig_features:
                            # Pad with zeros
                            padding = torch.zeros(
                                num_orig_features - avg_explanation.shape[0],
                                device=device_obj,
                            )
                            avg_explanation = torch.cat(
                                [avg_explanation, padding], dim=0
                            )
                        elif avg_explanation.shape[0] > num_orig_features:
                            # Truncate
                            avg_explanation = avg_explanation[:num_orig_features]

                        # Normalize to [0, 1] range
                        feature_mask = (
                            avg_explanation / (avg_explanation.max() + 1e-8)
                        ).unsqueeze(0)  # [1, F_orig]

                        # Calculate fidelity for each node in the subgraph
                        sample_fidelities = []
                        for node_idx in range(
                            min(10, sample_feats.shape[0])
                        ):  # Sample first 10 nodes
                            fid_score = fidelity(
                                model=gnn_model,
                                node_idx=node_idx,
                                full_feature_matrix=original_features.to(device_obj),
                                edge_index=full_edge_index.to(device_obj),
                                feature_mask=feature_mask,
                                samples=50,
                                random_seed=s + node_idx,  # Different seed per node
                                device=device_obj,
                                validity=False,
                            )
                            sample_fidelities.append(fid_score)

                        avg_sample_fidelity = np.mean(sample_fidelities)
                        batch_fidelities.append(avg_sample_fidelity)
                    except Exception as e:
                        print(
                            f"  Warning: Fidelity calculation failed for sample {sample_idx}: {e}"
                        )
                        batch_fidelities.append(0.0)

                count += 1
                if count % 10 == 0:
                    fid_info = (
                        f" | Avg Fid: {np.mean(batch_fidelities):.4f}"
                        if len(batch_fidelities) > 0
                        else ""
                    )
                    print(f"  Progress: {count}/{total_runs} runs completed{fid_info}")

            mean_ap = np.mean(batch_aps)
            std_ap = np.std(batch_aps)
            mean_auc = np.mean(batch_aucs)
            mean_deg_corr = np.mean(batch_deg_corrs)
            mean_micro_f1 = np.mean(batch_micro_f1s)
            mean_fidelity = (
                np.mean(batch_fidelities) if len(batch_fidelities) > 0 else 0.0
            )
            std_fidelity = (
                np.std(batch_fidelities) if len(batch_fidelities) > 0 else 0.0
            )

            fid_str = (
                f" | Fidelity: {mean_fidelity:.4f} ± {std_fidelity:.4f}"
                if len(batch_fidelities) > 0
                else ""
            )

            print(
                f"  -> Result: AP {mean_ap:.4f} ± {std_ap:.4f} | AUC {mean_auc:.4f} | DegCorr {mean_deg_corr:.4f} | MicroF1 {mean_micro_f1:.4f}{fid_str}"
            )

            results.append(
                {
                    "epsilon": str(eps),
                    "scale": scale,
                    "AP_mean": mean_ap,
                    "AP_std": std_ap,
                    "AUC_mean": mean_auc,
                    "DegCorr_mean": mean_deg_corr,
                    "MicroF1_mean": mean_micro_f1,
                    "Fidelity_mean": mean_fidelity,
                    "Fidelity_std": std_fidelity,
                }
            )

    # Save results
    df = pd.DataFrame(results)

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{dataset_name}_{explainer}_{backbone_tag}_{gnn_type}_{noise_type}_eps{train_epsilon}_ws{window_size}_split{train_pct}_{100 - train_pct}_n{num_test_samples}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"ablation_{run_name}.csv")
    df.to_csv(results_path, index=False)

    print(f"\nEvaluation Complete in {(time.time() - start_time) / 60:.1f} mins.")
    print(f"Results saved to '{results_path}'")
    if use_fidelity:
        print(f"Fidelity metrics included in results (per test sample)")

    return results_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="D3PM Graph Reconstruction Attack Pipeline with Feature Explanations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "data", "train", "eval"],
        help="Which stage(s) to run: data, train, eval, or all (explanations must already exist in saved_explanations/)",
    )

    # Explainer args
    parser.add_argument(
        "--explainer",
        type=str,
        default="Grad",
        choices=EXPLAINER_CHOICES + ["all"],
        help="Explainer to use for features",
    )
    parser.add_argument(
        "--use-combined",
        action="store_true",
        help="Use combined original + explanation features",
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
        help="Weight for sparsity regularization",
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
        default="gcn",
        choices=["gin", "gcn", "sage"],
        help="GNN type: 'gin', 'gcn', or 'sage' (GraphSAGE)",
    )
    parser.add_argument(
        "--explanation-backbone",
        type=str,
        default="GCN",
        help="Backbone folder under saved_explanations/{explainer}/ (GCN, GIN, GraphSAGE)",
    )
    parser.add_argument(
        "--train-epsilon",
        type=float,
        default=5.0,
        help="Epsilon for DP noise during training",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplacian", "renyi"],
        help="DP noise type",
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5, help="Delta parameter for DP"
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0, help="Alpha parameter for Renyi DP"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Epochs between logging"
    )

    # Evaluation args
    parser.add_argument(
        "--num-test-samples", type=int, default=50, help="Number of test samples"
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0,8.0,16.0",
        help="Comma-separated epsilon values",
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
        help="Directory for original datasets",
    )
    parser.add_argument(
        "--explanation-dir",
        type=str,
        default="../saved_explanations",
        help="Directory for saved explanations",
    )
    parser.add_argument(
        "--exp-data-dir",
        type=str,
        default="../data_exp",
        help="Directory for explanation-based data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models/results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument(
        "--force-recreate", action="store_true", help="Force recreation of dataset"
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Custom name for this run"
    )
    parser.add_argument(
        "--train-pct",
        type=int,
        default=20,
        help="Train split percentage (default: 20)",
    )
    parser.add_argument(
        "--num-sample-nodes",
        type=int,
        default=100,
        help="Number of nodes to sample for explainers",
    )
    parser.add_argument(
        "--no-balanced-eval",
        action="store_true",
        help="Disable balanced edge evaluation in reconstruction metrics (uses all off-diagonal edges)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    backbone_tag = normalize_backbone_name(args.explanation_backbone)

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "../result_exp", args.dataset, args.explainer, backbone_tag
        )

    print(f"\n{'#' * 60}")
    print("D3PM GRAPH RECONSTRUCTION - EXPLANATION PIPELINE")
    print(f"{'#' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Explainer: {args.explainer}")
    print(f"Explanation backbone: {backbone_tag}")
    print(f"Use combined features: {args.use_combined}")
    print(f"Stage: {args.stage}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.explanation_dir, exist_ok=True)
    os.makedirs(args.exp_data_dir, exist_ok=True)

    # ==================== DATA STAGE ====================
    if args.stage in ["all", "data"]:
        print(f"\n{'=' * 60}")
        print("STAGE 1: DATA PREPARATION (with explanations)")
        print(f"{'=' * 60}")

        explainers_to_process = (
            EXPLAINER_CHOICES if args.explainer == "all" else [args.explainer]
        )

        for exp in explainers_to_process:
            print(f"\nProcessing {exp.upper()} explanations...")
            create_explanation_dataset(
                dataset_name=args.dataset,
                explainer=exp,
                num_subgraphs=args.num_subgraphs,
                window_size=args.window_size,
                data_dir=args.data_dir,
                explanation_dir=args.explanation_dir,
                output_dir=args.exp_data_dir,
                seed=args.seed,
                force_recreate=args.force_recreate,
                use_combined=args.use_combined,
                train_pct=args.train_pct,
                explanation_backbone=backbone_tag,
            )

    # ==================== TRAINING STAGE ====================
    if args.stage in ["all", "train"]:
        print(f"\n{'=' * 60}")
        print("STAGE 2: TRAINING (with explanations)")
        print(f"{'=' * 60}")

        if args.explainer == "all":
            print("Training with all explainers...")
            for exp in EXPLAINER_CHOICES:
                print(f"\n--- Training with {exp.upper()} ---")
                train_with_explanations(
                    dataset_name=args.dataset,
                    explainer=exp,
                    data_dir=args.exp_data_dir,
                    output_dir=os.path.join(
                        "../result_exp", args.dataset, exp, backbone_tag
                    ),
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
                    use_combined=args.use_combined,
                    window_size=args.window_size,
                    train_pct=args.train_pct,
                    explanation_backbone=backbone_tag,
                )
        else:
            train_with_explanations(
                dataset_name=args.dataset,
                explainer=args.explainer,
                data_dir=args.exp_data_dir,
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
                use_combined=args.use_combined,
                window_size=args.window_size,
                train_pct=args.train_pct,
                explanation_backbone=backbone_tag,
            )

    # ==================== EVALUATION STAGE ====================
    if args.stage in ["all", "eval"]:
        print(f"\n{'=' * 60}")
        print("STAGE 3: EVALUATION (with explanations)")
        print(f"{'=' * 60}")

        # Parse epsilon values
        epsilons = []
        for e in args.epsilons.split(","):
            e = e.strip()
            if e:
                epsilons.append(float("inf") if e.lower() == "inf" else float(e))

        # Parse guidance scales
        scales = [
            float(s.strip()) for s in args.guidance_scales.split(",") if s.strip()
        ]

        if args.explainer == "all":
            print("Evaluating with all explainers...")
            for exp in EXPLAINER_CHOICES:
                print(f"\n--- Evaluating with {exp.upper()} ---")
                evaluate_with_explanations(
                    dataset_name=args.dataset,
                    explainer=exp,
                    data_dir=args.exp_data_dir,
                    output_dir=os.path.join(
                        "../result_exp", args.dataset, exp, backbone_tag
                    ),
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
                    use_combined=args.use_combined,
                    window_size=args.window_size,
                    train_pct=args.train_pct,
                    balanced_eval=not args.no_balanced_eval,
                    explanation_backbone=backbone_tag,
                    explanation_dir=args.explanation_dir,
                    kappa=args.kappa,
                    rho=args.rho,
                )
        else:
            evaluate_with_explanations(
                dataset_name=args.dataset,
                explainer=args.explainer,
                data_dir=args.exp_data_dir,
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
                use_combined=args.use_combined,
                window_size=args.window_size,
                train_pct=args.train_pct,
                balanced_eval=not args.no_balanced_eval,
                explanation_backbone=backbone_tag,
                explanation_dir=args.explanation_dir,
                kappa=args.kappa,
                rho=args.rho,
            )

    print(f"\n{'#' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
