import os
import argparse
import torch

# Fix for PyTorch 2.6+ weights_only issue with PyG/OGB datasets
# Add safe globals for torch_geometric classes
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass  # Older versions don't need this

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
from torch_geometric.utils import to_networkx, degree
import networkx as nx
import numpy as np
from tqdm import tqdm


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
        # AmazonBook heterogeneous dataset (user-book bipartite)
        # Returns HeteroData object with user and book nodes
        dataset = AmazonBook(root=os.path.join(data_dir, "AmazonBook"))
        data = dataset[0]
        # Note: This is a HeteroData object with:
        # - data['user'].num_nodes, data['book'].num_nodes
        # - data['user', 'rates', 'book'].edge_index
        # Pipeline needs to handle heterogeneous data separately
        return data

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
        # Returns HeteroData object with movie, director, actor nodes
        dataset = IMDB(root=os.path.join(data_dir, "IMDB"))
        data = dataset[0]
        # Note: This is a HeteroData object with:
        # - data['movie'].x, data['movie'].num_nodes
        # - data['director'].num_nodes, data['actor'].num_nodes
        # - data['movie', 'to', 'director'].edge_index
        # - data['movie', 'to', 'actor'].edge_index
        # Pipeline needs to handle heterogeneous data separately
        return data
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


def create_split_dataset_hetero(
    data,
    dataset_name,
    num_subgraphs=5000,
    window_size=32,
    data_dir="../data",
    seed=42,
    feature_dim=128,
    train_pct=20,
):
    """
    Generate train/test split for heterogeneous graphs (IMDB, AmazonBook).

    For heterogeneous graphs, we sample subgraphs that preserve the heterogeneous structure.
    Each subgraph contains nodes of different types and edges between them.
    """
    # Include window_size and split ratio in filename
    train_path = os.path.join(
        data_dir, f"{dataset_name}_train_data_{window_size}_{train_pct}.pt"
    )
    test_path = os.path.join(
        data_dir, f"{dataset_name}_test_data_{window_size}_{100 - train_pct}.pt"
    )

    # Get node types and edge types
    node_types = data.node_types
    edge_types = data.edge_types

    print(f"  Heterogeneous Graph:")
    print(f"    Node types: {node_types}")
    print(f"    Edge types: {edge_types}")

    # Calculate total nodes
    total_nodes = sum(data[nt].num_nodes for nt in node_types)
    print(f"    Total nodes: {total_nodes}")

    # Calculate total edges
    total_edges = sum(data[et].edge_index.size(1) for et in edge_types)
    print(f"    Total edges: {total_edges}")

    # Create a unified node index mapping
    # Maps (node_type, local_idx) -> global_idx
    node_offset = {}
    current_offset = 0
    for nt in node_types:
        node_offset[nt] = current_offset
        current_offset += data[nt].num_nodes

    # Build unified edge index for NetworkX graph
    all_edges = []
    for et in edge_types:
        src_type, _, dst_type = et
        edge_index = data[et].edge_index
        # Convert to global indices
        src_global = edge_index[0] + node_offset[src_type]
        dst_global = edge_index[1] + node_offset[dst_type]
        for i in range(edge_index.size(1)):
            all_edges.append((src_global[i].item(), dst_global[i].item()))

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    G.add_edges_from(all_edges)

    # Prepare features - concatenate all node features or generate if missing
    feature_list = []
    max_feat_dim = 0
    for nt in node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            max_feat_dim = max(max_feat_dim, data[nt].x.shape[1])

    if max_feat_dim == 0:
        max_feat_dim = feature_dim

    for nt in node_types:
        num_nodes_nt = data[nt].num_nodes
        if hasattr(data[nt], "x") and data[nt].x is not None:
            feat = data[nt].x
            # Pad if necessary
            if feat.shape[1] < max_feat_dim:
                padding = torch.zeros(num_nodes_nt, max_feat_dim - feat.shape[1])
                feat = torch.cat([feat, padding], dim=1)
        else:
            # Generate synthetic features
            feat = torch.randn(num_nodes_nt, max_feat_dim)
        feature_list.append(feat)

    all_features = torch.cat(feature_list, dim=0)

    # Create node type labels for each global node
    node_type_labels = []
    for i, nt in enumerate(node_types):
        node_type_labels.extend([i] * data[nt].num_nodes)
    node_type_labels = torch.tensor(node_type_labels, dtype=torch.long)

    # Now sample subgraphs similar to homogeneous case
    node_indices = list(range(total_nodes))

    processed_data = []
    seen_hashes = set()
    attempts = 0
    max_attempts = num_subgraphs * 20

    print(
        f"Generating {num_subgraphs} UNIQUE heterogeneous subgraphs (N={window_size})..."
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

        # Fill Adjacency
        induced = G.subgraph(selected_nodes)
        for u, v in induced.edges():
            sub_adj[mapping[u], mapping[v]] = 1.0
            sub_adj[mapping[v], mapping[u]] = 1.0

        # Check for duplicate
        adj_hash = hash(sub_adj.numpy().tobytes())
        if adj_hash in seen_hashes:
            continue
        seen_hashes.add(adj_hash)

        # Extract Features and node types
        mask = torch.tensor(selected_nodes, dtype=torch.long)
        sub_x = all_features[mask]
        sub_node_types = node_type_labels[mask]

        processed_data.append(
            {
                "adj": sub_adj,
                "x": sub_x,
                "node_types": sub_node_types,  # Store node type info for heterogeneous analysis
            }
        )
        pbar.update(1)

    pbar.close()

    if len(processed_data) < num_subgraphs:
        print(
            f"WARNING: Only generated {len(processed_data)} unique subgraphs (requested {num_subgraphs})"
        )

    # Split
    print(f"Splitting Data ({train_pct}% Train / {100 - train_pct}% Test)...")
    np.random.shuffle(processed_data)
    split_idx = int(len(processed_data) * (train_pct / 100.0))

    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    # Save metadata about heterogeneous structure
    metadata = {
        "node_types": node_types,
        "edge_types": edge_types,
        "node_offset": node_offset,
        "feature_dim": max_feat_dim,
        "is_hetero": True,
        "window_size": window_size,
        "train_pct": train_pct,
        "test_pct": 100 - train_pct,
    }

    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    torch.save(
        metadata,
        os.path.join(data_dir, f"{dataset_name}_metadata_{window_size}_{train_pct}.pt"),
    )

    print(f"Saved: {len(train_data)} Training graphs, {len(test_data)} Test graphs.")
    print(f"  -> {train_path}")
    print(f"  -> {test_path}")
    print(f"  -> {os.path.join(data_dir, f'{dataset_name}_metadata.pt')}")

    return train_path, test_path


def create_split_dataset(
    dataset_name="CiteSeer",
    num_subgraphs=5000,
    window_size=32,
    data_dir="../data",
    seed=42,
    force_recreate=False,
    train_pct=20,
):
    """Generate train/test split of subgraph data with unique samples.

    Args:
        dataset_name: Name of the dataset
        num_subgraphs: Number of subgraphs to generate
        window_size: Size of each subgraph (number of nodes)
        data_dir: Directory to save data
        seed: Random seed
        force_recreate: Force recreation even if files exist
        train_pct: Percentage of data for training (default 20%)
    """
    # Data directory for processed data
    os.makedirs(data_dir, exist_ok=True)

    # Include window_size and split ratio in filename
    train_path = os.path.join(
        data_dir, f"{dataset_name}_train_data_{window_size}_{train_pct}.pt"
    )
    test_path = os.path.join(
        data_dir, f"{dataset_name}_test_data_{window_size}_{100 - train_pct}.pt"
    )

    # Check if dataset already exists
    if not force_recreate and os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Dataset already exists in {data_dir}")
        print(f"  -> {train_path}")
        print(f"  -> {test_path}")
        print("Skipping data generation. Use --force-recreate to regenerate.")
        return train_path, test_path

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading {dataset_name}...")
    data = load_dataset(dataset_name, data_dir)

    # Check if data is heterogeneous (HeteroData)
    from torch_geometric.data import HeteroData

    is_hetero = isinstance(data, HeteroData)

    if is_hetero:
        # Handle heterogeneous graphs (IMDB, AmazonBook)
        return create_split_dataset_hetero(
            data=data,
            dataset_name=dataset_name,
            num_subgraphs=num_subgraphs,
            window_size=window_size,
            data_dir=data_dir,
            seed=seed,
            train_pct=train_pct,
        )

    print(f"  Nodes: {data.num_nodes}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Edges: {data.num_edges // 2} (undirected)")

    G = to_networkx(data, to_undirected=True)
    node_indices = list(range(data.num_nodes))

    processed_data = []
    seen_hashes = set()  # Track unique subgraphs
    attempts = 0
    max_attempts = num_subgraphs * 20  # Avoid infinite loop

    print(f"Generating {num_subgraphs} UNIQUE subgraphs (N={window_size})...")

    pbar = tqdm(total=num_subgraphs)
    while len(processed_data) < num_subgraphs and attempts < max_attempts:
        attempts += 1

        # 1. Ego-net sampling
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
        # 2. Induce Subgraph
        mapping = {node: idx for idx, node in enumerate(selected_nodes)}

        sub_adj = torch.zeros((window_size, window_size), dtype=torch.float)

        # Fill Adjacency
        induced = G.subgraph(selected_nodes)
        for u, v in induced.edges():
            sub_adj[mapping[u], mapping[v]] = 1.0
            sub_adj[mapping[v], mapping[u]] = 1.0

        # Check for duplicate using hash
        adj_hash = hash(sub_adj.numpy().tobytes())
        if adj_hash in seen_hashes:
            continue  # Skip duplicate
        seen_hashes.add(adj_hash)

        # 3. Extract Features
        mask = torch.tensor(selected_nodes, dtype=torch.long)
        sub_x = data.x[mask]

        processed_data.append({"adj": sub_adj, "x": sub_x})
        pbar.update(1)

    pbar.close()

    if len(processed_data) < num_subgraphs:
        print(
            f"WARNING: Only generated {len(processed_data)} unique subgraphs (requested {num_subgraphs})"
        )
        print(f"  Dataset may not have enough unique {window_size}-node subgraphs.")

    # SHUFFLE AND SPLIT
    print(f"Splitting Data ({train_pct}% Train / {100 - train_pct}% Test)...")
    np.random.shuffle(processed_data)
    split_idx = int(len(processed_data) * (train_pct / 100.0))

    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    # Save metadata
    metadata = {
        "window_size": window_size,
        "train_pct": train_pct,
        "test_pct": 100 - train_pct,
        "num_subgraphs": len(processed_data),
        "is_hetero": False,
    }

    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    torch.save(
        metadata,
        os.path.join(data_dir, f"{dataset_name}_metadata_{window_size}_{train_pct}.pt"),
    )
    print(f"Saved: {len(train_data)} Training graphs, {len(test_data)} Test graphs.")
    print(f"All samples are UNIQUE (no duplicates).")
    print(f"  -> {train_path}")
    print(f"  -> {test_path}")
    print(
        f"  -> {os.path.join(data_dir, f'{dataset_name}_metadata_{window_size}_{train_pct}.pt')}"
    )

    return train_path, test_path


def parse_args():
    parser = argparse.ArgumentParser(description="Data preparation for D3PM")
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
        default=50000,
        help="Number of subgraphs to generate",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Size of subgraph windows (number of nodes)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Directory to save data"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train-pct",
        type=int,
        default=20,
        help="Percentage of data for training (default 20%%, rest for testing)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of dataset even if it exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_split_dataset(
        dataset_name=args.dataset,
        num_subgraphs=args.num_subgraphs,
        window_size=args.window_size,
        data_dir=args.data_dir,
        seed=args.seed,
        force_recreate=args.force_recreate,
        train_pct=args.train_pct,
    )
