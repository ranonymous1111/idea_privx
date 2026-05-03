#!/usr/bin/env python3
"""
phase_01b_explain.py — GNN Explanation Extraction for PrivX

Computes per-node GNN explanations (Grad, GradInput, GNNExplainer, GraphLime)
for each subgraph sample in a dataset and saves a new dataset with a 'phi' key
containing explanation features alongside the original 'adj' and 'x' keys.

Supported explainers:
  - Grad        : gradient w.r.t. input features
  - GradInput   : gradient × input (saliency)
  - GNNExplainer: learnable edge/feature masks (PyG implementation)
  - GraphLime   : graph LIME with local linear model

Usage:
  python phase_01b_explain.py --dataset Cornell --explainer Grad
  python phase_01b_explain.py --dataset Wisconsin --explainer GNNExplainer
  python phase_01b_explain.py --dataset Squirrel --explainer GraphLime \\
         --gnn-backbone GCN --data-dir ./data --output-dir ./data \\
         --explanation-dir ../saved_explanations
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Fix PyTorch 2.6+ weights_only issue
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass

from torch_geometric.datasets import (
    Planetoid, Amazon, WebKB, WikipediaNetwork,
    Reddit, AmazonBook, AmazonProducts,
    HeterophilousGraphDataset, IMDB,
)
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import to_undirected


# ──────────────────────────────────────────────────────────────────────────────
# Dataset constants (must NOT convert heterophilic to homophilic)
# ──────────────────────────────────────────────────────────────────────────────
PLANETOID_DATASETS   = ["Cora", "CiteSeer", "PubMed"]
WEBKB_DATASETS       = ["Texas", "Cornell", "Wisconsin"]
WIKIPEDIA_DATASETS   = ["Chameleon", "Squirrel"]
HETEROPHILIC_DATASETS = WEBKB_DATASETS + WIKIPEDIA_DATASETS + ["Amazon-ratings"]
ALL_DATASETS = (
    PLANETOID_DATASETS + WEBKB_DATASETS + WIKIPEDIA_DATASETS
    + ["Amazon-ratings", "AmazonBook", "IMDB", "ogbn-arxiv", "Reddit", "Bitcoinalpha"]
)


# ──────────────────────────────────────────────────────────────────────────────
# Simple GCN backbone for computing explanations
# ──────────────────────────────────────────────────────────────────────────────
class GCNClassifier(torch.nn.Module):
    """2-layer GCN for node classification. Trained/loaded to compute explanations."""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


# ──────────────────────────────────────────────────────────────────────────────
# Load pre-saved per-node explanation files (already computed in saved_explanations/)
# ──────────────────────────────────────────────────────────────────────────────
def load_node_explanations(explanation_dir, dataset_name, explainer, backbone="GCN"):
    """
    Load per-node explanation tensors from saved_explanations/<explainer>/<backbone>/<dataset>/.

    Returns:
        node_explanations: dict {node_idx: tensor [F]} of feature importance scores
    """
    exp_folder = os.path.join(explanation_dir, explainer, backbone, dataset_name)
    if not os.path.exists(exp_folder):
        raise FileNotFoundError(
            f"Explanation folder not found: {exp_folder}\n"
            f"Run explainer first or check path."
        )

    node_explanations = {}
    pattern = os.path.join(exp_folder, "feature_masks_node=*.pt")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No explanation files found in {exp_folder} (expected feature_masks_node=N.pt)")

    for fpath in files:
        basename = os.path.basename(fpath)          # "feature_masks_node=42.pt"
        node_idx = int(basename.split("=")[1].replace(".pt", ""))
        mask = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(mask, dict):
            # Some savers store as dict with 'feature_mask' key
            mask = mask.get("feature_mask", mask.get("mask", list(mask.values())[0]))
        if mask is None:
            continue
        mask = mask.float().squeeze()
        node_explanations[node_idx] = mask

    return node_explanations


# ──────────────────────────────────────────────────────────────────────────────
# Compute Grad / GradInput explanations on-the-fly (for missing datasets)
# ──────────────────────────────────────────────────────────────────────────────
def compute_grad_explanation(model, x, edge_index, target_node, device, grad_input=False):
    """
    Gradient (or Gradient×Input) explanation for target_node.

    Args:
        model       : trained GNN
        x           : [N, F] feature matrix
        edge_index  : [2, E]
        target_node : node index
        grad_input  : if True, compute Gradient×Input (saliency); else plain gradient
    Returns:
        phi [F] explanation vector
    """
    x = x.to(device).float()
    x.requires_grad_(True)
    edge_index = edge_index.to(device)

    model.eval()
    logits = model(x, edge_index)
    pred_class = logits[target_node].argmax().item()
    score = logits[target_node, pred_class]
    score.backward()

    grad = x.grad[target_node].detach()  # [F]
    if grad_input:
        phi = (grad * x[target_node].detach()).abs()
    else:
        phi = grad.abs()
    return phi.cpu()


def compute_gnnexplainer_explanation(model, x, edge_index, target_node, device,
                                     epochs=100, lr=0.01):
    """GNNExplainer: learns feature mask via optimisation."""
    from torch_geometric.explain import Explainer, GNNExplainer as PygGNNExplainer

    x = x.to(device).float()
    edge_index = edge_index.to(device)

    explainer = Explainer(
        model=model,
        algorithm=PygGNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    explanation = explainer(x, edge_index, index=target_node)
    phi = explanation.node_mask[target_node].abs().cpu()  # [F]
    return phi


def compute_graphlime_explanation(model, x, edge_index, target_node, device,
                                  num_hops=2, rho=0.1):
    """GraphLIME: local linear explanation via HSIC Lasso."""
    from torch_geometric.explain import Explainer, GraphLIME as PygGraphLIME

    x = x.to(device).float()
    edge_index = edge_index.to(device)

    explainer = Explainer(
        model=model,
        algorithm=PygGraphLIME(num_hops=num_hops, rho=rho),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    explanation = explainer(x, edge_index, index=target_node)
    phi = explanation.node_mask[target_node].abs().cpu()  # [F]
    return phi


# ──────────────────────────────────────────────────────────────────────────────
# Save per-node explanation to disk (mirrors existing saved_explanations layout)
# ──────────────────────────────────────────────────────────────────────────────
def save_node_explanation(phi, explanation_dir, dataset_name, explainer, node_idx, backbone="GCN"):
    out_dir = os.path.join(explanation_dir, explainer, backbone, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"feature_masks_node={node_idx}.pt")
    torch.save(phi, out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Load dataset (homophily PRESERVED for heterophilic datasets)
# ──────────────────────────────────────────────────────────────────────────────
def load_full_dataset(dataset_name, data_dir):
    """Load full graph dataset. Does NOT convert heterophilic to homophilic."""
    if dataset_name in PLANETOID_DATASETS:
        ds = Planetoid(root=os.path.join(data_dir, dataset_name), name=dataset_name)
        data = ds[0]
    elif dataset_name in WEBKB_DATASETS:
        ds = WebKB(root=os.path.join(data_dir, "WebKB"), name=dataset_name)
        data = ds[0]
        if isinstance(data, list):
            data = data[0]  # WebKB has 5 splits; use first
    elif dataset_name in WIKIPEDIA_DATASETS:
        ds = WikipediaNetwork(root=os.path.join(data_dir, "WikipediaNetwork"), name=dataset_name.lower())
        data = ds[0]
    elif dataset_name == "Amazon-ratings":
        ds = HeterophilousGraphDataset(root=os.path.join(data_dir, "HeterophilousGraph"), name="Amazon-ratings")
        data = ds[0]
    elif dataset_name == "Amazon-Computers":
        ds = Amazon(root=os.path.join(data_dir, "Amazon"), name="Computers")
        data = ds[0]
    elif dataset_name == "Amazon-Photo":
        ds = Amazon(root=os.path.join(data_dir, "Amazon"), name="Photo")
        data = ds[0]
    elif dataset_name == "IMDB":
        ds = IMDB(root=os.path.join(data_dir, "IMDB"))
        data = ds[0]
    elif dataset_name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(data_dir, "OGB"))
        data = ds[0]
    elif dataset_name == "Reddit":
        ds = Reddit(root=os.path.join(data_dir, "Reddit"))
        data = ds[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Train or load GCN backbone on the full graph
# ──────────────────────────────────────────────────────────────────────────────
def get_gnn_model(data, dataset_name, data_dir, device, hidden_dim=64, epochs=200):
    """Train (or load cached) GCN backbone for explanation computation."""
    model_cache_dir = os.path.join(data_dir, "_gnn_cache")
    os.makedirs(model_cache_dir, exist_ok=True)
    model_path = os.path.join(model_cache_dir, f"{dataset_name}_GCN.pth")

    x = data.x.float().to(device)
    edge_index = data.edge_index.to(device)

    # Handle datasets with no labels
    if not hasattr(data, "y") or data.y is None:
        num_classes = 2
        y = torch.zeros(x.size(0), dtype=torch.long).to(device)
    else:
        y = data.y
        if y.dim() > 1:
            y = y.squeeze(-1)
        y = y.to(device)
        num_classes = int(y.max().item()) + 1

    model = GCNClassifier(x.size(1), hidden_dim, num_classes).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded cached GCN from {model_path}")
        model.eval()
        return model

    # Train mask: use available train_mask or random 80%
    if hasattr(data, "train_mask") and data.train_mask is not None:
        train_mask = data.train_mask
        if train_mask.dim() > 1:
            train_mask = train_mask[:, 0]  # WebKB has 5 splits
        train_mask = train_mask.to(device)
    else:
        perm = torch.randperm(x.size(0))
        train_mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
        train_mask[perm[:int(0.8 * x.size(0))]] = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for ep in tqdm(range(epochs), desc=f"Training GCN backbone for {dataset_name}", leave=False):
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)
    print(f"  Saved GCN backbone to {model_path}")
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Augment a dataset split (train or test) with explanation features 'phi'
# ──────────────────────────────────────────────────────────────────────────────
def augment_dataset_with_explanations(
    dataset_path,
    output_path,
    node_explanations,
    full_graph_x,
    fallback_zeros=True,
):
    """
    Add 'phi' key to each sample in a dataset split.

    phi[i] = explanation for node i in the subgraph window.
    If a node's explanation is missing, falls back to the raw feature or zeros.

    Args:
        dataset_path     : path to existing .pt file (list of dicts with 'adj', 'x')
        output_path      : path to write augmented .pt file
        node_explanations: dict {global_node_idx: tensor[F]}
        full_graph_x     : [N_global, F] original feature matrix
        fallback_zeros   : if True, use zeros for missing nodes; else use raw features
    """
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    F_dim = full_graph_x.size(1)

    augmented = []
    missing_count = 0

    for sample in tqdm(data, desc=f"Augmenting {os.path.basename(dataset_path)}", leave=False):
        adj = sample["adj"]
        x   = sample["x"]           # [N_window, F]
        N   = x.size(0)

        # node_ids: if sample has 'node_ids', use them; else skip (unknown global idx)
        node_ids = sample.get("node_ids", None)

        phi_list = []
        for local_i in range(N):
            if node_ids is not None:
                global_idx = int(node_ids[local_i])
            else:
                # Fallback: use local index (may be wrong for remapped subgraphs)
                global_idx = local_i

            if global_idx in node_explanations:
                phi_i = node_explanations[global_idx]
            else:
                missing_count += 1
                phi_i = torch.zeros(F_dim) if fallback_zeros else full_graph_x[global_idx]

            phi_list.append(phi_i)

        phi = torch.stack(phi_list, dim=0)  # [N_window, F]
        new_sample = dict(sample)
        new_sample["phi"] = phi
        augmented.append(new_sample)

    torch.save(augmented, output_path)
    if missing_count > 0:
        print(f"  WARNING: {missing_count} nodes had no explanation (used fallback).")
    print(f"  Saved augmented dataset to {output_path}  ({len(augmented)} samples)")


# ──────────────────────────────────────────────────────────────────────────────
# Compute missing per-node explanations and save them to disk
# ──────────────────────────────────────────────────────────────────────────────
def compute_and_save_explanations(
    dataset_name,
    explainer,
    data_dir,
    explanation_dir,
    backbone="GCN",
    device="auto",
    max_nodes=None,
    batch_size=1,
):
    """
    Compute GNN explanations for every node in a dataset and save per-node files.

    Skips nodes for which explanation files already exist.
    IMPORTANT: For heterophilic datasets, the graph structure is preserved as-is
               (no homophily conversion).
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"\n[phase_01b] Computing {explainer} explanations for {dataset_name} on {device}")

    # Load dataset — heterophily PRESERVED
    data = load_full_dataset(dataset_name, data_dir)

    x = data.x
    if x is None:
        print(f"  Dataset {dataset_name} has no node features — skipping explanation extraction.")
        return
    x = x.float()
    edge_index = data.edge_index

    if dataset_name in HETEROPHILIC_DATASETS:
        print(f"  [heterophilic] {dataset_name}: preserving original graph structure (no homophily forcing).")

    # Determine which nodes need explanations
    out_dir = os.path.join(explanation_dir, explainer, backbone, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    all_nodes = list(range(x.size(0)))
    if max_nodes is not None:
        all_nodes = all_nodes[:max_nodes]

    # Skip already-computed nodes
    todo_nodes = []
    for n in all_nodes:
        fpath = os.path.join(out_dir, f"feature_masks_node={n}.pt")
        if not os.path.exists(fpath):
            todo_nodes.append(n)

    print(f"  Total nodes: {len(all_nodes)} | Already done: {len(all_nodes) - len(todo_nodes)} | Remaining: {len(todo_nodes)}")
    if not todo_nodes:
        print("  All explanations already computed. Nothing to do.")
        return

    # Train / load GNN backbone
    model = get_gnn_model(data, dataset_name, data_dir, device)

    # Compute explanations
    x_gpu     = x.to(device)
    ei_gpu    = edge_index.to(device)

    for node_idx in tqdm(todo_nodes, desc=f"{explainer}/{dataset_name}"):
        try:
            if explainer == "Grad":
                phi = compute_grad_explanation(model, x_gpu, ei_gpu, node_idx, device, grad_input=False)
            elif explainer == "GradInput":
                phi = compute_grad_explanation(model, x_gpu, ei_gpu, node_idx, device, grad_input=True)
            elif explainer == "GNNExplainer":
                phi = compute_gnnexplainer_explanation(model, x_gpu, ei_gpu, node_idx, device)
            elif explainer == "GraphLime":
                phi = compute_graphlime_explanation(model, x_gpu, ei_gpu, node_idx, device)
            else:
                raise ValueError(f"Unknown explainer: {explainer}")

            save_node_explanation(phi, explanation_dir, dataset_name, explainer, node_idx, backbone)

        except Exception as e:
            print(f"  WARNING: Failed for node {node_idx}: {e}")
            # Save zeros as fallback so we don't retry broken nodes
            phi = torch.zeros(x.size(1))
            save_node_explanation(phi, explanation_dir, dataset_name, explainer, node_idx, backbone)

    print(f"  Done. Explanations saved to {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Compute GNN explanations for PrivX")
    parser.add_argument("--dataset", type=str, required=True, choices=ALL_DATASETS)
    parser.add_argument(
        "--explainer", type=str, required=True,
        choices=["Grad", "GradInput", "GNNExplainer", "GraphLime"],
    )
    parser.add_argument("--backbone", type=str, default="GCN", choices=["GCN"])
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument(
        "--explanation-dir", type=str, default="../saved_explanations",
        help="Directory for saving/loading per-node explanation files.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=None,
        help="Limit number of nodes to explain (for debugging).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_and_save_explanations(
        dataset_name=args.dataset,
        explainer=args.explainer,
        data_dir=args.data_dir,
        explanation_dir=args.explanation_dir,
        backbone=args.backbone,
        device=args.device,
        max_nodes=args.max_nodes,
    )
