"""
Train a model on combined data from all datasets (Cora, CiteSeer, PubMed).
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math
import time
from datetime import datetime
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
    """Factory function to get the appropriate sanitizer."""
    if noise_type == "gaussian":
        return GaussianDPSanitizer(epsilon, sensitivity)
    elif noise_type == "laplacian":
        return LaplacianDPSanitizer(epsilon, sensitivity)
    else:
        raise ValueError(
            f"Unknown noise type: {noise_type}. Choose 'gaussian' or 'laplacian'."
        )


def load_and_combine_datasets(datasets, data_dir, window_size=32, device="cpu"):
    """
    Load and combine training data from multiple datasets.
    Pads/truncates features to a common dimension.
    """
    all_adj = []
    all_feat = []

    # First pass: find the maximum feature dimension
    max_feat_dim = 0
    dataset_info = {}

    for dataset_name in datasets:
        train_path = os.path.join(data_dir, f"{dataset_name}_train_data.pt")
        if not os.path.exists(train_path):
            print(f"WARNING: {train_path} not found. Skipping {dataset_name}.")
            continue

        data = torch.load(train_path)
        feat_dim = data[0]["x"].shape[1]
        max_feat_dim = max(max_feat_dim, feat_dim)
        dataset_info[dataset_name] = {"data": data, "feat_dim": feat_dim}
        print(f"Loaded {dataset_name}: {len(data)} samples, feat_dim={feat_dim}")

    print(f"\nMax feature dimension: {max_feat_dim}")
    print(f"All features will be padded/truncated to {max_feat_dim} dimensions.\n")

    # Second pass: combine and pad features
    for dataset_name, info in dataset_info.items():
        data = info["data"]
        feat_dim = info["feat_dim"]

        for sample in data:
            adj = sample["adj"]
            feat = sample["x"]

            # Ensure adjacency is the right size
            if adj.shape[0] != window_size:
                print(
                    f"WARNING: Skipping sample with adj size {adj.shape[0]} != {window_size}"
                )
                continue

            # Pad or truncate features to max_feat_dim
            if feat_dim < max_feat_dim:
                # Pad with zeros
                padding = torch.zeros(window_size, max_feat_dim - feat_dim)
                feat = torch.cat([feat, padding], dim=1)
            elif feat_dim > max_feat_dim:
                # Truncate (shouldn't happen since we use max)
                feat = feat[:, :max_feat_dim]

            all_adj.append(adj)
            all_feat.append(feat)

    # Stack into tensors
    adj_tensor = torch.stack(all_adj).to(device)
    feat_tensor = torch.stack(all_feat).to(device)

    return adj_tensor, feat_tensor, max_feat_dim


def train_combined(
    datasets=["Cora", "CiteSeer", "PubMed"],
    data_dir="../data",
    output_dir="./results/combined",
    epochs=1000,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    warmup_pct=0.1,
    sparsity_weight=0.1,
    hidden_dim=128,
    num_layers=4,
    diffusion_steps=100,
    gnn_type="gcn",
    train_epsilon=5.0,
    noise_type="gaussian",
    window_size=32,
    log_interval=10,
    seed=42,
    device="auto",
):
    """Train the conditional diffusion model on combined datasets."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # Load and combine datasets
    print(f"\n{'=' * 60}")
    print("LOADING AND COMBINING DATASETS")
    print(f"{'=' * 60}")
    print(f"Datasets: {datasets}")

    adj_tensor, feat_tensor, feat_dim = load_and_combine_datasets(
        datasets, data_dir, window_size, device
    )

    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(adj_tensor)}")
    print(f"  Adjacency shape: {adj_tensor.shape}")
    print(f"  Feature shape: {feat_tensor.shape}")

    # Normalize features (zero mean, unit variance per feature dimension)
    feat_mean = feat_tensor.mean(dim=(0, 1), keepdim=True)
    feat_std = feat_tensor.std(dim=(0, 1), keepdim=True) + 1e-8
    feat_tensor = (feat_tensor - feat_mean) / feat_std
    print(
        f"Features normalized: mean={feat_tensor.mean().item():.4f}, std={feat_tensor.std().item():.4f}"
    )

    N = adj_tensor.shape[1]  # window_size
    F_dim = feat_tensor.shape[2]

    print(f"\nModel config: N={N}, F={F_dim}")

    diffusion = DiscreteDiffusionBase(num_steps=diffusion_steps, device=device)
    model = ConditionalDenseGNN(
        num_nodes=N,
        feature_dim=F_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # OPTIMIZER & SCHEDULER
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = (len(adj_tensor) + batch_size - 1) // batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1

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

    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Train epsilon: {train_epsilon}, Noise type: {noise_type}")

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(len(adj_tensor))
        total_loss = 0

        for i in range(0, len(adj_tensor), batch_size):
            # Batching
            idx = perm[i : i + batch_size]
            if len(idx) == 0:
                continue

            x_0 = adj_tensor[idx]
            batch_feats = feat_tensor[idx]
            B = x_0.size(0)

            # Attack Simulation with specified epsilon and noise type
            sanitizer = get_sanitizer(noise_type, epsilon=train_epsilon)
            noisy_feats = sanitizer.sanitize(batch_feats)

            t = torch.randint(0, diffusion.num_steps, (B,), device=device)
            x_t = diffusion.q_sample(x_0, t)

            # Forward
            logits = model(x_t, noisy_feats, t)

            # Primary loss
            loss = F.cross_entropy(logits.view(-1, 2), x_0.long().view(-1))

            # Sparsity regularization (match true edge density)
            edge_probs = F.softmax(logits, dim=-1)[:, :, :, 1]
            target_density = x_0.float().mean()
            pred_density = edge_probs.mean()
            sparsity_loss = (pred_density - target_density).abs()

            loss = loss + sparsity_weight * sparsity_loss

            optimizer.zero_grad()
            loss.backward()

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Logging
        current_lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{epochs}: Loss {total_loss:.4f} | LR {current_lr:.6f} | Time {elapsed / 60:.1f}m"
            )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(
        output_dir,
        f"model_combined_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}.pth",
    )
    torch.save(model.state_dict(), model_path)

    # Also save metadata
    metadata = {
        "datasets": datasets,
        "train_epsilon": train_epsilon,
        "noise_type": noise_type,
        "gnn_type": gnn_type,
        "num_samples": len(adj_tensor),
        "feat_dim": F_dim,
        "window_size": N,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "diffusion_steps": diffusion_steps,
        "epochs": epochs,
    }
    metadata_path = os.path.join(
        output_dir,
        f"model_combined_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}_metadata.pt",
    )
    torch.save(metadata, metadata_path)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Total training time: {(time.time() - start_time) / 60:.1f} minutes")

    return model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train D3PM model on combined datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="Cora,CiteSeer,PubMed",
        help="Comma-separated list of datasets to combine",
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--output-dir", type=str, default="./results/combined")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-pct", type=float, default=0.1)
    parser.add_argument("--sparsity-weight", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gcn",
        choices=["gin", "gcn", "sage"],
        help="GNN type: 'gin', 'gcn', or 'sage'",
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
        choices=["gaussian", "laplacian"],
        help="DP noise type: 'gaussian' or 'laplacian'",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    train_combined(
        datasets=datasets,
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
        window_size=args.window_size,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )
