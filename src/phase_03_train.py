import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math
import time
from datetime import datetime

# Local imports (when running from src folder)
try:
    from phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN
except ImportError:
    from .phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN


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

    For alpha > 1, the noise sigma for (alpha, epsilon)-RDP is:
        sigma = sqrt(alpha * sensitivity^2 / (2 * epsilon))

    RDP can be converted to (epsilon, delta)-DP using:
        epsilon_dp = epsilon_rdp + log(1/delta) / (alpha - 1)
    """

    def __init__(self, epsilon, sensitivity=1.0, delta=1e-5, alpha=10.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.alpha = alpha  # Renyi divergence order

        if epsilon == float("inf"):
            self.sigma = 0.0
        else:
            # For (alpha, epsilon_rdp)-RDP with Gaussian mechanism:
            # epsilon_rdp = alpha * sensitivity^2 / (2 * sigma^2)
            # Solving for sigma: sigma = sqrt(alpha * sensitivity^2 / (2 * epsilon_rdp))
            #
            # We need to convert from (epsilon, delta)-DP to epsilon_rdp:
            # epsilon_rdp = epsilon - log(1/delta) / (alpha - 1)
            epsilon_rdp = max(epsilon - np.log(1 / delta) / (alpha - 1), 0.01)
            self.sigma = np.sqrt(alpha * (sensitivity**2) / (2 * epsilon_rdp))

    def sanitize(self, x):
        return x + torch.randn_like(x) * self.sigma if self.sigma > 0 else x


def get_sanitizer(noise_type, epsilon, sensitivity=1.0, delta=1e-5, alpha=10.0):
    """Factory function to get the appropriate sanitizer.

    Args:
        noise_type: Type of DP mechanism ('gaussian', 'laplacian', or 'renyi')
        epsilon: Privacy budget (lower = more private)
        sensitivity: L2 sensitivity of the query
        delta: Failure probability for approximate DP (used by gaussian and renyi)
        alpha: Renyi divergence order (only used by renyi)

    Returns:
        DPSanitizer instance
    """
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


def train(
    dataset_name="CiteSeer",
    data_dir="../data",
    output_dir="./results/CiteSeer",
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
    window_size=32,
    train_pct=20,
    use_explanations=False,
    explainer="none",
    explanation_dir="../saved_explanations",
):
    """Train the conditional diffusion model."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # LOAD TRAIN SPLIT from data/ directory - use new naming convention
    train_path = os.path.join(
        data_dir, f"{dataset_name}_train_data_{window_size}_{train_pct}.pt"
    )
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run phase_01_data.py first.")
        return None
    data = torch.load(train_path)
    print(f"Loaded training data from {train_path}")

    adj_tensor = torch.stack([d["adj"] for d in data]).to(device)

    # PrivX mode: use GNN explanations as conditioning signal
    if use_explanations and explainer != "none":
        if all("phi" in d for d in data):
            feat_tensor = torch.stack([d["phi"] for d in data]).to(device)
            print(f"[PrivX] Using '{explainer}' explanations as conditioning (phi).")
        else:
            print(f"[PrivX] WARNING: 'phi' key missing in data — falling back to raw features.")
            feat_tensor = torch.stack([d["x"] for d in data]).to(device)
    else:
        feat_tensor = torch.stack([d["x"] for d in data]).to(device)

    # Normalize features (zero mean, unit variance per feature dimension)
    feat_mean = feat_tensor.mean(dim=(0, 1), keepdim=True)
    feat_std = feat_tensor.std(dim=(0, 1), keepdim=True) + 1e-8
    feat_tensor = (feat_tensor - feat_mean) / feat_std
    print(
        f"Features normalized: mean={feat_tensor.mean().item():.4f}, std={feat_tensor.std().item():.4f}"
    )

    N = adj_tensor.shape[1]
    F_dim = feat_tensor.shape[2]

    print(f"Loaded {len(adj_tensor)} graphs, N={N}, F={F_dim}")

    diffusion = DiscreteDiffusionBase(num_steps=diffusion_steps, device=device)
    model = ConditionalDenseGNN(
        num_nodes=N,
        feature_dim=F_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    ).to(device)

    # OPTIMIZER & SCHEDULER
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # OneCycleLR: Warms up to max_lr then decays. Excellent for Transformers.
    # steps_per_epoch needs to be accurate - use ceiling division to account for partial batches.
    steps_per_epoch = (len(adj_tensor) + batch_size - 1) // batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1

    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(warmup_pct * total_steps))

    def lr_lambda(step):
        # step is 0-based; during warmup ramp from 0->1 linearly
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        # cosine decay from 1 -> 0 over remaining steps
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=LR,
    #     epochs=EPOCHS,
    #     steps_per_epoch=steps_per_epoch,
    #     pct_start=0.1 # Warmup for first 10% of training
    # )

    print(f"Training on {len(adj_tensor)} graphs...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
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
            sanitizer = get_sanitizer(
                noise_type, epsilon=train_epsilon, delta=delta, alpha=alpha
            )
            noisy_feats = sanitizer.sanitize(batch_feats)

            t = torch.randint(0, diffusion.num_steps, (B,), device=device)
            x_t = diffusion.q_sample(x_0, t)

            # Forward
            logits = model(x_t, noisy_feats, t)

            # Primary loss
            loss = F.cross_entropy(logits.view(-1, 2), x_0.long().view(-1))

            # Fix #4: Sparsity regularization (match true edge density)
            edge_probs = F.softmax(logits, dim=-1)[:, :, :, 1]
            target_density = x_0.float().mean()
            pred_density = edge_probs.mean()
            sparsity_loss = (pred_density - target_density).abs()

            loss = loss + sparsity_weight * sparsity_loss

            optimizer.zero_grad()
            loss.backward()

            # Clip Gradients for Transformer stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  ##TODO: Check transformr part

            optimizer.step()
            scheduler.step()  # Update LR per step

            total_loss += loss.item()

        # Logging
        current_lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{epochs}: Loss {total_loss:.4f} | LR {current_lr:.6f} | Time {elapsed / 60:.1f}m"
            )

    os.makedirs(output_dir, exist_ok=True)

    # Generate unique model name: dataset_gnntype_eps_numsamples_timestamp
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_samples = len(adj_tensor)
        run_name = f"{dataset_name}_{gnn_type}_eps{train_epsilon:.1f}_n{num_samples}_{timestamp}"

    exp_tag = f"_{explainer}" if use_explanations and explainer != "none" else ""
    model_path = os.path.join(
        output_dir,
        f"model_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}{exp_tag}.pth",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Total training time: {(time.time() - start_time) / 60:.1f} minutes")

    return model_path, run_name


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train D3PM model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CiteSeer",
        choices=ALL_DATASETS,
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--output-dir", type=str, default="./results/CiteSeer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-pct", type=float, default=0.1)
    parser.add_argument("--sparsity-weight", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gin",
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
        choices=["gaussian", "laplacian", "renyi"],
        help="DP noise type: 'gaussian', 'laplacian', or 'renyi'",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Delta parameter for (epsilon, delta)-DP (used by gaussian and renyi)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Renyi divergence order for Renyi DP (only used when noise-type='renyi')",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Window size for subgraph sampling (default: 32)",
    )
    parser.add_argument(
        "--train-pct",
        type=int,
        default=20,
        help="Train split percentage (default: 20)",
    )
    parser.add_argument(
        "--use-explanations",
        action="store_true",
        help="Use GNN explanations as conditioning (PrivX mode). Default: raw features (PrivF mode).",
    )
    parser.add_argument(
        "--explainer",
        type=str,
        default="none",
        choices=["none", "Grad", "GradInput", "GNNExplainer", "GraphLime"],
        help="Explainer for PrivX mode (requires --use-explanations).",
    )
    parser.add_argument(
        "--explanation-dir",
        type=str,
        default="../saved_explanations",
        help="Directory containing saved explanations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_name=args.dataset,
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
        delta=args.delta,
        alpha=args.alpha,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        window_size=args.window_size,
        train_pct=args.train_pct,
        use_explanations=args.use_explanations,
        explainer=args.explainer,
        explanation_dir=args.explanation_dir,
    )
