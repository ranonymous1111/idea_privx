import torch
import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve, auc
from scipy.stats import spearmanr

# Local imports (when running from src folder)
try:
    from phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN
except ImportError:
    from .phase_02_model import DiscreteDiffusionBase, ConditionalDenseGNN


# def random_baseline(
#     seed=42,
#     sample_idx=0,
#     dataset_name="CiteSeer",
#     data_dir="./data",
#     device="auto",
#     window_size=32,
#     train_pct=20,
# ):
#     """Random baseline: predicts random probabilities for edges."""
#     # Device selection
#     if device == "auto":
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)

#     # LOAD TEST SPLIT from data/ directory - use new naming convention
#     test_pct = 100 - train_pct
#     test_path = os.path.join(
#         data_dir, f"{dataset_name}_test_data_{window_size}_{test_pct}.pt"
#     )
#     if not os.path.exists(test_path):
#         print(f"Run phase_01_data.py first! ({test_path} not found)")
#         return 0, 0, 0

#     data = torch.load(test_path)
#     if sample_idx >= len(data):
#         sample_idx = 0
#     sample = data[sample_idx]

#     true_adj = sample["adj"].to(device)
#     N = true_adj.shape[0]

#     # Random predictions
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random_probs = torch.rand(N, N, device=device)
#     random_probs = (random_probs + random_probs.T) / 2  # Symmetrize

#     # Evaluate
#     mask = ~torch.eye(N, dtype=torch.bool, device=device)
#     y_true = true_adj[mask].cpu().numpy()
#     y_score = random_probs[mask].cpu().numpy()

#     # Safety check for flat graphs
#     if len(np.unique(y_true)) < 2:
#         return 0.5, 0.5, 0.0, 0.0

#     ## TODO: Use micro averaging to be consistent with other metrics [To handle the imbanaced classes better]
#     ap = average_precision_score(y_true, y_score)
#     auc = roc_auc_score(y_true, y_score)

#     # Micro F1 score (binarize predictions at threshold 0.5)
#     y_pred = (y_score >= 0.5).astype(int)
#     micro_f1 = f1_score(y_true, y_pred, average="micro")

#     # Degree correlation metric
#     true_degrees = true_adj.sum(dim=1).cpu().numpy()
#     pred_degrees = random_probs.sum(dim=1).cpu().numpy()
#     degree_corr, _ = spearmanr(true_degrees, pred_degrees)

#     return ap, auc, degree_corr, micro_f1


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


def homophily_grad(logits, features, temperature=1.0, heterophilic=False):
    """
    Improved homophily/heterophily gradient guidance.
    - homophilic (default): edges connect similar nodes (positive similarity)
    - heterophilic: edges connect dissimilar nodes (anti-correlation signal)
    """
    probs = F.softmax(logits / temperature, dim=-1)[:, :, :, 1]

    # Normalize features for cosine similarity (robust to noise scale)
    feat_norm = F.normalize(features, p=2, dim=-1)
    similarity = torch.mm(feat_norm, feat_norm.T)  # [N, N], range [-1, 1]

    if heterophilic:
        # Heterophilic: edges connect dissimilar nodes — use anti-correlation
        energy = -(probs * (-similarity).unsqueeze(0)).sum()
    else:
        # Homophilic: edges connect similar nodes
        energy = -(probs * similarity.unsqueeze(0)).sum()

    return torch.autograd.grad(energy, logits)[0]


def load_model(
    dataset_name="CiteSeer",
    data_dir="../data",
    output_dir="./results/CiteSeer",
    hidden_dim=128,
    num_layers=4,
    diffusion_steps=100,
    gnn_type="gin",
    noise_type="gaussian",
    train_epsilon=5.0,
    device="auto",
    window_size=32,
    train_pct=20,
    use_explanations=False,
    explainer="none",
    explanation_backbone="GCN",
):
    """Load model and diffusion once for efficient batch evaluation."""
    # Device selection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load test data to get dimensions - use new naming convention
    test_pct = 100 - train_pct
    if use_explanations and explainer != "none":
        # PrivX: explanation-based data files include explainer and backbone in name
        test_path = os.path.join(
            data_dir,
            f"{dataset_name}_{explainer}_{explanation_backbone}_test_data_{window_size}_{test_pct}.pt",
        )
    else:
        test_path = os.path.join(
            data_dir, f"{dataset_name}_test_data_{window_size}_{test_pct}.pt"
        )
    if not os.path.exists(test_path):
        print(f"Run phase_01_data.py first! ({test_path} not found)")
        return None, None, None, None, None, None

    data = torch.load(test_path)
    sample = data[0]
    N = sample["adj"].shape[0]

    # PrivX mode: use phi (explanations) if available
    feat_key = "phi" if (use_explanations and explainer != "none" and "phi" in sample) else "x"
    if feat_key == "phi":
        print(f"[PrivX] load_model: using explanation key 'phi' (explainer={explainer})")
    F_dim = sample[feat_key].shape[1]

    # Pre-compute normalization stats
    all_feats = torch.stack([d[feat_key] for d in data])
    feat_mean = all_feats.mean(dim=(0, 1), keepdim=True)
    feat_std = all_feats.std(dim=(0, 1), keepdim=True) + 1e-8

    # Create model and diffusion
    diffusion = DiscreteDiffusionBase(num_steps=diffusion_steps, device=device)
    model = ConditionalDenseGNN(
        num_nodes=N,
        feature_dim=F_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    ).to(device)

    # Load model weights
    # PrivX: model saved as model_{explainer}_{gnn_type}_{epsilon}_{window}_{noise}.pth
    # PrivF: model saved as model_{gnn_type}_{epsilon}_{window}_{noise}.pth
    if use_explanations and explainer != "none":
        model_path = os.path.join(
            output_dir,
            f"model_{explainer}_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}.pth",
        )
    else:
        model_path = os.path.join(
            output_dir,
            f"model_{gnn_type}_{train_epsilon:.1f}_{window_size}_{noise_type}.pth",
        )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Run training first! ({model_path} not found or incompatible: {e})")
        return None, None, None, None, None, None
    model.eval()

    norm_stats = (feat_mean, feat_std)
    return model, diffusion, data, norm_stats, device, feat_key


def reconstruct_with_model(
    model,
    diffusion,
    data,
    norm_stats,
    epsilon,
    guidance_scale,
    seed=42,
    sample_idx=0,
    noise_type="gaussian",
    delta=1e-5,
    alpha=10.0,
    device="cuda",
    feat_key="x",
    kappa=0.0,
    temperature=1.0,
    heterophilic=False,
    balanced_eval=True,
    rho=1.0,
):
    """Reconstruct graph using pre-loaded model (efficient for batch evaluation).

    Args:
        feat_key: 'x' for PrivF mode, 'phi' for PrivX mode (explanation features)
        kappa: Attacker estimation error factor. Perturbs epsilon by ±kappa fraction.
               kappa=0 means perfect knowledge; kappa=1 means up to 100% error.
        temperature: Temperature for logit scaling before evaluation (default 1.0).
        heterophilic: If True, use anti-correlation guidance (for heterophilic graphs).
        rho: Partial observation fraction in (0, 1]. The attacker observes features for
             only rho fraction of nodes; the rest are zeroed out. rho=1.0 means full
             observation (default behaviour, no masking).
    """
    from sklearn.metrics import precision_recall_curve

    if sample_idx >= len(data):
        sample_idx = 0
    sample = data[sample_idx]

    feat_mean, feat_std = norm_stats

    true_adj = sample["adj"].to(device)
    # Use feat_key: 'phi' for PrivX, 'x' for PrivF
    true_feat = sample.get(feat_key, sample["x"]).to(device)
    # Apply normalization
    true_feat = (true_feat - feat_mean[0, 0].to(device)) / feat_std[0, 0].to(device)
    N = true_adj.shape[0]

    # Apply DP to Test Features
    # kappa: adaptive attacker with imperfect epsilon knowledge
    torch.manual_seed(seed)
    np.random.seed(seed)
    if kappa > 0:
        perturbation = 1.0 + kappa * (2 * np.random.rand() - 1)
        epsilon_used = epsilon * perturbation
    else:
        epsilon_used = epsilon
    sanitizer = get_sanitizer(noise_type, epsilon=epsilon_used, delta=delta, alpha=alpha)
    dp_feats = sanitizer.sanitize(true_feat).unsqueeze(0)  # [1, N, F]

    # rho: partial observation — zero out (1-rho) fraction of conditioning features
    # Simulates attacker observing only rho fraction of nodes (Table 3 ablation)
    if rho < 1.0:
        torch.manual_seed(seed + 9999)  # reproducible, distinct from kappa seed
        num_observed = max(1, int(round(rho * N)))
        perm = torch.randperm(N)
        unobserved = perm[num_observed:]
        dp_feats = dp_feats.clone()
        dp_feats[0, unobserved, :] = 0.0

    # Sampling
    x_t = torch.randint(0, 2, (1, N, N)).to(device)

    with torch.no_grad():
        for t_idx in reversed(range(diffusion.num_steps)):
            t = torch.full((1,), t_idx, device=device, dtype=torch.long)

            # 1. Conditional Prediction
            logits = model(x_t, dp_feats, t)

            # 2. Gradient Guidance (homophily or heterophily)
            if guidance_scale > 0:
                with torch.enable_grad():
                    logits.requires_grad = True
                    grad = homophily_grad(logits, dp_feats[0], heterophilic=heterophilic)
                logits = logits - guidance_scale * grad
                logits = logits.detach()

            # 3. Posterior Step
            log_post = diffusion.compute_posterior_logits(x_t, logits, t)
            x_t = torch.distributions.Categorical(torch.exp(log_post)).sample()

    # Evaluate — apply temperature scaling before softmax
    probs = F.softmax(logits / temperature, dim=-1)[0, :, :, 1]
    mask = ~torch.eye(N, dtype=torch.bool, device=device)

    y_true = true_adj[mask].cpu().numpy()
    y_score = probs[mask].cpu().numpy()

    if balanced_eval:
        # Match src_baseline-style balanced evaluation by downsampling negatives
        # so positives and negatives contribute equally to AUROC/AP.
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

    # Baseline-style AUROC/AP computation
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)

    # Optimal threshold via precision-recall curve (handles sparse graphs better)
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
    pred_degrees = probs.sum(dim=1).cpu().numpy()
    degree_corr, _ = spearmanr(true_degrees, pred_degrees)

    return ap, auc_score, degree_corr, micro_f1


## TODO: This is the old function that has been replaced by the above two functions for efficiency
# def reconstruct(
#     epsilon,
#     guidance_scale,
#     seed=42,
#     sample_idx=0,
#     dataset_name="CiteSeer",
#     data_dir="./data",
#     output_dir="./results/CiteSeer",
#     hidden_dim=128,
#     num_layers=4,
#     diffusion_steps=100,
#     gnn_type="gin",
#     train_epsilon=5.0,
#     device="auto",
# ):
#     """Reconstruct graph from DP-noised features using trained model."""
#     # Device selection
#     if device == "auto":
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)

#     # LOAD TEST SPLIT from data/ directory
#     test_path = os.path.join(data_dir, f"{dataset_name}_test_data.pt")
#     if not os.path.exists(test_path):
#         print(f"Run phase_01_data.py first! ({test_path} not found)")
#         return 0, 0, 0

#     data = torch.load(test_path)
#     if sample_idx >= len(data):
#         sample_idx = 0
#     sample = data[sample_idx]

#     # Normalize features (same as training)
#     all_feats = torch.stack([d["x"] for d in data])
#     feat_mean = all_feats.mean(dim=(0, 1), keepdim=True)
#     feat_std = all_feats.std(dim=(0, 1), keepdim=True) + 1e-8

#     true_adj = sample["adj"].to(device)
#     true_feat = sample["x"].to(device)
#     # Apply normalization
#     true_feat = (true_feat - feat_mean[0, 0].to(device)) / feat_std[0, 0].to(device)
#     N = true_adj.shape[0]
#     F_dim = true_feat.shape[1]

#     # Apply DP to Test Features
#     torch.manual_seed(seed)
#     sanitizer = DPSanitizer(epsilon=epsilon)
#     dp_feats = sanitizer.sanitize(true_feat).unsqueeze(0)  # [1, N, F]

#     # Load Model
#     diffusion = DiscreteDiffusionBase(num_steps=diffusion_steps, device=device)
#     model = ConditionalDenseGNN(
#         num_nodes=N,
#         feature_dim=F_dim,
#         hidden_dim=hidden_dim,
#         num_layers=num_layers,
#         gnn_type=gnn_type,
#     ).to(device)

#     # # Try to find a model file matching the gnn_type
#     # # Priority: 1. Exact match with gnn_type in name, 2. model_cond.pth (legacy), 3. Last resort fallback
#     # model_path = None
#     model_path = os.path.join(output_dir, f"model_{gnn_type}_{train_epsilon:.1f}.pth")
#     # if os.path.exists(output_dir):
#     #     model_files = [f for f in os.listdir(output_dir) if f.startswith("model_") and f.endswith(".pth")]

#     #     # Look for models with matching gnn_type in filename
#     #     matching_models = [f for f in model_files if f"_{gnn_type}_" in f or f"_{gnn_type.upper()}_" in f]
#     #     if matching_models:
#     #         # Use the most recent one (last alphabetically due to timestamp)
#     #         model_path = os.path.join(output_dir, sorted(matching_models)[-1])
#     #     elif "model_cond.pth" in model_files:
#     #         # Fallback to legacy model name
#     #         model_path = os.path.join(output_dir, "model_cond.pth")

#     # # Last resort if no model found
#     # if model_path is None:
#     #     model_path = os.path.join(output_dir, "model_cond.pth")

#     try:
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         print(f"Loaded model from {model_path}")
#     except Exception as e:
#         print(f"Run training first! ({model_path} not found or incompatible: {e})")
#         return 0, 0, 0
#     model.eval()

#     # Sampling
#     x_t = torch.randint(0, 2, (1, N, N)).to(device)  ##TODO: Noise featurSim

#     with torch.no_grad():
#         for t_idx in reversed(range(diffusion.num_steps)):
#             t = torch.full((1,), t_idx, device=device, dtype=torch.long)

#             # 1. Conditional Prediction
#             logits = model(x_t, dp_feats, t)

#             # 2. Gradient Guidance
#             if guidance_scale > 0:
#                 with torch.enable_grad():
#                     logits.requires_grad = True
#                     grad = homophily_grad(logits, dp_feats[0])
#                 logits = logits - guidance_scale * grad
#                 logits = logits.detach()

#             # 3. Posterior Step
#             log_post = diffusion.compute_posterior_logits(x_t, logits, t)
#             x_t = torch.distributions.Categorical(torch.exp(log_post)).sample()

#     # Evaluate
#     probs = F.softmax(logits, dim=-1)[0, :, :, 1]
#     mask = ~torch.eye(N, dtype=torch.bool, device=device)

#     y_true = true_adj[mask].cpu().numpy()
#     y_score = probs[mask].cpu().numpy()

#     # Safety check for flat graphs
#     if len(np.unique(y_true)) < 2:
#         return 0.5, 0.5, 0.0

#     ap = average_precision_score(y_true, y_score)
#     auc = roc_auc_score(y_true, y_score)

#     # Fix #5: Degree correlation - how well do we recover node degrees?
#     true_degrees = true_adj.sum(dim=1).cpu().numpy()
#     pred_degrees = probs.sum(dim=1).cpu().numpy()
#     degree_corr, _ = spearmanr(true_degrees, pred_degrees)

#     return ap, auc, degree_corr
