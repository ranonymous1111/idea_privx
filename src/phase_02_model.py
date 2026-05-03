import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, SAGEConv


class DiscreteDiffusionBase(nn.Module):
    def __init__(
        self, num_classes=2, num_steps=100, beta_min=1e-4, beta_max=0.02, device="cpu"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.device = device

        self.betas = torch.linspace(beta_min, beta_max, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.one_over_k = 1.0 / num_classes

    def q_sample(self, x_0, t):
        # x_0: [B, N, N] (0 or 1)
        alpha_bar_t = self.alphas_bar[t].view(-1, 1, 1, 1)
        x_0_onehot = F.one_hot(x_0.long(), num_classes=self.num_classes).float()
        probs = alpha_bar_t * x_0_onehot + (1.0 - alpha_bar_t) * self.one_over_k
        return torch.distributions.Categorical(probs).sample()

    def compute_posterior_logits(self, x_t, x_0_pred_logits, t):
        # Vectorized Factored Posterior for Uniform Diffusion
        x_0_probs = F.softmax(x_0_pred_logits, dim=-1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_bar_tm1 = self.alphas_bar[torch.clamp(t - 1, 0)].view(-1, 1, 1, 1)

        # q(x_t | x_{t-1}) terms
        prob_stay = (1.0 - beta_t) + beta_t * self.one_over_k
        prob_flip = beta_t * self.one_over_k
        x_t_onehot = F.one_hot(x_t.long(), self.num_classes).float()
        q_xt_xtm1 = x_t_onehot * prob_stay + (1.0 - x_t_onehot) * prob_flip

        # q(x_{t-1} | x_0) terms
        prob_stay_bar = alpha_bar_tm1 + (1.0 - alpha_bar_tm1) * self.one_over_k
        prob_flip_bar = (1.0 - alpha_bar_tm1) * self.one_over_k
        q_xtm1_x0 = x_0_probs * prob_stay_bar + (1.0 - x_0_probs) * prob_flip_bar

        post_probs = q_xt_xtm1 * q_xtm1_x0
        return torch.log(post_probs + 1e-8)


class ConditionalDenseGNN(nn.Module):
    def __init__(
        self, num_nodes, feature_dim=1433, hidden_dim=128, num_layers=4, gnn_type="gin"
    ):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        assert self.gnn_type in ["gin", "gcn", "sage"], (
            f"gnn_type must be 'gin', 'gcn', or 'sage', got {gnn_type}"
        )

        self.time_emb = nn.Embedding(1000, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_nodes, hidden_dim))

        # Feature Projector (Conditions on Noisy Features)
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Self-attention layers (adj-independent) - crucial for early diffusion steps
        self.num_attn_layers = num_layers // 2
        self.self_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                for _ in range(self.num_attn_layers)
            ]
        )
        self.attn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.num_attn_layers)]
        )

        # GNN layers (GINConv, GCNConv, or SAGEConv)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.gnn_layers.append(GINConv(mlp, train_eps=True))
            elif self.gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:  # sage
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, adj_t, features, t):
        # adj_t: [B, N, N]
        # features: [B, N, F]
        A = adj_t.float()
        B, N, _ = A.shape

        # Embeddings
        h = self.pos_emb.repeat(B, 1, 1)
        h = h + self.time_emb(t).unsqueeze(1)
        h = h + self.feat_proj(features)  # Conditional Injection

        # Self-attention layers (adj-independent) - works well at high noise levels
        for attn, norm in zip(self.self_attn_layers, self.attn_norms):
            h_attn, _ = attn(h, h, h)
            h = norm(h + h_attn)  # Pre-norm residual

        ## TODO: This is for sparse matrix
        # Convert adjacency to edge_index for PyG GNN layers
        # Process each graph in batch
        h_out = []
        for b in range(B):
            # Get edge_index from adjacency matrix
            edge_index = A[b].nonzero(as_tuple=False).t().contiguous()  # [2, num_edges]
            h_b = h[b]  # [N, hidden_dim]

            # GNN layers (GINConv or GCNConv)
            for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
                h_b = norm(h_b + gnn(h_b, edge_index))  # Residual + LayerNorm

            h_out.append(h_b)

        h = torch.stack(h_out, dim=0)  # [B, N, hidden_dim]

        # GNN layers using dense adjacency (batched, much faster than edge_index loop)

        ## TODO: For dense message passing
        # for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
        #     # Use dense message passing: h_new = A @ h @ W
        #     # For GCN/GIN-like behavior with dense adjacency
        #     A_norm = A / (A.sum(dim=-1, keepdim=True) + 1e-8) # Row-normalize
        #     h_msg = torch.bmm(A_norm, h)  # [B, N, hidden] - aggregate neighbor features

        #     # Apply the GNN's MLP/linear transformation
        #     if self.gnn_type == "gin":
        #         # GIN: h = MLP((1 + eps) * h + A @ h)
        #         h_combined = h + h_msg  # simplified, eps=0
        #         h_transformed = gnn.nn(h_combined)  # Apply MLP
        #     else:  # gcn
        #         # GCN: h = A @ h @ W
        #         h_transformed = gnn.lin(h_msg)

        #     h = norm(h + F.relu(h_transformed))  # Residual + LayerNorm

        h = self.norm(h)

        # Pairwise Pred
        h_row = h.unsqueeze(2).repeat(1, 1, N, 1)
        h_col = h.unsqueeze(1).repeat(1, N, 1, 1)
        h_pair = torch.cat([h_row, h_col], dim=-1)

        logits = self.final(h_pair)
        logits = (logits + logits.transpose(1, 2)) / 2  # Symmetrize
        return logits
