"""
PrivX Workflow Diagram  (v2 – wider canvas, improved spacing)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# ── palette ──────────────────────────────────────────────────────────────────
C = dict(
    data   ="#3A6BA8",
    priv   ="#C9683A",
    model  ="#3F8F58",
    attack ="#B03040",
    eval   ="#6B5EA8",
    expl   ="#7A5C38",
    arrow  ="#2A2A2A",
    bg     ="#F5F5F5",
    text   ="#1A1A1A",
)

# ── helpers ───────────────────────────────────────────────────────────────────
def rgba(name, alpha=0.88):
    r, g, b = mcolors.to_rgb(name)
    return (r, g, b, alpha)

def box(ax, x, y, w, h, title, sub=None, color=C["model"], fs=8.5):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.3, edgecolor=color,
        facecolor=rgba(color, 0.90), zorder=3,
    )
    ax.add_patch(rect)
    dy = 0.013 if sub else 0
    ax.text(x, y + dy, title,
            ha="center", va="center", fontsize=fs,
            fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(x, y - 0.022, sub,
                ha="center", va="center", fontsize=fs - 1.5,
                fontstyle="italic", color="white", zorder=4,
                alpha=0.92)

def arr(ax, x0, y0, x1, y1, color=C["arrow"], lw=1.4,
        style="arc3,rad=0.0", label=None, lfs=7.2, ldx=0.01, ldy=0.0):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, connectionstyle=style), zorder=5)
    if label:
        mx = (x0+x1)/2 + ldx
        my = (y0+y1)/2 + ldy
        ax.text(mx, my, label, ha="left", va="center",
                fontsize=lfs, color=color, zorder=6)

def section(ax, x, y, w, h, color, title, tfs=8):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        linewidth=1.0, edgecolor=color, linestyle="--",
        facecolor=rgba(color, 0.055), zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.010, title,
            ha="center", va="bottom", fontsize=tfs,
            color=color, fontweight="bold", zorder=2)

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

# title
ax.text(0.5, 0.972,
        "PrivX: Privacy-Aware Graph Reconstruction Attack via GNN Explanations",
        ha="center", va="center", fontsize=13.5,
        fontweight="bold", color=C["text"])
ax.text(0.5, 0.945,
        "PrivF uses raw node features (x)   ·   "
        "PrivX replaces them with explanation masks (φ)   ·   "
        "Both share the same 5-phase pipeline",
        ha="center", va="center", fontsize=8.5,
        color="#555555", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION BANDS  (x, y_bottom, width, height)
# ─────────────────────────────────────────────────────────────────────────────
# x-centres of columns:  c1=0.10  c2=0.295  c3=0.475  c4=0.665  c5=0.875
# column half-widths:    hw=0.085

c1, c2, c3, c4, c5 = 0.100, 0.295, 0.475, 0.665, 0.875
hw = 0.085   # box half-width

section(ax, 0.012, 0.04, 0.185, 0.87, C["data"],   "Phase 1 · Data Prep",       tfs=8)
section(ax, 0.025, 0.55, 0.165, 0.29, C["expl"],   "Phase 1b · Explanation\n(PrivX only)", tfs=7.5)
section(ax, 0.205, 0.04, 0.180, 0.87, C["model"],  "Phase 2 · Model",           tfs=8)
section(ax, 0.393, 0.04, 0.185, 0.87, C["priv"],   "Phase 3 · Training",        tfs=8)
section(ax, 0.586, 0.04, 0.220, 0.87, C["attack"], "Phase 4 · Attack",          tfs=8)
section(ax, 0.814, 0.04, 0.178, 0.87, C["eval"],   "Phase 5 · Evaluation",      tfs=8)

# ── row y-positions ────────────────────────────────────────────────────────
bh = 0.072   # standard box height
gap = 0.095  # row pitch
rows = [0.87 - i*gap for i in range(9)]   # rows[0]=top … rows[8]=bottom
# rows: 0.870 0.775 0.680 0.585 0.490 0.395 0.300 0.205 0.110

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 – Data
# ─────────────────────────────────────────────────────────────────────────────
r = rows
box(ax, c1, r[0], 2*hw, bh, "Graph Dataset",
    "Cora · CiteSeer · Amazon · Reddit …", color=C["data"])
box(ax, c1, r[1], 2*hw, bh, "BFS Subgraph Sampling",
    "N=32 nodes · K=5 000 subgraphs", color=C["data"])
box(ax, c1, r[2], 2*hw, 0.063, "Train / Test Split  (80 / 20)", color=C["data"], fs=8.5)

arr(ax, c1, r[0]-bh/2, c1, r[1]+bh/2)
arr(ax, c1, r[1]-bh/2, c1, r[2]+0.031)

# Phase 1b – PrivX
box(ax, c1, r[3], 2*hw, 0.063, "GNN Backbone",
    "GCN / GraphSAGE / GIN", color=C["expl"], fs=8.5)
box(ax, c1, r[4], 2*hw, 0.063, "Explainer  →  φ",
    "Grad · GradInput · GNNExplainer · LIME", color=C["expl"], fs=8.5)

arr(ax, c1, r[2]-0.031, c1, r[3]+0.031,
    color=C["expl"], label="PrivX only", lfs=7, ldx=0.012)
arr(ax, c1, r[3]-0.031, c1, r[4]+0.031, color=C["expl"])

# Feature Key
box(ax, c1, r[5], 2*hw, 0.063, "Feature Key",
    "PrivF: x   |   PrivX: φ", color=C["data"], fs=8.5)
arr(ax, c1, r[4]-0.031, c1, r[5]+0.031,
    color=C["expl"], label="φ", lfs=8, ldx=0.010)
# PrivF bypass arrow (left side)
ax.annotate("", xy=(c1-hw+0.012, r[5]+0.031),
            xytext=(c1-hw+0.012, r[2]-0.031),
    arrowprops=dict(arrowstyle="-|>", color=C["data"],
                    lw=1.1, connectionstyle="arc3,rad=0.0"), zorder=5)
ax.text(c1-hw-0.002, (r[2]+r[5])/2, "PrivF\n(x)",
        ha="right", va="center", fontsize=7, color=C["data"], style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 – Model
# ─────────────────────────────────────────────────────────────────────────────
box(ax, c2, r[0], 2*hw, bh,  "Discrete Diffusion",
    "β schedule · q(xₜ|x₀)",          color=C["model"])
box(ax, c2, r[1], 2*hw, bh,  "Self-Attention",
    "4 heads · LayerNorm",             color=C["model"])
box(ax, c2, r[2], 2*hw, bh,  "GNN Layers",
    "GIN / GCN / GraphSAGE",           color=C["model"])
box(ax, c2, r[3], 2*hw, bh,  "Feature Projector",
    "MLP:  F  →  hidden_dim",          color=C["model"])
box(ax, c2, r[4], 2*hw, 0.063, "Pairwise Decoder",
    "[hᵢ ‖ hⱼ]  →  edge logits",      color=C["model"], fs=8.5)
for i in range(4):
    arr(ax, c2, r[i]-bh/2, c2, r[i+1]+bh/2)
arr(ax, c2, r[3]-bh/2, c2, r[4]+0.031)

ax.text(c2 + hw + 0.008, (r[0]+r[4])/2,
        "ConditionalDenseGNN",
        ha="left", va="center", fontsize=7, color=C["model"],
        rotation=90, style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 – Training
# ─────────────────────────────────────────────────────────────────────────────
box(ax, c3, r[0], 2*hw, bh,  "Sample Subgraph",
    "{adj,  x / φ}",                   color=C["priv"])
box(ax, c3, r[1], 2*hw, bh,  "DP Noise  →  x̃",
    "Gaussian · Laplace · Rényi",      color=C["priv"])
box(ax, c3, r[2], 2*hw, bh,  "Forward Diffusion",
    "x₀  →  xₜ   t ~ Uniform[1,T]",   color=C["priv"])
box(ax, c3, r[3], 2*hw, bh,  "Model Forward Pass",
    "logits = f(xₜ, x̃, t ; θ)",       color=C["model"])
box(ax, c3, r[4], 2*hw, 0.063, "Loss = CE + λ · Sparsity",
                                        color=C["priv"], fs=8.5)
box(ax, c3, r[5], 2*hw, 0.060, "AdamW  +  Cosine-Warmup LR",
                                        color=C["model"], fs=8.5)
for y0, y1 in [(r[0]-bh/2, r[1]+bh/2),(r[1]-bh/2, r[2]+bh/2),
               (r[2]-bh/2, r[3]+bh/2),(r[3]-bh/2, r[4]+0.031),
               (r[4]-0.031, r[5]+0.030)]:
    arr(ax, c3, y0, c3, y1)

# back-prop arc
bx = c3 - hw - 0.012
arr(ax, bx, r[5]-0.030, bx, r[3]+bh/2, color=C["priv"], lw=1.0)
arr(ax, bx, r[3]+bh/2, c3-hw, r[3]+bh/2, color=C["priv"], lw=1.0)
ax.text(bx - 0.008, (r[3]+r[5])/2, "∂L/∂θ",
        ha="right", va="center", fontsize=7.5,
        color=C["priv"], style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 – Attack
# ─────────────────────────────────────────────────────────────────────────────
bw4 = 0.195  # slightly wider for attack column
box(ax, c4, r[0], bw4, bh,    "Test Sample",
    "{true_adj,  true_x / φ}",         color=C["attack"])
box(ax, c4, r[1], bw4, bh,    "DP Noise at  ε_test",
    "x̃_test  (may differ from ε_train)",color=C["attack"])
box(ax, c4, r[2], bw4, 0.065, "Init  xₜ ~ Categorical(0.5)",
                                        color=C["attack"], fs=8.5)
box(ax, c4, r[3], bw4, bh,    "Reverse Diffusion  (T steps)",
    "logits = f(xₜ, x̃_test, t ; θ*)", color=C["attack"])
box(ax, c4, r[4], bw4, 0.063, "Gradient Guidance (opt.)",
    "Homophily / Heterophily scale",    color=C["attack"], fs=8.5)
box(ax, c4, r[5], bw4, 0.063, "Posterior Sampling",
    "x_{t−1} ~ P(x_{t−1}|xₜ, x̂₀)",  color=C["attack"], fs=8.5)
box(ax, c4, r[6], bw4, 0.063, "Reconstructed Adjacency  x̂₀",
                                        color=C["attack"], fs=8.5)

for y0, y1 in [(r[0]-bh/2, r[1]+bh/2),(r[1]-bh/2, r[2]+0.032),
               (r[2]-0.032, r[3]+bh/2),(r[3]-bh/2, r[4]+0.031),
               (r[4]-0.031, r[5]+0.031),(r[5]-0.031, r[6]+0.031)]:
    arr(ax, c4, y0, c4, y1)

# loop arc
lx4 = c4 + bw4/2 + 0.010
arr(ax, lx4, r[5]-0.031, lx4, r[3]+bh/2, color=C["attack"], lw=1.0)
arr(ax, lx4, r[3]+bh/2, c4+bw4/2, r[3]+bh/2, color=C["attack"], lw=1.0)
ax.text(lx4 + 0.006, (r[3]+r[5])/2, "t = T−1 … 0",
        ha="left", va="center", fontsize=7, color=C["attack"], style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────
box(ax, c5, r[0], 2*hw, bh,    "Compare  x̂₀  vs  true_adj",
                                        color=C["eval"])
box(ax, c5, r[1], 2*hw, bh,    "AP  ·  AUC  ·  Micro-F1",
    "DegCorr (Spearman ρ)",            color=C["eval"])
box(ax, c5, r[2], 2*hw, bh,    "Ablation Grid",
    "ε ∈ {0.1, 0.5, 1, 2, 5, 8, 16, ∞}",color=C["eval"])
box(ax, c5, r[3], 2*hw, bh,    "Privacy-Utility Curve",
    "AP / AUC  vs  ε",                 color=C["eval"])
box(ax, c5, r[4], 2*hw, 0.063, "Save  results/…/ablation.csv",
                                        color=C["eval"], fs=8.5)
for i in range(4):
    arr(ax, c5, r[i]-bh/2, c5, r[i+1]+bh/2)
arr(ax, c5, r[3]-bh/2, c5, r[4]+0.031)

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PHASE ARROWS
# ─────────────────────────────────────────────────────────────────────────────
cross_y = r[5]  # feature key → training
arr(ax, c1+hw, cross_y, c3-hw, cross_y,
    color=C["data"], label="x / φ  (normalised)", lfs=7, ldx=0.008)

# Phase 2 model def → Phase 3 (row 2/3)
arr(ax, c2+hw, r[2], c3-hw, r[2], color=C["model"])

# Phase 3 → Phase 4  (trained θ*)
arr(ax, c3+hw, r[5]+0.030, c4-bw4/2, r[5]+0.030,
    color=C["model"], label="θ*", lfs=8.5, ldx=0.006)

# Phase 4 → Phase 5  (x̂₀)
arr(ax, c4+bw4/2, r[6], c5-hw, r[0],
    color=C["attack"], style="arc3,rad=-0.25",
    label="x̂₀", lfs=8.5, ldx=0.005, ldy=0.02)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C["data"],   "Data / Dataset"),
    (C["expl"],   "Explanation (PrivX)"),
    (C["model"],  "Model / Architecture"),
    (C["priv"],   "DP Noise / Training"),
    (C["attack"], "Attack / Reconstruction"),
    (C["eval"],   "Evaluation"),
]
lx0, ly = 0.025, 0.028
for i, (col, lbl) in enumerate(legend_items):
    px = lx0 + i * 0.162
    rect = FancyBboxPatch((px, ly - 0.010), 0.018, 0.018,
                          boxstyle="round,pad=0.002",
                          facecolor=col, edgecolor=col, zorder=6)
    ax.add_patch(rect)
    ax.text(px + 0.023, ly - 0.001, lbl,
            va="center", fontsize=7.5, color=C["text"], zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
base = "/mnt/kedargouri/nfs_shared/rishi/project/d3pm/d3pm2/docs/privx_workflow"
fig.savefig(base + ".pdf", bbox_inches="tight", dpi=200, facecolor=C["bg"])
fig.savefig(base + ".png", bbox_inches="tight", dpi=200, facecolor=C["bg"])
print(f"Saved PDF: {base}.pdf")
print(f"Saved PNG: {base}.png")
