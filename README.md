# Graph Structure Reconstruction from Differentially Private Explanations

This repository contains the implementation of **D3PM** (Diffusion-based Denoising for Privacy-aware Mechanisms), a unified framework for attacking graph neural networks through differentially private explanations and features.

## Overview

Graph structure reconstruction introduces two complementary attack variants:
- **PrivX**: Reconstructs graphs from differentially private GNN explanations
- **PrivF**: Reconstructs graphs from differentially private node features (no GNN/explanation needed)

Both variants leverage the key insight that Gaussian DP mechanisms are structurally equivalent to a single DDPM forward diffusion step, enabling graph reconstruction as a **reverse diffusion** problem conditioned on corrupted signals.

### Key Contributions

1. **Stratified Adversary Model**: A knowledge hierarchy that interpolates between oblivious and oracle attackers
2. **Partition-weighted AUC Bounds**: Tight theoretical bounds linking reconstruction accuracy to graph homophily, explanation fidelity, and privacy budget
3. **Fidelity Gap Discovery**: Surrogate-based explainers (GraphLIME, GNNExplainer) are both more interpretable AND more vulnerable under the same DP budget
4. **Heterophilic Graph Support**: Theoretical characterization showing anti-correlation signals survive DP noise
5. **Comprehensive Evaluation**: Seven benchmark datasets, four explainers, three DP mechanisms with full adaptive-attacker ablations

## Project Structure

```
d3pm2/
├── src/
│   ├── run.py                      # Main runner for PrivF attacks (feature-based)
│   ├── run_exp.py                  # Runner for PrivX attacks (explanation-based)
│   ├── phase_01_data.py            # Data preparation and subgraph extraction
│   ├── phase_03_train.py           # Model training with diffusion
│   ├── phase_05_ablation.py        # Evaluation and ablation studies
│   ├── models/
│   │   ├── diffusion.py            # Diffusion model implementations
│   │   └── gnn.py                  # GNN architectures (GCN, GIN, GraphSAGE)
│   └── utils/
│       ├── dp_mechanisms.py        # DP noise mechanisms (Gaussian, Laplace, Renyi)
│       ├── graph_utils.py          # Graph utilities and metrics
│       └── data_loaders.py         # Dataset loading utilities
├── data/                           # Pre-processed datasets (downloaded automatically)
├── data_exp/                       # Explanation-based datasets
├── saved_explanations/             # Pre-computed GNN explanations
├── results/                        # Training outputs and model checkpoints
├── neurips/
│   └── latex/
│       ├── main.tex               # Main paper (NeurIPS 2026 submission)
│       └── revised_theorems.tex   # Theoretical appendix with proofs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Supported Datasets

### Citation Networks (Homophilic)
- **Planetoid**: Cora, CiteSeer, PubMed

### E-commerce & Social (Mixed)
- **Amazon**: Amazon-Computers, Amazon-Photo
- **Amazon Heterogeneous**: AmazonBook, AmazonProducts

### Heterophilic Networks
- **WebKB**: Texas, Cornell, Wisconsin
- **Wikipedia**: Chameleon, Squirrel
- **Reddit**: Reddit
- **Others**: IMDB, Amazon-ratings

### Large-Scale
- **OGB**: ogbn-arxiv

## Supported Explainers

1. **Grad**: Gradient-based attribution (local)
2. **GradInput**: Gradient × Input attribution (local)
3. **GNNExplainer**: Learned masking (surrogate, global)
4. **GraphLIME**: Local linear approximation (surrogate, global)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support; CPU-only also supported)

### Setup

1. Clone the repository:
```bash
cd /mnt/kedargouri/rishi/project/d3pm/d3pm2
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### PrivF Attack (Feature-based Reconstruction)

Reconstruct graphs from differentially private **raw node features**:

```bash
# Full pipeline: data → train → eval
python src/run.py --dataset Cora --stage all

# Run only training
python src/run.py --dataset Cora --stage train --epochs 1000 --batch-size 64

# Run only evaluation with multiple DP budgets
python src/run.py --dataset Cora --stage eval \
  --epsilons 0.1,0.5,1.0,2.0,5.0,8.0 \
  --num-test-samples 5000
```

### PrivX Attack (Explanation-based Reconstruction)

Reconstruct graphs from differentially private **GNN explanations**:

```bash
# Full pipeline with GNNExplainer
python src/run_exp.py --dataset Cora --explainer GNNExplainer --stage all

# Training with GraphLIME explanations
python src/run_exp.py --dataset CiteSeer --explainer GraphLime \
  --stage train --epochs 1000 --gnn-type gcn

# Evaluation with Grad explanations at various epsilon values
python src/run_exp.py --dataset Cora --explainer Grad \
  --stage eval --epsilons 0.5,1.0,2.0,5.0
```

### Partial Observation and Adaptive Attackers

```bash
# Type II attacker with 50% node observation and 30% epsilon estimation error
python src/run.py --dataset Cora --stage eval \
  --rho 0.5 --kappa 0.3 --num-test-samples 5000

# Oracle attacker (full observation, exact epsilon)
python src/run.py --dataset Cora --stage eval \
  --rho 1.0 --kappa 0.0
```

### Advanced Configuration

```bash
# Custom DP mechanism (Laplace instead of Gaussian)
python src/run.py --dataset CiteSeer \
  --noise-type laplacian \
  --train-epsilon 2.0 \
  --stage train

# Renyi DP with alpha=20
python src/run.py --dataset PubMed \
  --noise-type renyi --alpha 20.0 \
  --train-epsilon 3.0 \
  --stage train

# Large-scale dataset with sparse attention
python src/run.py --dataset ogbn-arxiv \
  --window-size 128 \
  --num-subgraphs 10000 \
  --stage data --force-recreate
```

## Configuration Options

### Data Stage
| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | CiteSeer | Dataset name (see Supported Datasets above) |
| `--num-subgraphs` | 5000 | Number of subgraph samples to generate |
| `--window-size` | 32 | Size of subgraph windows (number of nodes) |
| `--train-pct` | 20 | Percentage for training (rest goes to test) |
| `--force-recreate` | False | Force recreation even if dataset exists |

### Training Stage
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | L2 regularization |
| `--warmup-pct` | 0.1 | Fraction of epochs for LR warmup |
| `--sparsity-weight` | 0.1 | Weight for sparsity loss |
| `--hidden-dim` | 128 | Hidden dimension for model |
| `--num-layers` | 4 | Number of GNN layers |
| `--diffusion-steps` | 100 | Number of diffusion timesteps |
| `--gnn-type` | gin | GNN architecture: 'gin', 'gcn', or 'sage' |
| `--train-epsilon` | 5.0 | Privacy budget ε for DP mechanism |
| `--noise-type` | gaussian | DP noise: 'gaussian', 'laplacian', or 'renyi' |
| `--delta` | 1e-5 | Privacy parameter δ (for Gaussian/Renyi) |
| `--alpha` | 10.0 | Renyi order (for Renyi DP) |

### Evaluation Stage
| Option | Default | Description |
|--------|---------|-------------|
| `--num-test-samples` | 50 | Number of test samples for evaluation |
| `--epsilons` | "0.1,...,16.0" | Comma-separated ε values to evaluate |
| `--guidance-scales` | "0.0" | Comma-separated guidance scale values |
| `--rho` | 1.0 | Fraction of nodes observed (0-1) |
| `--kappa` | 0.0 | Attacker's ε estimation error (0-1) |

## PrivX vs PrivF Comparison

| Aspect | PrivX | PrivF |
|--------|-------|-------|
| **Input** | DP-perturbed explanations | DP-perturbed node features |
| **Requires** | Trained GNN + explainer | Raw node features only |
| **Signal** | Explanation fidelity + homophily | Feature homophily only |
| **Use case** | Measure marginal privacy cost of releasing explanations | Baseline privacy leakage from features |
| **Typical AUC** | 0.6–0.9 (high risk) | 0.5–0.8 (moderate risk) |

## Theoretical Results

### Main Theorem: Partition-Weighted AUC Bounds

For a Type II (adaptive) attacker observing fraction ρ of nodes:

$$\underbrace{R_I + (R_{III} - R_I)\rho^2}_{\text{lower bound}} \leq \overline{R}_{II} \leq \underbrace{R_I + (R_{III} - R_I)(1-(1-\rho)^2)}_{\text{upper bound}}$$

Where:
- **$R_I$**: Oblivious attacker AUC (no DP knowledge)
- **$R_{III}$**: Oracle attacker AUC (full knowledge)
- **$\rho$**: Fraction of nodes observed

### Key Findings

1. **Gap at $\rho=0.5$**: Typically 0.1–0.15 AUC points
2. **Fidelity Gap**: $\gamma_{LIME} - \gamma_{Grad} = \Omega(1/(\sigma\sqrt{d}))$
3. **Heterophilic Bound**: $(1-h)|\rho_{XA}^-| - O(\sigma^2)$ for anti-correlated features
4. **Laplace Approximation**: TV distance to Gaussian is $\leq \min(1, \sqrt{d(\ln\pi-1)/4})$

## Output Format

### Trained Model
- Path: `results/{dataset}/model_{explainer}_{gnn_type}_{epsilon}_{window_size}_{noise_type}.pth`
- Contains: Denoising network weights for the diffusion model

### Evaluation Results
- Path: `results/{dataset}/eval_results_{explainer}_{noise_type}.json`
- Includes: AUC, AP, TPR, FPR across epsilon and guidance scale ranges

### Deprivatized Explanations (PrivX only)
- Path: `results/{dataset}/deprivatized_explanations/`
- Contains: Cleaned explanation features from diffusion denoiser

## References

**Main Paper**: See `neurips/latex/main.tex` (NeurIPS 2026 submission)

**Theoretical Appendix**: See `neurips/latex/revised_theorems.tex`

### Key Related Work

- [DDPM](https://arxiv.org/abs/2006.11239): Denoising Diffusion Probabilistic Models
- [DDRM](https://arxiv.org/abs/2201.11793): Denoising Diffusion Restoration Models
- [GNNExplainer](https://arxiv.org/abs/1905.13686): Explainability in GNNs
- [DP-SGD](https://arxiv.org/abs/1607.00133): Differentially Private Training

## Performance Benchmarks

### Typical Results on Cora (homophilic)

| Mechanism | ε=1.0 | ε=2.0 | ε=5.0 |
|-----------|-------|-------|-------|
| **PrivX (GNNExplainer)** | 0.72 | 0.78 | 0.85 |
| **PrivX (Grad)** | 0.58 | 0.64 | 0.71 |
| **PrivF** | 0.62 | 0.68 | 0.75 |

**Note**: AUC on full-set edges; partition-weighted bounds are tighter.

## System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16 GB minimum (32 GB for ogbn-arxiv)
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA 11.8+)
  - For large datasets (ogbn-arxiv): A100 or equivalent (40GB VRAM)
  - For small-medium (Cora, CiteSeer): RTX 3090 or equivalent (24GB VRAM)

### Disk Space
- **Total**: ~50–100 GB
  - Raw datasets: 5–10 GB
  - Subgraph datasets: 20–40 GB per dataset
  - Saved explanations: 10–20 GB (optional)
  - Models & results: 10–20 GB

## Troubleshooting

### Out of Memory Error
- Reduce `--batch-size` (e.g., 32 or 16)
- Reduce `--window-size` (e.g., 16 or 8)
- Reduce `--num-subgraphs` for data generation

### Slow Data Generation
- Use `--force-recreate` to skip existing datasets
- Reduce `--num-subgraphs` for testing
- Run in parallel by generating different window sizes separately

### Poor Reconstruction Results
- Ensure DP noise scale is not too high (start with `--train-epsilon 5.0`)
- Verify explainer files exist in `saved_explanations/` (for PrivX)
- Check homophily of dataset: heterophilic graphs may have lower AUC

### CUDA Out of Memory
- Use CPU: `--device cpu`
- Reduce model size: `--hidden-dim 64 --num-layers 2`
- Use mixed precision training (requires code modification)




