#!/usr/bin/env bash
# =============================================================================
# run_baseline.sh — Cosine baseline (PrivF-Cosine) across all datasets × 3 DP mechanisms
#
# Runs cosine_baseline_subgraph.py for all 11 datasets at 7 epsilon values,
# for all 3 DP mechanisms (gaussian, laplacian, renyi).
# All results saved to new files; existing results NOT overwritten.
#
# Usage:
#   bash run_baseline.sh
#   bash run_baseline.sh --dry-run
#
# GPU assignment: baselines are CPU-heavy, distribute across 4 GPUs
# =============================================================================

set -euo pipefail

CONDA_ENV="ddpy"
SRC_DIR="$(cd "$(dirname "$0")/src" && pwd)"
DATA_DIR="$SRC_DIR/data"
RESULTS_DIR="$SRC_DIR/../results_baseline"
LOG_DIR="$SRC_DIR/../logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

NUM_TEST=500
EPSILONS="0.1,0.5,1.0,2.0,5.0,8.0,16.0"
WINDOW=64
TRAIN_PCT=20

# Dataset → GPU mapping
declare -A DS_GPU
DS_GPU["Cora"]=0
DS_GPU["CiteSeer"]=0
DS_GPU["PubMed"]=1
DS_GPU["Texas"]=1
DS_GPU["Cornell"]=1
DS_GPU["Wisconsin"]=2
DS_GPU["Chameleon"]=2
DS_GPU["Amazon-ratings"]=2
DS_GPU["IMDB"]=3
DS_GPU["AmazonBook"]=3
DS_GPU["ogbn-arxiv"]=3

run_baseline() {
    local DS="$1"
    local GPU="$2"
    local NOISE="$3"
    local LOG="$LOG_DIR/baseline_${DS}_${NOISE}_$(date +%Y%m%d_%H%M%S).log"
    local OUT_DIR="$RESULTS_DIR/$DS"
    mkdir -p "$OUT_DIR"

    echo "  [$(date '+%H:%M:%S')] $DS | $NOISE | GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV --no-capture-output \
        python $SRC_DIR/cosine_baseline_subgraph.py \
        --dataset "$DS" \
        --epsilons "$EPSILONS" \
        --noise-type "$NOISE" \
        --num-test-samples "$NUM_TEST" \
        --window-size "$WINDOW" \
        --train-pct "$TRAIN_PCT" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$LOG" &
}


echo "=== Running PrivF-Cosine baselines across all datasets × 3 DP mechanisms ==="
echo ""

for NOISE in gaussian laplacian renyi; do
    echo "--- Noise: $NOISE ---"
    for DS in "${!DS_GPU[@]}"; do
        GPU="${DS_GPU[$DS]}"
        run_baseline "$DS" "$GPU" "$NOISE"
    done
    wait
    echo "  Noise=$NOISE done."
done

echo ""
echo "=== Baselines complete. ==="
echo "Results in: $RESULTS_DIR"
