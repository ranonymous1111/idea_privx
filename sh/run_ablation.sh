#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh — Ablation study runner
#
# Runs all ablation experiments from the checklist:
#   A4.1 Window size ablation
#   A4.2 Train/test split ablation (partial access ρ)
#   A4.3 DP mechanism ablation
#   A4.4 GNN backbone ablation
#   A4.5 Adaptive attacker κ ablation
#
# All results saved to NEW files (existing results not overwritten).
# Requires trained models from run_main.sh.
#
# Usage:
#   bash run_ablation.sh
#   ABLATION=window bash run_ablation.sh    # run only window ablation
# =============================================================================

set -euo pipefail

CONDA_ENV="ddpy"
SRC_DIR="$(cd "$(dirname "$0")/src" && pwd)"
DATA_DIR="$SRC_DIR/data"
RESULTS_DIR="$SRC_DIR/results"
LOG_DIR="$SRC_DIR/../logs"
mkdir -p "$LOG_DIR"

NUM_TEST=200          # fewer samples for ablation to keep runtime manageable
NOISE=gaussian
GNN=gin
TRAIN_EPS=5.0
WINDOW=64
TRAIN_PCT=20
EPSILONS="0.1,0.5,1.0,2.0,5.0,8.0,16.0"
GUIDE_SCALES="0.0,5.0"

eval_dataset() {
    local DS="$1"
    local GPU="$2"
    local OUT_DIR="$RESULTS_DIR/$DS"
    local EXTRA="$3"
    local LOG="$LOG_DIR/ablation_${DS}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$OUT_DIR"

    echo "  eval $DS (GPU $GPU) extra: $EXTRA"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV --no-capture-output \
        python $SRC_DIR/phase_05_ablation.py \
        --dataset $DS \
        --data-dir $DATA_DIR \
        --output-dir "$OUT_DIR" \
        --num-test-samples $NUM_TEST \
        --epsilons "$EPSILONS" \
        --guidance-scales "$GUIDE_SCALES" \
        --gnn-type $GNN \
        --train-epsilon $TRAIN_EPS \
        --noise-type $NOISE \
        $EXTRA \
        2>&1 | tee "$LOG" &
}


# ──────────────────────────────────────────────────────────────────────────────
# A4.1  Window size ablation  (Cora + Texas)
# ──────────────────────────────────────────────────────────────────────────────
run_window_ablation() {
    echo ""
    echo "=== A4.1 Window Size Ablation ==="
    for WS in 32 64 128 256; do
        for DS_GPU in "Cora:0" "Texas:1"; do
            DS="${DS_GPU%%:*}"
            GPU="${DS_GPU##*:}"
            OUT_DIR="$RESULTS_DIR/$DS"
            LOG="$LOG_DIR/ablation_ws${WS}_${DS}_$(date +%Y%m%d_%H%M%S).log"
            mkdir -p "$OUT_DIR"
            echo "  Window=$WS | $DS | GPU $GPU"
            CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV --no-capture-output \
                python $SRC_DIR/phase_05_ablation.py \
                --dataset "$DS" \
                --data-dir "$DATA_DIR" \
                --output-dir "$OUT_DIR" \
                --window-size "$WS" \
                --train-pct "$TRAIN_PCT" \
                --num-test-samples "$NUM_TEST" \
                --epsilons "$EPSILONS" \
                --guidance-scales "$GUIDE_SCALES" \
                --gnn-type "$GNN" \
                --train-epsilon "$TRAIN_EPS" \
                --noise-type "$NOISE" \
                2>&1 | tee "$LOG" &
        done
        wait
    done
    echo "  Window ablation done."
}


# ──────────────────────────────────────────────────────────────────────────────
# A4.2  Train/Test split ablation (CiteSeer)  — controls ρ in theory
# ──────────────────────────────────────────────────────────────────────────────
run_split_ablation() {
    echo ""
    echo "=== A4.2 Train/Test Split Ablation (CiteSeer) ==="
    for SPLIT in 20 40 60 80; do
        OUT_DIR="$RESULTS_DIR/CiteSeer"
        LOG="$LOG_DIR/ablation_split${SPLIT}_CiteSeer_$(date +%Y%m%d_%H%M%S).log"
        mkdir -p "$OUT_DIR"
        echo "  Split=${SPLIT}% | GPU 2"
        CUDA_VISIBLE_DEVICES=2 conda run -n $CONDA_ENV --no-capture-output \
            python $SRC_DIR/phase_05_ablation.py \
            --dataset CiteSeer \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUT_DIR" \
            --window-size "$WINDOW" \
            --train-pct "$SPLIT" \
            --num-test-samples "$NUM_TEST" \
            --epsilons "$EPSILONS" \
            --guidance-scales "$GUIDE_SCALES" \
            --gnn-type "$GNN" \
            --train-epsilon "$TRAIN_EPS" \
            --noise-type "$NOISE" \
            2>&1 | tee "$LOG" &
    done
    wait
    echo "  Split ablation done."
}


# ──────────────────────────────────────────────────────────────────────────────
# A4.3  DP Mechanism Ablation (Cora)
# ──────────────────────────────────────────────────────────────────────────────
run_noise_ablation() {
    echo ""
    echo "=== A4.3 DP Mechanism Ablation (Cora) ==="
    for NTYPE in gaussian laplacian renyi; do
        OUT_DIR="$RESULTS_DIR/Cora"
        LOG="$LOG_DIR/ablation_noise_${NTYPE}_Cora_$(date +%Y%m%d_%H%M%S).log"
        mkdir -p "$OUT_DIR"
        echo "  Noise=$NTYPE | GPU 0"
        CUDA_VISIBLE_DEVICES=0 conda run -n $CONDA_ENV --no-capture-output \
            python $SRC_DIR/phase_05_ablation.py \
            --dataset Cora \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUT_DIR" \
            --window-size "$WINDOW" \
            --train-pct "$TRAIN_PCT" \
            --num-test-samples "$NUM_TEST" \
            --epsilons "$EPSILONS" \
            --guidance-scales "$GUIDE_SCALES" \
            --gnn-type "$GNN" \
            --train-epsilon "$TRAIN_EPS" \
            --noise-type "$NTYPE" \
            2>&1 | tee "$LOG" &
    done
    wait
    echo "  DP mechanism ablation done."
}


# ──────────────────────────────────────────────────────────────────────────────
# A4.4  GNN Backbone Ablation (CiteSeer)
# ──────────────────────────────────────────────────────────────────────────────
run_backbone_ablation() {
    echo ""
    echo "=== A4.4 GNN Backbone Ablation (CiteSeer) ==="
    for GTYPE in gin gcn sage; do
        OUT_DIR="$RESULTS_DIR/CiteSeer"
        LOG="$LOG_DIR/ablation_gnn_${GTYPE}_CiteSeer_$(date +%Y%m%d_%H%M%S).log"
        mkdir -p "$OUT_DIR"
        echo "  GNN=$GTYPE | GPU 1"
        CUDA_VISIBLE_DEVICES=1 conda run -n $CONDA_ENV --no-capture-output \
            python $SRC_DIR/phase_05_ablation.py \
            --dataset CiteSeer \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUT_DIR" \
            --window-size "$WINDOW" \
            --train-pct "$TRAIN_PCT" \
            --num-test-samples "$NUM_TEST" \
            --epsilons "$EPSILONS" \
            --guidance-scales "$GUIDE_SCALES" \
            --gnn-type "$GTYPE" \
            --train-epsilon "$TRAIN_EPS" \
            --noise-type "$NOISE" \
            2>&1 | tee "$LOG" &
    done
    wait
    echo "  GNN backbone ablation done."
}


# ──────────────────────────────────────────────────────────────────────────────
# A4.5  Adaptive Attacker κ × ρ Sweep (Cora)
# ──────────────────────────────────────────────────────────────────────────────
run_attacker_ablation() {
    echo ""
    echo "=== A4.5 Adaptive Attacker Ablation (kappa × rho sweep) ==="
    for KAPPA in 0.0 0.1 0.3 1.0; do
        for RHO in 20 40 60 80; do
            OUT_DIR="$RESULTS_DIR/Cora"
            LOG="$LOG_DIR/ablation_kappa${KAPPA}_rho${RHO}_Cora_$(date +%Y%m%d_%H%M%S).log"
            mkdir -p "$OUT_DIR"
            echo "  kappa=$KAPPA rho=$RHO | GPU 3"
            CUDA_VISIBLE_DEVICES=3 conda run -n $CONDA_ENV --no-capture-output \
                python $SRC_DIR/phase_05_ablation.py \
                --dataset Cora \
                --data-dir "$DATA_DIR" \
                --output-dir "$OUT_DIR" \
                --window-size "$WINDOW" \
                --train-pct "$RHO" \
                --num-test-samples "$NUM_TEST" \
                --epsilons "5.0,8.0,16.0" \
                --guidance-scales "0.0,5.0" \
                --gnn-type "$GNN" \
                --train-epsilon "$TRAIN_EPS" \
                --noise-type "$NOISE" \
                --kappa "$KAPPA" \
                2>&1 | tee "$LOG" &
        done
    done
    wait
    echo "  Adaptive attacker ablation done."
}


# ──────────────────────────────────────────────────────────────────────────────
# Select which ablations to run
# ──────────────────────────────────────────────────────────────────────────────
ABLATION="${ABLATION:-all}"

case "$ABLATION" in
    window)   run_window_ablation ;;
    split)    run_split_ablation ;;
    noise)    run_noise_ablation ;;
    backbone) run_backbone_ablation ;;
    attacker) run_attacker_ablation ;;
    all)
        run_window_ablation
        run_split_ablation
        run_noise_ablation
        run_backbone_ablation
        run_attacker_ablation
        ;;
    *)
        echo "Unknown ABLATION=$ABLATION. Use: window|split|noise|backbone|attacker|all"
        exit 1
        ;;
esac

echo ""
echo "=== All ablation studies complete. ==="
echo "Run: python $RESULTS_DIR/collect_results.py to merge all CSVs."
