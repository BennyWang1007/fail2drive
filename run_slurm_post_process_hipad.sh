#!/bin/bash
#SBATCH --job-name=hipad_post_process
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

# --------------------------------------------------
# Environment setup
# --------------------------------------------------

mkdir -p logs

# Run from the directory where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fail2drive

# --------------------------------------------------
# Configuration
# --------------------------------------------------

export SAVE_PATH="${LOCAL:?LOCAL is not set}/results/hipad/test"

MODEL_NAME=hipad
SCRIPT="post_process_steering_features.py"
COLLECTION_ROOT="steering/${MODEL_NAME}"
OUTPUT_ROOT="steering/${MODEL_NAME}/post_process"

# Set FORCE=1 when submitting to bypass the cache.
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_FLAG=(--force)
fi

# --------------------------------------------------
# Run post-processing
# --------------------------------------------------

run_post_process() {
    local feature_name="$1"
    local action="$2"
    local positive_split="$3"
    local output_name="$4"
    local manual="$5"

    local manual_flag=()
    if [[ "$manual" == "1" ]]; then
        manual_flag=(--manual)
    fi

    echo "[$(date)] Processing:"
    echo "  Feature: ${feature_name}"
    echo "  Action:  ${action}"
    echo "  Output:  ${OUTPUT_ROOT}/${output_name}"
    echo "  Manual:  ${manual}"

    python -u "$SCRIPT" \
        --adapter hipad_plan \
        --model_name "$MODEL_NAME" \
        --model-count 1 \
        --feature_name "$feature_name" \
        --collection-root "$COLLECTION_ROOT" \
        --output-dir "${OUTPUT_ROOT}/${output_name}" \
        --action "$action" \
        "${manual_flag[@]}" \
        --positive-include-pattern "$positive_split" \
        --negative-include-pattern Normal \
        "${FORCE_FLAG[@]}"
}

FEATURE_NAME=align_query

# First pass: manual processing.
for LAYER in layer_00 layer_01 layer_02 layer_03 layer_04 layer_05; do
    FULL_FEATURE_NAME="${FEATURE_NAME}/${LAYER}"

    while read -r ACTION POSITIVE_SPLIT OUTPUT_NAME; do
        run_post_process \
            "$FULL_FEATURE_NAME" \
            "$ACTION" \
            "$POSITIVE_SPLIT" \
            "$OUTPUT_NAME" \
            1
    done <<'CONFIG'
brake Brake brake_manual
left_change_lane LaneChangeLeft left_manual
right_change_lane Right right_manual
CONFIG

done

# Second pass: non-manual processing.
for LAYER in layer_00 layer_01 layer_02 layer_03 layer_04 layer_05; do
    FULL_FEATURE_NAME="${FEATURE_NAME}/${LAYER}"

    while read -r ACTION POSITIVE_SPLIT OUTPUT_NAME; do
        run_post_process \
            "$FULL_FEATURE_NAME" \
            "$ACTION" \
            "$POSITIVE_SPLIT" \
            "$OUTPUT_NAME" \
            0
    done <<'CONFIG'
brake Brake brake
left_change_lane LaneChangeLeft left
right_change_lane Right right
CONFIG

done

echo "[$(date)] All HiP-AD feature post-processing jobs completed."
