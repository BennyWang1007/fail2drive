
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fail2drive

export SAVE_PATH=$LOCAL/results/tfv6_resnet34/test

MODEL_NAME=tfv6

# For each (feature, action) combination, post_process_steering_features.py will:
#   - look for steering/$MODEL_NAME/<ActionDir>/picked_frames_<FEATURE_NAME>.json first,
#     falling back to steering/$MODEL_NAME/<ActionDir>/picked_frames.json if that
#     feature-specific file doesn't exist.
#   - skip recomputation entirely (cache hit) if the picked frames and config are
#     unchanged since the last run for that exact feature/action combination.
# Set FORCE=1 to bypass the cache and always recompute.
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_FLAG=(--force)
fi

for FEATURE_NAME in decoder_layer4 decoder_layer5 decoder_layer6 fused_features; do
    for CONFIG in \
        "brake Brake brake_manual" \
        "left_change_lane Left left_manual" \
        "right_change_lane Right right_manual"
    do
        read -r ACTION POSITIVE_SPLIT OUTPUT_NAME <<< "$CONFIG"

        python post_process_steering_features.py \
            --adapter transfuser_target_speed \
            --model_name "$MODEL_NAME" \
            --feature_name "$FEATURE_NAME" \
            --collection-root "steering/$MODEL_NAME" \
            --output-dir "steering/$MODEL_NAME/post_process/$OUTPUT_NAME" \
            --action "$ACTION" \
            --manual \
            --positive-include-pattern "$POSITIVE_SPLIT" \
            --negative-include-pattern Normal \
            "${FORCE_FLAG[@]}"
    done
done