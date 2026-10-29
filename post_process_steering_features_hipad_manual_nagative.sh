
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fail2drive

export SAVE_PATH=$LOCAL/results/hipad/test

MODEL_NAME=hipad

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

for FEATURE_NAME in align_query; do
    for LAYER in layer_00 layer_01 layer_02 layer_03 layer_04 layer_05; do
        for CONFIG in \
            "brake Brake brake_manual" \
            "left_change_lane LaneChangeLeft left_manual" \
            "right_change_lane LaneChangeRight right_manual"
        do
            read -r ACTION POSITIVE_SPLIT OUTPUT_NAME <<< "$CONFIG"

            # --negative-include-pattern Normal
            # --negative-include-pattern SteeringVehicleOnRoad
            python post_process_steering_features.py \
                --adapter hipad_plan \
                --model_name "$MODEL_NAME" \
                --model-count 1 \
                --feature_name "$FEATURE_NAME" \
                --layer_name "$LAYER" \
                --collection-root "steering/$MODEL_NAME" \
                --output-dir "steering/$MODEL_NAME/post_process/$OUTPUT_NAME" \
                --action "$ACTION" \
                --manual \
                --positive-include-pattern "$POSITIVE_SPLIT" \
                --manual-negative-frames "steering/hipad/$POSITIVE_SPLIT/manual_negative_frames.json" \
                "${FORCE_FLAG[@]}"
        done
    done
done
