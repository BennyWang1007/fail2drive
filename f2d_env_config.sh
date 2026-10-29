#!/bin/bash
# ==============================================================================
# Fail2Drive Unified Environment Configuration (f2d_env_config.sh)
# ==============================================================================
# Usage in .sbatch or .sh scripts:
#   env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common hipad oracle"
#   eval "$env_var_cmd"
#
# You can also append per-run overrides after load_f2d_env:
#   env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common hipad oracle && export MAX_CARLA_FRAME=2000"
# ==============================================================================

# Base paths (fall back to current working directory if F2D is not set)
export F2D="${F2D:-$PWD}"
export REPO_ROOT="${REPO_ROOT:-$F2D}"

# ------------------------------------------------------------------------------
# 1. Common / General Environment
# ------------------------------------------------------------------------------
env_common() {
    export CARLA_ROOT="${CARLA_ROOT:-$F2D/f2d_carla}"
    export LIVE_VISU="${LIVE_VISU:-0}"
    export DEBUG_CHALLENGE="${DEBUG_CHALLENGE:-1}"
    export DEBUG_LOG_EGO_STAT="${DEBUG_LOG_EGO_STAT:-true}"
    export ORACLE_VERBOSE="${ORACLE_VERBOSE:-true}"
    # export MAX_CARLA_FRAME="${MAX_CARLA_FRAME:-2000}"
}

# ------------------------------------------------------------------------------
# 2. Model Presets (hipad / tfv6)
# ------------------------------------------------------------------------------
env_hipad() {
    export STEERING_VECTORS_DIR="$F2D/steering/hipad/post_process"
    export ACTIVATION_VECTOR_PATHS="$F2D/steering/hipad_new/post_process/brake/steering_vector.pt,$F2D/steering/hipad_new/post_process/left/steering_vector.pt,$F2D/steering/hipad_new/post_process/right/steering_vector.pt"
    export STEERING_FEATURE_NAME="align_query"
    export ACTIVATION_COSINE_GATE="false"
    export HIPAD_STEER_LAYERS="0,1,2,3,4,5"

    export STEERING_ALPHA="1.0"
    export BRAKE_ACTIVATION_ALPHA_SCALE="1.0"
    export LEFT_ACTIVATION_ALPHA_SCALE="3.0"
    export RIGHT_ACTIVATION_ALPHA_SCALE="2.0"
}

env_tfv6() {
    export STEERING_VECTORS_DIR="$F2D/steering/tfv6/post_process"
    export STEERING_FEATURE_NAME="fused_features"
    export STEERING_FEATURE_NAME_SHORT="fused"
    # export STEERING_FEATURE_NAME="decoder_layer4"
    # export STEERING_FEATURE_NAME_SHORT="decoder4"
    # export STEERING_FEATURE_NAME="decoder_layer5"
    # export STEERING_FEATURE_NAME_SHORT="decoder5"
    # export STEERING_FEATURE_NAME="decoder_layer6"
    # export STEERING_FEATURE_NAME_SHORT="decoder6"

    export STEERING_ALPHA="1.0"
    export BRAKE_ACTIVATION_ALPHA_SCALE="1.0"
    export LEFT_ACTIVATION_ALPHA_SCALE="1.1"   # 1.1 for fused/decoder4, 1.5 for decoder5, 0.9 for decoder6
    export RIGHT_ACTIVATION_ALPHA_SCALE="1.1"  # 1.1 for fused/decoder4, 1.5 for decoder5, 0.9 for decoder6
}

# ------------------------------------------------------------------------------
# 3. Policy Presets (none / oracle / oracle_break_only / depth_ttc / vlm)
# ------------------------------------------------------------------------------
env_no_policy() {
    export STEERING_POLICY="none"
    export ACTIVATION_POLICY="none"
}

env_oracle() {
    export STEERING_POLICY="oracle"
    export ACTIVATION_POLICY="pdm_oracle"
    export PDM_ORACLE_ACTION="auto"
    export PDM_ORACLE_ALPHA="1.0"
    export PDM_ORACLE_TRIGGER_DISTANCE="20"

    export PDM_ORACLE_HOLD_FRAMES="30"
    export PDM_ORACLE_HOLD_FRAMES_BRAKE="30"
    export PDM_ORACLE_HOLD_FRAMES_LEFT="10"
    export PDM_ORACLE_HOLD_FRAMES_RIGHT="10"

    export PDM_ORACLE_COOLDOWN_FRAMES="50"
    export PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE="70"
    export PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE="100"
    export PDM_ORACLE_SIDE_HAZARD_DISTANCE="25"
    export PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE="10"
    export PDM_ORACLE_ROADBLOCKED_DISTANCE="40"
    export PDM_ORACLE_PRIORITY_DISTANCE="25"
    export PDM_ORACLE_YIELD_EMERGENCY_DISTANCE="50"
    export PDM_ORACLE_GENERAL_BRAKE="1"

    export PDM_ORACLE_PARKING_EXIT_REAR_TIME_GAP="3.0"
    export PDM_ORACLE_SAME_LANE_LATERAL_MARGIN="2.0"
}

env_oracle_brake_only() {
    env_oracle
    export STEERING_POLICY="oracle_brake_only"
    export ACTIVATION_POLICY="oracle_brake_only"
}

env_depth_ttc() {
    export ACTIVATION_POLICY="depth_ttc"
    export STEERING_POLICY="depth_ttc"
    export DEPTH_TTC_STEERING="1"
    export DEPTH_TTC_VERBOSE="1"
    export DEPTH_TTC_EVERY_N="5"

    export DEPTH_TTC_HOLD_FRAMES="10"
    export DEPTH_TTC_HOLD_FRAMES_BRAKE="2"
    export DEPTH_TTC_HOLD_FRAMES_LEFT="10"
    export DEPTH_TTC_HOLD_FRAMES_RIGHT="10"

    export DEPTH_TTC_BRAKE_ALPHA="1.0"
    export DEPTH_TTC_LEFT_ALPHA="1.0"
    export DEPTH_TTC_RIGHT_ALPHA="1.0"

    export DEPTH_ROOT="${DEPTH_ROOT:-/mnt/bapve/thome/wangyuhao/Depth-Anything-V2}"
    export DEPTH_CHECKPOINT="${DEPTH_CHECKPOINT:-$DEPTH_ROOT/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth}"
}

env_vlm() {
    export ROOT="${ROOT:-$HOME}"
    export ALPAMAYO_ROOT="${ALPAMAYO_ROOT:-$ROOT/alpamayo1.5}"
    export ALPAMAYO_REPO="${ALPAMAYO_REPO:-$ALPAMAYO_ROOT}"
    export ALPAMAYO_PY="${ALPAMAYO_PY:-$ALPAMAYO_ROOT/a1_5_venv/bin/python}"
    export ALPAMAYO_MODEL="${ALPAMAYO_MODEL:-nvidia/Alpamayo-1.5-10B}"
    export ALPAMAYO_VERSION="${ALPAMAYO_VERSION:-1.5}"
    export ALPAMAYO_MAX_NEW_TOKENS="${ALPAMAYO_MAX_NEW_TOKENS:-256}"
    export ALPAMAYO_DECISION_MODE="${ALPAMAYO_DECISION_MODE:-vqa}"
    export ALPAMAYO_HOST="${ALPAMAYO_HOST:-127.0.0.1}"

    export VLM_STEERING="${VLM_STEERING:-1}"
    export VLM_BACKEND="${VLM_BACKEND:-alpamayo_server}"
    export ACTIVATION_POLICY="${ACTIVATION_POLICY:-vlm}"
    export VLM_BRAKE_ALPHA="${VLM_BRAKE_ALPHA:-1.0}"
    export VLM_LATERAL_ALPHA="${VLM_LATERAL_ALPHA:-1.0}"
    export VLM_MIN_SPEED_MPS="${VLM_MIN_SPEED_MPS:-0.0}"
    export VLM_EVERY_N="${VLM_EVERY_N:-5}"
    export VLM_DECISION_TTL_FRAMES="${VLM_DECISION_TTL_FRAMES:-5}"
    export VLM_SERVER_FRAME_OFFSETS="${VLM_SERVER_FRAME_OFFSETS:--6,-4,-2,0}"
    export VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-256}"

    if command -v hf >/dev/null 2>&1 && [ -z "${HF_TOKEN:-}" ]; then
        export HF_TOKEN="$(hf auth token 2>/dev/null || true)"
    fi
}

# ------------------------------------------------------------------------------
# 4. Debug / Closed-Loop Visualization Preset
# ------------------------------------------------------------------------------
env_debug() {
    export LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=true use_kalman_filter=true slower_for_stop_sign=true produce_debug_video=true produce_debug_image=false produce_input_video=false produce_input_image=false produce_frame_frequency=1"
    export ORACLE_VERBOSE="true"
    export VLM_VERBOSE="true"
    export DEBUG_LOG_EGO_STAT="true"
    export DEPTH_TTC_SERVER_VERBOSE="true"
}

# ------------------------------------------------------------------------------
# Profile Loader Entrypoint
# ------------------------------------------------------------------------------
load_f2d_env() {
    for profile in "$@"; do
        case "$profile" in
            common)            env_common ;;
            hipad)             env_hipad ;;
            tfv6)              env_tfv6 ;;
            none|no_policy)    env_no_policy ;;
            oracle)            env_oracle ;;
            oracle_brake_only) env_oracle_brake_only ;;
            depth_ttc)         env_depth_ttc ;;
            vlm)               env_vlm ;;
            debug)             env_debug ;;
            *)
                echo "[f2d_env_config.sh] Error: Unknown profile '$profile'" >&2
                return 1
                ;;
        esac
    done
}