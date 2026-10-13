#!/bin/bash
#SBATCH --job-name=tfv6_depth_ttc_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:0

# This is only the *controller* job (no GPU needed). It runs
# slurm_evaluate_server.py, which submits one 2-GPU job per route/seed.
# Each of those jobs starts its own Depth-Anything-V2 TTC server + CARLA on the
# same node, so DEPTH_TTC_SERVER_URL is set per job and must NOT be set here.

set -e

mkdir -p logs

# Make sure we are running from the project root
cd "$SLURM_SUBMIT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fail2drive

# ----------------------------------------------------------------------------
# Policy: Depth-TTC (replaces the PDM oracle used in run_tfv6_eval.sbatch)
# ----------------------------------------------------------------------------
export ACTIVATION_POLICY=depth_ttc
export STEERING_POLICY=depth_ttc
export DEPTH_TTC_STEERING=1
export DEPTH_TTC_VERBOSE=1          # set to 1 for per-frame TTC logs
export DEPTH_TTC_EVERY_N=5          # query the depth server every N frames

# Hold frames (kept in sync with the oracle values used in the oracle eval)
export DEPTH_TTC_HOLD_FRAMES=10
export DEPTH_TTC_HOLD_FRAMES_BRAKE=2
export DEPTH_TTC_HOLD_FRAMES_LEFT=10
export DEPTH_TTC_HOLD_FRAMES_RIGHT=10

# Per-action alphas are multiplied by the *_ACTIVATION_ALPHA_SCALE values
# below, so keep these at 1.0 to avoid double scaling.
export DEPTH_TTC_BRAKE_ALPHA=1.0
export DEPTH_TTC_LEFT_ALPHA=1.0
export DEPTH_TTC_RIGHT_ALPHA=1.0

export STEERING_ALPHA="1.0"
export BRAKE_ACTIVATION_ALPHA_SCALE=1.0

export LEFT_ACTIVATION_ALPHA_SCALE=1.1  # For fused and decoder4
export RIGHT_ACTIVATION_ALPHA_SCALE=1.1  # For fused and decoder4

# export LEFT_ACTIVATION_ALPHA_SCALE=1.5  # For decoder5
# export RIGHT_ACTIVATION_ALPHA_SCALE=1.5  # For decoder5

# export LEFT_ACTIVATION_ALPHA_SCALE=0.9  # For decoder6
# export RIGHT_ACTIVATION_ALPHA_SCALE=0.9  # For decoder6

# ----------------------------------------------------------------------------
# Depth Anything V2 (used by validate_mode_dependencies and by each job)
# ----------------------------------------------------------------------------
export DEPTH_ROOT="${DEPTH_ROOT:-/mnt/bapve/thome/wangyuhao/Depth-Anything-V2}"
export DEPTH_CHECKPOINT="${DEPTH_CHECKPOINT:-$DEPTH_ROOT/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth}"

# ----------------------------------------------------------------------------
# General environment
# ----------------------------------------------------------------------------
export CARLA_ROOT="$PWD/f2d_carla"
export LIVE_VISU=0
export DEBUG_CHALLENGE=0
export DEBUG_LOG_EGO_STAT=true
export LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=true use_kalman_filter=true slower_for_stop_sign=true produce_debug_video=true produce_debug_image=false produce_input_video=false produce_input_image=false produce_frame_frequency=5"

export PYTHONPATH="$PWD/team_code:$LEAD_ROOT:${PYTHONPATH:-}"

export STEERING_VECTORS_DIR="$PWD/steering/tfv6/post_process"

export STEERING_FEATURE_NAME="fused_features"
export STEERING_FEATURE_NAME_SHORT="fused"

# export STEERING_FEATURE_NAME="decoder_layer4"
# export STEERING_FEATURE_NAME_SHORT="decoder4"

# export STEERING_FEATURE_NAME="decoder_layer5"
# export STEERING_FEATURE_NAME_SHORT="decoder5"

# export STEERING_FEATURE_NAME="decoder_layer6"
# export STEERING_FEATURE_NAME_SHORT="decoder6"

export MAX_CARLA_FRAME=1600

env_cmd="export STEERING_FEATURE_NAME=$STEERING_FEATURE_NAME; \
export STEERING_FEATURE_NAME_SHORT=$STEERING_FEATURE_NAME_SHORT; \
export STEERING_POLICY=$STEERING_POLICY; \
export STEERING_ALPHA=$STEERING_ALPHA; \
export ACTIVATION_POLICY=$ACTIVATION_POLICY; \
export DEPTH_TTC_STEERING=$DEPTH_TTC_STEERING; \
export DEPTH_TTC_VERBOSE=$DEPTH_TTC_VERBOSE; \
export DEPTH_TTC_EVERY_N=$DEPTH_TTC_EVERY_N; \
export DEPTH_TTC_HOLD_FRAMES=$DEPTH_TTC_HOLD_FRAMES; \
export DEPTH_TTC_HOLD_FRAMES_BRAKE=$DEPTH_TTC_HOLD_FRAMES_BRAKE; \
export DEPTH_TTC_HOLD_FRAMES_LEFT=$DEPTH_TTC_HOLD_FRAMES_LEFT; \
export DEPTH_TTC_HOLD_FRAMES_RIGHT=$DEPTH_TTC_HOLD_FRAMES_RIGHT; \
export DEPTH_TTC_BRAKE_ALPHA=$DEPTH_TTC_BRAKE_ALPHA; \
export DEPTH_TTC_LEFT_ALPHA=$DEPTH_TTC_LEFT_ALPHA; \
export DEPTH_TTC_RIGHT_ALPHA=$DEPTH_TTC_RIGHT_ALPHA; \
export BRAKE_ACTIVATION_ALPHA_SCALE=$BRAKE_ACTIVATION_ALPHA_SCALE; \
export LEFT_ACTIVATION_ALPHA_SCALE=$LEFT_ACTIVATION_ALPHA_SCALE; \
export RIGHT_ACTIVATION_ALPHA_SCALE=$RIGHT_ACTIVATION_ALPHA_SCALE; \
export DEPTH_ROOT=$DEPTH_ROOT; \
export DEPTH_CHECKPOINT=$DEPTH_CHECKPOINT; \
export DEBUG_LOG_EGO_STAT=$DEBUG_LOG_EGO_STAT; \
export LEAD_CLOSED_LOOP_CONFIG=\"$LEAD_CLOSED_LOOP_CONFIG\"; \
export MAX_CARLA_FRAME=$MAX_CARLA_FRAME"

python -u slurm_evaluate_server.py \
  --eval-mode depth_ttc \
  --routes fail2drive_split \
  --seeds 3 \
  --retries 2 \
  --out-root results/tfv6_depth_ttc_$STEERING_FEATURE_NAME_SHORT \
  --agent-file "$LEAD_ROOT/lead/inference/sensor_agent.py" \
  --agent-config "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34"
  --env-var-cmd "$env_cmd"
