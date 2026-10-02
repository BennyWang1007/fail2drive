#!/bin/bash
#SBATCH --job-name=tfv6_steering_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:0

set -e

mkdir -p logs

# Make sure we are running from the project root
cd "$SLURM_SUBMIT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
set +u
conda activate hipad
set -u

export STEERING_POLICY="oracle"
export STEERING_ALPHA="1.0"

export ACTIVATION_POLICY=pdm_oracle
export PDM_ORACLE_ACTION=auto
export PDM_ORACLE_ALPHA=1.0
export PDM_ORACLE_TRIGGER_DISTANCE=20
export PDM_ORACLE_HOLD_FRAMES=30
export PDM_ORACLE_COOLDOWN_FRAMES=50
export PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE=70
export PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE=100
export PDM_ORACLE_SIDE_HAZARD_DISTANCE=25
export PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE=10
export PDM_ORACLE_ROADBLOCKED_DISTANCE=40
export PDM_ORACLE_PRIORITY_DISTANCE=25
export PDM_ORACLE_YIELD_EMERGENCY_DISTANCE=50
export PDM_ORACLE_GENERAL_BRAKE=1

export ACTIVATION_VECTOR_PATHS="$F2D/steering/hipad_new/post_process/brake/steering_vector.pt,$F2D/steering/hipad_new/post_process/left/steering_vector.pt,$F2D/steering/hipad_new/post_process/right/steering_vector.pt"

export BRAKE_ACTIVATION_ALPHA_SCALE=1.0
export LEFT_ACTIVATION_ALPHA_SCALE=3.0
export RIGHT_ACTIVATION_ALPHA_SCALE=2.0

# Your existing environment variables
export CARLA_ROOT="$PWD/f2d_carla"
export LIVE_VISU=0
export DEBUG_CHALLENGE=0


# export STEERING_FEATURE_NAME="fused_features"
python -u slurm_evaluate.py \
  --routes fail2drive_split \
  --seeds 3 \
  --retries 1 \
  --out_root results/hipad_oracle \
  --agent_file $F2D/team_code/hipad_f2d_agent.py \
  --agent_config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --lb_script leaderboard/leaderboard/leaderboard_evaluator_local.py
