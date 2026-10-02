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
conda activate fail2drive

export STEERING_POLICY="none"

export CARLA_ROOT="$PWD/f2d_carla"
export LIVE_VISU=0
export DEBUG_CHALLENGE=0
export DEBUG_LOG_EGO_STAT=true
export ORACLE_VERBOSE=true

export STEERING_FEATURE_NAME="fused_features"
env_cmd="export DEBUG_LOG_EGO_STAT=$DEBUG_LOG_EGO_STAT; export ORACLE_VERBOSE=$ORACLE_VERBOSE"

python -u slurm_evaluate.py \
  --routes fail2drive_split \
  --seeds 3 \
  --retries 1 \
  --out_root results/tfv6_fused_no_policy \
  --agent_file "$LEAD_ROOT/lead/inference/sensor_agent.py" \
  --agent_config "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34" \
  --lb_script leaderboard/leaderboard/leaderboard_evaluator_local.py
  --env_var_cmd "$env_cmd"
