#!/bin/bash
#SBATCH --job-name=tfv6_vlm_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:0

# Controller job. slurm_evaluate_server.py submits one two-GPU job per
# route/seed; each child starts Alpamayo, CARLA, and the TFv6 evaluator.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
set +u
conda activate fail2drive
set -u

export ROOT="${ROOT:-$HOME}"
export FAIL2DRIVE_ROOT="${FAIL2DRIVE_ROOT:-$PWD}"
export LEAD_ROOT="${LEAD_ROOT:-$ROOT/lead-cvpr2026}"


# export VLM_MIN_SPEED_MPS="${VLM_MIN_SPEED_MPS:-3.5}"
export VLM_MIN_SPEED_MPS="${VLM_MIN_SPEED_MPS:-0.0}"

# Load common + tfv6 + vlm + debug settings, and override DEBUG_CHALLENGE and VLM_MIN_SPEED_MPS
env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common tfv6 vlm debug && export DEBUG_CHALLENGE=0 && export VLM_MIN_SPEED_MPS=$VLM_MIN_SPEED_MPS"

dry_run=""
# dry_run=" --dry-run"

# In the current Controller Shell, execute the command to load all environment variables for this script 
eval "$env_var_cmd"

# Start the evaluation and pass the same load command to the child task
python -u slurm_evaluate_server.py \
  --eval-mode vlm \
  --routes "${ROUTES:-fail2drive_split}" \
  --seeds "${SEEDS:-3}" \
  --retries "${RETRIES:-2}" \
  --out-root "${OUT_ROOT:-results_shome/tfv6_vlm_${STEERING_FEATURE_NAME_SHORT}}" \
  --agent-file "${AGENT_FILE:-$LEAD_ROOT/lead/inference/sensor_agent.py}" \
  --agent-config "${AGENT_CONFIG:-$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34}" \
  --track "${EVAL_TRACK:-SENSORS}" \
  --timeout "${EVAL_TIMEOUT:-300}" \
  --template "$FAIL2DRIVE_ROOT/sbatch_evaluate_server.sh" \
  --env_var_cmd "$env_var_cmd" \
  --verbose \
  --job-prefix "F2D_TFV6_VLM_" \
  $dry_run
