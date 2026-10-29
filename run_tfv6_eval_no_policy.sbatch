#!/bin/bash
#SBATCH --job-name=tfv6_no_policy_eval
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
conda activate fail2drive
set -u

# Load common + tfv6 + no_policy settings, and override DEBUG_CHALLENGE
env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common tfv6 no_policy && export DEBUG_CHALLENGE=0"

dry_run=""
# dry_run=" --dry-run"

# In the current Controller Shell, execute the command to load all environment variables for this script
eval "$env_var_cmd"

# Start the evaluation and pass the same load command to the child task
python -u slurm_evaluate.py \
  --routes fail2drive_split \
  --seeds 3 \
  --retries 1 \
  --out_root "results_shome/tfv6_${STEERING_FEATURE_NAME_SHORT}_no_policy" \
  --agent_file "$LEAD_ROOT/lead/inference/sensor_agent.py" \
  --agent_config "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34" \
  --lb_script leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --env_var_cmd "$env_var_cmd" \
  --verbose \
  --job_prefix "F2D_TFV6_No_Policy_" \
  $dry_run
