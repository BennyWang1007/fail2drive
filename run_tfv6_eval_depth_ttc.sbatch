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
set +u
conda activate fail2drive
set -u

export PYTHONPATH="$PWD/team_code:$LEAD_ROOT:${PYTHONPATH:-}"

# Load common + tfv6 + depth_ttc + debug settings, and override DEBUG_CHALLENGE and MAX_CARLA_FRAME
env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common tfv6 depth_ttc debug && export DEBUG_CHALLENGE=0 && export MAX_CARLA_FRAME=1600"
# env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common tfv6 depth_ttc debug"

dry_run=""
# dry_run=" --dry-run"

# In the current Controller Shell, execute the command to load all environment variables for this script
eval "$env_var_cmd"

# Start the evaluation and pass the same load command to the child task
python -u slurm_evaluate_server.py \
  --eval-mode depth_ttc \
  --routes fail2drive_split \
  --seeds 3 \
  --retries 2 \
  --out-root results_shome/tfv6_depth_ttc_${STEERING_FEATURE_NAME_SHORT} \
  --agent-file "$LEAD_ROOT/lead/inference/sensor_agent.py" \
  --agent-config "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34" \
  --env-var-cmd "$env_var_cmd" \
  --verbose \
  --job-prefix "F2D_TFV6_Depth_TTC_" \
  $dry_run
