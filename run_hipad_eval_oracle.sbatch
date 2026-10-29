#!/bin/bash
#SBATCH --job-name=hipad_oracle_eval
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

# Load common + hipad + oracle settings, and override MAX_CARLA_FRAME
env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common hipad oracle && export MAX_CARLA_FRAME=2000"
# env_var_cmd="source $PWD/f2d_env_config.sh && load_f2d_env common hipad oracle"

dry_run=""
# dry_run=" --dry-run"

# In the current Controller Shell, execute the command to load all environment variables for this script
eval "$env_var_cmd"

# Start the evaluation and pass the same load command to the child task
python -u slurm_evaluate.py \
  --routes fail2drive_split \
  --seeds 3 \
  --retries 1 \
  --out_root results_shome/hipad_oracle \
  --agent_file $F2D/team_code/hipad_f2d_agent.py \
  --agent_config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
  --lb_script leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --env_var_cmd "$env_var_cmd" \
  --verbose \
  --job_prefix "F2D_Hipad_Oracle_" \
  $dry_run
