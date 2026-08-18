LOCAL="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
CARLA="$LOCAL/f2d_carla"
F2D="$LOCAL"
HIP="$LOCAL/../HiP-AD"

# conda env config vars set WORK_DIR=$LOCAL -n fail2drive_hipad
# conda env config vars set CARLA_ROOT=$CARLA -n fail2drive_hipad

# conda env config vars set LEADERBOARD_ROOT=$LOCAL/leaderboard -n fail2drive_hipad
# conda env config vars set SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner -n fail2drive_hipad

# conda env config vars set PYTHONPATH=$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner:$LOCAL/team_code -n fail2drive_hipad
# conda deactivate
# conda activate fail2drive_hipad

conda env config vars set F2D=$F2D -n hipad
conda env config vars set HIP=$HIP -n hipad
conda env config vars set WORK_DIR=$LOCAL -n hipad
conda env config vars set CARLA_ROOT=$CARLA -n hipad

conda env config vars set LEADERBOARD_ROOT=$LOCAL/leaderboard -n hipad
conda env config vars set SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner -n hipad

conda env config vars set IS_BENCH2DRIVE=1 -n hipad
conda env config vars set PYTHONPATH=$CARLA/PythonAPI:$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner:$LOCAL/team_code:$HIP -n hipad

conda deactivate
conda activate hipad

