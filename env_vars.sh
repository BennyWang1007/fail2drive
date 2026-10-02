LOCAL="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
CARLA="$LOCAL/f2d_carla"
LEAD_ROOT="$LOCAL/../lead-cvpr2026"

conda env config vars set F2D_ROOT=$LOCAL -n fail2drive
conda env config vars set LEAD_ROOT=$LOCAL/../lead-cvpr2026 -n fail2drive
conda env config vars set LEAD_PROJECT_ROOT=$LOCAL/../lead-cvpr2026 -n fail2drive

conda env config vars set IS_BENCH2DRIVE=0 -n fail2drive
conda env config vars set PLANNER_TYPE=only_traj -n fail2drive

conda env config vars set SAVE_PATH=$LOCAL/results/tfv6_resnet34/test -n fail2drive

# for visualization
# conda env config vars set LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=true use_kalman_filter=true slower_for_stop_sign=true produce_debug_video=true produce_debug_image=true produce_input_video=true produce_input_image=true produce_frame_frequency=5" -n fail2drive
conda env config vars set LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=true use_kalman_filter=true slower_for_stop_sign=true produce_debug_video=true produce_debug_image=false produce_input_video=false produce_input_image=false produce_frame_frequency=5" -n fail2drive
conda env config vars set BENCHMARK_ROUTE_ID="Base_BadParking_0002" -n fail2drive

conda env config vars set WORK_DIR=$LOCAL -n fail2drive
conda env config vars set CARLA_ROOT=$CARLA -n fail2drive

conda env config vars set LEADERBOARD_ROOT=$LOCAL/leaderboard -n fail2drive
conda env config vars set SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner -n fail2drive

conda env config vars set PYTHONPATH=$LEAD_ROOT:$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner:$LOCAL/team_code -n fail2drive


conda deactivate
conda activate fail2drive
