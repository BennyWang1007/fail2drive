LOCAL="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
CARLA="$LOCAL/f2d_carla"
F2D="$LOCAL"
export ALPAMAYO_ROOT="$F2D/../alpamayo1.5"
export ALPAMAYO_REPO="$ALPAMAYO_ROOT"
export ALPAMAYO_RAW_LOG_PATH="$F2D/results/alpamayo_vlm_test/local/alpamayo_raw_responses.txt"
export ALPAMAYO_INPUT_SAVE_DIR="$F2D/results/alpamayo_vlm_test/local/alpamayo_inputs"
# export ALPAMAYO_RAW_OUTPUT_ONLY=1  # debug prompts without Alpamayo action parser/Invalid reason
export CUDA_VISIBLE_DEVICES=7


export CUDA_VISIBLE_DEVICES=6

export LIVE_VISU=0
export DEBUG_CHALLENGE=0

export VLM_STEERING=1
export VLM_BACKEND=alpamayo_server
export VLM_SERVER_URL=http://127.0.0.1:8765/generate
export VLM_EVERY_N=5
export VLM_DECISION_TTL_FRAMES=5
export VLM_SERVER_FRAME_OFFSETS=-6,-4,-2,0
export VLM_MAX_NEW_TOKENS=256
export VLM_VERBOSE=0

export ACTIVATION_POLICY=vlm
export ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/Brake/steering_vector.pt,./steering/transfuser/post_process/left_change_lane/steering_vector.pt,./steering/transfuser/post_process/right_change_lane/steering_vector.pt"

export CARLA_ROOT=$CARLA

export LEADERBOARD_ROOT=$LOCAL/leaderboard
export SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner

export IS_BENCH2DRIVE=1
export PYTHONPATH=$CARLA/PythonAPI:$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner:$LOCAL/team_code:$HIP


source $ALPAMAYO_ROOT/a1_5_venv/bin/activate