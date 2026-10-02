#!/usr/bin/env python3
"""Submit a small, explicitly listed set of Fail2Drive validation jobs.

This is intentionally simpler than ``slurm_evaluate.py``: there is no route
discovery, retry, result inspection, or job monitoring.  Edit ``TESTS`` and
``SBATCH_OPTIONS`` below, then run:

    python slurm_quick_validate.py

Use ``--dry-run`` to only generate and print the sbatch files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "results" / "quick_validate"
# OUTPUT_ROOT = REPO_ROOT / "results" / "quick_validate_fused"
# OUTPUT_ROOT = REPO_ROOT / "results" / "quick_validate_dec4"
# OUTPUT_ROOT = REPO_ROOT / "results" / "quick_validate_dec5"
# OUTPUT_ROOT = REPO_ROOT / "results" / "quick_validate_dec6"
F2D = str(REPO_ROOT)

# Edit the Slurm resources here.
SBATCH_OPTIONS = {
    "partition": "devq",
    # "qos": "normal",
    "qos": "debug",
    "nodes": "1",
    "ntasks": "1",
    "cpus-per-task": "8",
    "mem": "16gb",
    "time": "2:00:00",
    "gres": "gpu:nvidia_geforce_rtx_4090:1",
    # "exclusive": "",  # optional, ensures no other user jobs share the GPU on that node
}

DEFULAT_TFV6_ENV_VARS = {
    "REPO_ROOT": str(REPO_ROOT),
    "STEERING_VECTORS_DIR": f"{str(REPO_ROOT)}/steering/tfv6/post_process",

    "STEERING_ALPHA": "1.0",
    "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
    "LEFT_ACTIVATION_ALPHA_SCALE": "1.1",
    "RIGHT_ACTIVATION_ALPHA_SCALE": "1.1",
    "MAX_CARLA_FRAME": "600",
}

DEFULAT_ORACLE_ENV_VARS = {
    "STEERING_POLICY": "oracle",
    "ACTIVATION_POLICY": "pdm_oracle",
    "PDM_ORACLE_ACTION": "auto",
    "PDM_ORACLE_ALPHA": "1.0",
    "PDM_ORACLE_TRIGGER_DISTANCE": "20",

    "PDM_ORACLE_HOLD_FRAMES": "31",
    "PDM_ORACLE_HOLD_FRAMES_BRAKE": "31",
    "PDM_ORACLE_HOLD_FRAMES_LEFT": "10",
    "PDM_ORACLE_HOLD_FRAMES_RIGHT": "10",

    "PDM_ORACLE_COOLDOWN_FRAMES": "20",
    "PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE": "70",
    "PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE": "100",
    "PDM_ORACLE_SIDE_HAZARD_DISTANCE": "25",
    "PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE": "10",
    "PDM_ORACLE_ROADBLOCKED_DISTANCE": "50",
    "PDM_ORACLE_PRIORITY_DISTANCE": "25",
    "PDM_ORACLE_YIELD_EMERGENCY_DISTANCE": "50",
    "PDM_ORACLE_GENERAL_BRAKE": "1",
}

DEFAULT_OCCUPANCY_ENV_VARS = {
    "STEERING_POLICY": "occupancy",
}

DEBUG_ENV_VARS = {
    "LEAD_CLOSED_LOOP_CONFIG": (
        "sensor_agent_creeping=true use_kalman_filter=true ",
        "slower_for_stop_sign=true produce_debug_video=true ",
        "produce_debug_image=false produce_input_video=false ",
        "produce_input_image=false produce_frame_frequency=1",
    ),
    "DEBUG_LOG_EGO_STAT": "true",
    # "DEBUG_LOG_STEERING_DIRECTION": "true",
}


def build_tfv6_test(steering_feature, routes):
    return Rf'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    STEERING_FEATURE_NAME="{steering_feature}" \
    python -u leaderboard/leaderboard/leaderboard_evaluator.py \
    --agent "$LEAD_ROOT/lead/inference/sensor_agent.py" \
    --agent-config "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34" \
    --routes {routes} \
    --track SENSORS \
    --checkpoint "$SAVE_PATH/result.json" \
    --debug-checkpoint "$SAVE_PATH/debug.txt" \
    --port "$FREE_WORLD_PORT" \
    --traffic-manager-port "$FREE_TM_PORT"'''


def run_tfv6_test(steering_feature: str, routes: str | list[str], env_vars: dict[str, str] | None = None):
    if isinstance(routes, str):
        routes = [routes]

    for route in routes:
        command = build_tfv6_test(steering_feature, route)
        route_name = route.split('/')[-1].split('.')[0]
        env_vars = env_vars or {}
        env_vars["BENCHMARK_ROUTE_ID"] = route_name
        env_vars["STEERING_FEATURE_NAME"] = steering_feature

        run = SlurmQuickValidateRun(
            name=f"f2d_tfv6_{steering_feature}_test_{route_name}",
            command=command,
            env="fail2drive",
            env_vars=env_vars
        )
        run_test(run)


@dataclass
class SlurmQuickValidateRun:
    name: str
    command: str
    env: str = "fail2drive"
    env_vars: dict[str, str] | None = None
    default_env_vars: dict[str, str] | None = None

    def get_env_vars_str(self) -> str:
        if self.env_vars is None:
            return ""

        env_vars_str = "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in self.env_vars.items()
        )
        return env_vars_str

    def get_default_env_vars_str(self) -> str:
        if self.default_env_vars is None:
            return ""

        return "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in self.default_env_vars.items()
        )


# Edit, add, or remove entries here.  Each command is submitted as one
# independent Slurm job.  Keeping the full command in every entry makes it easy
# to change agent/config/environment variables for just one test.
TESTS = [
    {
        "name": "f2d_1020_hipad",
        "command": R'''LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --agent ./team_code/hipad_f2d_agent.py \
    --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
    --routes ./fail2drive_split/Generalization_CustomObstacles_1020.xml \
    --port "$FREE_WORLD_PORT"''',
        "env": "hipad",
        "env_vars": {
            "STEERING_POLICY": "oracle",
            "STEERING_ALPHA": "1.0",
            "ACTIVATION_POLICY": "pdm_oracle",
            "PDM_ORACLE_ACTION": "auto",
            "PDM_ORACLE_ALPHA": "1.0",
            "PDM_ORACLE_TRIGGER_DISTANCE": "20",
            "PDM_ORACLE_HOLD_FRAMES": "30",
            "PDM_ORACLE_COOLDOWN_FRAMES": "50",
            "PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE": "70",
            "PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE": "100",
            "PDM_ORACLE_SIDE_HAZARD_DISTANCE": "25",
            "PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE": "10",
            "PDM_ORACLE_ROADBLOCKED_DISTANCE": "40",
            "PDM_ORACLE_PRIORITY_DISTANCE": "25",
            "PDM_ORACLE_YIELD_EMERGENCY_DISTANCE": "50",
            "PDM_ORACLE_GENERAL_BRAKE": "1",
            "ACTIVATION_VECTOR_PATHS": (
                f"{F2D}/steering/hipad_new/post_process/brake/steering_vector.pt,",
                f"{F2D}/steering/hipad_new/post_process/left/steering_vector.pt,",
                f"{F2D}/steering/hipad_new/post_process/right/steering_vector.pt",
            ),
            "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
            "LEFT_ACTIVATION_ALPHA_SCALE": "3.0",
            "RIGHT_ACTIVATION_ALPHA_SCALE": "2.0",
        }
    },
    # {
    #     "name": "f2d_tfv6",
    #     "command": R''' LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    # python -u leaderboard/leaderboard/leaderboard_evaluator.py \
    # --agent "$LEAD_ROOT/lead/inference/sensor_agent.py" \
    # --agent-config "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34" \
    # --routes ./fail2drive_split/Base_BadParking_0002.xml \
    # --track SENSORS \
    # --checkpoint "$SAVE_PATH/result.json" \
    # --debug-checkpoint "$SAVE_PATH/debug.txt" \
    # --port "$FREE_WORLD_PORT" \
    # --traffic-manager-port "$FREE_TM_PORT"''',
    # },
    {
        "name": "f2d_tfv6_occupancy_dec5_fix_brake",
        "command": build_tfv6_test("decoder_layer5", "./steering_split/Brake/SteeringVehicleOnRoad_1085.xml"),
        "env_vars": {
            "REPO_ROOT": str(REPO_ROOT),
            "STEERING_VECTORS_DIR": f"{REPO_ROOT}/steering/tfv6/post_process",
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            "STEERING_FEATURE_NAME": "decoder_layer5",
            "STEERING_ALPHA": "1.0",
            "STEERING_COMMAND": "brake",
            "ACTIVATION_START_FRAME": "5",
            "ACTIVATION_END_FRAME": "30",
            "NO_OTHER_VEHICLES": "1",
        }
    },
    {
        "name": "f2d_tfv6_occupancy_dec5_fix_left_steering",
        "command": build_tfv6_test("decoder_layer5", "./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml"),
        "env_vars": {
            "REPO_ROOT": str(REPO_ROOT),
            "STEERING_VECTORS_DIR": f"{str(REPO_ROOT)}/steering/tfv6/post_process",
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            "STEERING_FEATURE_NAME": "decoder_layer5",
            "STEERING_ALPHA": "1.0",
            "STEERING_COMMAND": "left",
            "ACTIVATION_START_FRAME": "5",
            "ACTIVATION_END_FRAME": "30",
            "NO_OTHER_VEHICLES": "1",
        }
    },
    {
        "name": "f2d_tfv6_occupancy_dec5_fix_right_steering",
        "command": build_tfv6_test("decoder_layer5", "./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml"),
        "env_vars": {
            "REPO_ROOT": str(REPO_ROOT),
            "STEERING_VECTORS_DIR": f"{str(REPO_ROOT)}/steering/tfv6/post_process",
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            "STEERING_FEATURE_NAME": "decoder_layer5",
            "STEERING_ALPHA": "1.0",
            "STEERING_COMMAND": "right",
            "ACTIVATION_START_FRAME": "5",
            "ACTIVATION_END_FRAME": "30",
            "NO_OTHER_VEHICLES": "1",
        }
    },
    {
        "name": "f2d_tfv6_dec5_oracle_test_PedestriansOnRoad_1085",
        "command": build_tfv6_test("decoder_layer5", "./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml"),
        "default_env_vars": {**DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS},
        "env_vars": {
            "REPO_ROOT": str(REPO_ROOT),
            "STEERING_VECTORS_DIR": f"{str(REPO_ROOT)}/steering/tfv6/post_process",
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            "STEERING_FEATURE_NAME": "decoder_layer5",
            "STEERING_POLICY": "oracle",

            # "STEERING_ALPHA": "1.0",
            # "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "2.0",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.1",

            # "PDM_ORACLE_HOLD_FRAMES": "31",
            # "PDM_ORACLE_HOLD_FRAMES_BRAKE": "31",
            # "PDM_ORACLE_HOLD_FRAMES_LEFT": "20",
            # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "9",

            # "NO_OTHER_VEHICLES": "1",
        }
    },

    # ================================= Fused features tests =================================
    {
        "name": "f2d_tfv6_fused_fix_brake",
        "command": build_tfv6_test("fused_features", "./steering_split/Brake/SteeringVehicleOnRoad_0000.xml"),
        "env_vars": {
            "REPO_ROOT": str(REPO_ROOT),
            "STEERING_VECTORS_DIR": f"{str(REPO_ROOT)}/steering/tfv6/post_process",
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            "STEERING_FEATURE_NAME": "fused_features",
            "STEERING_ALPHA": "1.5",
            "STEERING_COMMAND": "brake",
            "ACTIVATION_START_FRAME": "60",
            "ACTIVATION_END_FRAME": "80",
            "NO_OTHER_VEHICLES": "1",
            "MAX_CARLA_FRAME": "150",
        }
    },
    {
        "name": "f2d_tfv6_fused_fix_left_steering",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml"),
        "env_vars": {
            "REPO_ROOT": str(REPO_ROOT),
            "STEERING_VECTORS_DIR": f"{REPO_ROOT}/steering/tfv6/post_process",
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            "STEERING_FEATURE_NAME": "fused_features",
            "STEERING_ALPHA": "1.1",
            "STEERING_COMMAND": "left",
            # "ACTIVATION_START_FRAME": "94",
            "ACTIVATION_START_FRAME": "92",
            "ACTIVATION_END_FRAME": "100",
            "NO_OTHER_VEHICLES": "1",
            "MAX_CARLA_FRAME": "150",
        }
    },
    {
        "name": "f2d_tfv6_fused_fix_right_steering",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml"),
        "default_env_vars": {**DEFULAT_TFV6_ENV_VARS},
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            "STEERING_FEATURE_NAME": "fused_features",
            "STEERING_ALPHA": "1.1",
            "STEERING_COMMAND": "right",
            "ACTIVATION_START_FRAME": "92",
            "ACTIVATION_END_FRAME": "100",
            "NO_OTHER_VEHICLES": "1",
            "MAX_CARLA_FRAME": "150",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_Wall_1096",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Generalization_Wall_1096.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Generalization_Wall_1096",
            "STEERING_FEATURE_NAME": "fused_features",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_BadParking_0005",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_BadParking_0005.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_BadParking_0005",
            "STEERING_FEATURE_NAME": "fused_features",
            # "STEERING_ALPHA": "1.0",
            # "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "1.5",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.5",
            # "PDM_ORACLE_HOLD_FRAMES_LEFT": "20",
            # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "10",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_BadParking_0007",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_BadParking_0007.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_BadParking_0007",
            "STEERING_FEATURE_NAME": "fused_features",
            # "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "1.2",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.5",
            # "PDM_ORACLE_HOLD_FRAMES": "31",
            # "PDM_ORACLE_HOLD_FRAMES_BRAKE": "31",
            # "PDM_ORACLE_HOLD_FRAMES_LEFT": "20",
            # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "10",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_BadParking_0008",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_BadParking_0008.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_BadParking_0008",
            "STEERING_FEATURE_NAME": "fused_features",
            # "STEERING_POLICY": "oracle",

            # "STEERING_ALPHA": "1.0",
            # "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "2.0",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.5",

            # "PDM_ORACLE_HOLD_FRAMES": "31",
            # "PDM_ORACLE_HOLD_FRAMES_BRAKE": "31",
            # "PDM_ORACLE_HOLD_FRAMES_LEFT": "20",
            # # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "9",
            # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "10",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_ConstructionPedestrian_0011",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_ConstructionPedestrian_0011.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            # **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_ConstructionPedestrian_0011",
            "STEERING_FEATURE_NAME": "fused_features",

            "LEFT_ACTIVATION_ALPHA_SCALE": "1.5",
            "RIGHT_ACTIVATION_ALPHA_SCALE": "1.5",
            "ORACLE_VERBOSE": "true",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_HardBrake_0036",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_HardBrake_0036.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_HardBrake_0036",
            "STEERING_FEATURE_NAME": "fused_features",
            # "STEERING_POLICY": "oracle",

            # "STEERING_ALPHA": "1.0",
            # "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "2.0",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.1",

            # "PDM_ORACLE_HOLD_FRAMES": "31",
            # "PDM_ORACLE_HOLD_FRAMES_BRAKE": "31",
            # "PDM_ORACLE_HOLD_FRAMES_LEFT": "20",
            # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "9",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_CustomObstacles_0024",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_CustomObstacles_0024.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_CustomObstacles_0024",
            "STEERING_FEATURE_NAME": "fused_features",
            # "BRAKE_ACTIVATION_ALPHA_SCALE": "1.0",
            # # "LEFT_ACTIVATION_ALPHA_SCALE": "2.0",
            # # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.5",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "1.1",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.1",

            # "PDM_ORACLE_HOLD_FRAMES_LEFT": "10",
            # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "10",
        }
    },
    {
        "name": "f2d_tfv6_fused_oracle_test_BadParking_0000",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_BadParking_0000.xml"),
        "default_env_vars": {
            **DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS,
            **DEBUG_ENV_VARS
        },
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_BadParking_0000",
            "STEERING_FEATURE_NAME": "fused_features",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "1.2",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "1.5",
            # "PDM_ORACLE_HOLD_FRAMES_LEFT": "10",
            # "PDM_ORACLE_HOLD_FRAMES_RIGHT": "10",
        }
    },
    # ================================= No Policy tests =================================
    {
        "name": "f2d_tfv6_no_policy",
        "command": build_tfv6_test("fused_features", "./steering_split/Brake/SteeringVehicleOnRoad_0000.xml"),
        "default_env_vars": {**DEFULAT_TFV6_ENV_VARS},
        "env_vars": {
            "REPO_ROOT": str(REPO_ROOT),
            "ACTIVATION_POLICY": "none",
            "BENCHMARK_ROUTE_ID": "SteeringVehicleOnRoad_0000",
            "NO_OTHER_VEHICLES": "1",
        }
    },
    {
        "name": "f2d_tfv6_no_policy_test_CustomObstacles_0024",
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Base_CustomObstacles_0024.xml"),
        "default_env_vars": {**DEFULAT_TFV6_ENV_VARS},
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_CustomObstacles_0024",
            "STEERING_FEATURE_NAME": "fused_features",
            "STEERING_POLICY": "none",
        }
    },
    #     {
    #         "name": "f2d_0004_transfuser",
    #         "command": r''' LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    # python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    #   --agent ./team_code/sensor_agent.py \
    #   --agent-config ./checkpoints/tfpp \
    #   --routes ./fail2drive_split/Base_BadParking_0004.xml \
    #   --port "$FREE_WORLD_PORT"''',
    #     },
    #     {
    #         "name": "f2d_0012_transfuser",
    #         "command": r''' LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    # python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    #   --agent ./team_code/sensor_agent.py \
    #   --agent-config ./checkpoints/tfpp \
    #   --routes ./fail2drive_split/Base_ConstructionPedestrian_0012.xml \
    #   --port "$FREE_WORLD_PORT"''',
    #     },
    #     {
    #             "name": "f2d_0016_transfuser",
    #             "command": r''' LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    #     python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    #       --agent ./team_code/sensor_agent.py \
    #       --agent-config ./checkpoints/tfpp \
    #       --routes ./fail2drive_split/Base_ConstructionPermutations_0016.xml \
    #       --port "$FREE_WORLD_PORT"''',
    #         },
    #         {
    #                 "name": "f2d_0017_transfuser",
    #                 "command": r''' LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    #         python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    #           --agent ./team_code/sensor_agent.py \
    #           --agent-config ./checkpoints/tfpp \
    #           --routes ./fail2drive_split/Base_ConstructionPermutations_0017.xml \
    #           --port "$FREE_WORLD_PORT"''',
    #             },
    #     {
    #         "name": "f2d_0021_transfuser",
    #         "command": r''' LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    # python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    #   --agent ./team_code/sensor_agent.py \
    #   --agent-config ./checkpoints/tfpp \
    #   --routes ./fail2drive_split/Base_CustomObstacles_0021.xml \
    #   --port "$FREE_WORLD_PORT"''',
    #     },
    #     {
    #         "name": "f2d_0056_transfuser",
    #         "command": r''' LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    # python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    #   --agent ./team_code/sensor_agent.py \
    #   --agent-config ./checkpoints/tfpp \
    #   --routes ./fail2drive_split/Base_RightOfWay_0056.xml \
    #   --port "$FREE_WORLD_PORT"''',
    #     },
]

RUNS = [
    SlurmQuickValidateRun(
        name=test["name"],
        command=test["command"],
        env=test.get("env", "fail2drive"),
        env_vars=test.get("env_vars", {}),
        default_env_vars=test.get("default_env_vars", {}),
    )
    for test in TESTS
]


def safe_name(name: str) -> str:
    """Return a name safe for paths and Slurm job names."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-")
    if not value:
        raise ValueError(f"Invalid empty test name derived from {name!r}")
    return value


def make_job_script(run: SlurmQuickValidateRun, test_output: Path) -> str:
    name = safe_name(run.name)
    directives = "\n".join(
        f"#SBATCH --{key}={value}" for key, value in SBATCH_OPTIONS.items()
    )

    conda_env: str = run.env
    conda_activate_cmd = f"conda activate {conda_env}\n"

    default_env_vars_str = run.get_default_env_vars_str()
    env_vars_str = run.get_env_vars_str()

    return fr'''#!/bin/bash
#SBATCH --job-name=f2d_val_{name}
{directives}
#SBATCH --output={test_output}/slurm-%j.out.log
#SBATCH --error={test_output}/slurm-%j.err.log

set -euo pipefail

REPO_ROOT={shlex.quote(str(REPO_ROOT))}
TEST_OUTPUT={shlex.quote(str(test_output))}
mkdir -p "$TEST_OUTPUT/viz_vehicle"
mkdir -p "$(dirname "$SAVE_PATH/result.json")"
cd "$REPO_ROOT"

{default_env_vars_str}
{env_vars_str}
echo "Env vars:"
printenv | grep -E '^(REPO_ROOT|STEERING_|ACTIVATION_|PDM_|BENCHMARK_ROUTE_ID|NO_OTHER_VEHICLES|MAX_CARLA_FRAME|.*_ACTIVATION_ALPHA_SCALE)' | sort
BENCHMARK_ROUTE_ID="${{BENCHMARK_ROUTE_ID:-BENCHMARK_ROUTE_ID}}_${{SLURM_JOB_ID:-manual}}"
echo "Job ID: ${{SLURM_JOB_ID:-manual}}"
echo "Node: $(hostname)"
echo "Test: {name}"
echo "Output: $TEST_OUTPUT"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u
{conda_activate_cmd}
set -u

: "${{CARLA_ROOT:?CARLA_ROOT is not set. Submit from the activated fail2drive environment.}}"
# : "${{HIP:?HIP is not set. Export HIP before submitting.}}"

free_port() {{
  local start=$1
  comm -23 <(seq "$start" "$((start + 49))" | sort) \
    <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) \
    | shuf | head -n 1
}}

WORLD_PORT_BASE=$((10000 + (SLURM_JOB_ID % 200) * 50))
STREAMING_PORT_BASE=$((20000 + (SLURM_JOB_ID % 200) * 50))
TM_PORT_BASE=$((30000 + (SLURM_JOB_ID % 200) * 50))

FREE_WORLD_PORT=$(free_port "$WORLD_PORT_BASE")
FREE_STREAMING_PORT=$(free_port "$STREAMING_PORT_BASE")
FREE_TM_PORT=$(free_port "$TM_PORT_BASE")

if [[ -z "$FREE_WORLD_PORT" || -z "$FREE_STREAMING_PORT" || -z "$FREE_TM_PORT" ]]; then
  echo "Could not find a free CARLA port in the assigned window ($WORLD_PORT_BASE/$STREAMING_PORT_BASE/$TM_PORT_BASE)" >&2
  exit 1
fi

CARLA_LOG="$TEST_OUTPUT/carla.log"
"$CARLA_ROOT/CarlaUE4.sh" \
  -carla-rpc-port="$FREE_WORLD_PORT" \
  -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 \
  -carla-streaming-port="$FREE_STREAMING_PORT" \
  >"$CARLA_LOG" 2>&1 &
CARLA_PID=$!

cleanup() {{
  kill "$CARLA_PID" 2>/dev/null || true
  wait "$CARLA_PID" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

sleep 60
if ! kill -0 "$CARLA_PID" 2>/dev/null; then
  echo "CARLA exited before validation started. Last log lines:" >&2
  tail -100 "$CARLA_LOG" >&2 || true
  exit 1
fi

{run.command}
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without calling sbatch.",
    )
    return parser.parse_args()


def run_test(run: SlurmQuickValidateRun) -> None:
    run_dir = OUTPUT_ROOT / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(run.name)

    test_output = OUTPUT_ROOT / name
    test_output.mkdir(parents=True, exist_ok=True)
    job_file = run_dir / f"{name}.sh"
    job_file.write_text(make_job_script(
        run, test_output), encoding="utf-8")
    job_file.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(job_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"{name}: {result.stdout.strip()}")


def main() -> None:
    args = parse_args()
    run_dir = OUTPUT_ROOT / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    names_to_test = [
        # "f2d_1020_hipad",
        # "f2d_tfv6_occupancy_dec5_fix_brake",
        # "f2d_tfv6_occupancy_dec5_fix_left_steering",
        # "f2d_tfv6_occupancy_dec5_fix_right_steering",
        # "f2d_tfv6_occupancy_dec5_oracle",

        # "f2d_tfv6_fused_fix_brake",
        # "f2d_tfv6_fused_fix_left_steering",
        # "f2d_tfv6_fused_fix_right_steering",

        # "f2d_tfv6_fused_oracle_test_BadParking_0000",
        # "f2d_tfv6_fused_oracle_test_BadParking_0005",
        # "f2d_tfv6_fused_oracle_test_BadParking_0007",
        # "f2d_tfv6_fused_oracle_test_BadParking_0008",
        # "f2d_tfv6_fused_oracle_test_ConstructionPedestrian_0011",
        # "f2d_tfv6_fused_oracle_test_CustomObstacles_0024",
        # "f2d_tfv6_fused_oracle_test_HardBrake_0036",
        # "f2d_tfv6_fused_oracle_test_Wall_1096",

        # "f2d_tfv6_no_policy",
        # "f2d_tfv6_no_policy_test_CustomObstacles_0024",
    ]

    for run in RUNS:
        name = safe_name(run.name)
        if name not in names_to_test:
            continue

        test_output = OUTPUT_ROOT / name
        test_output.mkdir(parents=True, exist_ok=True)
        job_file = run_dir / f"{name}.sh"
        job_file.write_text(make_job_script(
            run, test_output), encoding="utf-8")
        job_file.chmod(0o755)

        if args.dry_run:
            print(f"[dry-run] {job_file}")
            continue

        result = subprocess.run(
            ["sbatch", str(job_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"{name}: {result.stdout.strip()}")

    run_tfv6_test(
        "fused_features",
        # "decoder_layer4",
        # "decoder_layer5",
        # steering_feature="decoder_layer6",
        routes=[
            "./fail2drive_split/Base_BadParking_0001.xml",
            "./fail2drive_split/Base_BadParking_0005.xml",
            "./fail2drive_split/Base_ConstructionPedestrian_0011.xml",
            "./fail2drive_split/Base_ConstructionPedestrian_0014.xml",
            "./fail2drive_split/Base_CustomObstacles_0020.xml",
            "./fail2drive_split/Base_CustomObstacles_0021.xml",
            "./fail2drive_split/HardBrake_0036.xml",
            "./fail2drive_split/Wall_1096.xml",
        ],
        env_vars={**DEFULAT_TFV6_ENV_VARS, **DEFULAT_ORACLE_ENV_VARS, **DEBUG_ENV_VARS,
                  "ORACLE_VERBOSE": "true",
                  #  "MAX_CARLA_FRAME": "550"
                  }
    )


if __name__ == "__main__":
    main()
