#!/usr/bin/env python3
"""Submit a small, explicitly listed set of Fail2Drive validation jobs.

This script loads shared environment presets from ``f2d_env_config.sh`` via
``profiles`` (e.g. ``["common", "hipad", "oracle", "debug"]``), and layers
per-test ``env_vars`` on top.

This is intentionally simpler than ``slurm_evaluate.py``: there is no route
discovery, retry, result inspection, or job monitoring.  Edit ``TESTS`` and
``SBATCH_OPTIONS`` below.

Use ``--dry-run`` to only generate and print the sbatch files.

Run:
    python slurm_quick_validate.py
    python slurm_quick_validate.py --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
import subprocess
import textwrap
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "results"
CONFIG_SCRIPT = REPO_ROOT / "f2d_env_config.sh"
# F2D = str(REPO_ROOT)

DEPTH_ROOT = "/mnt/bapve/thome/wangyuhao/Depth-Anything-V2"
DEPTH_CHECKPOINT = f"{DEPTH_ROOT}/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"

# Edit the Slurm resources here.
SBATCH_OPTIONS = {
    "partition": "devq",
    "qos": "normal",
    "time": "2:00:00",
    # "qos": "debug",
    # "time": "0:30:00",
    "nodes": "1",
    "ntasks": "1",
    "cpus-per-task": "8",
    "mem": "16gb",
    "gres": "gpu:nvidia_geforce_rtx_4090:1",
    # "exclusive": None,  # do not share the allocated GPU with another job
}


def build_tfv6_test(steering_feature: str, routes: str) -> str:
    return Rf'''PYTHONPATH="$REPO_ROOT/team_code:${{PYTHONPATH:-}}" \
    LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
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


def build_hipad_test(steering_feature: str, routes: str) -> str:
    return Rf'''PYTHONPATH="$REPO_ROOT/team_code:$LEAD_ROOT:${{PYTHONPATH:-}}" \
    LIVE_VISU=0 SAVE_PATH="$TEST_OUTPUT/viz_vehicle" VISUAL_OUTPUT_PATH="$TEST_OUTPUT/viz_vehicle" DEBUG_CHALLENGE=1 \
    STEERING_FEATURE_NAME="{steering_feature}" \
    python -u leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --agent ./team_code/hipad_f2d_agent.py \
    --agent-config "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d" \
    --routes {routes} \
    --port "$FREE_WORLD_PORT" \
    --traffic-manager-port "$FREE_TM_PORT"'''


@dataclass
class SlurmQuickValidateRun:
    name: str
    command: str
    env: str = "fail2drive"
    profiles: list[str] = field(default_factory=list)
    env_vars: dict[str, str] | None = None
    default_env_vars: dict[str, str] | None = None  # Obsolete, use profiles instead
    model_name: str | None = None
    policy_name: str | None = None
    steering_feature: str | None = None
    log_name: str | None = None
    env_type: str = "conda"  # or "uv"

    def get_env_activate_cmd(self) -> str:
        if self.env_type == "conda":
            return textwrap.dedent(f"""
                CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
                source "$CONDA_BASE/etc/profile.d/conda.sh"
                set +u
                conda activate {self.env}
                set -u
            """).strip()
        elif self.env_type == "uv":
            return f"source {self.env}/bin/activate"
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")

    def get_profiles_cmd(self) -> str:
        """load f2d_env_config.sh command string"""
        if not self.profiles:
            return ""
        profiles_str = " ".join(shlex.quote(p) for p in self.profiles)
        return (
            f"source {shlex.quote(str(CONFIG_SCRIPT))}\n"
            f"load_f2d_env {profiles_str}"
        )

    def get_env_vars_str(self) -> str:
        if not self.env_vars:
            return ""

        return "\n".join(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in self.env_vars.items()
        )

    def get_default_env_vars_str(self) -> str:
        if not self.default_env_vars:
            return ""

        return "\n".join(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in self.default_env_vars.items()
        )

    def get_output_dir(self) -> Path:
        output_dir = OUTPUT_ROOT
        if self.model_name:
            output_dir = output_dir / f"quick_validate_{self.model_name}"
        else:
            output_dir = output_dir / "quick_validate"

        if self.policy_name:
            output_dir = output_dir / self.policy_name

        if self.steering_feature:
            output_dir = output_dir / self.steering_feature

        return output_dir


def run_tfv6_test(
    steering_feature: str,
    routes: str | list[str],
    profiles: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    if isinstance(routes, str):
        routes = [routes]

    active_profiles = profiles if profiles is not None else ["common", "tfv6", "oracle", "debug"]

    for route in routes:
        command = build_tfv6_test(steering_feature, route)
        route_name = Path(route).stem
        merged_env_vars = {
            **(env_vars or {}),
            "BENCHMARK_ROUTE_ID": route_name,
            "STEERING_FEATURE_NAME": steering_feature,
        }
        policy_name = merged_env_vars.get("STEERING_POLICY") or (
            "depth_ttc" if "depth_ttc" in active_profiles
            else "vlm" if "vlm" in active_profiles
            else "oracle" if "oracle" in active_profiles
            else None
        )

        run = SlurmQuickValidateRun(
            name=f"f2d_tfv6_{steering_feature}_test_{route_name}",
            command=command,
            env="fail2drive",
            profiles=active_profiles,
            env_vars=merged_env_vars,
            model_name="tfv6",
            policy_name=policy_name,
            steering_feature=steering_feature,
        )
        time.sleep(3)  # Avoid submitting too quickly to Slurm
        run_test(run, dry_run=dry_run)


def run_hipad_test(
    steering_feature: str,
    routes: str | list[str],
    profiles: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    if isinstance(routes, str):
        routes = [routes]

    active_profiles = profiles if profiles is not None else ["common", "hipad", "oracle", "debug"]

    for route in routes:
        command = build_hipad_test(steering_feature, route)
        route_name = Path(route).stem
        merged_env_vars = {
            **(env_vars or {}),
            "BENCHMARK_ROUTE_ID": route_name,
            "STEERING_FEATURE_NAME": steering_feature,
        }
        policy_name = merged_env_vars.get("STEERING_POLICY") or (
            "depth_ttc" if "depth_ttc" in active_profiles
            else "vlm" if "vlm" in active_profiles
            else "oracle" if "oracle" in active_profiles
            else None
        )

        run = SlurmQuickValidateRun(
            name=f"f2d_hipad_test_{route_name}",
            command=command,
            env="hipad",
            profiles=active_profiles,
            env_vars=merged_env_vars,
            model_name="hipad",
            policy_name=policy_name,
            steering_feature=steering_feature,
        )
        time.sleep(5)
        run_test(run, dry_run=dry_run)


# Edit, add, or remove entries here.  Each command is submitted as one
# independent Slurm job.  Keeping the full command in every entry makes it easy
# to change agent/config/environment variables for just one test.
TESTS_TFV6: list[dict[str, Any]] = [
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
        "name": "f2d_tfv6_vlm_alpamayo_ConstructionPedestrian_0011",
        "command": build_tfv6_test(
            "fused_features",
            "./fail2drive_split/Base_ConstructionPedestrian_0011.xml",
        ),
        "env": "fail2drive",
        "model_name": "tfv6_vlm_alpamayo",
        "policy_name": "vlm",
        "steering_feature": "fused_features",
        "profiles": ["common", "tfv6", "vlm", "debug"],
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_ConstructionPedestrian_0011",
            "STEERING_FEATURE_NAME": "fused_features",
        },
    },
    {
        "name": "f2d_tfv6_depth_ttc_fused_BadParking_0002",
        "command": build_tfv6_test(
            "fused_features",
            "./fail2drive_split/Base_BadParking_0002.xml",
        ),
        "profiles": ["common", "tfv6", "depth_ttc", "debug"],
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Base_BadParking_0002",
            "STEERING_FEATURE_NAME": "fused_features",
        },
    },
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
        "profiles": ["common", "tfv6", "oracle", "debug"],
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
        "command": build_tfv6_test("fused_features", "./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml"),
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
        "profiles": ["common", "tfv6", "debug"],
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
        "profiles": ["common", "tfv6", "oracle", "debug"],
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Generalization_Wall_1096",
            "STEERING_FEATURE_NAME": "fused_features",
        }
    },
    # ================================= No Policy tests =================================
    {
        "name": "f2d_tfv6_no_policy",
        "command": build_tfv6_test("fused_features", "./steering_split/Brake/SteeringVehicleOnRoad_0000.xml"),
        "profiles": ["common", "tfv6", "debug"],
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
        "profiles": ["common", "tfv6", "debug"],
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

TESTS_HIPAD: list[dict[str, Any]] = [
    {
        "name": "f2d_hipad_oracle_test_CustomObstacles_1020",
        "command": build_hipad_test("align_query", "./fail2drive_split/Generalization_CustomObstacles_1020.xml"),
        "env": "hipad",
        "profiles": ["common", "hipad", "oracle"],
        "env_vars": {}
    },
    # --------------------------- HipAD Fixed Tests ---------------------------
    {
        "name": "f2d_hipad_fixed_1085_brake_all_layers",
        "command": build_hipad_test("align_query", "./steering_split/Brake/SteeringVehicleOnRoad_0000.xml"),
        "env": "hipad", "model_name": "hipad", "policy_name": "fixed",
        "profiles": ["common", "hipad"],
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Generalization_PedestriansOnRoad_1085",
            # "STEERING_ALPHA": "1.5",
            # "BRAKE_ACTIVATION_ALPHA_SCALE": "1.5",
            "STEERING_COMMAND": "brake",
            "ACTIVATION_START_FRAME": "60",
            "ACTIVATION_END_FRAME": "80",
            "NO_OTHER_VEHICLES": "1",
            "MAX_CARLA_FRAME": "150",
        }
    },
    {
        "name": "f2d_hipad_oracle_1095",
        "command": build_hipad_test("align_query", "./fail2drive_split/Generalization_Wall_1095.xml"),
        "env": "hipad", "model_name": "hipad", "policy_name": "oracle",
        "profiles": ["common", "hipad", "oracle", "debug"],
        "env_vars": {
            "BENCHMARK_ROUTE_ID": "Generalization_Wall_1095",

            # "HIPAD_STEER_LAYERS": "0",
            "HIPAD_STEER_LAYERS": "0,1,2,3,4,5",
            "BRAKE_ACTIVATION_ALPHA_SCALE": "0.3",
            # "LEFT_ACTIVATION_ALPHA_SCALE": "3.0",
            # "RIGHT_ACTIVATION_ALPHA_SCALE": "3.0",
            "MAX_CARLA_FRAME": "450",
            "RUN_NAME": "auto-all-alpha0.3",
        },
        # "log_name": "manual-all-alpha0.5",
        "log_name": "auto-all-alpha0.3",
    }
]


def build_hipad_test_steering_feature_run(
    route_file: str = "./fail2drive_split/Base_ConstructionPedestrian_0012.xml",
    alpha: float = 1.0,
    layers: str = "0,1,2,3,4,5",
    action: str = "left",
    max_frame: int | None = None,
) -> SlurmQuickValidateRun:
    route_name = Path(route_file).stem
    route_id = route_name.split("_")[-1]

    layers_str = "layer_all" if layers == "0,1,2,3,4,5" else f"layer_{layers.replace(',', '_')}"
    run_name = f"manual-{layers_str}-alpha{alpha}"

    env_vars = {
        "RUN_NAME": run_name,
        "BENCHMARK_ROUTE_ID": route_name,
        "HIPAD_STEER_LAYERS": layers,
        "STEERING_ALPHA": str(alpha),
        "BRAKE_ACTIVATION_ALPHA_SCALE": str(alpha),
        "LEFT_ACTIVATION_ALPHA_SCALE": str(alpha),
        "RIGHT_ACTIVATION_ALPHA_SCALE": str(alpha),
    }
    if max_frame is not None:
        env_vars["MAX_CARLA_FRAME"] = str(max_frame)

    return SlurmQuickValidateRun(
        name=f"f2d_hipad_test_turning_{action}_{route_id}",
        command=build_hipad_test("align_query", route_file),
        env="hipad",
        profiles=["common", "hipad", "oracle", "debug"],
        model_name="hipad",
        policy_name="oracle",
        steering_feature="align_query",
        env_vars=env_vars,
        log_name=run_name,
    )


def build_hipad_test_stopping_run(
    route_file: str = "./fail2drive_split/Generalization_Wall_1095.xml",
    alpha: float = 1.0,
    layers: str = "0,1,2,3,4,5",
) -> SlurmQuickValidateRun:
    return build_hipad_test_steering_feature_run(
        route_file=route_file,
        alpha=alpha,
        layers=layers,
        action="brake",
        max_frame=150,
    )


def test_hipad_steering_feature_runs(dry_run: bool = False) -> None:

    hipad_test_steering_feature_runs = []

    # --- Stopping ---
    hipad_test_steering_feature_runs += [
        build_hipad_test_stopping_run(
            route_file="./fail2drive_split/Generalization_Wall_1095.xml",
            alpha=alpha,
            layers=layers,
        )
        for alpha in [1.5, 2.0, 3.0]
        for layers in ["0,1,2,3,4,5"]
    ]

    # --- Left ---
    hipad_test_steering_feature_runs += [
        build_hipad_test_steering_feature_run(
            route_file="./fail2drive_split/Base_ConstructionPedestrian_0012.xml",
            alpha=alpha,
            layers=layers,
            action="left",
            max_frame=750,
        )
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]
        for layers in ["0,1,2,3,4,5", "5", "0"]
    ]

    # --- Right ---
    hipad_test_steering_feature_runs += [
        build_hipad_test_steering_feature_run(
            route_file="./fail2drive_split/Base_ConstructionPedestrian_0011.xml",
            alpha=alpha,
            layers=layers,
            action="right",
            max_frame=1000,
        )
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]
        for layers in ["0,1,2,3,4,5", "5", "0"]
    ]

    print(f"Generated {len(hipad_test_steering_feature_runs)} HipAD steering feature test runs.")

    for run in hipad_test_steering_feature_runs:
        run_test(run, dry_run=dry_run)
        time.sleep(3)


TESTS = TESTS_TFV6 + TESTS_HIPAD

RUNS = [
    SlurmQuickValidateRun(
        name=test["name"],
        command=test["command"],
        env=test.get("env", "fail2drive"),
        env_vars=test.get("env_vars", {}),
        profiles=test.get("profiles", []),
        default_env_vars=test.get("default_env_vars", {}),
        model_name=test.get("model_name", None),
        policy_name=test.get("policy_name", None),
        steering_feature=test.get("steering_feature", None),
        log_name=test.get("log_name", None),
        env_type=test.get("env_type", "conda"),
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

    # Add extra resources for VLM and Depth-TTC tests (checks both name and profiles)
    opts = dict(SBATCH_OPTIONS)
    if (
        "vlm" in run.name
        or "depth" in run.name
        or "vlm" in run.profiles
        or "depth_ttc" in run.profiles
    ):
        opts.update({
            "cpus-per-task": "8",
            "mem": "48gb",
            "gres": "gpu:nvidia_geforce_rtx_4090:2",
        })

    directives = "\n".join(
        f"#SBATCH --{key}" if value is None else f"#SBATCH --{key}={value}"
        for key, value in opts.items()
    )

    env_activate_cmd = run.get_env_activate_cmd()

    log_name_str = "" if run.log_name is None else f"-{run.log_name}"
    # log_name_str = "-manual-alpha3.0"

    profiles_cmd = run.get_profiles_cmd()
    default_env_vars_str = run.get_default_env_vars_str()
    env_vars_str = run.get_env_vars_str()

    return fr'''#!/bin/bash
#SBATCH --job-name=f2d_val_{name}
{directives}
#SBATCH --output={test_output}/slurm-%j{log_name_str}.out.log
#SBATCH --error={test_output}/slurm-%j{log_name_str}.err.log

set -euo pipefail

export REPO_ROOT={shlex.quote(str(REPO_ROOT))}
export F2D="$REPO_ROOT"
TEST_OUTPUT={shlex.quote(str(test_output))}
SAVE_PATH="${{SAVE_PATH:-$TEST_OUTPUT/viz_vehicle}}"
export SAVE_PATH
mkdir -p "$TEST_OUTPUT/viz_vehicle"
mkdir -p "$(dirname "$SAVE_PATH/result.json")"
cd "$REPO_ROOT"

# 1. Load shared profile presets from f2d_env_config.sh
{profiles_cmd}
{default_env_vars_str}

# 2. Apply test-specific overrides
{env_vars_str}

BENCHMARK_ROUTE_ID="${{BENCHMARK_ROUTE_ID:-BENCHMARK_ROUTE_ID}}_${{SLURM_JOB_ID:-manual}}"
echo "Job ID: ${{SLURM_JOB_ID:-manual}}"
echo "Node: $(hostname)"
echo "Test: {name}"
echo "Profiles: {run.profiles}"
echo "Output: $TEST_OUTPUT"

{env_activate_cmd}

: "${{CARLA_ROOT:?CARLA_ROOT is not set. Submit from the activated fail2drive environment.}}"

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

# --- Resolve allocated GPUs once, before any override ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALLOC_GPUS="${{CUDA_VISIBLE_DEVICES:-${{SLURM_JOB_GPUS:-${{SLURM_STEP_GPUS:-}}}}}}"
if [[ -z "$ALLOC_GPUS" ]]; then
  ALLOC_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
IFS=',' read -r -a GPU_LIST <<< "$ALLOC_GPUS"
EVALUATOR_GPU="${{GPU_LIST[0]}}"
AUX_GPU="${{GPU_LIST[1]:-${{GPU_LIST[0]}}}}"   # Alpamayo / depth; shares GPU 0 if only one was allocated
echo "Allocated GPUs: $ALLOC_GPUS  evaluator=$EVALUATOR_GPU aux=$AUX_GPU"

# Start Depth Anything V2 only for a Depth-TTC policy.
DEPTH_PID=""
ALPAMAYO_PID=""
if [[ "${{VLM_BACKEND:-}}" == "alpamayo_server" ]]; then
  : "${{ALPAMAYO_PY:?ALPAMAYO_PY is not set.}}"
  if [[ ! -x "$ALPAMAYO_PY" ]]; then
    echo "Missing Alpamayo Python: $ALPAMAYO_PY" >&2
    exit 1
  fi
  if [[ -z "${{HF_TOKEN:-}}" ]] && command -v hf >/dev/null 2>&1; then
    HF_TOKEN="$(hf auth token 2>/dev/null || true)"
    export HF_TOKEN
  fi

  ALPAMAYO_PORT_BASE=$((5000 + (SLURM_JOB_ID % 100) * 50))
  ALPAMAYO_PORT=$(free_port "$ALPAMAYO_PORT_BASE")
  if [[ -z "$ALPAMAYO_PORT" ]]; then
    echo "Could not find a free Alpamayo port near $ALPAMAYO_PORT_BASE" >&2
    exit 1
  fi
  export ALPAMAYO_PORT

  ALPAMAYO_LOG="$TEST_OUTPUT/alpamayo_server.log"
  CUDA_VISIBLE_DEVICES="$AUX_GPU" \
    "$ALPAMAYO_PY" "$REPO_ROOT/tools/alpamayo_vlm_server.py" \
      --host "${{ALPAMAYO_HOST:-127.0.0.1}}" \
      --port "$ALPAMAYO_PORT" \
      --device cuda:0 \
      --model "${{ALPAMAYO_MODEL:-nvidia/Alpamayo-1.5-10B}}" \
      --alpamayo-version "${{ALPAMAYO_VERSION:-1.5}}" \
      --max-new-tokens "${{ALPAMAYO_MAX_NEW_TOKENS:-256}}" \
      >"$ALPAMAYO_LOG" 2>&1 &
  ALPAMAYO_PID=$!

  ALPAMAYO_URL="http://${{ALPAMAYO_HOST:-127.0.0.1}}:$ALPAMAYO_PORT"
  echo "Started Alpamayo server pid=$ALPAMAYO_PID port=$ALPAMAYO_PORT log=$ALPAMAYO_LOG"
  for _ in $(seq 1 180); do
    if curl -sf "$ALPAMAYO_URL/health" >/dev/null; then
      break
    fi
    if ! kill -0 "$ALPAMAYO_PID" 2>/dev/null; then
      echo "Alpamayo server exited before becoming ready. Last log lines:" >&2
      tail -100 "$ALPAMAYO_LOG" >&2 || true
      exit 1
    fi
    sleep 5
  done
  if ! curl -sf "$ALPAMAYO_URL/health" >/dev/null; then
    echo "Timed out waiting for Alpamayo server at $ALPAMAYO_URL" >&2
    tail -100 "$ALPAMAYO_LOG" >&2 || true
    exit 1
  fi
  export VLM_SERVER_URL="$ALPAMAYO_URL/generate"
  export ALPAMAYO_SERVER_URL="$VLM_SERVER_URL"
  export ALPAMAYO_RAW_LOG_PATH="${{ALPAMAYO_RAW_LOG_PATH:-$TEST_OUTPUT/alpamayo_raw_responses.txt}}"
  export ALPAMAYO_INPUT_SAVE_DIR="${{ALPAMAYO_INPUT_SAVE_DIR:-$TEST_OUTPUT/alpamayo_inputs}}"
  mkdir -p "$ALPAMAYO_INPUT_SAVE_DIR"
fi

if [[ "${{ACTIVATION_POLICY:-}}" == "depth_ttc" || "${{STEERING_POLICY:-}}" == "depth_ttc" ]]; then
  DEPTH_PORT_BASE=$((40000 + (SLURM_JOB_ID % 200) * 100))
  DEPTH_PORT=$(free_port "$DEPTH_PORT_BASE")
  if [[ -z "$DEPTH_PORT" ]]; then
    echo "Could not find a free Depth-TTC port near $DEPTH_PORT_BASE" >&2
    exit 1
  fi
  export DEPTH_PORT
  export DEPTH_TTC_SERVER_URL="http://127.0.0.1:${{DEPTH_PORT}}/score"
  export DEPTH_TTC_EXTERNAL_SERVER=1
  export DEPTH_TTC_DEBUG_DIR="${{DEPTH_TTC_DEBUG_DIR:-$TEST_OUTPUT/depth_ttc_debug}}"
  export DEPTH_TTC_DEBUG_SAVE_EVERY="${{DEPTH_TTC_DEBUG_SAVE_EVERY:-1}}"
  export DEPTH_TTC_DEBUG_MAX_POINTS="${{DEPTH_TTC_DEBUG_MAX_POINTS:-80000}}"
  mkdir -p "$DEPTH_TTC_DEBUG_DIR"

  DEPTH_LOG="$TEST_OUTPUT/depth_ttc_server.log"
  CUDA_VISIBLE_DEVICES="$AUX_GPU" \
    python -u tools/depth_ttc_server.py \
    --host 127.0.0.1 \
    --port "$DEPTH_PORT" \
    --device cuda:0 \
    --depth-anything-root "$DEPTH_ROOT" \
    --checkpoint "$DEPTH_CHECKPOINT" \
    --encoder vitl \
    --max-depth 80 \
    >"$DEPTH_LOG" 2>&1 &
  DEPTH_PID=$!

  sleep 5
  if ! kill -0 "$DEPTH_PID" 2>/dev/null; then
    echo "Depth-TTC server failed to start:" >&2
    tail -100 "$DEPTH_LOG" >&2 || true
    exit 1
  fi
  echo "Depth-TTC server listening at $DEPTH_TTC_SERVER_URL"
fi

echo "Env vars:"
# printenv | grep -E '^(REPO_ROOT|STEERING_|ACTIVATION_|PDM_|BENCHMARK_ROUTE_ID|NO_OTHER_VEHICLES|MAX_CARLA_FRAME|DEPTH_|.*_ACTIVATION_ALPHA_SCALE)' | sort
printenv | sort
echo

if [[ "${{VLM_BACKEND:-}}" == "alpamayo_server" ]]; then
  export CUDA_VISIBLE_DEVICES="$EVALUATOR_GPU"
fi

# For debugging GPU memory usage, print the GPU list and memory usage before starting CARLA.
nvidia-smi -L; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
vulkaninfo --summary 2>&1 | grep -E 'deviceName|GPU[0-9]'

CARLA_LOG_DIR="$TEST_OUTPUT/carla_log"
mkdir -p "$CARLA_LOG_DIR"
CARLA_LOG="$CARLA_LOG_DIR/carla_log_${{SLURM_JOB_ID:-manual}}.log"

"$CARLA_ROOT/CarlaUE4.sh" \
  -carla-rpc-port="$FREE_WORLD_PORT" \
  -nosound -RenderOffScreen -carla-primary-port=0 \
  -graphicsadapter="${{CARLA_GRAPHICS_ADAPTER:-$EVALUATOR_GPU}}" \
  -carla-streaming-port="$FREE_STREAMING_PORT" \
  >"$CARLA_LOG" 2>&1 &

CARLA_PID=$!
echo "Started CARLA pid=$CARLA_PID log=$CARLA_LOG"

cleanup() {{
  if [[ -n "${{ALPAMAYO_PID:-}}" ]]; then
    kill "$ALPAMAYO_PID" 2>/dev/null || true
    wait "$ALPAMAYO_PID" 2>/dev/null || true
  fi
  kill "$CARLA_PID" 2>/dev/null || true
  wait "$CARLA_PID" 2>/dev/null || true
  if [[ -n "${{DEPTH_PID:-}}" ]]; then
    kill "$DEPTH_PID" 2>/dev/null || true
    wait "$DEPTH_PID" 2>/dev/null || true
  fi
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


def run_test(run: SlurmQuickValidateRun, dry_run: bool = False) -> None:
    output_dir = run.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(run.name)

    test_output = output_dir / name
    test_output.mkdir(parents=True, exist_ok=True)

    job_file = run_dir / f"{name}.sh"
    job_file.write_text(make_job_script(run, test_output), encoding="utf-8")
    job_file.chmod(0o755)

    if dry_run:
        print(f"[dry-run] {job_file}")
        return

    result = subprocess.run(
        ["sbatch", str(job_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"{name}: {result.stdout.strip()}")


def main() -> None:
    args = parse_args()

    names_to_test: list[str] = [
        # "f2d_tfv6_vlm_alpamayo_ConstructionPedestrian_0011",
        # "f2d_1020_hipad",
        # "f2d_tfv6_occupancy_dec5_fix_brake",
        # "f2d_tfv6_occupancy_dec5_fix_left_steering",
        # "f2d_tfv6_occupancy_dec5_fix_right_steering",
        # "f2d_tfv6_occupancy_dec5_oracle",

        # "f2d_tfv6_fused_fix_brake",
        # "f2d_tfv6_fused_fix_left_steering",
        # "f2d_tfv6_fused_fix_right_steering",

        # "f2d_tfv6_no_policy",
        # "f2d_tfv6_no_policy_test_CustomObstacles_0024",

        # "f2d_tfv6_depth_ttc_fused_BadParking_0002",
        # "f2d_hipad_fixed_1085_brake_all_layers",
        # "f2d_hipad_oracle_1095",
    ]

    for run in RUNS:
        if safe_name(run.name) in names_to_test:
            run_test(run, dry_run=args.dry_run)

    # test_hipad_steering_feature_runs(args.dry_run)

    # run_tfv6_test(
    #     steering_feature="fused_features",
    #     # steering_feature="decoder_layer4",
    #     # steering_feature="decoder_layer5",
    #     # steering_feature="decoder_layer6",
    #     routes=[
    #         # "./fail2drive_split/Base_BadParking_0001.xml",
    #         # "./fail2drive_split/Base_BadParking_0005.xml",
    #         # "./fail2drive_split/Base_ConstructionPedestrian_0010.xml",
    #         # "./fail2drive_split/Base_ConstructionPedestrian_0011.xml",
    #         "./fail2drive_split/Base_ConstructionPedestrian_0012.xml",  # pass but slow
    #         # "./fail2drive_split/Base_ConstructionPedestrian_0013.xml",  # fail
    #         # "./fail2drive_split/Base_ConstructionPedestrian_0014.xml",
    #         # "./fail2drive_split/Base_CustomObstacles_0020.xml",
    #         # "./fail2drive_split/Base_CustomObstacles_0021.xml",
    #         # "./fail2drive_split/Base_HardBrake_0036.xml",  # test for emergency brake
    #         # "./fail2drive_split/Generalization_Wall_1096.xml",
    #         # "./fail2drive_split/Generalization_Wall_1095.xml",
    #     ],
    #     profiles=["common", "tfv6", "oracle", "debug"],
    #     env_vars={
    #         "MAX_CARLA_FRAME": "1600",
    #     },
    #     dry_run=args.dry_run,
    # )

    run_hipad_test(
        steering_feature="align_query",
        routes=[
            # "./fail2drive_split/Base_BadParking_0001.xml",
            "./fail2drive_split/Base_BadParking_0004.xml",
            # "./fail2drive_split/Base_BadParking_0005.xml",
            # "./fail2drive_split/Base_ConstructionPedestrian_0010.xml",
            # "./fail2drive_split/Base_ConstructionPedestrian_0011.xml",
            # "./fail2drive_split/Base_ConstructionPedestrian_0012.xml",
            # "./fail2drive_split/Base_ConstructionPedestrian_0013.xml",
            # "./fail2drive_split/Base_ConstructionPedestrian_0014.xml",
            "./fail2drive_split/Base_ConstructionPedestrian_0016.xml",
            # "./fail2drive_split/Base_CustomObstacles_0020.xml",
            # "./fail2drive_split/Base_CustomObstacles_0021.xml",
            # "./fail2drive_split/Base_HardBrake_0036.xml",  # test for emergency brake
            # "./fail2drive_split/Generalization_Wall_1096.xml",
        ],
        profiles=["common", "hipad", "oracle", "debug"],
        env_vars={
            "MAX_CARLA_FRAME": "1100",
        },
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
