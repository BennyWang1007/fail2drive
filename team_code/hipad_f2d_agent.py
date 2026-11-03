"""Fail2Drive adapter for HiP-AD's Bench2Drive closed-loop agent."""


import copy
import importlib.util
import json
import math
import os
import sys
import time
import types
from collections import deque
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from PIL import Image

from depth_ttc_gate import DepthTTCClient
from vlm_gate import AsyncVLMGate
from activation_steering.policy import (
    DepthTTCPolicy,
    depth_ttc_enabled_from_env,
    vlm_enabled_from_env,
    policy_from_env,
)
from activation_steering.feature_collectors.hipad_feature_collector import HiPADFeatureCollector
from activation_steering.steering_manager.hipad_steering_manager import HiPADSteeringManager


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAIL2DRIVE_ROOT = _REPO_ROOT / "fail2drive"
_HIPAD_ROOT = _REPO_ROOT / "HiP-AD"
_HIPAD_LEADERBOARD = _HIPAD_ROOT / "bench2drive" / "leaderboard"
_HIPAD_TEAM_CODE = _HIPAD_LEADERBOARD / "team_code"
_HIPAD_AGENT = _HIPAD_LEADERBOARD / "team_code" / "hipad_b2d_agent.py"
_F2D_LEADERBOARD = _FAIL2DRIVE_ROOT / "leaderboard"
_F2D_SCENARIO_RUNNER = _FAIL2DRIVE_ROOT / "scenario_runner"
_F2D_CARLA_API = _FAIL2DRIVE_ROOT / "f2d_carla" / "PythonAPI"
_F2D_CARLA_AGENTS = _F2D_CARLA_API / "carla"


def _prepend_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


for _path in reversed(
    (
        _F2D_LEADERBOARD,
        _HIPAD_LEADERBOARD,
        _HIPAD_ROOT,
        _F2D_SCENARIO_RUNNER,
        _F2D_CARLA_API,
        _F2D_CARLA_AGENTS,
    )
):
    _prepend_path(_path)

os.environ.setdefault("IS_BENCH2DRIVE", "1")
os.environ.setdefault("SAVE_PATH", str(_FAIL2DRIVE_ROOT / "results" / "hipad_f2d" / "vis"))
os.environ.setdefault("LEADERBOARD_ROOT", str(_F2D_LEADERBOARD))
os.environ.setdefault("SCENARIO_RUNNER_ROOT", str(_F2D_SCENARIO_RUNNER))

if not _HIPAD_AGENT.exists():
    raise FileNotFoundError(f"HiP-AD agent not found: {_HIPAD_AGENT}")

_team_code_module = types.ModuleType("team_code")
_team_code_module.__path__ = [str(_HIPAD_TEAM_CODE)]
sys.modules["team_code"] = _team_code_module

_spec = importlib.util.spec_from_file_location("_hipad_b2d_agent", _HIPAD_AGENT)
_hipad_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _hipad_module
_spec.loader.exec_module(_hipad_module)

_BaseHiPADAgent = _hipad_module.SparseAgent
_BaseHiPADAgent = cast(type, _BaseHiPADAgent)
from bench2drive.leaderboard.team_code.hipad_b2d_agent import SparseAgent as _BaseHiPADAgent


def _env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "t", "yes", "y")


# ---- Depth-TTC adapter constants (HiP-AD side) ----
# Output names of HiP-AD's plan heads (same ones hipad_b2d_agent.run_step feeds to its PID controller).
_PLAN_TEMP = "plan_speed_5hz"  # temporal plan, one point every 0.2 s, lidar frame (x right, y forward)
_PLAN_SPAT = "plan_spat_2m"  # spatial plan, one point every 2 m, same frame
_PLAN_DT = 0.2
_DEPTH_TTC_SPEED_HORIZON_STEPS = 10  # speed = path length of the first 2 s of the temporal plan / 2 s
# DepthTTCPolicy reads these from `training_config` (TFv6 has them in its config, HiP-AD does not).
# CAM_FRONT of hipad_b2d_agent.sensors(): 1600x900, fov 70, x=0.8 y=0 z=1.6.  Ego box 1.84 x 4.89 (visualize()).
_DEPTH_TTC_CAMERA = {"fov": 70.0, "pos": [0.80, 0.0, 1.60], "width": 1600, "height": 900}
_DEPTH_TTC_EGO_EXTENT = (4.89 / 2.0, 1.84 / 2.0)  # half extents (x forward, y sideways), like CARLA's bounding_box.extent
_DEPTH_TTC_SPEED_CLASSES = [round(0.5 * i, 1) for i in range(0, 41)]  # 0 .. 20 m/s; plan speed is one-hot on these
# HiP-AD plans are (x right, y forward); the Depth-TTC server was written for TFv6 ego coordinates
# (x forward, y right). Set HIPAD_DEPTH_TTC_SWAP_XY=0 if the server turns out to expect HiP-AD's frame.
_DEPTH_TTC_SWAP_XY = _env_flag("HIPAD_DEPTH_TTC_SWAP_XY", "1")
# Non-parameter state of an nn.Module lives in vars(module) minus these (see _snapshot_model_state).
_MODULE_INTERNALS = frozenset(vars(torch.nn.Module()).keys())


def get_entry_point():
    return "Fail2DriveHiPADAgent"


class Fail2DriveHiPADAgent(_BaseHiPADAgent):
    """Adapts HiP-AD to fail2drive's local leaderboard API."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.route_index = route_index
        self.traffic_manager = traffic_manager
        self.visual_save_path = None

        # Everything destroy() touches is initialised FIRST. The leaderboard calls destroy() even
        # when setup() raised; before this, a failure in the policy setup below was followed by
        # "AttributeError: ... no attribute '_steering_hooks'", which hid the real error.
        self._base_setup_done = False
        self.feature_collector = None
        self.steering_manager = None
        self.steering_manager_alpha = None
        self._steering_hooks = []
        self._steering_vectors = {}
        self.activation_policy = None
        self.depth_ttc_client = None
        self._depth_ttc_input = None  # raw CARLA input_data of the step being executed
        self._depth_ttc_busy = False  # True while candidate forwards run inside the model pre-hook
        self._depth_ttc_expected = None  # plan of the selected candidate, to verify the real forward
        self._depth_ttc_snapshot_s = None

        self.vlm_steering = False
        self.depth_ttc_steering = False
        self.vlm_gate: AsyncVLMGate | None = None
        self.vlm_last_decision = None
        self.vlm_last_decision_key = None
        self.vlm_decision_start_frame = None
        self.vlm_decision_age_frames = None
        self.vlm_history_steps = int(os.environ.get("VLM_ALPAMAYO_HISTORY_STEPS", 16))
        self.vlm_history_stride = int(os.environ.get("VLM_ALPAMAYO_HISTORY_STRIDE_FRAMES", 2))
        self.vlm_min_speed = float(os.environ.get("VLM_MIN_SPEED_MPS", 3.5))
        # tick_data["imgs"]["CAM_FRONT"] in SparseAgent is already RGB; set VLM_INPUT_IS_BGR=1 only if feeding BGR.
        self.vlm_input_is_bgr = _env_flag("VLM_INPUT_IS_BGR", "0")
        self._ego_pose_history: deque[tuple[float, float, float]] = deque(
            maxlen=max(64, self.vlm_history_steps * self.vlm_history_stride + 4)
        )
        self._last_tick_data: dict | None = None

        # Agent-side collection/steering (TFv6-style). Set HIPAD_LEGACY_HOOKS=1 to fall back
        # to the old behaviour where the HiP-AD model itself dumps features / injects vectors.
        self.legacy_hooks = _env_flag("HIPAD_LEGACY_HOOKS", "0")

        # activation_steering
        self.activation_steering = os.environ.get("STEERING_VECTORS_DIR") is not None
        self.activation_action_log = None
        self.hipad_activation_alpha = None
        self.hipad_activation_plan_shift = 0.0

        if self.activation_steering:
            print("[Fail2DriveHiPADAgent] Activation steering enabled (STEERING_VECTORS_DIR is set).", flush=True)
            # Same order as SensorAgent.setup: the Depth-TTC policy needs its client (it talks to the
            # depth server started by the job script); policy_from_env raises without it.
            self.depth_ttc_steering = depth_ttc_enabled_from_env()
            self.vlm_steering = vlm_enabled_from_env()
            print("[Fail2DriveHiPADAgent] DEPTH_TTC_STEERING:", self.depth_ttc_steering, flush=True)
            print("[Fail2DriveHiPADAgent] VLM_STEERING:", self.vlm_steering, flush=True)
            if self.depth_ttc_steering:
                self.depth_ttc_client = DepthTTCClient.from_env()
                print(f"[Fail2DriveHiPADAgent] Depth TTC server: {self.depth_ttc_client.server_url}", flush=True)
            self.activation_policy = policy_from_env(
                vlm_enabled=self.vlm_steering,
                depth_ttc_client=self.depth_ttc_client,
                # Only DepthTTCPolicy reads it; TFv6's training_config does not exist for HiP-AD.
                training_config=self._depth_ttc_training_config() if self.depth_ttc_steering else None,
            )
            print(
                f"[Fail2DriveHiPADAgent] Loaded activation policy: {self.activation_policy.__class__.__name__}",
                flush=True,
            )

        # HiP-AD plan feature collection
        self.save_plan_features = _env_flag("SAVE_HIPAD_PLAN_FEATURES")
        os.environ["ROUTES"] = str(route_index or os.environ.get("ROUTES", "fail2drive_hipad"))
        save_path = Path(os.environ["SAVE_PATH"]).resolve()
        os.environ["SAVE_PATH"] = str(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        parts = path_to_conf_file.split("+")
        if len(parts) < 2:
            raise ValueError("HiP-AD agent config must be '<config.py>+<checkpoint.pth>[+save_name]'.")
        if len(parts) == 2:
            path_to_conf_file = path_to_conf_file + "+hipad_f2d"

        cwd = os.getcwd()
        os.chdir(_HIPAD_ROOT)
        try:
            super().setup(path_to_conf_file)
        finally:
            os.chdir(cwd)
        self._base_setup_done = True
        self.is_visualize = _env_flag("DEBUG_CHALLENGE")
        self.front_traj_vis_only = _env_flag("FRONT_TRAJ_VIS_ONLY", "0")
        if not self.is_visualize:
            self._remove_empty_log_visual_dirs()
        self.activation_action_log = self.save_path / "activation_actions.jsonl"
        self._configure_visual_output_path()
        if not self.legacy_hooks:
            self._setup_feature_collector()  # attach first: its pre-hook sees the un-steered inputs
            if self.activation_steering:
                self._setup_steering_manager()  # attach second: its pre-hook then shifts the plan inputs
        if isinstance(self.activation_policy, DepthTTCPolicy):
            self._setup_depth_ttc()
        if self.vlm_steering:
            print(
                f"[Fail2DriveHiPADAgent] VLM_STEERING enabled (VLM_MIN_SPEED_MPS={self.vlm_min_speed})",
                flush=True,
            )
            self.vlm_gate = AsyncVLMGate.from_env()
            self.vlm_gate.start()

    # ------------------------------------------------------------------ VLM & Ego History
    def _update_ego_history(self, tick_data: dict) -> None:
        pos = tick_data.get("pos")
        if pos is None:
            return
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos_arr.size < 2:
            return
        compass = tick_data.get("compass", 0.0)
        raw_theta = 0.0 if (compass is None or math.isnan(float(compass))) else float(compass)
        # In CARLA GPS/compass frame: +x_world is North (pos[0]), +y_world is East (-pos[1] in SparseAgent's convention)
        # SparseAgent uses ego_pos = [pos[0], -pos[1]] and theta_to_lidar = raw_theta.
        self._ego_pose_history.append((float(pos_arr[0]), float(-pos_arr[1]), raw_theta))

    @staticmethod
    def _yaw_rotation_matrix(yaw: float) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _vlm_ego_history(self):
        """Ego history in current ego frame [x_forward, y_right, z] for Alpamayo VLM."""
        if not self._ego_pose_history:
            return None, None
        curr_x, curr_y, curr_theta = self._ego_pose_history[-1]
        cos_t = math.cos(curr_theta)
        sin_t = math.sin(curr_theta)

        steps, stride = self.vlm_history_steps, self.vlm_history_stride
        xyz = np.zeros((steps, 3), dtype=np.float32)
        rot = np.zeros((steps, 3, 3), dtype=np.float32)
        n = len(self._ego_pose_history)
        for k in range(steps):
            back = stride * (steps - 1 - k)
            idx = max(0, n - 1 - back)
            px, py, ptheta = self._ego_pose_history[idx]
            dx = px - curr_x
            dy = py - curr_y
            # Match SparseAgent's rotation_matrix:
            # x_lidar (right) = cos_t * dx - sin_t * dy
            # y_lidar (forward) = sin_t * dx + cos_t * dy
            x_forward = sin_t * dx + cos_t * dy
            y_right = cos_t * dx - sin_t * dy
            rel_yaw = ptheta - curr_theta
            xyz[k, 0] = x_forward
            xyz[k, 1] = y_right
            rot[k] = self._yaw_rotation_matrix(rel_yaw)
        return xyz.reshape(1, 1, steps, 3).tolist(), rot.reshape(1, 1, steps, 3, 3).tolist()

    def _front_rgb_image(self, tick_data: dict) -> np.ndarray:
        image = np.asarray(tick_data["imgs"]["CAM_FRONT"])
        if self.vlm_input_is_bgr:
            image = image[..., ::-1]
        return np.ascontiguousarray(image)

    @staticmethod
    def _vlm_command_id(tick_data: dict) -> int:
        cmd = int(tick_data.get("command", 4))
        if cmd < 1 or cmd > 6:
            cmd = 4
        return cmd

    @staticmethod
    def _ego_target_point(tick_data: dict) -> np.ndarray:
        """Compute target_point in ego frame [x_forward, y_right]."""
        pos = np.asarray(tick_data["pos"], dtype=np.float32).reshape(-1)
        target_raw = np.asarray(tick_data["target_xy"], dtype=np.float32).reshape(-1)
        ego_pos = [float(pos[0]), float(-pos[1])]
        target_xy = [float(target_raw[0]), float(-target_raw[1])]
        compass = tick_data.get("compass", 0.0)
        raw_theta = 0.0 if (compass is None or np.isnan(float(compass))) else float(compass)
        rel_xy = np.array([target_xy[0] - ego_pos[0], target_xy[1] - ego_pos[1]], dtype=np.float32)
        rotation_matrix = np.array(
            [
                [np.cos(raw_theta), -np.sin(raw_theta)],
                [np.sin(raw_theta), np.cos(raw_theta)],
            ],
            dtype=np.float32,
        )
        # In SparseAgent, target_lidar = [x_right, y_forward]
        target_lidar = rotation_matrix @ rel_xy
        return np.array([float(target_lidar[1]), float(target_lidar[0])], dtype=np.float32)

    @classmethod
    def _alpha_active(cls, activation_alpha) -> bool:
        values = cls._json_activation_alpha(activation_alpha)
        if values is None:
            return False
        if isinstance(values, list):
            return any(abs(v) > 1e-6 for v in values)
        return abs(values) > 1e-6

    def _vlm_policy_alpha(self, frame: int):
        assert self.vlm_gate is not None
        decision = self.vlm_gate.latest()
        age = None
        if decision is not None:
            decision_key = (decision.frame_id, decision.timestamp)
            if decision_key != self.vlm_last_decision_key:
                self.vlm_last_decision_key = decision_key
                self.vlm_decision_start_frame = frame
            age = frame - self.vlm_decision_start_frame
        self.vlm_last_decision = decision
        self.vlm_decision_age_frames = age

        alpha = self.activation_policy.alpha(frame, decision, decision_age_frames=age)
        if self.vlm_gate.verbose:
            if decision is None:
                print(f"[VLMGate] use step={frame} no completed decision yet", flush=True)
            elif self._alpha_active(alpha):
                print(
                    f"[VLMGate] use step={frame} decision_frame={decision.frame_id} "
                    f"age={age} action={getattr(decision, 'action', 'none')} reason={decision.reason}",
                    flush=True,
                )
        return alpha

    def _vlm_submit_frame(self, frame: int, tick_data: dict) -> None:
        if self.vlm_gate is None:
            return
        front_image = self._front_rgb_image(tick_data)
        speed = float(tick_data.get("speed", 0.0))
        if self.vlm_min_speed <= 0.0 or speed > self.vlm_min_speed:
            history_xyz, history_rot = self._vlm_ego_history()
            self.vlm_gate.submit(
                frame_id=frame,
                rgb_image=front_image,
                speed=speed,
                command=self._vlm_command_id(tick_data),
                target_point=self._ego_target_point(tick_data),
                ego_history_xyz=history_xyz,
                ego_history_rot=history_rot,
            )
        else:
            self.vlm_gate.observe(frame, front_image)

    def _vlm_log_fields(self) -> dict:
        if self.vlm_gate is None:
            return {}
        decision = self.vlm_last_decision
        return {
            "vlm_decision_frame": None if decision is None else int(decision.frame_id),
            "vlm_decision_age_frames": self.vlm_decision_age_frames,
            "vlm_action": None if decision is None else str(decision.action),
            "vlm_reason": None if decision is None else str(decision.reason)[:240],
            "vlm_error": None if decision is None else str(decision.error),
        }

    # ------------------------------------------------------------------ Depth-TTC
    # SensorAgent.run_step hands DepthTTCPolicy.decide() a predict(alpha) closure and runs control() on
    # the winner. HiP-AD's run_step builds its model inputs and calls the model in one piece, so the same
    # thing is done from a forward pre-hook on the model: the hook sees the exact inputs of the real
    # forward, runs one steered forward per candidate, lets the policy pick, and leaves the chosen
    # steering vectors in place for the real forward (which then feeds the unchanged PID controller).
    @staticmethod
    def _depth_ttc_training_config():
        return types.SimpleNamespace(
            camera_calibration={1: dict(_DEPTH_TTC_CAMERA)},
            ego_extent_x=_DEPTH_TTC_EGO_EXTENT[0],
            ego_extent_y=_DEPTH_TTC_EGO_EXTENT[1],
            target_speed_classes=list(_DEPTH_TTC_SPEED_CLASSES),
        )

    def _setup_depth_ttc(self):
        if self.legacy_hooks or self.steering_manager is None:
            raise ValueError(
                "Depth-TTC needs agent-side steering: set HIPAD_LEGACY_HOOKS=0 and provide "
                "STEERING_VECTORS_DIR with brake/left/right vectors."
            )
        try:
            # Works on PyTorch >= 1.13
            self._steering_hooks.append(
                self.model.register_forward_pre_hook(self._depth_ttc_pre_hook, with_kwargs=True)
            )
        except TypeError as exc:
            if "with_kwargs" not in str(exc):
                raise
            # Fallback for PyTorch < 1.13: pre-hooks do not receive kwargs,
            # so we intercept forward() directly to capture both args and kwargs.
            orig_forward = self.model.forward

            def wrapped_forward(*args, **kwargs):
                self._depth_ttc_pre_hook(self.model, args, kwargs)
                return orig_forward(*args, **kwargs)

            self.model.forward = wrapped_forward

            # Register a dummy handle so destroy() can cleanly restore the original forward
            class _ForwardRestoreHandle:
                def __init__(self, model, orig):
                    self.model = model
                    self.orig = orig

                def remove(self):
                    self.model.forward = self.orig

            self._steering_hooks.append(_ForwardRestoreHandle(self.model, orig_forward))

        print(
            f"[Fail2DriveHiPADAgent] Depth-TTC candidate selection attached "
            f"(swap_xy={_DEPTH_TTC_SWAP_XY}, snapshot={_env_flag('HIPAD_DEPTH_TTC_SNAPSHOT', '1')})"
        )

    @staticmethod
    def _snapshot_model_state(model):
        """Deep copy of every non-parameter attribute of every submodule (temporal caches live there)."""
        snapshot = []
        for module in model.modules():
            state = {}
            for key, value in vars(module).items():
                if key in _MODULE_INTERNALS or isinstance(value, (torch.nn.Module, torch.nn.Parameter)):
                    continue
                try:
                    state[key] = copy.deepcopy(value)
                except Exception:  # noqa: BLE001 - not copyable -> left as is (not restored)
                    pass
            snapshot.append((module, state))
        return snapshot

    @staticmethod
    def _restore_model_state(snapshot):
        for module, state in snapshot:
            for key, value in state.items():
                vars(module)[key] = copy.deepcopy(value)  # copy again: the snapshot must stay pristine

    @staticmethod
    def _depth_ttc_front_image(input_data):
        """Raw CAM_FRONT, first 3 channels (BGR, like SensorAgent's rgb_2)."""
        cam = input_data["CAM_FRONT"]
        image = cam[1] if isinstance(cam, (tuple, list)) else cam
        return np.asarray(image)[:, :, :3]

    @staticmethod
    def _plan_speed(plan_temp):
        """Mean speed [m/s] over the first 2 s of the temporal plan (points start at the ego origin)."""
        pts = plan_temp.detach().float().cpu().numpy()[:_DEPTH_TTC_SPEED_HORIZON_STEPS, :2]
        if len(pts) == 0:
            return 0.0
        pts = np.vstack([np.zeros((1, 2), dtype=pts.dtype), pts])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() / (len(pts) - 1) / _PLAN_DT)

    def _candidate_prediction(self, outputs):
        """HiP-AD model output -> the two fields DepthTTCPolicy._planner_candidates() reads.

        pred_route: spatial plan (path), pred_target_speed_distribution: one-hot of the temporal plan's speed
        (so the server's uncertainty terms are zero for HiP-AD candidates).
        """
        bbox = outputs[0]["img_bbox"]
        plan_temp = bbox.get(_PLAN_TEMP)
        if plan_temp is None:
            return None  # the policy skips candidates without a route -> scoring is skipped
        plan_route = bbox.get(_PLAN_SPAT, plan_temp)
        route = plan_route.detach().float().cpu()[:, :2]
        if _DEPTH_TTC_SWAP_XY:
            route = route[:, [1, 0]]
        speed_probs = torch.zeros(len(_DEPTH_TTC_SPEED_CLASSES))
        nearest = int(np.abs(np.asarray(_DEPTH_TTC_SPEED_CLASSES) - self._plan_speed(plan_temp)).argmin())
        speed_probs[nearest] = 1.0
        return types.SimpleNamespace(
            pred_route=route,
            pred_target_speed_distribution=speed_probs,
            plan_temp=plan_temp.detach().float().cpu().numpy(),
        )

    def _depth_ttc_pre_hook(self, module, args, kwargs):
        if self._depth_ttc_busy or self._depth_ttc_input is None:
            return None  # candidate forward / not inside run_step
        self._depth_ttc_busy = True
        try:
            self._depth_ttc_decide(args, kwargs)
        finally:
            self._depth_ttc_busy = False
        return None  # inputs of the real forward stay untouched

    def _depth_ttc_decide(self, args, kwargs):
        input_data = self._depth_ttc_input
        policy = cast(DepthTTCPolicy, self.activation_policy)
        steering_manager = cast(HiPADSteeringManager, self.steering_manager)

        frame = int(self.step)  # base tick() already incremented it for this step
        speed = float(input_data["SPEED"][1]["speed"])
        # More than one candidate only on scoring frames; otherwise the real forward below is the only
        # forward that is needed, so predict() must not spend one (decide() ignores that result).
        run_candidates = len(policy.candidates(frame, speed)) > 1
        snapshot = None
        if run_candidates and _env_flag("HIPAD_DEPTH_TTC_SNAPSHOT", "1"):
            started = time.perf_counter()
            snapshot = self._snapshot_model_state(self.model)
            if self._depth_ttc_snapshot_s is None:
                self._depth_ttc_snapshot_s = time.perf_counter() - started
                print(f"[Fail2DriveHiPADAgent] model state snapshot took {self._depth_ttc_snapshot_s * 1000:.0f} ms")
        plans = {}

        def predict(alpha):
            if not run_candidates:
                return None
            if snapshot is not None:
                self._restore_model_state(snapshot)  # every candidate sees the state of the previous frame
            self._steering_vectors = steering_manager.get_layer_vectors(alpha)
            call_args, call_kwargs = copy.deepcopy((args, kwargs))  # the model may edit its inputs in place
            prediction = self._candidate_prediction(self.model(*call_args, **call_kwargs))
            plans[tuple(float(a) for a in alpha)] = None if prediction is None else prediction.plan_temp
            return prediction

        decision = policy.decide(
            frame=frame,
            speed=speed,
            predict=predict,
            get_image=lambda: self._depth_ttc_front_image(input_data),
            input_data=input_data,
        )
        if snapshot is not None:
            self._restore_model_state(snapshot)  # the real forward must start from the pre-step state

        alpha = decision.alpha
        self.hipad_activation_alpha = alpha
        self._steering_vectors = steering_manager.get_layer_vectors(alpha)
        self.steering_manager_alpha = steering_manager.get_steering_alphas(alpha)
        if not any(a > 0.0 for a in self.steering_manager_alpha):
            self.steering_manager_alpha = None
        self._depth_ttc_expected = plans.get(tuple(float(a) for a in alpha))

    def _check_depth_ttc_forward(self):
        """The real forward must reproduce the selected candidate's plan; if not, candidate forwards
        are not side-effect free for this model and the numbers of this run cannot be trusted."""
        expected = self._depth_ttc_expected
        self._depth_ttc_expected = None
        if expected is None:
            return
        actual = np.asarray(self.pid_metadata["plan_temp"], dtype=np.float32)
        if actual.shape != expected.shape:
            diff = float("inf")
        else:
            diff = float(np.abs(actual - expected).max())
        if diff > 1e-3 or _env_flag("HIPAD_AGENT_VERBOSE"):
            tag = "MISMATCH" if diff > 1e-3 else "ok"
            print(
                f"[Fail2DriveHiPADAgent][DepthTTC] re-entrancy check {tag}: frame={self.step} "
                f"max|real - candidate| = {diff:.2e}",
                flush=True,
            )

    def _setup_feature_collector(self):
        if not self.save_plan_features:
            return
        # The model must not dump the same file again.
        os.environ["SAVE_HIPAD_PLAN_FEATURES"] = "0"
        root = os.environ.get("FEATURES_ROOT") or os.environ.get("FUSED_FEATURES_PATH")
        root = Path(root) if root else self.save_path / "fused_features"
        self.feature_collector = HiPADFeatureCollector(
            output_root=root / self.save_path.name,
        )
        self.feature_collector.attach(self.model)
        print(f"[Fail2DriveHiPADAgent] Saving plan features to {root / self.save_path.name}")

    def _setup_steering_manager(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reads STEERING_VECTORS_DIR (root) and HIPAD_STEER_LAYERS (default "5").
        self.steering_manager = HiPADSteeringManager.from_env(device=device)
        if self.steering_manager is None:
            print("[Fail2DriveHiPADAgent] No steering vectors configured; steering disabled.")
            return
        layers = HiPADFeatureCollector.find_plan_layers(self.model)
        for layer_idx in self.steering_manager.steer_layers:
            if layer_idx >= len(layers):
                raise ValueError(f"steer layer {layer_idx} out of range (plan_refine has {len(layers)} layers).")
            self._steering_hooks.append(
                self._register_steering_pre_hook(layers[layer_idx], self._make_steering_pre_hook(layer_idx))
            )
        print(f"[Fail2DriveHiPADAgent] Steering plan layers {self.steering_manager.steer_layers}")

    @staticmethod
    def _register_steering_pre_hook(layer, hook):
        try:
            return layer.register_forward_pre_hook(hook, with_kwargs=True)
        except TypeError as exc:  # older PyTorch: pre-hook gets only (module, args)
            if "with_kwargs" not in str(exc):
                raise

            def legacy_hook(module, args):
                result = hook(module, args, {})
                return None if result is None else result[0]

            return layer.register_forward_pre_hook(legacy_hook)

    def _make_steering_pre_hook(self, layer_idx):
        """Shift ``align_query`` of one plan-refine layer by the steering vector.

        ``align_query`` is not an output of the refine module: it is
        sum(chunk(instance_feature + anchor_embed)[temp/spat groups]) computed inside
        forward (see HiPADFeatureCollector._align_query).  A forward hook on the module
        output therefore cannot steer it, so we edit ``instance_feature`` on the way in.
        """

        def hook(module, args, kwargs):
            vec = self._steering_vectors.get(layer_idx)
            if vec is None:
                return None
            if args:
                instance_feature = args[0]
            elif "instance_feature" in kwargs:
                instance_feature = kwargs["instance_feature"]
            else:
                raise RuntimeError("HiP-AD plan refinement called without instance_feature.")

            steered = self._shift_align_query(module, instance_feature, vec)
            if args:
                return (steered, *args[1:]), kwargs
            return args, {**kwargs, "instance_feature": steered}

        return hook

    @staticmethod
    def _shift_align_query(module, instance_feature, vec):
        """Return instance_feature such that align_query changes by exactly ``vec``.

        The delta is split evenly over the temp/spat anchor groups (the ones summed into
        align_query), so sum(groups) moves by ``vec`` while no single group moves by more
        than vec / n_groups.  Non-temp/spat groups are left untouched.
        """
        group = module.anchor_group
        indices = [i for i, t in enumerate(module.anchor_types) if t[0] in ("temp", "spat")]
        if not indices:
            raise RuntimeError("HiP-AD plan refinement contains no temporal/spatial anchor groups.")
        chunks = list(instance_feature.chunk(group, dim=1))
        if len(chunks) != group:
            raise RuntimeError(
                f"instance_feature of shape {tuple(instance_feature.shape)} does not split into {group} groups."
            )

        # Some offline HiP-AD feature exports retain an extra singleton
        # dimension (for example [1, 1, modes, embed_dims]) while the live
        # align-query input is [batch, modes, embed_dims].  Normalize only
        # those harmless wrapper dimensions; a non-singleton mismatch means
        # the vector was generated for a different feature.
        if vec.ndim > chunks[0].ndim:
            extra_dims = vec.ndim - chunks[0].ndim
            if any(size != 1 for size in vec.shape[:extra_dims]):
                raise ValueError(
                    f"Steering vector shape {tuple(vec.shape)} has incompatible "
                    f"rank for live align_query group shape {tuple(chunks[0].shape)}."
                )
            vec = vec.reshape(vec.shape[extra_dims:])
        elif vec.ndim < chunks[0].ndim:
            raise ValueError(
                f"Steering vector shape {tuple(vec.shape)} has incompatible "
                f"rank for live align_query group shape {tuple(chunks[0].shape)}."
            )
        if tuple(vec.shape[-2:]) != tuple(chunks[0].shape[-2:]):
            raise ValueError(
                f"Steering vector {tuple(vec.shape)} does not match align_query group shape "
                f"{tuple(chunks[0].shape)}."
            )
        share = (vec / len(indices)).to(device=instance_feature.device, dtype=instance_feature.dtype)
        for i in indices:
            chunks[i] = chunks[i] + share
        return torch.cat(chunks, dim=1)

    def tick(self, input_data):
        if "bev" not in input_data:
            input_data = dict(input_data)
            input_data["bev"] = (None, np.zeros((512, 512, 4), dtype=np.uint8))
        tick_data = super().tick(input_data)
        self._last_tick_data = tick_data
        self._update_ego_history(tick_data)
        return tick_data

    def run_step(self, input_data, timestamp: float):
        if _env_flag("HIPAD_AGENT_VERBOSE"):
            print(f"[Fail2DriveHiPADAgent] run_step called at step {getattr(self, 'step', -1)}")
        self._set_plan_feature_env()
        if self.activation_steering:
            self._set_activation_env()
        self._depth_ttc_input = input_data
        self._depth_ttc_expected = None
        try:
            control = super().run_step(input_data, timestamp)
        finally:
            self._depth_ttc_input = None
        self._steering_vectors = {}  # never leak vectors into the next forward pass
        self._check_depth_ttc_forward()
        if self.vlm_gate is not None and self._last_tick_data is not None:
            self._vlm_submit_frame(frame=int(self.step), tick_data=self._last_tick_data)
        if self.feature_collector is not None:
            self.feature_collector.collect(
                model=self.model, frame_idx=int(self.step)
            )
        self._log_activation_action(control)
        return control

    def _set_plan_feature_env(self):
        if getattr(self, "save_path", None) is not None:
            os.environ["HIPAD_PLAN_FEATURE_RUN_ID"] = self.save_path.name
        os.environ["HIPAD_PLAN_FEATURE_FRAME"] = str(int(getattr(self, "step", -1)) + 1)

    def _set_activation_env(self):
        frame = int(getattr(self, "step", -1)) + 1
        if isinstance(self.activation_policy, DepthTTCPolicy):
            # The alpha is decided inside the model pre-hook (_depth_ttc_pre_hook): decide() needs the
            # model inputs that the base run_step only builds right before its forward. Until then run
            # un-steered, and keep the in-model ActivationInjector off.
            os.environ["HIPAD_ACTIVATION_ALPHA"] = "0.0"
            os.environ["HIPAD_ACTIVATION_PLAN_SHIFT"] = "0.0"
            self._steering_vectors = {}
            self.hipad_activation_alpha = None
            self.steering_manager_alpha = None
            self.hipad_activation_plan_shift = 0.0
            return

        if self.vlm_gate is not None:
            self.hipad_activation_alpha = self._vlm_policy_alpha(frame)
        else:
            self.hipad_activation_alpha = self.activation_policy.alpha(frame)

        if self.legacy_hooks:
            os.environ["HIPAD_ACTIVATION_ALPHA"] = self._format_activation_alpha(self.hipad_activation_alpha)
            print(
                f"[Fail2DriveHiPADAgent] Activation alpha (legacy) at frame {frame}: {self.hipad_activation_alpha}",
                flush=True,
            )
        else:
            # The agent-side hooks inject now; keep the in-model ActivationInjector off (alpha=0)
            # so the vectors are not added twice.
            assert self.steering_manager is not None, "Steering manager must be initialized when legacy_hooks=False"
            os.environ["HIPAD_ACTIVATION_ALPHA"] = "0.0"
            steering_alphas = self.steering_manager.get_steering_alphas(self.hipad_activation_alpha)
            print(f"[Fail2DriveHiPADAgent] Activation alpha at frame {frame}: {steering_alphas}", flush=True)
            self._steering_vectors = self.steering_manager.get_layer_vectors(self.hipad_activation_alpha)
            self.steering_manager_alpha = steering_alphas
            if not any(a > 0.0 for a in self.steering_manager_alpha):
                self.steering_manager_alpha = None

        self.hipad_activation_plan_shift = self._activation_plan_shift(self.hipad_activation_alpha)
        os.environ["HIPAD_ACTIVATION_PLAN_SHIFT"] = str(float(self.hipad_activation_plan_shift))

    def _configure_visual_output_path(self):
        if not getattr(self, "is_visualize", False):
            return
        visual_root = os.environ.get("VISUAL_OUTPUT_PATH")
        if visual_root is None or getattr(self, "save_path", None) is None:
            return
        self.visual_save_path = Path(visual_root).resolve() / self.save_path.name
        self.visual_save_path.mkdir(parents=True, exist_ok=True)
        (self.visual_save_path / "images").mkdir(exist_ok=True)
        self._remove_empty_log_visual_dirs()

    def _remove_empty_log_visual_dirs(self):
        for folder in ("images",):  # (was ("images"), a plain str -> looped over its letters)
            try:
                (self.save_path / folder).rmdir()
            except OSError:
                pass

    def visualize(self, tick_data, input_batch, output_batch, pred_planning, target_point):
        if self.front_traj_vis_only:
            return self._visualize_front_temporal_plan(
                tick_data, output_batch, pred_planning
            )
        return self._visualize_legacy(
            tick_data, input_batch, output_batch, pred_planning, target_point
        )

    def _visualize_legacy(self, *args, **kwargs):
        """Run HiP-AD's original six-camera + BEV visualization unchanged."""
        if self.visual_save_path is None:
            return super().visualize(*args, **kwargs)
        log_save_path = self.save_path
        self.save_path = self.visual_save_path

        self.save_path.mkdir(parents=True, exist_ok=True)
        (self.save_path / "images").mkdir(parents=True, exist_ok=True)

        try:
            return super().visualize(*args, **kwargs)
        finally:
            self.save_path = log_save_path

    def _visualize_front_temporal_plan(
        self, tick_data, output_batch, fallback_planning
    ):
        """Save the front camera with only the temporal ego plan overlaid."""
        image = np.ascontiguousarray(tick_data["imgs"]["CAM_FRONT"].copy())
        predictions = output_batch[0]["img_bbox"]
        planning = predictions.get("plan_temp_2hz", fallback_planning)
        if hasattr(planning, "detach"):
            planning = planning.detach().cpu().numpy()
        planning = np.asarray(planning, dtype=np.float32)
        if planning.ndim == 3:
            planning = planning[0]

        if planning.ndim == 2 and planning.shape[1] >= 2 and len(planning) > 0:
            planning = np.concatenate(
                [np.zeros((1, 2), dtype=np.float32), planning[:, :2]], axis=0
            )
            coord3d = np.concatenate(
                [
                    planning,
                    np.full((len(planning), 1), -1.8, dtype=np.float32),
                    np.ones((len(planning), 1), dtype=np.float32),
                ],
                axis=1,
            )
            projected = (self.lidar2img["CAM_FRONT"] @ coord3d.T).T
            valid = projected[:, 2] > 1e-5
            projected = projected[valid, :2] / projected[valid, 2:3]
            height, width = image.shape[:2]
            in_image = (
                (projected[:, 0] >= 0)
                & (projected[:, 0] < width)
                & (projected[:, 1] >= 0)
                & (projected[:, 1] < height)
            )
            points = np.rint(projected[in_image]).astype(np.int32)
            if len(points) >= 2:
                polyline = points.reshape(-1, 1, 2)
                cv2.polylines(image, [polyline], False, (20, 20, 20), 11, cv2.LINE_AA)
                cv2.polylines(image, [polyline], False, (0, 220, 255), 6, cv2.LINE_AA)
            for point in points:
                center = tuple(point)
                cv2.circle(image, center, 7, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(image, center, 4, (0, 220, 255), -1, cv2.LINE_AA)

        output_path = self.visual_save_path or self.save_path
        frame = self.step
        Image.fromarray(image).save(output_path / "images" / ("%04d.png" % frame))

    def _visualize_video(self):
        """Stitch saved frames into a browser/VSCode-compatible H.264 MP4."""
        if not hasattr(self, "is_visualize"):
            print("[Fail2DriveHiPADAgent] _visualize_video skipped: is_visualize attribute not found.")
            return

        print(
            f"[Fail2DriveHiPADAgent] _visualize_video called. "
            f"is_visualize={self.is_visualize}, "
            f"visual_save_path={self.visual_save_path}"
        )

        if not self.is_visualize or self.visual_save_path is None:
            return

        images_dir = self.visual_save_path / "images"
        if not images_dir.exists():
            return

        image_files = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")])
        if not image_files:
            return

        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None:
            return

        height, width = first_frame.shape[:2]

        vid_name = f"viz_{self.save_path.name}.mp4"

        video_path = self.visual_save_path / vid_name
        tmp_video_path = self.visual_save_path / f"_{vid_name}"

        # OpenCV first writes a temporary video.
        # It is converted to H.264 below using ffmpeg.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 20.0

        out = cv2.VideoWriter(
            str(tmp_video_path),
            fourcc,
            fps,
            (width, height),
        )

        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer: {tmp_video_path}")

        try:
            for frame_idx, img_file in enumerate(image_files):
                frame = cv2.imread(str(img_file))

                if frame is None:
                    print(
                        f"[Fail2DriveHiPADAgent] "
                        f"Warning: failed to read {img_file}"
                    )
                    continue

                # Make sure all frames have the same resolution.
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))

                # --------------------------------------------------
                # Draw frame number in the upper-right corner.
                # --------------------------------------------------
                text = f"Frame: {frame_idx}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                margin = 20

                (text_width, text_height), _ = cv2.getTextSize(
                    text,
                    font,
                    font_scale,
                    thickness,
                )

                x = width - text_width - margin
                y = text_height + margin

                # Black outline for readability.
                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 3,
                    cv2.LINE_AA,
                )

                # White text.
                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                out.write(frame)

        finally:
            out.release()

        # Convert to H.264 MP4.
        import subprocess

        print(f"[Fail2DriveHiPADAgent] Encoding H.264 video: {video_path}")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is not installed or not available in PATH. "
                "Please install ffmpeg."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ffmpeg failed to encode the visualization video:\n"
                f"{exc.stderr}"
            ) from exc
        finally:
            # Remove the intermediate mp4v video.
            if tmp_video_path.exists():
                tmp_video_path.unlink()

        if video_path.exists() and video_path.stat().st_size > 0:
            pass
        else:
            raise RuntimeError(
                f"Video encoding did not produce a valid output: {video_path}. "
                f"Original images preserved at {images_dir}"
            )

        print(
            f"[Fail2DriveHiPADAgent] "
            f"Saved visualization video: {video_path}"
        )

    def destroy(self, results=None):
        if getattr(self, "vlm_gate", None) is not None:
            self.vlm_gate.stop()
        for hook in getattr(self, "_steering_hooks", []):
            hook.remove()
        getattr(self, "_steering_hooks", []).clear()
        if getattr(self, "feature_collector", None) is not None:
            self.feature_collector.remove_hooks()
        if not getattr(self, "_base_setup_done", False):
            # setup() failed before HiP-AD's own setup finished: nothing else to clean up, and the
            # base destroy() would only raise a second, misleading error.
            return
        try:
            self._visualize_video()
        finally:
            super().destroy()

    def _log_activation_action(self, control):
        if self.activation_action_log is None:
            return
        metadata = getattr(self, "pid_metadata", {})
        frame = int(getattr(self, "step", -1))
        alpha = self.hipad_activation_alpha if self.legacy_hooks else self.steering_manager_alpha
        record = {
            "frame": frame,
            "speed": float(metadata.get("speed", 0.0)),
            "pred_target_speed": float(metadata.get("desired_speed", 0.0)),
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "stop_for_stop_sign": False,
            "activation_alpha": self._json_activation_alpha(alpha),
            "activation_plan_shift": float(self.hipad_activation_plan_shift),
            "feature_path": self._plan_feature_path(frame),
        }
        if isinstance(self.activation_policy, DepthTTCPolicy):
            policy = self.activation_policy
            held = int(self.step) <= int(policy.active_until)  # active_candidate stays set after the hold expired
            record["depth_ttc_candidate"] = policy.active_candidate if held else "original"
            record["depth_ttc_active_until"] = int(policy.active_until)
        record.update(self._vlm_log_fields())
        with open(self.activation_action_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _json_activation_alpha(activation_alpha):
        if activation_alpha is None:
            return None
        if hasattr(activation_alpha, "detach"):
            activation_alpha = activation_alpha.detach().cpu().flatten().tolist()
        elif isinstance(activation_alpha, np.ndarray):
            activation_alpha = activation_alpha.reshape(-1).tolist()
        elif isinstance(activation_alpha, (list, tuple)):
            activation_alpha = list(activation_alpha)
        else:
            return float(activation_alpha)
        return [float(value) for value in activation_alpha]

    @classmethod
    def _format_activation_alpha(cls, activation_alpha):
        value = cls._json_activation_alpha(activation_alpha)
        if value is None:
            return "0.0"
        if isinstance(value, list):
            return ",".join(str(float(item)) for item in value)
        return str(float(value))

    @classmethod
    def _activation_plan_shift(cls, activation_alpha):
        return 0.0

    @staticmethod
    def _activation_plan_shift_from_action_alpha(alpha_values):
        if len(alpha_values) < 3:
            return 0.0
        left_alpha = abs(float(alpha_values[1]))
        right_alpha = abs(float(alpha_values[2]))
        if left_alpha == 0.0 and right_alpha == 0.0:
            return 0.0
        return 0.0

    @classmethod
    def _infer_scalar_activation_direction(cls):
        for name in ("HIPAD_ACTIVATION_ACTION", "ACTIVATION_ACTION", "ORACLE_ACTION", "STEERING_ACTION"):
            action = cls._normalize_activation_direction(os.environ.get(name))
            if action is not None:
                return action

        for name in (
            "HIPAD_ACTIVATION_VECTOR_PATH",
            "ACTIVATION_VECTOR_PATH",
            "LEFT_ACTIVATION_VECTOR_PATH",
            "RIGHT_ACTIVATION_VECTOR_PATH",
            "ACTIVATION_VECTOR_PATH_LEFT",
            "ACTIVATION_VECTOR_PATH_RIGHT",
        ):
            action = cls._normalize_activation_direction(os.environ.get(name))
            if action is not None:
                return action
        return None

    @staticmethod
    def _normalize_activation_direction(value):
        if value is None:
            return None
        normalized = str(value).lower().replace("-", "_")
        if "right_change_lane" in normalized or "change_lane_right" in normalized:
            return "right"
        if "left_change_lane" in normalized or "change_lane_left" in normalized:
            return "left"
        parts = [part for part in normalized.replace("/", "_").split("_") if part]
        if "right" in parts or "rchange" in parts:
            return "right"
        if "left" in parts or "lchange" in parts:
            return "left"
        return None

    def _plan_feature_path(self, frame):
        # (env is forced to 0 once the agent-side collector takes over, so use the cached flag)
        if not self.save_plan_features:
            return None
        if self.feature_collector is not None:
            return str(self.feature_collector.resolve_feature_path(frame_idx=frame))
        root = os.environ.get("FUSED_FEATURES_PATH")
        if root is None or getattr(self, "save_path", None) is None:
            return None
        return str(Path(root) / self.save_path.name / f"{int(frame):06d}.pt")

    def get_metric_info(self):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        hero_actor = None
        for actor in CarlaDataProvider.get_world().get_actors():
            if actor.attributes.get("role_name") == "hero":
                hero_actor = actor
                break

        if hero_actor is None:
            return {}

        def vector2list(vector, rotation=False):
            if rotation:
                return [vector.roll, vector.pitch, vector.yaw]
            return [vector.x, vector.y, vector.z]

        transform = hero_actor.get_transform()
        return {
            "acceleration": vector2list(hero_actor.get_acceleration()),
            "angular_velocity": vector2list(hero_actor.get_angular_velocity()),
            "forward_vector": vector2list(transform.get_forward_vector()),
            "right_vector": vector2list(transform.get_right_vector()),
            "location": vector2list(hero_actor.get_location()),
            "rotation": vector2list(transform.rotation, rotation=True),
            "speed": hero_actor.get_velocity().length(),
        }
