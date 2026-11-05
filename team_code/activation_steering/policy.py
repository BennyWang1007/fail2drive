from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union, List

import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from depth_ttc_gate import DepthTTCClient, DepthTTCDecision

Trigger = Tuple[str, str, Optional[int], float]
Latch = Tuple[str, int]
Alpha = Optional[Union[float, List[float]]]
AlphaVec = List[float]  # [brake, left, right]
# predict(alpha) -> open-loop prediction (no PID / controller side effects)
PredictFn = Callable[[Alpha], Any]
ImageFn = Callable[[], np.ndarray]


@dataclass
class PolicyDecision:
    """What a policy hands back to the agent for one frame."""

    name: str  # selected candidate name, e.g. "original" / "brake"
    alpha: Alpha  # alpha that produced `prediction`
    prediction: Any  # prediction of the selected candidate
    info: Any = None  # optional extra info (e.g. Depth-TTC decision)


def _env_true(name: str) -> bool:
    return str(os.environ.get(name, "")).lower() in ("1", "true", "t", "yes", "y", "on")


def depth_ttc_enabled_from_env() -> bool:
    """Single source of truth for 'is Depth-TTC requested?' (used by agent + policy_from_env)."""
    policy_name = os.environ.get(
        "ACTIVATION_POLICY", os.environ.get("STEERING_POLICY", "")
    ).lower()
    return _env_true("DEPTH_TTC_STEERING") or policy_name in (
        "depth_ttc", "depth-ttc", "depthttc")


def vlm_enabled_from_env() -> bool:
    """Single source of truth for 'is VLM requested?' (used by agent + policy_from_env)."""
    policy_name = os.environ.get(
        "ACTIVATION_POLICY", os.environ.get("STEERING_POLICY", "")
    ).lower()
    return _env_true("VLM_STEERING") or policy_name in ("vlm", "vlm_gate")


def steering_override_enabled_from_env() -> bool:
    """STEERING_OVERRIDE=1 -> ablation: skip activation steering, override control / trajectory directly."""
    return _env_true("STEERING_OVERRIDE")


def classify_override_alpha(alpha, eps: float = 1e-6):
    """alpha -> None | "brake" | "left" | "right".  Order is [brake, left, right].

    None            : alpha is None or all zeros (policy wants no intervention)
    [a,0,0], a>0    : "brake"
    [0,a,0], a>0    : "left"
    [0,0,a], a>0    : "right"
    anything else   : ValueError
    """
    if alpha is None:
        return None
    if hasattr(alpha, "detach"):
        alpha = alpha.detach().cpu().numpy()
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"STEERING_OVERRIDE expects alpha of length 3 [brake,left,right], got {alpha!r}")
    if np.all(np.abs(a) <= eps):
        return None
    active = np.flatnonzero(a > eps)
    if len(active) == 1 and np.all(np.abs(np.delete(a, active[0])) <= eps):
        return ("brake", "left", "right")[int(active[0])]
    raise ValueError(f"STEERING_OVERRIDE: unsupported alpha {a.tolist()} (need one positive entry, others 0)")


class OverrideTrajectory:
    """Lateral offset profile used for the left/right override.  Convention: lateral > 0 is RIGHT.

    Per side:
      1. if the file exists -> (N, 2) .npy, columns [forward_distance_m, lateral_offset_m (right +)]
      2. otherwise          -> smoothstep lane-change profile (fallback)
                               0 -> STEERING_OVERRIDE_LATERAL [m] over STEERING_OVERRIDE_LENGTH [m]

    Env:
      STEERING_OVERRIDE_TRAJ_LEFT / STEERING_OVERRIDE_TRAJ_RIGHT   optional .npy paths
      STEERING_OVERRIDE_LATERAL (default 3.5), STEERING_OVERRIDE_LENGTH (default 30.0)
    """

    def __init__(self, left_file=None, right_file=None, lateral=3.5, length=30.0):
        self.lateral_max = float(lateral)
        self.length = float(length)
        self.tables = {"left": None, "right": None}
        for side, path in (("left", left_file), ("right", right_file)):
            if path and os.path.isfile(path):
                arr = np.load(path).astype(np.float64)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError(f"{path}: expected (N,2) [forward, lateral], got {arr.shape}")
                self.tables[side] = arr[np.argsort(arr[:, 0])]
                print(f"[OverrideTrajectory] {side}: loaded {path} ({len(arr)} rows)", flush=True)
            else:
                reason = f"file not found: {path}" if path else "no file configured"
                print(
                    f"[OverrideTrajectory] {side}: {reason} -> smoothstep fallback "
                    f"(lateral={self.lateral_max} m over {self.length} m)",
                    flush=True,
                )

    @classmethod
    def from_env(cls):
        return cls(
            left_file=os.environ.get("STEERING_OVERRIDE_TRAJ_LEFT"),
            right_file=os.environ.get("STEERING_OVERRIDE_TRAJ_RIGHT"),
            lateral=float(os.environ.get("STEERING_OVERRIDE_LATERAL", 3.5)),
            length=float(os.environ.get("STEERING_OVERRIDE_LENGTH", 30.0)),
        )

    # def lateral(self, s, side: str):
    #     """s: forward distance [m] (array) -> lateral offset [m] (array, right +)."""
    #     s = np.asarray(s, dtype=np.float64)
    #     table = self.tables[side]
    #     if table is not None:
    #         # np.interp clamps outside the table: the offset stays at the last value beyond its end
    #         return np.interp(s, table[:, 0], table[:, 1])
    #     t = np.clip(s / self.length, 0.0, 1.0)
    #     profile = self.lateral_max * (3.0 * t**2 - 2.0 * t**3)  # smoothstep
    #     return profile if side == "right" else -profile

    def lateral(self, s, side: str):
        """s: forward distance [m] -> lateral offset [m], right +."""

        s = np.asarray(s, dtype=np.float64)
        table = self.tables[side]

        if table is not None:
            return np.interp(s, table[:, 0], table[:, 1])

        t = np.clip(s / self.length, 0.0, 1.0)

        # Front-loaded Beta(3, 6) CDF
        profile = (
            56.0 * t**3
            - 210.0 * t**4
            + 336.0 * t**5
            - 280.0 * t**6
            + 120.0 * t**7
            - 21.0 * t**8
        )

        return (
            self.lateral_max * profile
            if side == "right"
            else -self.lateral_max * profile
        )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _strtobool(value: str | None) -> bool:
    return str(value).lower() in ("1", "true", "t", "yes", "y", "on")


def _parse_points(spec: str) -> list[tuple[float, float]]:
    points = []
    for item in spec.split(","):
        if not item.strip():
            continue
        x_raw, y_raw = item.split(":", maxsplit=1)
        points.append((_clip01(float(x_raw)), float(y_raw)))
    if not points:
        raise ValueError(
            "VLM_ALPHA_POINTS must contain at least one x:y point")
    return sorted(points)


def _interp(points: list[tuple[float, float]], x: float) -> float:
    x = _clip01(x)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


class ActivationPolicy:
    """Maps each agent frame to an activation steering alpha.

    Two levels of API:
      * ``alpha(frame)``  - what simple policies implement (one alpha per frame).
      * ``decide(...)``   - what the agent calls. The default runs ONE forward with
        ``alpha(frame)``. Policies that need several forwards (Depth-TTC) override it,
        so the agent never needs to know which policy is active.
    """

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> Alpha:
        return None  # None == no steering

    def decide(self, frame: int, speed: float, predict: PredictFn,
               get_image: ImageFn | None = None) -> PolicyDecision:
        alpha = self.alpha(frame)
        return PolicyDecision("original", alpha, predict(alpha))

    @staticmethod
    def to_alpha_vec(alpha: Alpha) -> AlphaVec:
        """Convert alpha to a 3-element vector [brake, left, right]."""
        if alpha is None:
            return [0.0, 0.0, 0.0]
        if isinstance(alpha, (float, int)):
            return [float(alpha), float(alpha), float(alpha)]
        if isinstance(alpha, (list, tuple)) and len(alpha) == 3:
            return [float(a) for a in alpha]
        raise ValueError(f"Invalid alpha {alpha!r}: must be None, float, or list/tuple of length 3")


class FixedAfterFramePolicy(ActivationPolicy):
    def __init__(self, alpha: float, start_frame: int, end_frame: int | None = None):
        self.fixed_alpha = float(alpha)
        self.start_frame = int(start_frame)
        self.end_frame = None if end_frame is None or int(
            end_frame) < 0 else int(end_frame)
        steering_command = os.environ.get("STEERING_COMMAND", "none").strip().lower()
        if steering_command not in ("brake", "left", "right", "none"):
            raise ValueError(
                f"STEERING_COMMAND must be one of 'brake', 'left', or 'right', got '{steering_command}'")

        self.steering_vec: AlphaVec
        if steering_command == "brake":
            self.steering_vec = [self.fixed_alpha, 0.0, 0.0]
        elif steering_command == "left":
            self.steering_vec = [0.0, self.fixed_alpha, 0.0]
        elif steering_command == "right":
            self.steering_vec = [0.0, 0.0, self.fixed_alpha]
        else:
            self.steering_vec = [self.fixed_alpha, self.fixed_alpha, self.fixed_alpha]

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> AlphaVec:
        if frame < self.start_frame:
            return [0.0, 0.0, 0.0]
        if self.end_frame is not None and frame > self.end_frame:
            return [0.0, 0.0, 0.0]
        return self.steering_vec


class DepthTTCPolicy(ActivationPolicy):
    """Select an activation-steered planner candidate using Depth-TTC."""

    ACTIONS = ("original", "brake", "left", "right")

    def __init__(self, client: DepthTTCClient, training_config):
        self.client = client
        self.training_config = training_config
        self.every_n = max(1, int(os.environ.get("DEPTH_TTC_EVERY_N", 1)))
        self.hold_frames = max(1, int(os.environ.get("DEPTH_TTC_HOLD_FRAMES", 1)))
        self.hold_frames_brake = max(1, int(os.environ.get("DEPTH_TTC_HOLD_FRAMES_BRAKE", 1)))
        self.hold_frames_left = max(1, int(os.environ.get("DEPTH_TTC_HOLD_FRAMES_LEFT", 1)))
        self.hold_frames_right = max(1, int(os.environ.get("DEPTH_TTC_HOLD_FRAMES_RIGHT", 1)))
        self.last_score_frame: int | None = None
        self.active_until = -1
        self.active_candidate = "original"

        self._cached_alphas = [
            ("original", [0.0, 0.0, 0.0]),
            ("brake", [float(os.environ.get("DEPTH_TTC_BRAKE_ALPHA", 3.0)), 0.0, 0.0]),
            ("right", [0.0, 0.0, float(os.environ.get("DEPTH_TTC_RIGHT_ALPHA", 1.0))]),
            ("left", [0.0, float(os.environ.get("DEPTH_TTC_LEFT_ALPHA", 1.0)), 0.0]),
        ]
        print(
            "[DepthTTCPolicy] loaded; this policy owns candidate generation, "
            "server scoring, and candidate selection",
            flush=True,
        )

    # @staticmethod
    def _candidate_alphas(self) -> list[tuple[str, list[float]]]:
        return self._cached_alphas
        return [
            ("original", [0.0, 0.0, 0.0]),
            ("brake", [float(os.environ.get("DEPTH_TTC_BRAKE_ALPHA", 3.0)), 0.0, 0.0]),
            ("right", [0.0, 0.0, float(os.environ.get("DEPTH_TTC_RIGHT_ALPHA", 1.0))]),
            ("left", [0.0, float(os.environ.get("DEPTH_TTC_LEFT_ALPHA", 1.0)), 0.0]),
        ]

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> Alpha:
        """Alpha of the candidate currently being held (None when nothing is held).

        Depth-TTC normally goes through decide(); this exists so alpha() is always safe to call.
        """
        if frame <= self.active_until and self.active_candidate in ("brake", "left", "right"):
            return dict(self._candidate_alphas())[self.active_candidate]
        return None

    def decide(self, frame: int, speed: float, predict: PredictFn,
               get_image: ImageFn | None = None, input_data: dict | None = None) -> PolicyDecision:
        should_score = self.should_score(frame, speed)
        candidates = self.candidates(frame, speed)
        print(
            f"[DepthTTC] frame={frame} speed={speed:.3f} score={'yes' if should_score else 'no'} "
            f"last_score_frame={self.last_score_frame} active_candidate={self.active_candidate} "
            f"candidates={candidates}",
            flush=True,
        )
        results = {name: predict(alpha) for name, alpha in candidates}
        alphas = dict(candidates)

        selected, info = next(iter(results)), None
        if should_score and len(results) > 1 and get_image is not None:
            try:
                selected, info = self.select(frame, speed, results, get_image(), input_data)
            except Exception as exc:  # scoring server down / bad reply -> keep driving
                print(f"[DepthTTCPolicy] frame={frame} scoring failed ({exc!r}); "
                      "falling back to original", flush=True)
                selected, info = "original", None
                self._activate(frame, selected)
        print(f"[DepthTTC] frame={frame} selected={selected} alpha={alphas[selected]} "
              f"active_until={self.active_until}", flush=True)
        return PolicyDecision(selected, alphas[selected], results[selected], info)

    def should_score(self, frame: int, speed: float) -> bool:
        if float(speed) < float(os.environ.get("DEPTH_TTC_MIN_SPEED_M_S", 0.0)):
            return False
        return (
            self.last_score_frame is None
            or frame - self.last_score_frame >= self.every_n
        )

    def candidates(self, frame: int, speed: float) -> list[tuple[str, list[float]]]:
        if self.should_score(frame, speed):
            return self._candidate_alphas()
        candidate = self.active_candidate
        if frame > self.active_until or candidate not in {"brake", "left", "right"}:
            candidate = "original"
        alpha = dict(self._candidate_alphas()).get(candidate, [0.0, 0.0, 0.0])
        return [(candidate, alpha)]

    @staticmethod
    def _planner_candidates(candidate_results) -> dict:
        planner_candidates = {}
        for name, prediction in candidate_results.items():
            checkpoints = prediction.pred_route
            target_speed_probs = prediction.pred_target_speed_distribution
            if checkpoints is None or target_speed_probs is None:
                continue
            checkpoints = checkpoints.detach().float().cpu().numpy()
            target_speed_probs = target_speed_probs.detach().float().cpu().numpy()
            if checkpoints.ndim == 3:
                checkpoints = checkpoints[0]
            target_speed_probs = target_speed_probs.reshape(-1)
            if checkpoints.ndim != 2 or checkpoints.shape[1] < 2 or not len(target_speed_probs):
                continue
            planner_candidates[name] = {
                "checkpoints": checkpoints[:, :2].tolist(),
                "target_speed_probs": target_speed_probs.tolist(),
            }
        return planner_candidates

    def _activate(self, frame: int, candidate: str) -> None:
        if candidate not in {"brake", "left", "right"}:
            self.active_candidate = "original"
            self.active_until = frame
            return
        self.active_candidate = candidate
        self.active_until = frame + getattr(self, f"hold_frames_{candidate}", self.hold_frames) - 1

    def select(self, frame: int, speed: float, candidate_results, image: np.ndarray, input_data: dict | None = None) -> tuple[str, DepthTTCDecision | None]:
        """Score candidates and return ``(selected_name, decision)``."""
        selected = next(iter(candidate_results))
        should_score = self.should_score(frame, speed)
        planner_candidates = self._planner_candidates(candidate_results)
        if should_score and len(candidate_results) > 1 and planner_candidates:
            self.last_score_frame = frame
            calibration = self.training_config.camera_calibration[1]
            image_height, image_width = image.shape[:2]
            fov = float(calibration["fov"])
            focal_length = image_width / (2.0 * math.tan(math.radians(fov) / 2.0))
            decision = self.client.score(
                frame_id=frame,
                rgb_image=image,
                speed=speed,
                camera={
                    "fx": focal_length,
                    "fy": focal_length,
                    "cx": (image_width - 1) / 2.0,
                    "cy": (image_height - 1) / 2.0,
                    "fov": fov,
                    "position": [float(value) for value in calibration["pos"]],
                },
                ego_extent={
                    "x": float(self.training_config.ego_extent_x),
                    "y": float(self.training_config.ego_extent_y),
                },
                planner_candidates={
                    "candidates": planner_candidates,
                    "target_speeds": [
                        float(value) for value in self.training_config.target_speed_classes
                    ],
                    "uncertainty_weight": 1,
                    "brake_uncertainty_threshold": 0.5,
                },
                # radar=input_data.get("radar", None) if input_data is not None else None
                radar=None
            )
            selected = getattr(decision, "selected_trajectory", "original")
            if selected not in candidate_results:
                print(
                    f"[DepthTTCPolicy] frame={frame} invalid selection {selected!r}; "
                    "falling back to original",
                    flush=True,
                )
                selected = "original"
            self._activate(frame, selected)
            print(
                f"[DepthTTCPolicy] frame={frame} decision={decision} "
                f"selected={selected} active_until={self.active_until}",
                flush=True,
            )
            if selected == "break":
                self.last_score_frame = None  # allow brake to be re-evaluated every frame
            return selected, decision

        if should_score and len(candidate_results) > 1:
            print(
                f"[DepthTTCPolicy] frame={frame} scoring skipped: no valid "
                "planner candidates",
                flush=True,
            )
        return selected, None


class VLMPolicy(ActivationPolicy):
    ACTIONS = ("brake", "left", "right")
    ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}

    def __init__(
        self,
        alpha_max: float = 1.0,
        mapping: str = "linear",
        deadzone: float = 0.0,
        gamma: float = 1.0,
        points: str | None = None,
        decay_frames: int = 10,
        ttl_frames: int | None = None,
    ):
        # VLM steering is binary and action-specific: [brake, left, right].
        # Keep the old constructor arguments accepted so existing launch scripts do
        # not fail, but do not map VLM scores into fractional alphas anymore.
        self.ttl_frames = int(
            decay_frames if ttl_frames is None else ttl_frames)
        self.brake_alpha = float(os.environ.get("VLM_BRAKE_ALPHA", 2.0))
        self.weak_brake_alpha = float(
            os.environ.get("VLM_WEAK_BRAKE_ALPHA", 0.5))
        self.lateral_alpha = float(os.environ.get("VLM_LATERAL_ALPHA", 1.0))

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> AlphaVec:
        print("Running VLMPolicy.alpha with vlm_decision:", vlm_decision, "decision_age_frames:", decision_age_frames)
        alpha_vector = [0.0 for _ in self.ACTIONS]
        if vlm_decision is None:
            return alpha_vector
        if self.ttl_frames > 0 and decision_age_frames is not None and decision_age_frames > self.ttl_frames:
            return alpha_vector

        action = self._decision_action(vlm_decision)
        if action == "brake_weak":
            action_index = self.ACTION_INDEX["brake"]
        elif action in self.ACTION_INDEX:
            action_index = self.ACTION_INDEX[action]
        else:
            return alpha_vector
        if getattr(vlm_decision, "steering_alpha", 0.0) <= 0.0 and not getattr(vlm_decision, "enable_steering", False):
            return alpha_vector

        if action == "brake":
            alpha_vector[action_index] = self.brake_alpha
        elif action == "brake_weak":
            alpha_vector[action_index] = self.weak_brake_alpha
        else:
            alpha_vector[action_index] = self.lateral_alpha
        return alpha_vector

    @classmethod
    def _decision_action(cls, vlm_decision) -> str | None:
        action = getattr(vlm_decision, "action", None)
        if action is None:
            return "brake" if getattr(vlm_decision, "steering_alpha", 0.0) > 0.0 else None
        normalized = str(action).strip().lower()
        if normalized in ("brake_weak", "brakeweak", "weak_brake", "weakbrake", "gentle_brake", "gentlebrake"):
            return "brake_weak"
        if normalized in ("brake", "stop", "yield", "emergency_brake", "emergencybrake"):
            return "brake"
        if normalized in ("left", "l_change", "left_change", "left_change_lane", "change_lane_left"):
            return "left"
        if normalized in ("right", "r_change", "right_change", "right_change_lane", "change_lane_right"):
            return "right"
        return None


@dataclass
class SteerGuardConfig:
    """Safety / assist rules wrapped around a held left/right activation steer.

    1. Heading stop : the steer is not started / is stopped when the ego heading is more than
       ``max_heading_deg`` off the lane direction AND that offset points to the steer side
       (i.e. the ego is already turning that way).  <= 0 disables.
    2. Stall assist : if, after ``stall_grace`` frames, the ego moved less than ``stall_eps``
       metres toward the steer side over the last ``stall_window`` frames, alpha is raised by
       ``alpha_step`` (capped at ``alpha_max``) and ``active_until`` is extended by one frame
       (at most ``max_extend_frames`` times per steer).  alpha_step <= 0 / max_extend_frames <= 0
       disable the respective half.
    """

    max_heading_deg: float = 35.0
    stall_window: int = 3
    stall_eps: float = 0.05
    stall_grace: int = 2
    alpha_step: float = 0.25
    alpha_max: float = 3.0
    max_extend_frames: int = 30

    @classmethod
    def from_env(cls) -> "SteerGuardConfig":
        env = os.environ.get
        return cls(
            max_heading_deg=float(env("PDM_ORACLE_STEER_MAX_HEADING_DEG", 35.0)),
            stall_window=max(1, int(env("PDM_ORACLE_STEER_STALL_WINDOW", 3))),
            stall_eps=float(env("PDM_ORACLE_STEER_STALL_EPS", 0.05)),
            stall_grace=max(0, int(env("PDM_ORACLE_STEER_STALL_GRACE", 2))),
            alpha_step=float(env("PDM_ORACLE_STEER_ALPHA_STEP", 0.25)),
            alpha_max=float(env("PDM_ORACLE_STEER_ALPHA_MAX", 3.0)),
            max_extend_frames=int(env("PDM_ORACLE_STEER_MAX_EXTEND", 30)),
        )


class _BaseOraclePolicy(ActivationPolicy):
    """Shared state machine and geometry helpers for online oracle policies.

    CarlaDataProvider.active_scenarios stores currently relevant scenario actors.
    Subclasses map those signals into [brake, left, right] action alphas.
    """

    ACTIONS = ("brake", "left", "right")
    ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}

    def __init__(
        self,
        alpha: float,
        action: str = "auto",
        trigger_distance: float = 35.0,
        min_distance: float = 0.0,
        brake_hazard_distance: float = 20.0,
        brake_hazard_lateral_margin: float = 2.5,
        brake_reaction_time: float = 0.4,
        brake_deceleration: float = 6.0,
        brake_distance_margin: float = 2.0,
        brake_ttc_threshold: float = 2.0,
        brake_min_closing_speed: float = 0.5,
        hold_frames: int = 12,
        hold_frames_brake: int = 12,
        hold_frames_left: int = 12,
        hold_frames_right: int = 12,
        cooldown_frames: int = 30,
        allow_multi_action: bool = False,
        verbose: bool = False,
        steer_guard: SteerGuardConfig | None = None,
    ):
        self.fixed_alpha = float(alpha)
        self.steer_guard = steer_guard if steer_guard is not None else SteerGuardConfig.from_env()
        # Per-action state of the maneuver in progress (see _steer_begin / _steer_alpha).
        self._steer: dict[str, dict] = {}
        self.action = action.lower().strip()
        self.trigger_distance = float(trigger_distance)
        self.min_distance = float(min_distance)
        self.brake_hazard_distance = float(brake_hazard_distance)
        self.brake_hazard_lateral_margin = float(brake_hazard_lateral_margin)
        self.brake_reaction_time = float(brake_reaction_time)
        self.brake_deceleration = max(float(brake_deceleration), 1e-3)
        self.brake_distance_margin = float(brake_distance_margin)
        self.brake_ttc_threshold = float(brake_ttc_threshold)
        self.brake_min_closing_speed = float(brake_min_closing_speed)
        self.hold_frames = max(1, int(hold_frames))
        self.hold_frames_brake = max(1, int(hold_frames_brake))
        self.hold_frames_left = max(1, int(hold_frames_left))
        self.hold_frames_right = max(1, int(hold_frames_right))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.allow_multi_action = bool(allow_multi_action)
        self.verbose = bool(verbose)
        self._active_until = [-1 for _ in self.ACTIONS]
        self._cooldown_until = [-1 for _ in self.ACTIONS]
        self._last_trigger_key = None

        # ---- decision tracing -------------------------------------------------
        # Every helper appends short "why" notes to ``_trace`` during one frame.
        # ``alpha()`` prints ONE summary (+ the trace) only when the decision
        # changes (or every frame when verbose=True), instead of per-frame spam.
        self._trace: list[str] = []
        self._reason_key = "idle"      # stable text (no per-frame numbers) -> change detection
        self._reason_detail = ""       # numbers for the summary line
        self._last_emit_key = None
        self._lane_block: str | None = None  # set by _lane_clear: why the lane is NOT clear

    # ------------------------------------------------------------------ tracing
    def _note(self, msg: str) -> None:
        self._trace.append(msg)

    def _set_reason(self, key: str, detail: str = "") -> None:
        self._reason_key = key
        self._reason_detail = detail

    def _emit_decision(self, frame: int, alpha_vector: AlphaVec) -> None:
        """Print one line per decision *change*; trace lines only accompany it."""
        key = (tuple(alpha_vector), self._reason_key)
        changed = key != self._last_emit_key
        self._last_emit_key = key
        if not (changed or self.verbose):
            return
        tag = f"[{self.__class__.__name__}] frame={frame}"
        print(f"{tag} alpha={alpha_vector} reason={self._reason_key} {self._reason_detail}".rstrip(), flush=True)
        for line in self._trace:
            print(f"    - {line}", flush=True)

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> AlphaVec:
        self._trace = []
        self._set_reason("idle")
        alpha_vector = self._compute_alpha(frame)
        if self.fixed_alpha > 0.0:
            self._emit_decision(frame, alpha_vector)
        return alpha_vector

    def _compute_alpha(self, frame: int) -> AlphaVec:
        alpha_vector = [0.0 for _ in self.ACTIONS]
        if self.fixed_alpha <= 0.0:
            return alpha_vector

        data_provider = CarlaDataProvider()
        ego = data_provider.get_hero_actor()
        if ego is None or not getattr(ego, "is_alive", True):
            self._set_reason("no_ego")
            return alpha_vector

        # self._debug_both_lane_clear(ego, data_provider)  # For debugging, check both lane clearance and print debug info
        # self._debug_current_lane_clear_distance(ego, data_provider)  # For debugging, check current lane clearance distance and print debug info

        triggers = self._oracle_triggers()
        selected = self._select_triggers(triggers) if triggers else []
        self._note(f"triggers={self._fmt_triggers(triggers)} selected={self._fmt_triggers(selected)}")

        # Braking is a live safety decision, not a lane-change maneuver. Recheck it
        # every frame so an obstacle that remains in the ego path cannot fall into
        # brake cooldown. Scenario-specific live brakes interrupt a held lateral
        # action; the generic fallback is handled below so it cannot cancel a
        # lane-change maneuver that is already underway.
        brake_trigger = next(
            (trigger for trigger in selected if trigger[0] == "brake"), None)

        lateral_hold_active = any(
            frame <= self._active_until[self.ACTION_INDEX[action]] for action in ("left", "right")
        )
        lateral_scenarios = (
            getattr(self, "TWO_WAY_LATERAL_SCENARIOS", frozenset()) |
            getattr(self, "ONE_WAY_LATERAL_SCENARIOS", frozenset()) |
            frozenset({"HazardAtSideLane", "HazardAtSideLaneTwoWays"})
        )

        if (
            brake_trigger is not None and
            lateral_hold_active and
            (
                brake_trigger[1] == "GeneralActorBrake" or
                brake_trigger[1] in getattr(self, "TWO_WAY_LATERAL_SCENARIOS", frozenset())
            )
        ):
            # Once a two-way detour has been declared clear and its lateral action
            # is underway, do not let a transient re-evaluation of that same class
            # or the generic actor-brake fallback cancel the maneuver mid-hold.
            # Scenario-specific safety brakes (for example pedestrians or road
            # blocks) retain priority over lateral steering.
            self._note(f"brake trigger {brake_trigger[1]} suppressed: lateral hold active")
            if hasattr(self, "_two_way_brake_latches"):
                self._two_way_brake_latches.clear()
            brake_trigger = None

        brake_index = self.ACTION_INDEX["brake"]
        if brake_trigger is not None:
            for action in ("left", "right"):
                action_index = self.ACTION_INDEX[action]
                self._active_until[action_index] = -1
                self._cooldown_until[action_index] = -1
            self._steer.clear()
            self._active_until[brake_index] = frame
            self._cooldown_until[brake_index] = frame
            alpha_vector[brake_index] = self.fixed_alpha
            self._set_reason(
                f"brake<-{brake_trigger[1]}(actor={brake_trigger[2]})",
                f"d={brake_trigger[3]:.1f}m")
            return alpha_vector

        # The brake hazard has cleared.  Do not let a previous brake hold or
        # cooldown block a lateral intervention that is now safe to execute.
        self._active_until[brake_index] = -1
        self._cooldown_until[brake_index] = -1

        # Lateral actions retain the original hold/cooldown behavior to avoid
        # steering chatter.  A newly selected lateral action is considered before
        # an old hold, allowing a safe left/right intervention to replace it.
        for action, scenario_type, actor_id, distance in selected:
            if action == "brake":
                continue

            action_index = self.ACTION_INDEX[action]

            if not self._lane_clear_new(ego, action, data_provider):
                self._note(f"{action} <- {scenario_type} rejected: target lane not clear ({self._lane_block})")
                continue

            exceeded, angle = self._heading_exceeded(action, ego, data_provider)
            if exceeded:
                self._note(f"{action} <- {scenario_type} rejected: ego heading {angle:+.1f}deg off lane "
                           f"(right+) already points to the {action} side > {self.steer_guard.max_heading_deg:.0f}deg")
                continue

            if frame < self._cooldown_until[action_index]:
                self._note(f"{action} <- {scenario_type} rejected: cooldown until frame "
                           f"{self._cooldown_until[action_index]}")
                continue

            if not self.allow_multi_action:
                other_action = "right" if action == "left" else "left"
                other_index = self.ACTION_INDEX[other_action]
                self._active_until[other_index] = -1
                self._cooldown_until[other_index] = -1
            # self._active_until[action_index] = frame + self.hold_frames - 1
            self._active_until[action_index] = frame + getattr(self, f"hold_frames_{action}", self.hold_frames) - 1
            self._cooldown_until[action_index] = self._active_until[action_index] + \
                self.cooldown_frames
            alpha_vector[action_index] = self._steer_begin(action, ego, frame)
            self._set_reason(
                f"{action}<-{scenario_type}(actor={actor_id})",
                f"d={distance:.1f}m hold_until={self._active_until[action_index]} "
                f"cooldown_until={self._cooldown_until[action_index]}")

            if not self.allow_multi_action:
                break

        if any(value > 0.0 for value in alpha_vector):
            return alpha_vector

        for action in ("left", "right"):
            if self._active_until[self.ACTION_INDEX[action]] != -1:
                if self._lateral_hold_invalidated(ego, action, data_provider):
                    self._set_reason(f"release_{action}_hold", f"({self._lane_block or 'lane unavailable'})")
                    self._active_until[self.ACTION_INDEX[action]] = -1
                    self._cooldown_until[self.ACTION_INDEX[action]] = -1
                    self._steer.pop(action, None)
                    return [0.0, 0.0, 0.0]

        for action in ("left", "right"):
            action_index = self.ACTION_INDEX[action]
            if frame <= self._active_until[action_index]:
                exceeded, angle = self._heading_exceeded(action, ego, data_provider)
                if exceeded:
                    # Stop steering.  _cooldown_until is kept on purpose so the (still active)
                    # trigger cannot restart the steer on the very next frame.
                    self._active_until[action_index] = -1
                    self._steer.pop(action, None)
                    self._set_reason(f"stop_{action}_heading",
                                     f"heading={angle:+.1f}deg > {self.steer_guard.max_heading_deg:.0f}deg toward {action}")
                    continue
                alpha_vector[action_index] = self._steer_alpha(action, ego, frame)
                self._set_reason(
                    f"hold_{action}",
                    f"until={self._active_until[action_index]} alpha={alpha_vector[action_index]:.2f}")
        return alpha_vector

    # ------------------------------------------------------------------ steer guard
    def _heading_vs_lane_deg(self, ego, data_provider) -> float | None:
        """Signed angle (deg) from the lane direction to the ego heading; + = rotated to the RIGHT."""
        world_map = self._get_map(data_provider)
        if world_map is None:
            return None
        try:
            waypoint = world_map.get_waypoint(ego.get_location())
        except Exception:
            return None
        if waypoint is None:
            return None
        lane_f = waypoint.transform.get_forward_vector()
        ego_f = ego.get_transform().get_forward_vector()
        lx, ly = lane_f.x, lane_f.y
        if ego_f.x * lx + ego_f.y * ly < 0.0:   # borrowing the oncoming lane: compare with its flow
            lx, ly = -lx, -ly
        # CARLA is left-handed (+y is right): positive cross = ego rotated to the right.
        return math.degrees(math.atan2(lx * ego_f.y - ly * ego_f.x, lx * ego_f.x + ly * ego_f.y))

    def _heading_exceeded(self, action: str, ego, data_provider) -> tuple[bool, float | None]:
        """(True, angle) if the ego is > max_heading_deg off the lane toward the ``action`` side."""
        limit = self.steer_guard.max_heading_deg
        if limit <= 0.0:
            return False, None
        angle = self._heading_vs_lane_deg(ego, data_provider)
        if angle is None:
            return False, None
        toward = angle if action == "right" else -angle
        return toward > limit, angle

    def _steer_begin(self, action: str, ego, frame: int) -> float:
        """Anchor the stall detector at the current pose; returns the initial alpha."""
        loc = ego.get_location()
        fwd = ego.get_transform().get_forward_vector()
        norm = math.hypot(fwd.x, fwd.y) or 1.0
        fx, fy = fwd.x / norm, fwd.y / norm
        self._steer[action] = {
            "origin": (loc.x, loc.y),
            "right": (-fy, fx),        # CARLA: right = forward rotated +90deg
            "start": frame,
            "alpha": self.fixed_alpha,
            "lat": [0.0],              # lateral displacement toward the steer side [m]
            "extended": 0,
        }
        return self.fixed_alpha

    def _steer_alpha(self, action: str, ego, frame: int) -> float:
        """Alpha for a held steer.  Raises alpha / extends the hold while the ego is not moving."""
        cfg = self.steer_guard
        state = self._steer.get(action)
        if state is None:
            return self.fixed_alpha

        loc = ego.get_location()
        rx, ry = state["right"]
        sign = 1.0 if action == "right" else -1.0
        lateral = sign * ((loc.x - state["origin"][0]) * rx + (loc.y - state["origin"][1]) * ry)
        hist = state["lat"]
        hist.append(lateral)
        del hist[:-(cfg.stall_window + 1)]

        elapsed = frame - state["start"]
        if elapsed < cfg.stall_grace or len(hist) <= cfg.stall_window:
            self._note(f"steer {action}: lat={lateral:+.2f}m (warming up, {elapsed}f) alpha={state['alpha']:.2f}")
            return state["alpha"]

        progress = hist[-1] - hist[0]
        if progress >= cfg.stall_eps:
            self._note(f"steer {action}: lat={lateral:+.2f}m moving ({progress:+.2f}m/{cfg.stall_window}f) "
                       f"alpha={state['alpha']:.2f}")
            return state["alpha"]

        index = self.ACTION_INDEX[action]
        old_alpha = state["alpha"]
        if cfg.alpha_step > 0.0:
            state["alpha"] = min(old_alpha + cfg.alpha_step, max(cfg.alpha_max, self.fixed_alpha))
        extended = False
        if state["extended"] < cfg.max_extend_frames:
            self._active_until[index] += 1
            self._cooldown_until[index] += 1
            state["extended"] += 1
            extended = True
        self._note(f"steer {action}: lat={lateral:+.2f}m STALLED ({progress:+.2f}m/{cfg.stall_window}f < {cfg.stall_eps}m) "
                   f"alpha {old_alpha:.2f}->{state['alpha']:.2f}, "
                   f"active_until={self._active_until[index]}{' (+1)' if extended else ' (extension cap reached)'} "
                   f"[{state['extended']}/{cfg.max_extend_frames}]")
        return state["alpha"]

    @staticmethod
    def _fmt_triggers(triggers) -> str:
        if not triggers:
            return "[]"
        return "[" + ", ".join(
            f"{a}<-{s}(actor={i}, d={d:.1f})" for a, s, i, d in triggers) + "]"

    def _lateral_hold_invalidated(self, ego, action: str, data_provider: CarlaDataProvider) -> bool:
        """Whether a currently held lateral action should be aborted."""
        # data_provider = CarlaDataProvider()
        # ego = data_provider.get_hero_actor()
        world_map = self._get_map(data_provider)

        self._lane_block = None
        if world_map is None:
            return False

        if ego is None or not getattr(ego, "is_alive", True):
            return False

        try:
            ego_waypoint = world_map.get_waypoint(ego.get_location())
        except Exception as e:
            self._lane_block = f"ego waypoint error: {e}"
            return True

        if ego_waypoint is None:
            return False

        # [新增] 已跨線進入對向車道借道中，不再往下檢查對向的鄰車道/路肩，直接維持 hold
        ego_forward = ego.get_transform().get_forward_vector()
        wp_forward = ego_waypoint.transform.get_forward_vector()
        if (ego_forward.x * wp_forward.x + ego_forward.y * wp_forward.y) < 0.0:
            self._note(f"hold {action}: ego already in the oncoming lane -> lane re-check skipped")
            return False

        if action == "left":
            target_lane = ego_waypoint.get_left_lane()
        elif action == "right":
            target_lane = ego_waypoint.get_right_lane()
        else:
            self._lane_block = f"invalid action {action}"
            return True

        if target_lane is None:
            self._lane_block = f"no {action} lane"
            return True

        if not self._lane_clear(ego, target_lane, data_provider):
            # self._lane_block was filled by _lane_clear
            return True

        return False

    def _lane_clear(self, ego, target_lane, data_provider: CarlaDataProvider) -> bool:
        raise NotImplementedError

    def _lane_clear_new(self, ego, action: str, data_provider: CarlaDataProvider) -> bool:
        raise NotImplementedError

    def _debug_both_lane_clear(self, ego, data_provider: CarlaDataProvider) -> tuple[bool, bool]:
        raise NotImplementedError

    def _debug_current_lane_clear_distance(self, ego, data_provider: CarlaDataProvider,
                                           vehicles: list | None = None, only_front: bool = True) -> float:
        raise NotImplementedError

    @staticmethod
    def _get_map(data_provider: CarlaDataProvider):
        try:
            return data_provider.get_map()
        except Exception:
            return None

    def _oracle_triggers(self) -> list[Trigger]:
        raise NotImplementedError

    def _select_triggers(self, triggers: list[Trigger]) -> list[Trigger]:
        if self.allow_multi_action:
            best_by_action: dict[str, Trigger] = {}
            for trigger in triggers:
                action = trigger[0]
                if action not in best_by_action or trigger[3] < best_by_action[action][3]:
                    best_by_action[action] = trigger
            return [best_by_action[action] for action in self.ACTIONS if action in best_by_action]

        # Keep the intervention one-hot. Braking has priority because an incorrect
        # lateral intervention is usually worse than staying conservative.
        action_priority = {"brake": 0, "left": 1, "right": 1}
        return [min(triggers, key=lambda item: (action_priority[item[0]], item[3]))]

    def _action_allowed(self, action: str) -> bool:
        if self.action in ("auto", "any", "all"):
            return True
        if self.action == action:
            return True
        if self.action in ("lane", "lane_change", "change_lane") and action in ("left", "right"):
            return True
        return False

    def _scenario_actor_brake_hazard(self, ego, actor) -> bool:
        aid = getattr(actor, "id", None)
        tag = f"hazard(actor={aid})"
        ego_transform = ego.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()

        # [核心新增] 如果對方是車輛，且與自車「同車道、方向相反(相向)」，直接忽略煞車！
        if getattr(actor, "type_id", "").startswith("vehicle."):
            try:
                actor_forward = actor.get_transform().get_forward_vector()
                # 1. 先看兩車是否相向 (內積 < 0)
                if (forward.x * actor_forward.x + forward.y * actor_forward.y) < 0.0:
                    world_map = self._get_map(CarlaDataProvider)
                    if world_map is not None:
                        ego_wp = world_map.get_waypoint(ego_location)
                        actor_wp = world_map.get_waypoint(actor.get_location())
                        # 2. 如果自車與該逆向車已經在同一個 lane_id (或自車本身已處於逆向借道狀態: ego 與車道方向相反)
                        ego_in_oncoming_lane = (
                            ego_wp is not None and
                            (forward.x * ego_wp.transform.get_forward_vector().x +
                             forward.y * ego_wp.transform.get_forward_vector().y) < 0.0
                        )
                        same_lane = (
                            ego_wp is not None and actor_wp is not None and
                            ego_wp.road_id == actor_wp.road_id and ego_wp.lane_id == actor_wp.lane_id
                        )
                        if same_lane or ego_in_oncoming_lane:
                            self._note(f"{tag} NO: ignored oncoming vehicle in same/borrowed lane")
                            return False
            except Exception:
                pass

        actor_location = actor.get_location()
        diff_x = actor_location.x - ego_location.x
        diff_y = actor_location.y - ego_location.y
        longitudinal = diff_x * forward.x + diff_y * forward.y
        if longitudinal < self.min_distance or longitudinal > self.brake_hazard_distance:
            self._note(f"{tag} NO: longitudinal={longitudinal:.1f} outside "
                       f"[{self.min_distance:.1f}, {self.brake_hazard_distance:.1f}]")
            return False

        lateral = abs(diff_x * right.x + diff_y * right.y)
        # if lateral > self.brake_hazard_lateral_margin:
        #     self._note(f"{tag} NO: lateral={lateral:.2f} > margin={self.brake_hazard_lateral_margin:.2f}")
        #     return False

        lateral_margin = self.brake_hazard_lateral_margin
        if any(self._active_until[self.ACTION_INDEX[a]] >= 0 for a in ("left", "right")):
            lateral_margin = min(lateral_margin, 1.3)

        ego_velocity = ego.get_velocity()
        actor_velocity = actor.get_velocity()

        # TODO

        if lateral > lateral_margin:
            self._note(f"{tag} NO: lateral={lateral:.2f} > margin={lateral_margin:.2f}")
            return False

        ego_forward_speed = ego_velocity.x * forward.x + ego_velocity.y * forward.y
        actor_forward_speed = actor_velocity.x * \
            forward.x + actor_velocity.y * forward.y
        closing_speed = ego_forward_speed - actor_forward_speed
        if closing_speed < self.brake_min_closing_speed:
            self._note(f"{tag} NO: closing={closing_speed:.2f} < min={self.brake_min_closing_speed:.2f} "
                       f"(v_ego={ego_forward_speed:.2f}, v_actor={actor_forward_speed:.2f})")
            return False

        ttc = longitudinal / max(closing_speed, 1e-3)
        stopping_distance = (
            max(ego_forward_speed, 0.0) * self.brake_reaction_time +
            max(ego_forward_speed, 0.0)**2 / (2.0 * self.brake_deceleration) +
            self.brake_distance_margin)

        by_ttc = ttc <= self.brake_ttc_threshold
        by_dist = longitudinal <= stopping_distance
        result = by_ttc or by_dist
        why = []
        if by_ttc:
            why.append(f"ttc={ttc:.2f}<={self.brake_ttc_threshold:.2f}")
        if by_dist:
            why.append(f"long={longitudinal:.1f}<=stop_dist={stopping_distance:.1f}")
        self._note(
            f"{tag} {'YES' if result else 'NO'}: long={longitudinal:.1f} lat={lateral:.2f} "
            f"v_ego={ego_forward_speed:.2f} v_actor={actor_forward_speed:.2f} closing={closing_speed:.2f} "
            f"ttc={ttc:.2f} stop_dist={stopping_distance:.1f}" + (f" -> {', '.join(why)}" if why else ""))
        return result

    @staticmethod
    def _first_alive_actor(scenario_data):
        if scenario_data is None:
            return None
        for item in scenario_data:
            if hasattr(item, "get_location") and getattr(item, "is_alive", True):
                return item
        return None

    @staticmethod
    def _scenario_direction(scenario_data) -> str | None:
        if scenario_data is None:
            return None
        for item in scenario_data:
            if isinstance(item, str) and item in ("left", "right"):
                return item
        return None

    @staticmethod
    def _invading_turn_offset(scenario_data) -> float | None:
        if scenario_data is None or len(scenario_data) < 3:
            return None
        item = scenario_data[2]
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)) and abs(float(item)) > 1e-6:
            return float(item)
        return None

    @staticmethod
    def _horizontal_distance(location_a, location_b) -> float:
        dx = float(location_a.x - location_b.x)
        dy = float(location_a.y - location_b.y)
        return math.hypot(dx, dy)

    @staticmethod
    def _ego_has_passed_actor(ego, actor) -> bool:
        """Return whether the target actor is behind the ego's current heading."""
        ego_location = ego.get_location()
        actor_location = actor.get_location()
        forward = ego.get_transform().get_forward_vector()
        diff_x = actor_location.x - ego_location.x
        diff_y = actor_location.y - ego_location.y
        longitudinal = diff_x * forward.x + diff_y * forward.y
        return longitudinal < 0.0


class PDMOraclePolicy(_BaseOraclePolicy):
    """Conservative PDM-like oracle using only online CARLA actor/scenario state.

    This policy does not consume PDM-Lite labels or expert trajectories. It maps
    scenario-runner active_scenarios plus live actor geometry into
    [brake, left, right]. Two-way obstacle scenarios do not trigger lateral
    steering unless a simple opposite-lane clearance check passes. If the path is
    not clear, the policy stays inactive unless the obstacle is an imminent brake
    hazard.
    """

    LANE_NOTE_RADIUS = 60.0  # _lane_clear only reports vehicles within this distance of the ego
    ONE_WAY_LATERAL_SCENARIOS = frozenset({
        "Accident",
        "ConstructionObstacle",
        "ParkedObstacle",
    })
    TWO_WAY_LATERAL_SCENARIOS = frozenset({
        "AccidentTwoWays",
        "ConstructionObstacleTwoWays",
        "ParkedObstacleTwoWays",
        "VehicleOpensDoorTwoWays",
    })
    PRIORITY_BRAKE_SCENARIOS = frozenset({
        "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight",
        "NormalVehicleTakingPriority",
        "NormalVehicleRunningRedLight",
    })
    # A DynamicObjectCrossing actor can be hidden until it enters the road.  The
    # generic actor-geometry brake check is therefore often too late; brake as
    # soon as scenario runner marks this scenario active.
    ACTIVE_BRAKE_SCENARIOS = frozenset({
        "DynamicObjectCrossing",
    })

    def __init__(
        self,
        alpha: float,
        action: str = "auto",
        trigger_distance: float = 50.0,
        min_distance: float = 0.0,
        brake_hazard_distance: float = 20.0,
        brake_hazard_lateral_margin: float = 2.5,
        brake_reaction_time: float = 0.4,
        brake_deceleration: float = 6.0,
        brake_distance_margin: float = 2.0,
        brake_ttc_threshold: float = 2.0,
        brake_min_closing_speed: float = 0.5,
        hold_frames: int = 8,
        hold_frames_brake: int = 8,
        hold_frames_left: int = 8,
        hold_frames_right: int = 8,
        cooldown_frames: int = 20,
        allow_multi_action: bool = False,
        verbose: bool = False,
        two_way_clear_distance: float = 70.0,
        lane_key_search_distance: float = 90.0,
        side_hazard_distance: float = 25.0,
        side_hazard_two_way_distance: float = 10.0,
        roadblocked_distance: float = 40.0,
        priority_distance: float = 10.0,
        priority_path_lateral_margin: float = 3.0,
        yield_emergency_distance: float = 50.0,
        parking_exit_left: bool = True,
        parking_exit_distance_to_driving_lane: float = 3.0,
        parking_exit_clear_front_distance: float = 25.0,
        parking_exit_clear_rear_distance: float = 12.0,
        parking_exit_side_margin: float = 1.0,
        parking_exit_rear_time_gap: float = 3.0,
        general_brake: bool = False,
        # [NEW] Same-lane check for lateral (left/right) triggers.
        #   "lateral": skip steering if |lateral offset| > same_lane_lateral_margin
        #   "lane":    skip steering if ego and obstacle are on different (road_id, lane_id)
        #   "both":    skip if either of the above says "different lane"
        #   "off":     disable (old behaviour)
        same_lane_mode: str = "lateral",
        same_lane_lateral_margin: float = 3.0,
        # [NEW] Speed-scaled trigger distance for RoadBlocked / YieldToEmergencyVehicle.
        # The trigger radius becomes min(static_distance, stopping_distance(v) + margin)
        # instead of a fixed radius, so a slow ego no longer brakes/steers 40-50 m out.
        speed_scaled_limits: bool = True,
        speed_scaled_margin: float = 6.0,
    ):
        super().__init__(
            alpha=alpha,
            action=action,
            trigger_distance=trigger_distance,
            min_distance=min_distance,
            brake_hazard_distance=brake_hazard_distance,
            brake_hazard_lateral_margin=brake_hazard_lateral_margin,
            brake_reaction_time=brake_reaction_time,
            brake_deceleration=brake_deceleration,
            brake_distance_margin=brake_distance_margin,
            brake_ttc_threshold=brake_ttc_threshold,
            brake_min_closing_speed=brake_min_closing_speed,
            hold_frames=hold_frames,
            hold_frames_brake=hold_frames_brake,
            hold_frames_left=hold_frames_left,
            hold_frames_right=hold_frames_right,
            cooldown_frames=cooldown_frames,
            allow_multi_action=allow_multi_action,
            verbose=verbose,
        )
        self.speed_scaled_limits = bool(speed_scaled_limits)
        self.speed_scaled_margin = float(speed_scaled_margin)
        self.same_lane_mode = str(same_lane_mode).lower().strip()
        self.same_lane_lateral_margin = float(same_lane_lateral_margin)
        self.two_way_clear_distance = float(two_way_clear_distance)
        self.lane_key_search_distance = float(lane_key_search_distance)
        self.side_hazard_distance = float(side_hazard_distance)
        self.side_hazard_two_way_distance = float(side_hazard_two_way_distance)
        self.roadblocked_distance = float(roadblocked_distance)
        self.priority_distance = float(priority_distance)
        self.priority_path_lateral_margin = float(priority_path_lateral_margin)
        self.yield_emergency_distance = float(yield_emergency_distance)
        self.parking_exit_left = bool(parking_exit_left)
        self.parking_exit_distance_to_driving_lane = float(
            parking_exit_distance_to_driving_lane)
        self.parking_exit_clear_front_distance = float(
            parking_exit_clear_front_distance)
        self.parking_exit_clear_rear_distance = float(
            parking_exit_clear_rear_distance)
        self.parking_exit_side_margin = float(parking_exit_side_margin)
        self.parking_exit_rear_time_gap = float(parking_exit_rear_time_gap)
        self.general_brake = bool(general_brake)
        self._two_way_brake_latches: set[Latch] = set()
        # Set each frame the parking-exit clearance check fails, so a held left
        # steer can be aborted instead of running to completion blindly.
        self._parking_exit_blocked = False

        print(f"[{self.__class__.__name__}] self.ACTIVE_BRAKE_SCENARIOS={self.ACTIVE_BRAKE_SCENARIOS}")

    @classmethod
    def from_env(cls) -> "PDMOraclePolicy":
        action = os.environ.get("PDM_ORACLE_ACTION", os.environ.get(
            "ORACLE_ACTION", os.environ.get("ACTIVATION_ACTION", "auto")))
        alpha = float(
            os.environ.get("PDM_ORACLE_ALPHA",
                           os.environ.get("ORACLE_ALPHA", os.environ.get("STEERING_ALPHA", os.environ.get("ACTIVATION_ALPHA", 1.0)))))
        return cls(
            alpha=alpha,
            action=action,
            trigger_distance=float(os.environ.get(
                "PDM_ORACLE_TRIGGER_DISTANCE", os.environ.get("ORACLE_TRIGGER_DISTANCE", 50.0))),
            min_distance=float(os.environ.get(
                "PDM_ORACLE_MIN_DISTANCE", os.environ.get("ORACLE_MIN_DISTANCE", 0.0))),
            brake_hazard_distance=float(
                os.environ.get("PDM_ORACLE_BRAKE_HAZARD_DISTANCE", os.environ.get("ORACLE_BRAKE_HAZARD_DISTANCE", 15.0))),
            brake_hazard_lateral_margin=float(
                os.environ.get("PDM_ORACLE_BRAKE_HAZARD_LATERAL_MARGIN",
                               os.environ.get("ORACLE_BRAKE_HAZARD_LATERAL_MARGIN", 2.5))),
            brake_reaction_time=float(
                os.environ.get("PDM_ORACLE_BRAKE_REACTION_TIME", os.environ.get("ORACLE_BRAKE_REACTION_TIME", 0.4))),
            brake_deceleration=float(
                os.environ.get("PDM_ORACLE_BRAKE_DECELERATION", os.environ.get("ORACLE_BRAKE_DECELERATION", 6.0))),
            brake_distance_margin=float(
                os.environ.get("PDM_ORACLE_BRAKE_DISTANCE_MARGIN", os.environ.get("ORACLE_BRAKE_DISTANCE_MARGIN", 2.0))),
            brake_ttc_threshold=float(
                os.environ.get("PDM_ORACLE_BRAKE_TTC_THRESHOLD", os.environ.get("ORACLE_BRAKE_TTC_THRESHOLD", 2.0))),
            brake_min_closing_speed=float(
                os.environ.get("PDM_ORACLE_BRAKE_MIN_CLOSING_SPEED",
                               os.environ.get("ORACLE_BRAKE_MIN_CLOSING_SPEED", 0.5))),
            hold_frames=int(os.environ.get(
                "PDM_ORACLE_HOLD_FRAMES", os.environ.get("ORACLE_HOLD_FRAMES", 8))),
            hold_frames_brake=int(os.environ.get(
                "PDM_ORACLE_HOLD_FRAMES_BRAKE", os.environ.get("ORACLE_HOLD_FRAMES_BRAKE", 8))),
            hold_frames_left=int(os.environ.get(
                "PDM_ORACLE_HOLD_FRAMES_LEFT", os.environ.get("ORACLE_HOLD_FRAMES_LEFT", 8))),
            hold_frames_right=int(os.environ.get(
                "PDM_ORACLE_HOLD_FRAMES_RIGHT", os.environ.get("ORACLE_HOLD_FRAMES_RIGHT", 8))),
            cooldown_frames=int(os.environ.get(
                "PDM_ORACLE_COOLDOWN_FRAMES", os.environ.get("ORACLE_COOLDOWN_FRAMES", 20))),
            allow_multi_action=_strtobool(os.environ.get(
                "PDM_ORACLE_ALLOW_MULTI_ACTION", os.environ.get("ORACLE_ALLOW_MULTI_ACTION"))),
            verbose=_strtobool(os.environ.get(
                "PDM_ORACLE_VERBOSE", os.environ.get("ORACLE_VERBOSE"))),
            two_way_clear_distance=float(os.environ.get(
                "PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE", 70.0)),
            lane_key_search_distance=float(os.environ.get(
                "PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE", 90.0)),
            side_hazard_distance=float(os.environ.get(
                "PDM_ORACLE_SIDE_HAZARD_DISTANCE", 25.0)),
            side_hazard_two_way_distance=float(os.environ.get(
                "PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE", 10.0)),
            roadblocked_distance=float(os.environ.get(
                "PDM_ORACLE_ROADBLOCKED_DISTANCE", 40.0)),
            priority_distance=float(os.environ.get(
                "PDM_ORACLE_PRIORITY_DISTANCE", 10.0)),
            priority_path_lateral_margin=float(os.environ.get(
                "PDM_ORACLE_PRIORITY_PATH_LATERAL_MARGIN", 3.0)),
            yield_emergency_distance=float(os.environ.get(
                "PDM_ORACLE_YIELD_EMERGENCY_DISTANCE", 50.0)),
            parking_exit_left=_strtobool(os.environ.get(
                "PDM_ORACLE_PARKING_EXIT_LEFT", "1")),
            parking_exit_distance_to_driving_lane=float(
                os.environ.get("PDM_ORACLE_PARKING_EXIT_DISTANCE_TO_DRIVING_LANE", 3.0)),
            parking_exit_clear_front_distance=float(os.environ.get(
                "PDM_ORACLE_PARKING_EXIT_CLEAR_FRONT_DISTANCE", 25.0)),
            parking_exit_clear_rear_distance=float(os.environ.get(
                "PDM_ORACLE_PARKING_EXIT_CLEAR_REAR_DISTANCE", 0.2)),
            parking_exit_side_margin=float(os.environ.get(
                "PDM_ORACLE_PARKING_EXIT_SIDE_MARGIN", 1.0)),
            parking_exit_rear_time_gap=float(os.environ.get(
                "PDM_ORACLE_PARKING_EXIT_REAR_TIME_GAP", 3.0)),
            general_brake=_strtobool(
                os.environ.get("PDM_ORACLE_GENERAL_BRAKE")),
            same_lane_mode=os.environ.get("PDM_ORACLE_SAME_LANE_MODE", "lateral"),
            same_lane_lateral_margin=float(os.environ.get(
                "PDM_ORACLE_SAME_LANE_LATERAL_MARGIN", 2.0)),
            speed_scaled_limits=_strtobool(os.environ.get(
                "PDM_ORACLE_SPEED_SCALED_LIMITS", "1")),
            speed_scaled_margin=float(os.environ.get(
                "PDM_ORACLE_SPEED_SCALED_MARGIN", 6.0)),
        )

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> AlphaVec:
        # print(f"[{self.__class__.__name__}.alpha] frame={frame}, vlm_decision={vlm_decision}, decision_age_frames={decision_age_frames}")
        return super().alpha(frame, vlm_decision, decision_age_frames)

    def _oracle_triggers(self) -> list[Trigger]:
        self._parking_exit_blocked = False
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=import-outside-toplevel
        except Exception:
            return []

        try:
            ego = CarlaDataProvider.get_hero_actor()
        except Exception:
            ego = None
        if ego is None or not getattr(ego, "is_alive", True):
            return []

        ego_location = ego.get_location()
        scenarios = []
        triggers: list[Trigger] = []
        for scenario_type, scenario_data in list(getattr(CarlaDataProvider, "active_scenarios", [])):
            actor = self._first_alive_actor(scenario_data)
            # print(f"[{self.__class__.__name__}._oracle_triggers] scenario_type={scenario_type}, actor={actor}")
            # print(f"[{self.__class__.__name__}._oracle_triggers] scenario_data={scenario_data}")
            if actor is None:
                # Keep the rule tied to scenario-runner state rather than an actor's
                # visibility: crossing actors can be spawned underground or behind a
                # blocker when the scenario becomes active.
                if scenario_type in self.ACTIVE_BRAKE_SCENARIOS and self._action_allowed("brake"):
                    triggers.append(("brake", scenario_type, None, 0.0))
                continue
            distance = self._horizontal_distance(ego_location, actor.get_location())
            scenarios.append((distance, scenario_type, scenario_data, actor))

        scenarios.sort(key=lambda item: item[0])
        if not scenarios and not triggers:
            self._note("no active scenario with a live actor")

        for distance, scenario_type, scenario_data, actor in scenarios:
            sc = f"scenario {scenario_type}(actor={getattr(actor, 'id', None)}, d={distance:.1f})"
            action = self._action_from_pdm_state(
                scenario_type, scenario_data, ego, actor, distance, CarlaDataProvider)
            if action is None:
                self._note(f"{sc} -> no action")
                continue
            if not self._action_allowed(action):
                self._note(f"{sc} -> {action} filtered by action={self.action}")
                continue
            # ``trigger_distance`` is radial, so an actor can stay within range long
            # after the ego has passed it. Do not restart a lateral maneuver for a
            # target that is now behind the ego. While it remains ahead, retries are
            # still allowed when an initially unsafe adjacent lane becomes clear.
            if action in ("left", "right") and self._ego_has_passed_actor(ego, actor):
                self._note(f"{sc} -> {action} skipped: ego already passed actor")
                continue
            # [NEW] After a lane change the ego is in a different lane from the
            # obstacle but still behind it, so the distance check below would
            # re-trigger another steer. Skip lateral actions when the obstacle is
            # no longer in the ego's lane.
            if (
                action in ("left", "right") and
                scenario_type in self.LANE_CHECK_SCENARIOS and
                not self._actor_in_ego_lane(ego, actor)
            ):
                self._note(f"{sc} -> {action} skipped: obstacle not in ego lane")
                continue
            limit = self._distance_limit_for_scenario(scenario_type, ego)
            if self.min_distance <= distance <= limit:
                actor_id = getattr(actor, "id", None)
                triggers.append((action, scenario_type, actor_id, distance))
                self._note(f"{sc} -> {action} TRIGGER (d<={limit:.1f})")
            else:
                self._note(f"{sc} -> {action} out of range: d={distance:.1f} not in "
                           f"[{self.min_distance:.1f}, {limit:.1f}]")

        if triggers:
            return triggers

        parking_exit_trigger: Trigger | None = self._general_parking_exit_left_trigger(
            ego, CarlaDataProvider)
        if parking_exit_trigger is not None and self._action_allowed("left"):
            triggers.append(parking_exit_trigger)
            self._note("fallback GeneralParkingExitLeft: ego off driving lane and left lane clear")
            return triggers

        if not self.general_brake:
            return triggers

        brake_actor = self._general_brake_actor(ego, CarlaDataProvider)
        if brake_actor is None:
            self._note("fallback GeneralActorBrake: no vehicle/walker is a brake hazard")
            return triggers
        self._note(f"fallback GeneralActorBrake: actor={getattr(brake_actor, 'id', None)} "
                   f"({getattr(brake_actor, 'type_id', '?')}) is a brake hazard "
                   f"(general_brake=True, no scenario trigger)")
        distance = self._horizontal_distance(
            ego_location, brake_actor.get_location())
        if self._action_allowed("brake"):
            triggers.append(("brake", "GeneralActorBrake",
                            getattr(brake_actor, "id", None), distance))
        return triggers

    # def _lateral_hold_invalidated(self) -> bool:
    #     return self._parking_exit_blocked

    # [NEW] Scenarios whose lateral trigger is gated by the same-lane check.
    # Edit this set if some scenario's obstacle legitimately sits off the ego lane.
    LANE_CHECK_SCENARIOS = ONE_WAY_LATERAL_SCENARIOS | TWO_WAY_LATERAL_SCENARIOS

    def _actor_in_ego_lane(self, ego, actor) -> bool:
        """[NEW] Whether the obstacle is still in the ego's lane (True = may steer).

        Lateral offset is measured along the right vector of the ego's lane
        waypoint (not the ego heading), so it is not distorted by the ego's yaw
        while it is mid lane-change. If map data is unavailable the check passes.
        """
        mode = self.same_lane_mode
        if mode in ("off", "none", "false", "0", ""):
            return True

        ego_location = ego.get_location()
        actor_location = actor.get_location()
        world_map = self._get_map(CarlaDataProvider)
        ego_wp = self._get_waypoint_driving(world_map, ego_location) if world_map is not None else None
        actor_wp = self._get_waypoint_driving(world_map, actor_location) if world_map is not None else None

        if mode in ("lateral", "both"):
            try:
                if ego_wp is not None:
                    right = ego_wp.transform.get_right_vector()
                else:
                    right = ego.get_transform().get_right_vector()
                lateral = abs((actor_location.x - ego_location.x) * right.x +
                              (actor_location.y - ego_location.y) * right.y)
            except Exception:
                lateral = 0.0

            if lateral > self.same_lane_lateral_margin:
                self._note(f"same-lane(lateral): {lateral:.2f} > margin={self.same_lane_lateral_margin:.2f}")
                return False

        if mode in ("lane", "both"):
            if ego_wp is not None and actor_wp is not None:
                # Lane ids are not comparable inside junctions; do not block there.
                if not (getattr(ego_wp, "is_junction", False) or getattr(actor_wp, "is_junction", False)):
                    same_lane = (ego_wp.road_id == actor_wp.road_id and
                                 ego_wp.lane_id == actor_wp.lane_id)
                    if not same_lane:
                        self._note(f"same-lane(lane): ego=(road {ego_wp.road_id}, lane {ego_wp.lane_id}) "
                                   f"actor=(road {actor_wp.road_id}, lane {actor_wp.lane_id})")
                        return False

        return True

    def _speed_scaled_limit(self, static_limit: float, ego) -> float:
        """[NEW] min(static_limit, v*t_react + v^2/(2a) + margin).

        Uses the ego's current speed and the existing brake_reaction_time /
        brake_deceleration. ``speed_scaled_margin`` should cover the two vehicle
        half-lengths (the distance used here is center-to-center) plus a gap.
        Never larger than the original fixed limit.
        """
        if not self.speed_scaled_limits or ego is None:
            return static_limit
        try:
            velocity = ego.get_velocity()
            speed = math.hypot(float(velocity.x), float(velocity.y))
        except Exception:
            return static_limit
        stopping = (speed * self.brake_reaction_time +
                    speed ** 2 / (2.0 * self.brake_deceleration) +
                    self.speed_scaled_margin)
        return min(static_limit, stopping)

    def _distance_limit_for_scenario(self, scenario_type: str, ego=None) -> float:
        if scenario_type in self.ACTIVE_BRAKE_SCENARIOS:
            # Activation itself is the trigger for this rule, so do not discard it
            # merely because the crossing actor is still off the driving lane.
            return float("inf")
        if scenario_type in ("HazardAtSideLane",):
            return self.side_hazard_distance
        if scenario_type in ("HazardAtSideLaneTwoWays",):
            return self.side_hazard_two_way_distance
        if scenario_type == "RoadBlocked":
            # return self.roadblocked_distance
            # return self._speed_scaled_limit(self.yield_emergency_distance, ego)
            return max(self._speed_scaled_limit(self.roadblocked_distance, ego), 8.0)
        if scenario_type in self.PRIORITY_BRAKE_SCENARIOS:
            return self.priority_distance
        if scenario_type == "YieldToEmergencyVehicle":
            # return self.yield_emergency_distance
            return self._speed_scaled_limit(self.yield_emergency_distance, ego)
        return self.trigger_distance

    def _action_from_pdm_state(self, scenario_type, scenario_data, ego, actor, distance, data_provider) -> str | None:
        if scenario_type in self.ACTIVE_BRAKE_SCENARIOS:
            return "brake"

        if scenario_type == "InvadingTurn":
            offset = self._invading_turn_offset(scenario_data)
            if offset is None:
                return None
            return "left" if offset > 0.0 else "right"

        if scenario_type in self.ONE_WAY_LATERAL_SCENARIOS:
            return self._lateral_action_from_direction(self._scenario_direction(scenario_data))

        if scenario_type in self.TWO_WAY_LATERAL_SCENARIOS:
            direction = self._scenario_direction(scenario_data)
            latch_key = self._two_way_latch_key(scenario_type, scenario_data, actor)
            # Evaluate once (it used to be evaluated twice per frame for logging).
            path_clear = self._two_way_path_clear(ego, scenario_data, direction, data_provider)

            if path_clear:
                self._two_way_brake_latches.discard(latch_key)
                return self._lateral_action_from_direction(direction)

            if latch_key in self._two_way_brake_latches:
                self._note(f"two-way {scenario_type}: brake latched until target lane clears")
                return "brake"

            active_brake_hazard = self._scenario_actor_brake_hazard(ego, actor)

            if active_brake_hazard:
                # Once braking starts for a blocked two-way detour, keep braking until
                # the target lane is clear.  Slowing down otherwise makes the closing
                # speed fall below the instantaneous hazard threshold and prematurely
                # releases the brake.
                self._two_way_brake_latches.add(latch_key)
                self._note(f"two-way {scenario_type}: target lane blocked + brake hazard -> brake latched")
                return "brake"
            return None

        if scenario_type == "HazardAtSideLane":
            # PDM-Lite shifts around the bicycles as if the obstacle is on the right.
            return "left"

        if scenario_type == "HazardAtSideLaneTwoWays":
            if self._two_way_path_clear(ego, scenario_data, "right", data_provider):
                return "left"
            if self._scenario_actor_brake_hazard(ego, actor):
                return "brake"
            return None

        if scenario_type == "YieldToEmergencyVehicle":
            return self._yield_emergency_action(ego, data_provider)

        if scenario_type == "RoadBlocked":
            # if distance <= self.roadblocked_distance:
            if distance <= self._distance_limit_for_scenario("RoadBlocked", ego):
                self._note(f"RoadBlocked: d={distance:.1f} within speed-scaled limit")
                return "brake"
            return None

        if scenario_type in self.PRIORITY_BRAKE_SCENARIOS:
            if self._ego_in_actor_path(ego, actor):
                # The ego is already in the cross-traffic actor's lane corridor.  A
                # brake intervention would leave it in that path; let the planner
                # continue clearing the conflict instead.
                self._note(f"{scenario_type}: ego already in actor path -> do not brake")
                return None
            if distance <= self.priority_distance and self._priority_actor_conflict(ego, actor, scenario_data):
                self._note(f"{scenario_type}: cross traffic conflict within {self.priority_distance:.1f}m")
                return "brake"
            return None

        if self._scenario_actor_brake_hazard(ego, actor):
            return "brake"
        return None

    @staticmethod
    def _lateral_action_from_direction(direction: str | None) -> str | None:
        if direction == "right":
            return "left"
        if direction == "left":
            return "right"
        return None

    def _two_way_path_clear(self, ego, scenario_data, direction: str | None, data_provider) -> bool:
        self._lane_block = None
        if direction not in ("left", "right"):
            self._note(f"two-way path: unknown scenario direction={direction!r} -> not clear")
            return False

        world_map = self._get_map(data_provider)
        if world_map is None:
            self._note("two-way path: no map -> not clear")
            return False

        try:
            ego_waypoint = world_map.get_waypoint(ego.get_location())
        except Exception:
            self._note("two-way path: ego waypoint error -> not clear")
            return False
        if ego_waypoint is None:
            self._note("two-way path: ego waypoint None -> not clear")
            return False

        target_lane = ego_waypoint.get_left_lane(
        ) if direction == "right" else ego_waypoint.get_right_lane()
        if target_lane is None:
            self._note(f"two-way path: no {'left' if direction == 'right' else 'right'} lane -> not clear")
            return False

        lane_keys = self._collect_lane_keys(
            target_lane, self.lane_key_search_distance)
        if not lane_keys:
            self._note("two-way path: no lane keys -> not clear")
            return False

        ignored_ids = {getattr(actor, "id", None)
                       for actor in self._alive_actors(scenario_data)}
        ignored_ids.add(getattr(ego, "id", None))
        for _actor in self._alive_actors(scenario_data):
            try:
                _wp = self._get_waypoint_any(world_map, _actor.get_location())
                if _wp is not None and (_wp.road_id, _wp.lane_id) in lane_keys:
                    self._note(f"two-way path: IGNORING scenario actor {getattr(_actor, 'id', None)} although it is "
                               f"ON the target lane {(_wp.road_id, _wp.lane_id)} (scenario actors are never counted here)")
            except Exception:
                pass
        # ego_location = ego.get_location()
        # ego_forward = ego.get_transform().get_forward_vector()

        # for vehicle in self._vehicle_actors(data_provider):
        #     if getattr(vehicle, "id", None) in ignored_ids or not getattr(vehicle, "is_alive", True):
        #         continue
        #     try:
        #         vehicle_waypoint = world_map.get_waypoint(
        #             vehicle.get_location())
        #     except Exception:
        #         continue
        #     if vehicle_waypoint is None:
        #         continue
        #     if (vehicle_waypoint.road_id, vehicle_waypoint.lane_id) not in lane_keys:
        #         continue

        #     vehicle_location = vehicle.get_location()
        #     diff = vehicle_location - ego_location
        #     longitudinal = diff.x * ego_forward.x + diff.y * ego_forward.y
        #     if -5.0 <= longitudinal <= self.two_way_clear_distance:
        #         return False
        # return True

        vehicles = self._vehicle_actors(data_provider)
        vehicles = [v for v in vehicles if getattr(v, "id", None) not in ignored_ids and getattr(v, "is_alive", True)]
        lane_clear = self._lane_clear(ego, target_lane, data_provider, vehicles=vehicles)
        side = "left" if direction == "right" else "right"
        self._note(f"two-way path (target={side} lane): "
                   f"{'clear' if lane_clear else 'BLOCKED by ' + str(self._lane_block)}")
        return lane_clear

    @staticmethod
    def _two_way_latch_key(scenario_type: str, scenario_data, actor) -> Latch:
        actor_id = getattr(actor, "id", None)
        return scenario_type, int(actor_id) if actor_id is not None else id(scenario_data)

    def _collect_lane_keys(self, waypoint, distance: float) -> set[tuple[int, int]]:
        lane_keys = set()
        step = 2.0

        def add_chain(start, method_name):
            current = start
            traveled = 0.0
            while current is not None and traveled <= distance:
                lane_keys.add((current.road_id, current.lane_id))
                try:
                    next_waypoints = getattr(current, method_name)(step)
                except Exception:
                    break
                if not next_waypoints:
                    break
                current = next_waypoints[0]
                traveled += step

        add_chain(waypoint, "next")
        add_chain(waypoint, "previous")
        return lane_keys

    def _yield_emergency_action(self, ego, data_provider) -> str | None:
        world_map = self._get_map(data_provider)
        if world_map is None:
            return None
        try:
            ego_waypoint = world_map.get_waypoint(ego.get_location())
        except Exception:
            return None
        if ego_waypoint is None:
            return None

        # Mirrors the PDM-Lite rule: shift left unless the current waypoint only
        # allows a right lane change. Avoid importing carla at module import time.
        try:
            import carla  # pylint: disable=import-outside-toplevel
            if ego_waypoint.lane_change != carla.LaneChange.Right:  # pyright: ignore[reportAttributeAccessIssue]
                return "left"
            return "right"
        except Exception:
            left_lane = ego_waypoint.get_left_lane()
            return "left" if left_lane is not None else "right"

    def _priority_actor_conflict(self, ego, actor, scenario_data) -> bool:
        if actor is None or not getattr(actor, "is_alive", True):
            return False
        if actor.get_velocity().length() <= 0.1:
            return False

        ego_transform = ego.get_transform()
        actor_location = actor.get_location()
        ego_location = ego_transform.location
        diff_x = actor_location.x - ego_location.x
        diff_y = actor_location.y - ego_location.y
        right = ego_transform.get_right_vector()
        lateral = diff_x * right.x + diff_y * right.y
        actor_side = "right" if lateral > 0.0 else "left"
        direction = self._scenario_direction(scenario_data)
        return direction is None or direction == actor_side

    def _ego_in_actor_path(self, ego, actor) -> bool:
        """Whether the ego is already ahead in the target actor's driving corridor."""
        if actor is None or not getattr(actor, "is_alive", True):
            return False
        try:
            actor_transform = actor.get_transform()
            actor_location = actor_transform.location
            ego_location = ego.get_location()
            forward = actor_transform.get_forward_vector()
            right = actor_transform.get_right_vector()
        except Exception:
            return False

        diff_x = ego_location.x - actor_location.x
        diff_y = ego_location.y - actor_location.y
        longitudinal = diff_x * forward.x + diff_y * forward.y
        lateral = abs(diff_x * right.x + diff_y * right.y)
        return (
            0.0 <= longitudinal <= self.priority_distance and
            lateral <= self.priority_path_lateral_margin
        )

    def _general_parking_exit_left_trigger(self, ego, data_provider):
        if not self.parking_exit_left:
            return None

        world_map = self._get_map(data_provider)
        if world_map is None:
            return None

        ego_location = ego.get_location()
        ego_waypoint = self._get_waypoint_any(world_map, ego_location)
        if ego_waypoint is None:
            return None
        if getattr(ego_waypoint, "is_junction", False):
            return None
        if self._near_junction(ego_waypoint, distance=10.0):
            return None

        driving_waypoint = self._get_waypoint_driving(world_map, ego_location)
        distance_to_driving_lane = 0.0
        if driving_waypoint is not None:
            distance_to_driving_lane = self._horizontal_distance(
                ego_location, driving_waypoint.transform.location)

        parking_exit_like = (
            not self._waypoint_is_driving(ego_waypoint) or
            distance_to_driving_lane > self.parking_exit_distance_to_driving_lane)
        if not parking_exit_like:
            return None

        target_lane = self._left_driving_lane(ego_waypoint)
        if target_lane is None and driving_waypoint is not None:
            target_lane = driving_waypoint
        if target_lane is None:
            return None

        if not self._lane_clear(
            ego,
            target_lane,
            data_provider,
            # front_distance=self.parking_exit_clear_front_distance,
            # rear_distance=self.parking_exit_clear_rear_distance,
        ):
            self._parking_exit_blocked = True
            return None

        return ("left", "GeneralParkingExitLeft", None, distance_to_driving_lane)

    def _debug_both_lane_clear(self, ego, data_provider: CarlaDataProvider) -> tuple[bool, bool]:

        world_map = self._get_map(data_provider)
        if world_map is None:
            print(f"[{self.__class__.__name__}._debug_both_lane_clear] World map is None")
            return False, False

        ego_waypoint = world_map.get_waypoint(ego.get_location())
        if ego_waypoint is None:
            print(f"[{self.__class__.__name__}._debug_both_lane_clear] Ego waypoint is None")
            return False, False

        left_lane = ego_waypoint.get_left_lane()
        right_lane = ego_waypoint.get_right_lane()

        left_clear = self._lane_clear(ego, left_lane, data_provider) if left_lane is not None else None
        right_clear = self._lane_clear(ego, right_lane, data_provider) if right_lane is not None else None
        print(f"[{self.__class__.__name__}._debug_both_lane_clear] Left lane clear: {left_clear}, Right lane clear: {right_clear}")

        return left_clear or False, right_clear or False

    def _debug_current_lane_clear_distance(
        self,
        ego,
        data_provider: CarlaDataProvider,
        vehicles: list | None = None,
        only_front: bool = True,
    ) -> float:
        """
        Returns the distance to the nearest vehicle in the current lane, or
        float('inf') if no vehicle is detected within the search distance.

        距離定義與 _lane_clear 一致：沿著「車道方向」投影的中心距，再扣掉
        兩車在該方向上的投影半長（bumper-to-bumper gap，最小為 0）。
        only_front=True 時只看主車前方的車輛。
        """
        tag = f"[{self.__class__.__name__}._debug_current_lane_clear_distance]"

        world_map = self._get_map(data_provider)
        if world_map is None:
            print(f"{tag} World map is None")
            return float("inf")

        if ego is None or not getattr(ego, "is_alive", True):
            print(f"{tag} Ego actor is None or not alive")
            return float("inf")

        ego_waypoint = world_map.get_waypoint(ego.get_location())
        if ego_waypoint is None:
            print(f"{tag} Ego waypoint is None")
            return float("inf")

        # 與 _lane_clear 相同：lane-key 鏈至少要涵蓋整個 clearance window，
        # 否則位於下一段 road segment 的車輛會被提前濾掉。
        search_distance = max(
            self.parking_exit_clear_front_distance,
            self.parking_exit_clear_rear_distance,
            self.lane_key_search_distance,
        )
        lane_keys = self._collect_lane_keys(ego_waypoint, search_distance)
        if not lane_keys:
            print(f"{tag} No lane keys collected")
            return float("inf")

        # 沿車道方向投影，而非沿 ego 方向：BadParking 的 ego 是歪斜停放的。
        try:
            lane_forward = ego_waypoint.transform.get_forward_vector()
        except Exception:
            print(f"{tag} Failed to get lane forward vector, using ego's forward vector instead")
            lane_forward = ego.get_transform().get_forward_vector()

        ego_location = ego.get_location()
        ego_forward = ego.get_transform().get_forward_vector()
        ego_id = getattr(ego, "id", None)
        ego_half = self._projected_half_length(ego, lane_forward)

        if vehicles is None:
            vehicles = self._vehicle_actors(data_provider)

        nearest_distance = float("inf")
        nearest_vehicle_id = None

        for vehicle in vehicles:
            if getattr(vehicle, "id", None) == ego_id or not getattr(vehicle, "is_alive", True):
                continue

            vehicle_waypoint = self._get_waypoint_any(world_map, vehicle.get_location())
            if vehicle_waypoint is None:
                continue
            if (vehicle_waypoint.road_id, vehicle_waypoint.lane_id) not in lane_keys:
                continue

            diff = vehicle.get_location() - ego_location
            center_offset_lane = diff.x * lane_forward.x + diff.y * lane_forward.y
            longitudinal_ego = diff.x * ego_forward.x + diff.y * ego_forward.y

            if only_front and longitudinal_ego <= 0.0:
                continue

            vehicle_half = self._projected_half_length(vehicle, lane_forward)
            gap = max(abs(center_offset_lane) - (ego_half + vehicle_half), 0.0)

            if gap > search_distance:
                continue

            if gap < nearest_distance:
                nearest_distance = gap
                nearest_vehicle_id = getattr(vehicle, "id", None)

        if nearest_vehicle_id is None:
            print(f"{tag} No vehicle found in current lane within {search_distance:.2f} m")
        else:
            print(f"{tag} Nearest vehicle {nearest_vehicle_id} at {nearest_distance:.2f} m "
                  f"({'front only' if only_front else 'front or rear'})")

        return nearest_distance

    # def _lane_clear_new(self, action: str) -> bool:
    def _lane_clear_new(self, ego, action: str, data_provider: CarlaDataProvider) -> bool:
        self._lane_block = None
        world_map = self._get_map(data_provider)

        if world_map is None:
            print(f"[{self.__class__.__name__}._lane_clear_new] warn: World map is None")
            self._note("lane_clear_new: no map -> treated as CLEAR")
            return True  # TOFIX: If we can't get the map, assume lane is clear

        if ego is None or not getattr(ego, "is_alive", True):
            print(f"[{self.__class__.__name__}._lane_clear_new] warn: Ego actor is None or not alive")
            self._note("lane_clear_new: ego missing -> treated as CLEAR")
            return True  # TOFIX: If we can't get the ego, assume lane is clear

        ego_waypoint = world_map.get_waypoint(ego.get_location())
        if ego_waypoint is None:
            print(f"[{self.__class__.__name__}._lane_clear_new] warn: Ego waypoint is None")
            self._note("lane_clear_new: ego waypoint None -> treated as CLEAR")
            return True  # TOFIX: If we can't find the ego's waypoint, assume lane is clear
        if action == "left":
            target_lane = ego_waypoint.get_left_lane()
        elif action == "right":
            target_lane = ego_waypoint.get_right_lane()
        else:
            print(f"[{self.__class__.__name__}._lane_clear_new] warn: Invalid action: {action}")
            self._note(f"lane_clear_new: invalid action {action!r} -> treated as CLEAR")
            return True  # TOFIX: If the action is not left or right, assume lane is clear

        if target_lane is None:
            self._note(f"lane_clear_new: ego waypoint (road={ego_waypoint.road_id}, lane={ego_waypoint.lane_id}) "
                       f"has no {action} lane -> treated as CLEAR")
            return True
        if not self._waypoint_is_driving(target_lane):
            self._note(f"lane_clear_new: {action} lane (road={target_lane.road_id}, lane={target_lane.lane_id}) is "
                       f"{target_lane.lane_type}, not Driving -> traffic on the next DRIVING lane is NOT checked")
        if not self._lane_clear(ego, target_lane, data_provider):
            return False  # self._lane_block explains why

        return True

    def _lane_clear(self, ego, target_lane, data_provider: CarlaDataProvider, vehicles: list | None = None) -> bool:
        # On a False return, ``self._lane_block`` says why (read by callers' notes).
        self._lane_block = None
        world_map = self._get_map(data_provider)
        if world_map is None:
            self._lane_block = "no map"
            self._note("lane_clear: BLOCKED (no map)")
            return False

        front_distance = self.parking_exit_clear_front_distance
        rear_distance = self.parking_exit_clear_rear_distance

        # The lane-key chain has to span at least as far as the clearance window,
        # otherwise a vehicle sitting on the next road segment is silently dropped
        # before the longitudinal test ever runs.
        search_distance = max(
            front_distance, rear_distance, self.lane_key_search_distance)
        lane_keys = self._collect_lane_keys(target_lane, search_distance)
        if not lane_keys:
            self._lane_block = "no lane keys"
            self._note("lane_clear: BLOCKED (no lane keys)")
            return False

        # Project along the target lane, not the ego. In BadParking the ego is
        # parked askew by construction, so its forward vector is skewed relative
        # to traffic flow and a vehicle that is physically abreast projects to a
        # nonzero longitudinal offset.
        try:
            lane_forward = target_lane.transform.get_forward_vector()
        except Exception:
            lane_forward = ego.get_transform().get_forward_vector()

        ego_location = ego.get_location()
        ego_forward = ego.get_transform().get_forward_vector()
        ego_id = getattr(ego, "id", None)
        ego_half = self._projected_half_length(ego, lane_forward)

        # [新增] 取得自車「當前所在車道」的 lane_keys，用來判斷對方是否已經與自車處於同一車道
        ego_waypoint = self._get_waypoint_any(world_map, ego_location)
        ego_lane_keys = (
            self._collect_lane_keys(ego_waypoint, search_distance)
            if ego_waypoint is not None else set()
        )

        lane_same_direction = ego_forward.x * lane_forward.x + ego_forward.y * lane_forward.y > 0.0
        lane_dir = "same-dir" if lane_same_direction else "oncoming"

        if vehicles is None:
            vehicles = self._vehicle_actors(data_provider)
        self._note(
            f"lane_clear[{lane_dir}]: target road={getattr(target_lane, 'road_id', '?')} "
            f"lane={getattr(target_lane, 'lane_id', '?')} type={getattr(target_lane, 'lane_type', '?')} "
            f"keys={sorted(lane_keys)[:6]}{'...' if len(lane_keys) > 6 else ''} "
            f"front<={front_distance:.1f}m rear<={rear_distance:.1f}m vehicles={len(vehicles)} "
            f"(only vehicles within {self.LANE_NOTE_RADIUS:.0f}m are reported)")

        # for vehicle in self._vehicle_actors(data_provider):
        for vehicle in vehicles:
            if getattr(vehicle, "id", None) == ego_id or not getattr(vehicle, "is_alive", True):
                continue
            vid = getattr(vehicle, "id", None)
            _vloc = vehicle.get_location()
            _dist = math.hypot(_vloc.x - ego_location.x, _vloc.y - ego_location.y)
            _near = _dist <= self.LANE_NOTE_RADIUS
            vehicle_waypoint = self._get_waypoint_any(
                world_map, vehicle.get_location())
            if vehicle_waypoint is None:
                if _near:
                    self._note(f"lane_clear: skip veh {vid} d={_dist:.1f}m: no waypoint")
                continue
            # if (vehicle_waypoint.road_id, vehicle_waypoint.lane_id) not in lane_keys:
            #     continue

            v_lane_key = (vehicle_waypoint.road_id, vehicle_waypoint.lane_id)
            if v_lane_key not in lane_keys:
                if _near:
                    self._note(f"lane_clear: skip veh {vid} d={_dist:.1f}m: its lane road/lane={v_lane_key} "
                               f"({getattr(vehicle_waypoint, 'lane_type', '?')}) not in target lane keys")
                continue

            # [核心新增] 只有當該車與自車「當前處於同一車道 (v_lane_key in ego_lane_keys)」
            # 且「與自車相向 (內積 < 0)」時，才忽略它（代表自車已借道進入對向，不再因對向車而中斷或停下）
            if v_lane_key in ego_lane_keys:
                try:
                    vehicle_forward = vehicle.get_transform().get_forward_vector()
                    if (ego_forward.x * vehicle_forward.x + ego_forward.y * vehicle_forward.y) < 0.0:
                        self._note(f"lane_clear: skip veh {vid} d={_dist:.1f}m: oncoming AND its lane "
                                   f"{v_lane_key} is also in the EGO's own lane keys (ego treated as already in it)")
                        continue
                except Exception:
                    pass

            vehicle_location = vehicle.get_location()
            diff = vehicle_location - ego_location

            # 1. 幾何長度與重疊判定（使用 lane_forward 確保車身形狀計算準確）
            center_offset_lane = diff.x * lane_forward.x + diff.y * lane_forward.y
            # half_sum = ego_half + self._projected_half_length(vehicle, lane_forward)

            longitudinal_ego = diff.x * ego_forward.x + diff.y * ego_forward.y
            # gap = abs(center_offset_lane) - half_sum
            gap = abs(center_offset_lane)
            vehicle_half = self._projected_half_length(vehicle, lane_forward)
            # if self.verbose:
            #     print(f"[{self.__class__.__name__}._lane_clear] Vehicle {getattr(vehicle, 'id', None)}: "
            #         f"center_offset_lane: {center_offset_lane:.2f}, longitudinal_ego: {longitudinal_ego:.2f}, "
            #         f"gap: {gap:.2f}, vehicle_half: {vehicle_half:.2f}")

            # if lane_same_direction:
            #     front_distance = ego_half + vehicle_half - 0.2

            vehicle_speed_along_lane = self._longitudinal_speed(vehicle, lane_forward)

            # [新增 1] 過濾偏離車道中心 > 0.8m 的靜止路邊停車，允許並排擦身而過
            if abs(vehicle_speed_along_lane) < 0.5:
                wp_center = vehicle_waypoint.transform.location
                wp_right = vehicle_waypoint.transform.get_right_vector()
                lat_from_center = abs(
                    (vehicle_location.x - wp_center.x) * wp_right.x +
                    (vehicle_location.y - wp_center.y) * wp_right.y
                )
                if lat_from_center > 0.8:
                    self._note(f"lane_clear: skip veh {vid} d={_dist:.1f}m: static (|v|={abs(vehicle_speed_along_lane):.2f}m/s) "
                               f"and {lat_from_center:.2f}m off lane centre > 0.8m")
                    continue

            # [新增 2] 區分同向、對向移動來車、對向靜止車的前方淨空距離
            if lane_same_direction:
                effective_front_dist = ego_half + vehicle_half - 0.2
                rule = "same-dir: body overlap only (speed ignored)"
            else:
                if vehicle_speed_along_lane > 0.5:
                    # 真正迎面駛來的對向車：動態 TTC (2.5s) 與預設 front_distance 取小
                    ego_speed = max(self._longitudinal_speed(ego, ego_forward), 0.0)
                    dynamic_dist = (ego_speed + vehicle_speed_along_lane) * 2.5 + ego_half + vehicle_half
                    effective_front_dist = min(front_distance, dynamic_dist)
                    rule = (f"oncoming moving: min(front={front_distance:.1f}m, "
                            f"2.5s-TTC={dynamic_dist:.1f}m)")
                else:
                    # 對向車道上的靜止車：使用物理車身邊界 (-0.2 抵銷旋轉投影膨脹)
                    effective_front_dist = ego_half + vehicle_half - 0.2
                    rule = "oncoming static: body overlap only"

            self._note(f"lane_clear: consider veh {vid} d={_dist:.1f}m long_ego={longitudinal_ego:+.1f}m "
                       f"gap={gap:.1f}m v_lane={vehicle_speed_along_lane:+.1f}m/s eff_front={effective_front_dist:.1f}m "
                       f"[{rule}]")
            if longitudinal_ego > 0.0:
                if gap <= effective_front_dist:
                    self._lane_block = (f"front vehicle {vid} gap={gap:.2f}m <= {effective_front_dist:.2f}m "
                                        f"({lane_dir} lane)")
                    self._note(f"lane_clear: BLOCKED ({self._lane_block})")
                    return False
                self._note(f"lane_clear: front veh {vid} NOT blocking: gap {gap:.1f}m > eff_front {effective_front_dist:.1f}m")

            # vid = getattr(vehicle, "id", None)
            # if longitudinal_ego > 0.0:
            #     # 確實在主車前方
            #     if gap <= front_distance:
            #         self._lane_block = (f"front vehicle {vid} gap={gap:.2f}m <= {front_distance:.2f}m "
            #                             f"({lane_dir} lane)")
            #         return False
            else:
                # 確實在主車後方
                if gap <= rear_distance:
                    self._lane_block = (f"rear vehicle {vid} gap={gap:.2f}m <= {rear_distance:.2f}m "
                                        f"({lane_dir} lane)")
                    self._note(f"lane_clear: BLOCKED ({self._lane_block})")
                    return False

                # 確認車輛是否正在接近主車，並計算接近速度，計算lane_forward方向與ego的方向
                vehicle_speed = self._longitudinal_speed(vehicle, lane_forward)
                # is_closing = vehicle_speed > 0.1 and ego_forward.x * lane_forward.x + ego_forward.y * lane_forward.y > 0.0

                is_closing = vehicle_speed > 0.1 and (
                    (lane_same_direction and longitudinal_ego < 0.0) or
                    (not lane_same_direction and longitudinal_ego > 0.0)
                )

                # NOTE: previously this ``continue`` lived inside ``if self.verbose and ...``,
                # so with verbose=False non-closing vehicles were NOT skipped. Fixed.
                if not is_closing:
                    self._note(f"lane_clear: rear veh {vid} NOT closing (v={vehicle_speed:+.2f}m/s) -> ignored")
                    continue

                current_lane = world_map.get_waypoint(ego.get_location())
                if current_lane is None:
                    self._note(f"lane_clear: rear veh {vid} ignored: ego waypoint None")
                    continue

                current_lane_forward = current_lane.transform.get_forward_vector()
                ego_speed = self._longitudinal_speed(ego, current_lane_forward)

                if lane_same_direction:
                    closing_speed = vehicle_speed - ego_speed
                else:
                    closing_speed = vehicle_speed + ego_speed

                # 後方車輛接近速度判斷
                # closing_speed = self._longitudinal_speed(vehicle, lane_forward)
                # if closing_speed > 0.1 and gap / closing_speed <= self.parking_exit_rear_time_gap:
                #     print("Parking exit blocked (rear closing)...")
                #     return False
                max_time_gap = self.parking_exit_rear_time_gap  # default: 3.0
                if lane_same_direction:
                    max_time_gap *= 0.6  # Allow more time gap for same-direction traffic

                if closing_speed > 0.1 and gap / closing_speed <= self.parking_exit_rear_time_gap:
                    self._lane_block = (f"rear vehicle {vid} closing: gap={gap:.2f}m "
                                        f"closing={closing_speed:.2f}m/s "
                                        f"t={gap / closing_speed:.2f}s <= {self.parking_exit_rear_time_gap:.2f}s")
                    self._note(f"lane_clear: BLOCKED ({self._lane_block})")
                    return False
                self._note(f"lane_clear: rear veh {vid} closing but OK: gap={gap:.1f}m closing={closing_speed:.2f}m/s "
                           f"> {self.parking_exit_rear_time_gap:.1f}s margin")

        self._note("lane_clear: CLEAR")
        return True

        #     longitudinal = diff.x * ego_forward.x + diff.y * ego_forward.y
        #     if -rear_distance <= longitudinal <= front_distance:
        #         return False
        # return True

    @staticmethod
    def _projected_half_length(actor, axis) -> float:
        """Half the actor's footprint measured along ``axis``."""
        try:
            extent = actor.bounding_box.extent
        except Exception:
            return 2.5
        try:
            transform = actor.get_transform()
            actor_forward = transform.get_forward_vector()
            actor_right = transform.get_right_vector()
        except Exception:
            return float(getattr(extent, "x", 2.5))
        return (
            abs(extent.x * (actor_forward.x * axis.x + actor_forward.y * axis.y)) +
            abs(extent.y * (actor_right.x * axis.x + actor_right.y * axis.y))
        )

    @staticmethod
    def _longitudinal_speed(actor, axis) -> float:
        try:
            velocity = actor.get_velocity()
        except Exception:
            return 0.0
        return velocity.x * axis.x + velocity.y * axis.y

    @staticmethod
    def _get_waypoint_any(world_map, location):
        try:
            import carla  # pylint: disable=import-outside-toplevel
            lane_type = getattr(carla.LaneType, "Any", None)  # pyright: ignore[reportAttributeAccessIssue]
            if lane_type is None and hasattr(carla, "libcarla"):
                lane_type = getattr(carla.libcarla.LaneType, "Any", None)  # pyright: ignore[reportAttributeAccessIssue]
            if lane_type is not None:
                return world_map.get_waypoint(location, lane_type=lane_type)
        except Exception:
            pass
        try:
            return world_map.get_waypoint(location)
        except Exception:
            return None

    @staticmethod
    def _get_waypoint_driving(world_map, location):
        try:
            import carla  # pylint: disable=import-outside-toplevel
            return world_map.get_waypoint(location, lane_type=carla.LaneType.Driving)  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            try:
                return world_map.get_waypoint(location)
            except Exception:
                return None

    @classmethod
    def _waypoint_is_driving(cls, waypoint) -> bool:
        try:
            import carla  # pylint: disable=import-outside-toplevel
            return waypoint.lane_type == carla.LaneType.Driving  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            return "Driving" in str(getattr(waypoint, "lane_type", ""))

    def _left_driving_lane(self, waypoint):
        current = waypoint
        for _ in range(4):
            try:
                current = current.get_left_lane()
            except Exception:
                return None
            if current is None:
                return None
            if self._waypoint_is_driving(current):
                return current
        return None

    @staticmethod
    def _near_junction(waypoint, distance: float) -> bool:
        if waypoint is None:
            return False
        if getattr(waypoint, "is_junction", False):
            return True

        step = 2.0

        def scan(method_name: str) -> bool:
            current = waypoint
            traveled = 0.0
            while traveled < distance:
                try:
                    next_waypoints = getattr(current, method_name)(step)
                except Exception:
                    return False
                if not next_waypoints:
                    return False
                current = next_waypoints[0]
                traveled += step
                if getattr(current, "is_junction", False):
                    return True
            return False

        return scan("next") or scan("previous")

    def _general_brake_actor(self, ego, data_provider):
        for actor in self._vehicle_actors(data_provider) + self._walker_actors(data_provider):
            if getattr(actor, "id", None) == getattr(ego, "id", None) or not getattr(actor, "is_alive", True):
                continue
            if self._scenario_actor_brake_hazard(ego, actor):
                return actor
        return None

    @staticmethod
    def _get_map(data_provider):
        try:
            return data_provider.get_map()
        except Exception:
            return None

    @staticmethod
    def _vehicle_actors(data_provider):
        try:
            world = data_provider.get_world()
            if world is not None:
                return list(world.get_actors().filter("vehicle.*"))
        except Exception:
            pass
        try:
            return [actor for _, actor in data_provider.get_actors() if getattr(actor, "type_id", "").startswith("vehicle.")]
        except Exception:
            return []

    @staticmethod
    def _walker_actors(data_provider):
        try:
            world = data_provider.get_world()
            if world is not None:
                return list(world.get_actors().filter("walker.*"))
        except Exception:
            pass
        try:
            return [actor for _, actor in data_provider.get_actors() if getattr(actor, "type_id", "").startswith("walker.")]
        except Exception:
            return []

    @staticmethod
    def _alive_actors(scenario_data):
        if scenario_data is None:
            return []
        return [item for item in scenario_data if hasattr(item, "get_location") and getattr(item, "is_alive", True)]


class PDMOracleBrakeOnlyPolicy(PDMOraclePolicy):
    """A PDMOraclePolicy that always brakes when a hazard is detected."""

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> AlphaVec:
        alpha = super().alpha(frame, vlm_decision, decision_age_frames)
        alpha = [alpha[0], 0.0, 0.0]  # Always brake, never steer
        return alpha


class NoPolicy(ActivationPolicy):
    """A policy that always returns None."""

    def __init__(self):
        super().__init__()

    def alpha(self, frame: int, vlm_decision=None, decision_age_frames=None) -> Alpha:
        return None


def policy_from_env(
    vlm_enabled: bool = False,
    depth_ttc_client: DepthTTCClient | None = None,
    training_config=None,
) -> ActivationPolicy:

    if vlm_enabled:
        return VLMPolicy(
            ttl_frames=int(os.environ.get("VLM_DECISION_TTL_FRAMES",
                           os.environ.get("VLM_DECAY_FRAMES", 10))),
        )

    policy_name = os.environ.get(
        "ACTIVATION_POLICY", os.environ.get("STEERING_POLICY", "")).lower()

    if depth_ttc_enabled_from_env():
        if depth_ttc_client is None:
            raise ValueError("Depth-TTC policy requested but no DepthTTCClient was provided")
        if training_config is None:
            raise ValueError("DepthTTCPolicy requires the training configuration")
        return DepthTTCPolicy(depth_ttc_client, training_config)

    if policy_name == "none" or _strtobool(os.environ.get("NO_POLICY")):
        return NoPolicy()

    if (
        policy_name in ("pdm_oracle", "pdm-oracle", "pdmoracle") or
        _strtobool(os.environ.get("PDM_ORACLE_STEERING")) or
        _strtobool(os.environ.get("PDM_ORACLE_POLICY")) or
        policy_name == "oracle" or
        _strtobool(os.environ.get("ORACLE_STEERING")) or
        _strtobool(os.environ.get("ORACLE_POLICY"))
    ):
        return PDMOraclePolicy.from_env()

    if policy_name == "pdm_oracle_brake_only" or policy_name == "oracle_brake_only":
        return PDMOracleBrakeOnlyPolicy.from_env()

    alpha = float(os.environ.get("STEERING_ALPHA",
                  os.environ.get("ACTIVATION_ALPHA", 0.0)))
    print("[policy_from_env] STEERING_ALPHA:", alpha)

    if alpha > 0.0 and (policy_name == "fixed_after_frame" or policy_name == "fixed"):
        start_frame = int(os.environ.get("START_STEERING_FRAME", os.environ.get("ACTIVATION_START_FRAME", 0)))
        end_frame_raw = os.environ.get("END_STEERING_FRAME", os.environ.get("ACTIVATION_END_FRAME"))
        end_frame = None if end_frame_raw is None or end_frame_raw == "" else int(end_frame_raw)
        print("[policy_from_env] START_STEERING_FRAME:", start_frame)
        print("[policy_from_env] END_STEERING_FRAME:", end_frame)
        return FixedAfterFramePolicy(alpha=alpha, start_frame=start_frame, end_frame=end_frame)

    return NoPolicy()
