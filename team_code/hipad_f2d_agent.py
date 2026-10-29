"""Fail2Drive adapter for HiP-AD's Bench2Drive closed-loop agent."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from activation_steering.policy import policy_from_env
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
# from bench2drive.leaderboard.team_code.hipad_b2d_agent import SparseAgent as _BaseHiPADAgent


def _env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "t", "yes", "y")


def get_entry_point():
    return "Fail2DriveHiPADAgent"


class Fail2DriveHiPADAgent(_BaseHiPADAgent):
    """Adapts HiP-AD to fail2drive's local leaderboard API."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.route_index = route_index
        self.traffic_manager = traffic_manager
        self.visual_save_path = None

        # activation_steering
        self.activation_steering = os.environ.get("STEERING_VECTORS_DIR") is not None
        self.activation_action_log = None
        self.hipad_activation_alpha = None
        self.hipad_activation_plan_shift = 0.0

        if self.activation_steering:
            print("[Fail2DriveHiPADAgent] Activation steering enabled (STEERING_VECTORS_DIR is set).")
            self.activation_policy = policy_from_env(vlm_enabled=False)
            print(f"[Fail2DriveHiPADAgent] Loaded activation policy: {self.activation_policy.__class__.__name__}")

        # Agent-side collection/steering (TFv6-style). Set HIPAD_LEGACY_HOOKS=1 to fall back
        # to the old behaviour where the HiP-AD model itself dumps features / injects vectors.
        self.legacy_hooks = _env_flag("HIPAD_LEGACY_HOOKS", "0")

        self.feature_collector = None
        self.steering_manager = None
        self.steering_manager_alpha = None
        self._steering_hooks = []
        self._steering_vectors = {}

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
        # self.feature_collector.attach(self._hipad_model())
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

            # --- Steering Log ---
            # vec_norm = float(vec.norm().item())
            # frame = int(getattr(self, "step", -1)) + 1
            # alpha = self.hipad_activation_alpha if self.legacy_hooks else \
            #     self.steering_manager.get_steering_alphas(self.hipad_activation_alpha)
            # print(
            #     f"[Fail2DriveHiPADAgent] [STEERED] Frame {frame} | "
            #     f"Layer {layer_idx} | "
            #     f"Vector L2 Norm: {vec_norm:.4f} | "
            #     f"Alpha: {alpha}"
            # )
            # --------------------

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
            raise RuntimeError(f"instance_feature of shape {tuple(instance_feature.shape)} does not split into {group} groups.")

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
        return super().tick(input_data)

    def run_step(self, input_data, timestamp: float):
        print(f"[Fail2DriveHiPADAgent] run_step called at step {getattr(self, 'step', -1)}")
        self._set_plan_feature_env()
        if self.activation_steering:
            self._set_activation_env()
        control = super().run_step(input_data, timestamp)
        self._steering_vectors = {}  # never leak vectors into the next forward pass
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
        self.hipad_activation_alpha = self.activation_policy.alpha(frame)
        if self.legacy_hooks:
            os.environ["HIPAD_ACTIVATION_ALPHA"] = self._format_activation_alpha(self.hipad_activation_alpha)
            print(f"[Fail2DriveHiPADAgent] Activation alpha (legacy) at frame {frame}: {self.hipad_activation_alpha}")
        else:
            # The agent-side hooks inject now; keep the in-model ActivationInjector off (alpha=0)
            # so the vectors are not added twice.
            assert self.steering_manager is not None, "Steering manager must be initialized when legacy_hooks=False"
            os.environ["HIPAD_ACTIVATION_ALPHA"] = "0.0"
            print(f"[Fail2DriveHiPADAgent] Activation alpha at frame {frame}: {self.steering_manager.get_steering_alphas(self.hipad_activation_alpha)}")
            self._steering_vectors = self.steering_manager.get_layer_vectors(self.hipad_activation_alpha)
            self.steering_manager_alpha = self.steering_manager.get_steering_alphas(self.hipad_activation_alpha)
            if not any(a > 0.0 for a in self.steering_manager_alpha):
                self.steering_manager_alpha = None

        self.hipad_activation_plan_shift = self._activation_plan_shift(self.hipad_activation_alpha)
        os.environ["HIPAD_ACTIVATION_PLAN_SHIFT"] = str(float(self.hipad_activation_plan_shift))
        # print(f"[Fail2DriveHiPADAgent] HIPAD_ACTIVATION_ALPHA: {self._format_activation_alpha(self.hipad_activation_alpha)}")
        # print(f"[Fail2DriveHiPADAgent] Activation plan shift: {self.hipad_activation_plan_shift}")

    def _configure_visual_output_path(self):
        if not getattr(self, "is_visualize", False):
            return
        visual_root = os.environ.get("VISUAL_OUTPUT_PATH")
        if visual_root is None or getattr(self, "save_path", None) is None:
            return
        self.visual_save_path = Path(visual_root).resolve() / self.save_path.name
        self.visual_save_path.mkdir(parents=True, exist_ok=True)
        # (self.visual_save_path / "metas").mkdir(exist_ok=True)
        (self.visual_save_path / "images").mkdir(exist_ok=True)
        self._remove_empty_log_visual_dirs()

    def _remove_empty_log_visual_dirs(self):
        # for folder in ("images", "metas"):
        for folder in ("images"):
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
        # (self.save_path / "metas").mkdir(parents=True, exist_ok=True)

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
        # with open(output_path / "metas" / ("%04d.json" % frame), "w", encoding="utf-8") as outfile:
        #     json.dump(self.pid_metadata, outfile, indent=4)

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
            import shutil
            shutil.rmtree(images_dir)
            print(
                f"[Fail2DriveHiPADAgent] "
                f"Encoding succeeded. Deleted images directory: {images_dir}"
            )
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
        for hook in self._steering_hooks:
            hook.remove()
        self._steering_hooks.clear()
        if self.feature_collector is not None:
            self.feature_collector.remove_hooks()
        self._visualize_video()
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
        value = cls._json_activation_alpha(activation_alpha)
        if value is None:
            return 0.0
        if isinstance(value, list):
            return cls._activation_plan_shift_from_action_alpha(value)
        if float(value) <= 0.0:
            return 0.0
        direction = cls._infer_scalar_activation_direction()
        # if direction == "left":
        #     return -1.6
        # if direction == "right":
        #     return 1.6
        return 0.0

    @staticmethod
    def _activation_plan_shift_from_action_alpha(alpha_values):
        if len(alpha_values) < 3:
            return 0.0
        left_alpha = abs(float(alpha_values[1]))
        right_alpha = abs(float(alpha_values[2]))
        if left_alpha == 0.0 and right_alpha == 0.0:
            return 0.0
        # if right_alpha > left_alpha:
        #     return 1.6
        # if left_alpha > right_alpha:
        #     return -1.6
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
        # if str(os.environ.get("SAVE_HIPAD_PLAN_FEATURES", "")).lower() not in ("1", "true", "t", "yes", "y"):
        #     return None
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
