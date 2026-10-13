from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from activation_steering.steering_manager.base_steering_manager import BaseSteeringManager


class HiPADSteeringManager(BaseSteeringManager):
    """Steering-vector store for HiP-AD's plan-refinement ``align_query``.

    Vectors are read from a *root* with the layout written by the post-process
    step::

        <root>/<feature_name>/layer_<nn>/steering_vector.pt     # [1, 48, 256]

    e.g. ``steering/hipad/post_process/left_manual/align_query/layer_05/...``

    Only the layers listed in ``steer_layers`` are loaded (default: ``(5,)``).

    Which action does a root belong to?
      * single-action root (``<root>/<feature_name>`` exists, e.g. ``left_manual``):
        the action is ``action`` if given, otherwise inferred from the folder name
        (a ``brake`` / ``left`` / ``right`` token, split on ``_`` and ``-``).
      * multi-action root: ``<root>/{brake,left,right}/<feature_name>/...``;
        every action sub-folder that exists is loaded.

    Alpha semantics (``get_steering_alphas``):
      * length-3 sequence -> ``[brake, left, right]``
      * scalar            -> applied to the only loaded action; if several actions
                             are loaded it is brake-only (legacy ActivationInjector).
    """

    ACTIONS = ("brake", "left", "right")
    DEFAULT_STEER_LAYERS = (5,)
    VECTOR_FILENAME = "steering_vector.pt"

    def __init__(
        self,
        steering_root: Union[str, Path],
        device: torch.device,
        feature_name: str = "align_query",
        steer_layers: Sequence[int] = DEFAULT_STEER_LAYERS,
        action: str | None = None,
    ):
        if steering_root is None:
            raise ValueError("HiPADSteeringManager needs a steering_root.")
        self.steering_root = Path(steering_root)
        self.device = device
        self.feature_name = feature_name
        self.steer_layers: tuple[int, ...] = tuple(sorted({int(i) for i in steer_layers}))
        if not self.steer_layers:
            raise ValueError("steer_layers must contain at least one layer index.")
        if action is not None and action not in self.ACTIONS:
            raise ValueError(f"action must be one of {self.ACTIONS}, got {action!r}.")
        self.action = action
        self.internal_multipliers = self._multipliers_from_env()

        # vectors[action][layer_idx] -> tensor on `device`, float32
        self.vectors: dict[str, dict[int, torch.Tensor]] = {a: {} for a in self.ACTIONS}
        self._load_all()
        print(
            f"[HiPADSteeringManager] root={self.steering_root} feature={self.feature_name} "
            f"steer_layers={self.steer_layers} loaded_actions={self.loaded_actions}"
        )

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_env(
        cls,
        device: torch.device,
        feature_name: str = "align_query",
    ):
        """Build from env, or return None when STEERING_VECTORS_DIR is unset.

        STEERING_VECTORS_DIR   root described in the class docstring
        HIPAD_STEER_LAYERS     comma separated layer indices (default "5")
        HIPAD_STEERING_ACTION  brake|left|right, only needed if it can't be
                               inferred from the root folder name
        """
        root = os.environ.get("STEERING_VECTORS_DIR")
        if not root:
            return None
        layers = tuple(
            int(v) for v in os.environ.get("HIPAD_STEER_LAYERS", "5").split(",") if v.strip()
        )
        print(f"[HiPADSteeringManager.from_env] STEERING_VECTORS_DIR={root}, HIPAD_STEER_LAYERS={layers}, feature_name={feature_name}")
        return cls(
            steering_root=root,
            device=device,
            feature_name=feature_name,
            steer_layers=layers,
            action=os.environ.get("HIPAD_STEERING_ACTION") or None,
        )

    @classmethod
    def _multipliers_from_env(cls) -> list[float]:
        values = [1.0, 1.0, 1.0]
        spec = os.environ.get("ACTIVATION_ACTION_ALPHA_SCALES")
        if spec:
            values = [float(v) for v in spec.split(",")]
            if len(values) != len(cls.ACTIONS):
                raise ValueError("ACTIVATION_ACTION_ALPHA_SCALES must be 'brake,left,right'.")
        for i, action in enumerate(cls.ACTIONS):
            name = f"{action.upper()}_ACTIVATION_ALPHA_SCALE"
            if name in os.environ:
                values[i] = float(os.environ[name])
        return values

    # ---------------------------------------------------------------- loading
    @property
    def loaded_actions(self) -> list[str]:
        return [a for a in self.ACTIONS if self.vectors[a]]

    def _action_roots(self) -> dict[str, Path]:
        print(f"[HiPADSteeringManager] Determining action roots under {self.steering_root} for feature '{self.feature_name}'")
        if not self.steering_root.is_dir():
            raise FileNotFoundError(
                f"Steering root does not exist: {self.steering_root}"
            )

        # 1. Single-action root check: <root>/<feature_name>
        # if (self.steering_root / self.feature_name).is_dir():
        #     action = self.action or self._infer_action(self.steering_root.name)
        #     print(f"[HiPADSteeringManager] Single-action  inferred action: {action}")
        #     if action is None:
        #         raise ValueError(
        #             f"Cannot infer the action from '{self.steering_root.name}'. "
        #             "Set HIPAD_STEERING_ACTION (brake|left|right)."
        #         )
        #     print(f"[HiPADSteeringManager] Found steering vectors for action '{action}' in {self.steering_root}")
        #     return {action: self.steering_root}

        # 2. Multi-action search: check <root>/{action} and <root>/{action}_manual
        roots: dict[str, Path] = {}
        for a in self.ACTIONS:
            # for candidate in (a, f"{a}_manual"):
            for candidate in (f"{a}_manual",):
                cand_dir = self.steering_root / candidate
                print(f"[HiPADSteeringManager] Checking for steering vectors for action '{a}' in {cand_dir}")
                if (cand_dir / self.feature_name).is_dir():
                    print(f"[HiPADSteeringManager] Found steering vectors for action '{a}' in {cand_dir}")
                    roots[a] = cand_dir
                    break

        # 3. Fallback to os.environ["ACTIVATION_VECTOR_PATHS"] for missing actions
        missing_actions = [a for a in self.ACTIONS if a not in roots]
        if missing_actions:
            env_map = self._env_fallback_paths()
            for a in missing_actions:
                if a in env_map:
                    roots[a] = env_map[a]
                    print(f"[HiPADSteeringManager] Found fallback steering vectors for action '{a}' in {env_map[a]}")

        if not roots:
            raise FileNotFoundError(
                f"No steering vectors found under {self.steering_root} or via "
                f"ACTIVATION_VECTOR_PATHS. Expected <root>/{{action,action_manual}}/{self.feature_name}."
            )

        return roots

    @classmethod
    def _infer_action(cls, name: str) -> str | None:
        tokens = name.lower().replace("-", "_").split("_")
        for action in cls.ACTIONS:
            if action in tokens:
                return action
        return None

    @classmethod
    def _env_fallback_paths(cls) -> dict[str, Path]:
        """Parses ACTIVATION_VECTOR_PATHS as [brake, left, right].

        Raises ValueError if set and length is not exactly 3.
        """
        raw_env = os.environ.get("ACTIVATION_VECTOR_PATHS", "").strip()
        if not raw_env:
            return {}

        paths = [
            p.strip()
            for p in raw_env.split(",")
            if p.strip() and not p.strip().startswith("#")
        ]
        if len(paths) != len(cls.ACTIONS):
            raise ValueError(
                f"ACTIVATION_VECTOR_PATHS must contain exactly 3 paths (brake, left, right), "
                f"got {len(paths)}: {raw_env}"
            )

        return {action: Path(p) for action, p in zip(cls.ACTIONS, paths)}

    # @staticmethod
    # def _get_env_override_path(action: str) -> Path | None:
    #     """Returns the override path for brake (index 0) or right (index 2).

    #     Raises ValueError if ACTIVATION_VECTOR_PATHS is set but does not have
    #     exactly 3 paths.
    #     """
    #     raw_env = os.environ.get("ACTIVATION_VECTOR_PATHS", "").strip()
    #     if not raw_env:
    #         return None

    #     paths = [p.strip() for p in raw_env.split(",") if p.strip()]
    #     if len(paths) != 3:
    #         raise ValueError(
    #             f"ACTIVATION_VECTOR_PATHS must contain exactly 3 paths (brake, left, right), "
    #             f"got {len(paths)}: {raw_env}"
    #         )

    #     # Index mapping: 0 -> brake, 1 -> left, 2 -> right
    #     overrides = {
    #         "brake": Path(paths[0]),
    #         "right": Path(paths[2]),
    #     }
    #     return overrides.get(action)

    def _load_all(self) -> None:
        # override_actions = {"brake", "right"}
        for action, path_or_root in self._action_roots().items():
            # override_file = (
            #     self._get_env_override_path(action) if action in override_actions else None
            # )
            for layer_idx in self.steer_layers:
                if path_or_root.is_file():
                    file = path_or_root
                # if override_file is not None:
                #     file = override_file
                else:
                    file = path_or_root / self.feature_name / f"layer_{layer_idx:02d}" / self.VECTOR_FILENAME
                if not file.is_file():
                    raise FileNotFoundError(f"Steering vector for '{action}' layer {layer_idx} not found: {file}")
                vector = torch.load(file, map_location="cpu")
                if not torch.is_tensor(vector):
                    raise TypeError(f"{file} does not contain a tensor (got {type(vector).__name__}).")
                if self.feature_name == "align_query" and vector.ndim != 3:
                    raise ValueError(
                        f"Steering vector for '{action}' layer {layer_idx} has shape "
                        f"{tuple(vector.shape)}, but align_query vectors must be "
                        f"[batch, modes, embed_dims]. This usually means a speed-query "
                        f"vector was configured instead of an align_query vector: {file}"
                    )
                self.vectors[action][layer_idx] = vector.detach().to(device=self.device, dtype=torch.float32)
                print(
                    f"[HiPADSteeringManager] '{action}' layer {layer_idx}: "
                    f"shape={tuple(vector.shape)} norm={vector.float().norm().item():.4f} <- {file}"
                )

    # ------------------------------------------------------------------ query
    def get_steering_alphas(self, alpha) -> list[float]:
        """Return [brake, left, right] alphas after the internal per-action multipliers."""
        values = torch.as_tensor(alpha).detach().cpu().flatten().float().tolist()
        if len(values) == 1:
            loaded = self.loaded_actions
            target = loaded[0] if len(loaded) == 1 else "brake"
            alphas = [0.0] * len(self.ACTIONS)
            alphas[self.ACTIONS.index(target)] = values[0]
        elif len(values) == len(self.ACTIONS):
            alphas = list(values)
        else:
            raise ValueError("alpha must be a scalar or length 3 [brake, left, right].")
        return [a * m for a, m in zip(alphas, self.internal_multipliers)]

    def get_steering_vector(
        self,
        layer_idx: int,
        alpha_input: float | list[float] | None,
    ) -> Optional[torch.Tensor]:
        """Weighted sum of the action vectors for one layer (None if nothing to add)."""
        if alpha_input is None:
            return None
        total: Optional[torch.Tensor] = None
        for action, alpha in zip(self.ACTIONS, self.get_steering_alphas(alpha_input)):
            vector = self.vectors[action].get(layer_idx)
            if alpha != 0.0 and vector is not None:
                total = alpha * vector if total is None else total + alpha * vector
        return total

    def get_layer_vectors(
        self,
        alpha_input: float | list[float] | None,
    ) -> dict[int, torch.Tensor]:
        """{layer_idx: vector} for every layer in ``steer_layers`` that has a non-zero vector."""
        result: dict[int, torch.Tensor] = {}
        for layer_idx in self.steer_layers:
            vector = self.get_steering_vector(layer_idx, alpha_input)
            if vector is not None:
                result[layer_idx] = vector
        return result
