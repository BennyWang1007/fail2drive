from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union
import torch


from activation_steering.steering_manager.base_steering_manager import BaseSteeringManager


class Tfv6SteeringManager(BaseSteeringManager):
    """Steering vector manager for TransFuser v6 across ensemble models and decoder layers."""

    ACTIONS = ("brake", "left", "right")
    ACTION_SUBDIRS = {
        "brake": "brake_manual",
        "left": "left_manual",
        "right": "right_manual",
    }

    INTERNAL_ALPHA_MULTIPLIERS: dict[str, list[float]] = {
        "decoder_layer5": [1.0, 3.0, 2.5],
        "decoder_layer6": [1.0, 2.5, 2.5],
        "fused_features": [1.0, 1.1, 1.1],
        # "fused_features": [1.0, 1.5, 1.5],
    }

    def __init__(
        self,
        steering_root: Union[str, Path],
        device: torch.device,
        feature_name: str = "decoder_layer5",
    ):
        print(f"[Tfv6SteeringManager.__init__] Initializing with steering_root: {steering_root}, device: {device}, feature_name: {feature_name}")
        self.steering_root = Path(steering_root)
        self.device = device
        self.feature_name = feature_name  # e.g., 'decoder_layer4', 'decoder_layer5', 'decoder_layer6', 'fused_features'

        assert self.feature_name in self.INTERNAL_ALPHA_MULTIPLIERS, \
               f"Feature name '{self.feature_name}' is not recognized. Valid options are: {list(self.INTERNAL_ALPHA_MULTIPLIERS.keys())}"
        self.internal_multipliers = self.INTERNAL_ALPHA_MULTIPLIERS[self.feature_name]

        # if exist, BRAKE/LEFT/RIGHT_ACTIVATION_ALPHA_SCALE
        if "BRAKE_ACTIVATION_ALPHA_SCALE" in os.environ:
            self.internal_multipliers[0] = float(os.environ["BRAKE_ACTIVATION_ALPHA_SCALE"])
        if "LEFT_ACTIVATION_ALPHA_SCALE" in os.environ:
            self.internal_multipliers[1] = float(os.environ["LEFT_ACTIVATION_ALPHA_SCALE"])
        if "RIGHT_ACTIVATION_ALPHA_SCALE" in os.environ:
            self.internal_multipliers[2] = float(os.environ["RIGHT_ACTIVATION_ALPHA_SCALE"])

        # Structure: vectors[action][model_idx] = torch.Tensor
        self.vectors: dict[str, dict[int, torch.Tensor]] = {a: {} for a in self.ACTIONS}
        self._load_vectors()

    def _load_vectors(self) -> None:
        from lead.training.config_training import TrainingConfig
        config = TrainingConfig()
        for action, subdir in self.ACTION_SUBDIRS.items():
            print(f"[Tfv6SteeringManager] Loading steering vectors for action '{action}' from subdirectory '{subdir}'")
            action_dir = self.steering_root / subdir
            # ls the action_dir
            print(f"[Tfv6SteeringManager] Listing contents of {action_dir}:")
            for item in action_dir.iterdir():
                print(f"  {item}")

            if not action_dir.exists():
                continue

            for model_dir in sorted(action_dir.glob("model_*")):
                try:
                    print(f"[Tfv6SteeringManager] Processing model directory: {model_dir}")
                    model_idx = int(model_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                print(f"[Tfv6SteeringManager] Loading steering vector for action '{action}', model {model_idx}, feature '{self.feature_name}' from {model_dir}")

                # Locate vector file inside target feature subfolder: .../model_xx/{feature_name}/steering_vector.pt
                feature_dir = model_dir / self.feature_name
                vec_file = feature_dir / "steering_vector.pt"

                if vec_file.exists():
                    vec = torch.load(vec_file, map_location=self.device)
                    # self.vectors[action][model_idx] = vec.to(dtype=torch.float32)
                    self.vectors[action][model_idx] = vec.to(dtype=config.torch_float_type)  # Use the same dtype as the model's parameters
                    print(f"[Tfv6SteeringManager] Loaded steering vector for action '{action}', \
                            model {model_idx}, feature '{self.feature_name}'")
                    # print(f"  Vector shape: {vec.shape}, dtype: {vec.dtype}, device: {vec.device}")
                    # print(f"  {self.vectors[action][model_idx]}")
                else:
                    print(f"[Tfv6SteeringManager] Steering vector file not found for action '{action}', \
                            model {model_idx}, feature '{self.feature_name}': {vec_file}")
                    raise FileNotFoundError(f"Steering vector file not found: {vec_file}")
        # raise NotImplementedError("Testing Exit")

    def get_steering_vector(
        self,
        model_idx: int,
        alpha_input: float | list[float] | None,
    ) -> Optional[torch.Tensor]:
        """Calculates the weighted linear combination of steering vectors for the given model index."""
        if alpha_input is None:
            return None

        # if isinstance(alpha_input, (int, float)):
        #     # if alpha_input <= 0.0:
        #     #     return None
        #     alphas = [float(alpha_input), float(alpha_input), float(alpha_input)]
        # else:
        #     alphas = [float(a) for a in alpha_input]

        # if not any(a > 0.0 for a in alphas):
        #     return None

        # if self.feature_name in self.INTERNAL_ALPHA_MULTIPLIERS:
        #     internal_multipliers = self.INTERNAL_ALPHA_MULTIPLIERS[self.feature_name]
        #     alphas = [a * m for a, m in zip(alphas, internal_multipliers)]

        alphas = self.get_steering_alphas(alpha_input)
        if not any(a > 0.0 for a in alphas):
            return None

        total_steer: Optional[torch.Tensor] = None
        for action, alpha in zip(self.ACTIONS, alphas):
            if alpha > 0.0 and model_idx in self.vectors[action]:
                vec = self.vectors[action][model_idx]
                steer_component = alpha * vec
                total_steer = (
                    steer_component
                    if total_steer is None
                    else total_steer + steer_component
                )

        return total_steer

    def get_steering_alphas(self, alpha: float | list[float]) -> list[float]:
        """Returns the adjusted alphas based on the feature name and internal multipliers."""
        if isinstance(alpha, (int, float)):
            alphas = [float(alpha), float(alpha), float(alpha)]
        else:
            alphas = [float(a) for a in alpha]

        # if self.feature_name in self.INTERNAL_ALPHA_MULTIPLIERS:
        #     internal_multipliers = self.INTERNAL_ALPHA_MULTIPLIERS[self.feature_name]
        #     alphas = [a * m for a, m in zip(alphas, internal_multipliers)]

        if hasattr(self, "internal_multipliers"):
            alphas = [a * m for a, m in zip(alphas, self.internal_multipliers)]
        else:
            raise RuntimeError("internal_multipliers not set.")

        return alphas
