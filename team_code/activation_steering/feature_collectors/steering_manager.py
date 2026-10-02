from pathlib import Path
from typing import Optional, Union
import torch


class EnsembleSteeringManager:
    """Loads and applies steering vectors across ensemble models and actions."""

    ACTIONS = ("brake", "left", "right")
    ACTION_SUBDIRS = {
        "brake": "brake_manual",
        "left": "left_manual",
        "right": "right_manual",
    }

    def __init__(self, steering_root: Union[str, Path], device: torch.device):
        self.steering_root = Path(steering_root)
        self.device = device
        # Cache vectors as: vectors[action][model_idx] = Tensor
        self.vectors: dict[str, dict[int, torch.Tensor]] = {a: {} for a in self.ACTIONS}
        self._load_vectors()

    def _load_vectors(self):
        for action, subdir in self.ACTION_SUBDIRS.items():
            action_dir = self.steering_root / subdir
            if not action_dir.exists():
                continue
            for model_dir in sorted(action_dir.glob("model_*")):
                try:
                    model_idx = int(model_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                vec_file = model_dir / "steering_vector.pt"
                if vec_file.exists():
                    vec = torch.load(vec_file, map_location=self.device)
                    self.vectors[action][model_idx] = vec.to(dtype=torch.float32)

    def get_steering_vector(
        self,
        model_idx: int,
        alpha_input: Optional[float | list[float]]
    ) -> Optional[torch.Tensor]:
        """Combines loaded vectors based on the alpha vector [brake, left, right]."""
        if alpha_input is None:
            return None

        if isinstance(alpha_input, (int, float)):
            if alpha_input <= 0.0:
                return None
            alphas = [float(alpha_input), 0.0, 0.0]  # default to brake
        else:
            alphas = [float(a) for a in alpha_input]

        if not any(a > 0.0 for a in alphas):
            return None

        total_steer = None
        for action, alpha in zip(self.ACTIONS, alphas):
            if alpha > 0.0 and model_idx in self.vectors[action]:
                vec = self.vectors[action][model_idx]
                steer_component = alpha * vec
                total_steer = steer_component if total_steer is None else total_steer + steer_component

        return total_steer
