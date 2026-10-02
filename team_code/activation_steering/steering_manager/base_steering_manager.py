from __future__ import annotations

from typing import Optional, Union
import torch


class BaseSteeringManager:
    """Abstract base class for managing and applying steering vectors."""

    def get_steering_vector(
        self,
        model_idx: int,
        alpha_input: Optional[Union[float, list[float]]],
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError
