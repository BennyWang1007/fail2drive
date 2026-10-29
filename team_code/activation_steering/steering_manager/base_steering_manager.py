from __future__ import annotations

from typing import List, Optional, Union
import torch


Alpha = Optional[Union[float, List[float]]]
AlphaVec = List[float]  # (brake, left, right)


class BaseSteeringManager:
    """Abstract base class for managing and applying steering vectors."""

    ACTIONS = ("brake", "left", "right")

    def get_steering_vector(
        self,
        *args,
        **kwargs
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _to_alpha_list(alpha: Alpha) -> AlphaVec:
        """Convert a single float or a list of floats into a 1D torch.Tensor."""
        if isinstance(alpha, float):
            return [alpha, alpha, alpha]
        elif isinstance(alpha, List):
            if len(alpha) == 3:
                return alpha
            else:
                raise ValueError(f"Alpha must be a list of 3 floats, got {len(alpha)}")
        else:
            raise ValueError(f"Alpha must be a float or a list of floats, got {type(alpha)}")
