# feature_collectors/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class FeatureCollector(ABC):
    """Base interface for model-specific activation feature collection."""

    DEFAULT_FEATURE_NAME: str = "fused_features"

    def __init__(self, output_root: str | Path, model_idx: int = -1):
        self.output_root = Path(output_root)
        # self.output_root = self.resolve_model_root(self.output_root, model_idx)
        self.model_idx = model_idx
        self._fused_feature: torch.Tensor | None = None

    def set_fused_feature(self, fused_features: torch.Tensor) -> None:
        """Set fused features for collection, to be saved later."""
        self._fused_feature = fused_features.detach()

    @abstractmethod
    def collect(
        self,
        model: torch.nn.Module,
        frame_idx: int,
        model_idx: int = 0,
    ) -> None:
        """Collect activations from a model for one inference frame."""
        raise NotImplementedError

    def save_feature(
        self,
        feature: torch.Tensor,
        feature_name: str,
        frame_idx: int,
        model_idx: int = -1,
    ) -> Path:

        save_path = self.resolve_feature_path(
            feature_dir=self.output_root,
            frame_idx=frame_idx,
            feature_name=feature_name,
            model_idx=model_idx
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature.detach().cpu(), save_path)

        return save_path

    @classmethod
    def resolve_feature_path(
        cls,
        feature_dir: Path,
        frame_idx: int,
        feature_name: str | None = None,
        model_idx: int = -1,
    ) -> Path:
        """
        Resolve the exact file path where a feature should be saved or loaded.
        Example:
            feature_dir/model_01/decoder_layer5/000042.pt
        """
        resolved_name = feature_name or cls.DEFAULT_FEATURE_NAME

        target_dir = cls.resolve_model_root(feature_dir, model_idx)
        target_dir = target_dir / resolved_name

        return target_dir / f"{int(frame_idx):06d}.pt"

    @staticmethod
    def resolve_model_root(feature_dir: Path, model_idx: int = -1) -> Path:
        """
        Resolve the directory path for a specific model index.
        Add the model index to the path if it's not -1.
        """
        if model_idx != -1:
            return feature_dir / f"model_{model_idx:02d}"
        return feature_dir

    @staticmethod
    def _detach(feature: Any) -> torch.Tensor | None:
        if feature is None:
            return None

        if not isinstance(feature, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, got {type(feature).__name__}"
            )

        return feature.detach()
