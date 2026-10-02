# feature_collectors/default_collector.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import torch
import torch.nn as nn

from activation_steering.feature_collectors.base import FeatureCollector


class DefaultFeatureCollector(FeatureCollector):
    """
    Default feature collector for TransFuser / LidarCenterNet models.

    By default, captures the fused representations (`fused_features`) produced
    during forward passes, matching the legacy saving behavior of SensorAgent.
    """

    DEFAULT_FEATURE_NAME: str = "fused_features"

    def __init__(
        self,
        output_root: str | Path,
        model_idx: int = -1,
        save_fused: bool = True,
        fused_feature_name: str = "fused_features",
    ):
        super().__init__(output_root, model_idx)
        self.save_fused = save_fused
        self.fused_feature_name = fused_feature_name

        self._captured_features: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: nn.Module) -> None:
        """Attach forward hooks to automatically capture fused features."""
        self.detach_hooks()

        if not self.save_fused:
            return

        # # 1. Standard TransFuser / BEV architectures: hook the fusion join module
        # if hasattr(model, "join") and isinstance(model.join, nn.Module):
        #     handle = model.join.register_forward_hook(
        #         self._create_hook(self.fused_feature_name)
        #     )
        #     self._hook_handles.append(handle)
        # # 2. TransFuser backbone variants: hook the transformer/bev fusion backbone
        # elif hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        #     handle = model.backbone.register_forward_hook(
        #         self._create_hook(self.fused_feature_name)
        #     )
        #     self._hook_handles.append(handle)

    def _create_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Any, output: Any):
            # If the layer outputs a tuple/list, take the primary fused tensor
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tensor, torch.Tensor):
                self._captured_features[name] = tensor.detach()

        return hook

    def collect(
        self,
        model: nn.Module,
        frame_idx: int,
        model_idx: int = 0,
    ) -> None:
        """
        Collect and persist fused features for one inference step.

        Supports both direct tensor passing (from legacy return_fused_features)
        and hook-based interception.
        """
        if self.save_fused and self._fused_feature is not None:
            self.save_feature(
                feature=self._fused_feature,
                feature_name=self.fused_feature_name,
                frame_idx=frame_idx,
                model_idx=model_idx,
            )
            self._fused_feature = None  # Clear after saving
            return

    def detach_hooks(self) -> None:
        """Clean up attached PyTorch forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._captured_features.clear()

    def __del__(self):
        self.detach_hooks()
