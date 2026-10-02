# feature_collectors/tfv6.py

from __future__ import annotations

from pathlib import Path

import torch

from .base import FeatureCollector


class TFv6FeatureCollector(FeatureCollector):

    DEFAULT_FEATURE_NAME: str = "decoder_layer5"

    """
    Feature collector for TFv6.

    Saves:
        - fused_features
        - decoder_layer4
        - decoder_layer5
        - decoder_layer6
    """

    def __init__(
        self,
        output_root: str | Path,
        model_idx: int = -1,
        save_fused: bool = True,
        decoder_layers: tuple[int, ...] = (4, 5, 6),
    ):
        super().__init__(output_root, model_idx)

        self.save_fused = save_fused
        self.decoder_layers = decoder_layers

        self._fuse_feature: torch.Tensor | None = None
        self._activations: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: torch.nn.Module) -> None:
        self.remove_hooks()

        decoder = self._find_decoder(model)

        assert hasattr(decoder, "transformer_decoder") and isinstance(decoder.transformer_decoder, torch.nn.Module)
        assert hasattr(decoder.transformer_decoder, "layers")

        layers = decoder.transformer_decoder.layers

        assert isinstance(layers, torch.nn.ModuleList), "Expected decoder.transformer_decoder.layers to be a ModuleList."

        for layer_number in self.decoder_layers:
            layer_index = layer_number - 1

            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(
                    f"TFv6 decoder has {len(layers)} layers, "
                    f"cannot collect layer {layer_number}."
                )

            hook = layers[layer_index].register_forward_hook(
                self._make_hook(layer_number)
            )

            self._hooks.append(hook)

    def _make_hook(self, layer_number: int):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._activations[layer_number] = output.detach()

        return hook

    def collect(
        self,
        model: torch.nn.Module,
        frame_idx: int,
        model_idx: int = -1,
    ) -> None:
        if not self._hooks:
            self.attach(model)

        for layer_number in self.decoder_layers:
            feature = self._activations.get(layer_number)

            if feature is None:
                raise RuntimeError(
                    f"Decoder layer {layer_number} activation was not captured."
                )

            self.save_feature(
                feature=feature,
                feature_name=f"decoder_layer{layer_number}",
                frame_idx=frame_idx,
                model_idx=model_idx,
            )

        self._activations.clear()

        if self.save_fused and self._fuse_feature is not None:
            self.save_feature(
                feature=self._fuse_feature,
                feature_name="fused_features",
                frame_idx=frame_idx,
                model_idx=model_idx,
            )
            self._fuse_feature = None

    def set_fused_feature(self, fused_features: torch.Tensor) -> None:
        if not self.save_fused:
            return

        self._fuse_feature = fused_features.detach()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()

        self._hooks.clear()
        self._activations.clear()

    @staticmethod
    def _find_decoder(model: torch.nn.Module) -> torch.nn.Module:
        """
        Locate the TFv6 PlanningDecoder.

        Adjust this lookup only if the TFv6 model's module hierarchy differs.
        """
        candidates = (
            "planning_decoder",
            "decoder",
            "planning_decoder_module",
        )

        for name in candidates:
            decoder = getattr(model, name, None)

            if decoder is not None and hasattr(
                decoder,
                "transformer_decoder",
            ):
                return decoder

        for module in model.modules():
            if (
                hasattr(module, "transformer_decoder")
                and hasattr(module.transformer_decoder, "layers")
            ):
                return module

        raise AttributeError(
            "Could not locate TFv6 PlanningDecoder containing "
            "`transformer_decoder.layers`."
        )

    def __del__(self):
        self.remove_hooks()
