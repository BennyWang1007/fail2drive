from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import FeatureCollector


class HiPADFeatureCollector(FeatureCollector):
    """Collect the planning features written by the legacy HiP-AD model code.

    HiP-AD does not return ``align_query`` from its refinement module.  The
    legacy implementation saved it inside
    ``SparsePlanAlignRefinementModule.forward`` after adding ``anchor_embed``
    and summing the temporal/spatial anchor groups.  A hook on the module's
    output therefore cannot reproduce the old feature.  This collector uses a
    forward-pre-hook to perform that same calculation from the module inputs.

    Files are written as::

        <output_root>/<feature_name>/layer_<nn>/<frame:06d>.pt
    """

    DEFAULT_FEATURE_NAME = "align_query"

    def __init__(
        self,
        output_root: str | Path,
        feature_name: str = DEFAULT_FEATURE_NAME,
    ):
        super().__init__(output_root)

        self.feature_name = feature_name
        self._plan_layers: torch.nn.ModuleList | None = None
        self._activations: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: torch.nn.Module) -> None:
        self.remove_hooks()

        layers = self.find_plan_layers(model)
        self._plan_layers = layers
        for layer_index, layer in enumerate(layers):
            hook_fn = self._make_hook(layer_index)
            try:
                # Newer PyTorch: hook receives positional and keyword arguments.
                handle = layer.register_forward_pre_hook(
                    hook_fn,
                    with_kwargs=True,
                )
            except TypeError as exc:
                # Older PyTorch: pre-hook receives only (module, inputs).
                if "with_kwargs" not in str(exc):
                    raise

                def legacy_hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    _hook_fn=hook_fn,
                ) -> None:
                    _hook_fn(module, inputs, {})

                handle = layer.register_forward_pre_hook(legacy_hook)

            self._hooks.append(handle)

    def _make_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            values = list(inputs)
            if len(values) < 3:
                values.extend(
                    kwargs.get(name)
                    for name in ("instance_feature", "anchor", "anchor_embed")
                    if name in kwargs
                )
            if len(values) < 3:
                raise RuntimeError(
                    "HiP-AD plan refinement expected "
                    "(instance_feature, anchor, anchor_embed) inputs."
                )

            instance_feature, _, anchor_embed = values[:3]
            if not torch.is_tensor(instance_feature) or not torch.is_tensor(anchor_embed):
                raise TypeError(
                    "HiP-AD plan refinement inputs must contain tensors for "
                    "instance_feature and anchor_embed."
                )

            if self.feature_name == "pre_instance_feature":
                feature = instance_feature
            else:
                use_plan_anchor_embed = True
                # The fourth positional argument is the legacy flag.  Keyword
                # arguments are uncommon here, but are handled by the wrapper
                # below when a model invokes the module that way.
                if len(values) >= 4:
                    use_plan_anchor_embed = bool(values[3])
                elif "use_plan_anchor_embed" in kwargs:
                    use_plan_anchor_embed = bool(kwargs["use_plan_anchor_embed"])
                instance_with_embed = (
                    instance_feature + anchor_embed
                    if use_plan_anchor_embed
                    else instance_feature
                )

                if self.feature_name == "instance_feature_with_anchor_embed":
                    feature = instance_with_embed
                elif self.feature_name == "align_query":
                    feature = self._align_query(module, instance_with_embed)
                else:
                    raise ValueError(
                        f"Unsupported HiP-AD feature name: {self.feature_name}"
                    )

            self._activations[layer_index] = feature.detach()

        return hook

    @staticmethod
    def _align_query(
        module: torch.nn.Module,
        instance_feature: torch.Tensor,
    ) -> torch.Tensor:
        anchor_types = getattr(module, "anchor_types", None)
        anchor_group = getattr(module, "anchor_group", None)
        if not anchor_types or not isinstance(anchor_group, int):
            raise AttributeError(
                "HiP-AD plan refinement module is missing anchor_types/anchor_group."
            )

        instance_features = torch.stack(
            instance_feature.chunk(anchor_group, dim=1)
        )
        align_query = [
            instance_features[index]
            for index, anchor_type in enumerate(anchor_types)
            if anchor_type[0] in ("temp", "spat")
        ]
        if not align_query:
            raise RuntimeError(
                "HiP-AD plan refinement contains no temporal/spatial anchor groups."
            )
        return sum(align_query)

    def collect(
        self,
        model: torch.nn.Module,
        frame_idx: int,
    ) -> None:
        if not self._hooks:
            self.attach(model)

        if self._plan_layers is None:
            raise RuntimeError("HiP-AD plan refinement layers are not attached.")
        missing = [
            index
            for index in range(len(self._plan_layers))
            if index not in self._activations
        ]
        if missing:
            raise RuntimeError(
                f"HiP-AD plan layer(s) {missing} activation was not captured."
            )

        for layer_index in range(len(self._plan_layers)):
            path = self._layer_feature_path(
                frame_idx=frame_idx,
                feature_name=self.feature_name,
                layer_index=layer_index,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._activations[layer_index].cpu(), path)

        self._activations.clear()

    def _layer_feature_path(
        self,
        frame_idx: int,
        feature_name: str,
        layer_index: int,
        feature_dir: str | Path | None = None,
    ) -> Path:
        # The legacy HiP-AD writer did not add a model_<nn> component.  There
        # is one planner in this agent.
        root = Path(feature_dir) if feature_dir is not None else self._root
        return root / feature_name / f"layer_{layer_index:02d}" / f"{int(frame_idx):06d}.pt"

    def resolve_feature_path(
        self,
        feature_dir: str | Path | None = None,
        frame_idx: int = 0,
        feature_name: str | None = None,
        layer_index: int = 0,
        model_idx: int = -1,
    ) -> Path:
        assert model_idx == -1, "HiP-AD does not support multiple models; model_idx must be -1"
        # Keep the base interface usable by callers that need a deterministic
        # path; layer-specific files are written by collect().
        return self._layer_feature_path(
            frame_idx=frame_idx,
            feature_name=feature_name or self.feature_name,
            layer_index=layer_index,
            feature_dir=feature_dir,
        )

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()
        self._plan_layers = None

    @staticmethod
    def find_plan_layers(
        model: torch.nn.Module,
    ) -> torch.nn.ModuleList:
        try:
            layers = model.head.onedecoder_head.plan_refine
        except AttributeError as exc:
            raise AttributeError(
                "Could not locate HiP-AD plan refinement layers at "
                "model.head.onedecoder_head.plan_refine."
            ) from exc
        if not isinstance(layers, torch.nn.ModuleList):
            raise TypeError(
                "model.head.onedecoder_head.plan_refine must be a ModuleList."
            )
        if not layers or not all(
            hasattr(layer, "anchor_types") and hasattr(layer, "anchor_group")
            for layer in layers
        ):
            raise TypeError(
                "HiP-AD plan_refine must contain refinement modules with "
                "anchor_types and anchor_group."
            )
        return layers

    def __del__(self):
        self.remove_hooks()
