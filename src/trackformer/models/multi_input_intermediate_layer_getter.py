from collections import OrderedDict
from typing import Dict, Tuple
import torch
import torch.nn as nn


class MultiInputIntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model with multiple inputs.

    This supports models where inputs are split into multiple branches before merging.

    Args:
        model (nn.Module): Model from which intermediate layers are extracted.
        return_layers (Dict[str, str]): Dictionary where keys are layer names to extract,
                                       and values are the new names for returned activations.
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers.copy()
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out
