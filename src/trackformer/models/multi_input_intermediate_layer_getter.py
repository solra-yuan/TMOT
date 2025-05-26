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
        """
        Forward pass that supports multiple inputs.
        
        Assumes that the first input is processed in one branch and the second input in another.
        """
        out = OrderedDict()
        x1, x2 = x[:,0:3,:,:], x[:,3:,:,:]   # slicing 4-channel into two inputs
        
        for name, module in self.items():
            if name == "process_rgb":
                x1 = module(x1)
            elif name == "process_t":
                x2 = module(x2)
            elif name == "process_rgb2":
                x1_out = module(torch.cat([x1, x2], dim=1))
            elif name == "process_t2":
                x2_out = module(torch.cat([x1, x2], dim=1))
            elif name == "fusion_inplane_to_rgb":
                x = torch.cat((x1_out, x2_out), dim=1)  # Concatenation of branches
                x = module(x)
            else:
                x = module(x)  # Process normally if not part of a branching structure
            
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        
        return out
