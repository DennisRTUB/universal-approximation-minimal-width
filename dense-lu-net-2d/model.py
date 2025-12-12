import torch
import torch.nn as nn
from functions import Identity, LeakySoftplus
from functools import partial
        
class LUNet(nn.Module):
    def __init__(self, num_lu_blocks=1, layer_size = 2, leaky_learn=False, raw_alpha=0.1, device="cuda:0"):
        """init LUNet with given number of blocks of LU layers"""
        super(LUNet, self).__init__()
        
        """masks to zero out gradients"""
        self.mask_triu = torch.triu(torch.ones(layer_size, layer_size)).bool()
        self.mask_tril = torch.tril(torch.ones(layer_size, layer_size)).bool().fill_diagonal_(False)
        
        """create LU modules"""
        self.intermediate_lu_blocks = nn.ModuleList()
        """adding number of LU Blocks"""
        for _ in range(num_lu_blocks):
            """init upper triangular weight matrix U without bias"""
            self.intermediate_lu_blocks.append(nn.Linear(layer_size, layer_size, bias=False))
            upper = self.intermediate_lu_blocks[-1]
            with torch.no_grad():
                upper.weight.copy_(torch.triu(upper.weight))
            upper.weight.register_hook(get_zero_grad_hook(self.mask_triu, device))
            """init lower triangular weight matrix L with bias"""
            self.intermediate_lu_blocks.append(nn.Linear(layer_size, layer_size))
            lower = self.intermediate_lu_blocks[-1]
            with torch.no_grad():
                lower.weight.copy_(torch.tril(lower.weight))
                lower.weight.copy_(lower.weight.fill_diagonal_(1))
            lower.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))
            self.intermediate_lu_blocks.append(LeakySoftplus(
                alpha=raw_alpha, num_features=layer_size, leaky_learn=leaky_learn))
          
        """Adding one final LU block = extra block"""
        self.final_lu_block = nn.ModuleList()
        """init upper triangular weight matrix U without bias"""
        self.final_lu_block.append(nn.Linear(layer_size, layer_size, bias=False))
        upper = self.final_lu_block[-1]
        with torch.no_grad():
            upper.weight.copy_(torch.triu(upper.weight))
        upper.weight.register_hook(get_zero_grad_hook(self.mask_triu, device))
        """init lower triangular weight matrix L with bias"""
        self.final_lu_block.append(nn.Linear(layer_size, layer_size))
        lower = self.final_lu_block[-1]
        with torch.no_grad():
            lower.weight.copy_(torch.tril(lower.weight))
            lower.weight.copy_(lower.weight.fill_diagonal_(1))
        lower.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))

        """adding some identity layers to access activations"""
        self.storage = nn.ModuleList()
        for _ in range(num_lu_blocks+1):
            self.storage.append(Identity())
        
        print("... initialized LUNet")

    def forward(self, x):
        """build network"""
        x = torch.flatten(x, 1)
        i = 0
        for layer in self.intermediate_lu_blocks:
            if isinstance(layer, LeakySoftplus):
                x = self.storage[i](x)
                i = i+1
            x = layer(x)
        """final LU block without activation"""
        for layer in self.final_lu_block:
            x = layer(x)
        return x


def get_zero_grad_hook(mask, device="cuda:0"):
    """zero out gradients"""
    def hook(grad):
        return grad * mask.to(device)
    return hook


"""
* helper functions to store activations and parameters in intermediate layers of the model
* use forward hooks for this, which are functions executed automatically during forward pass
* in PyTorch hooks are registered for nn.Module and are triggered by forward pass of object
"""

def save_activations(activations_dict, name, blu, bla, out):
    activations_dict[name] = out

def register_activation_hooks(model, layer_name):
    """register forward hooks in specified layers"""
    hooks = []
    activations_dict = {}
    for name, module in model.named_modules():
        if layer_name + "." in name:
            hooks.append(module.register_forward_hook(partial(save_activations, activations_dict, name)))
    return activations_dict, hooks