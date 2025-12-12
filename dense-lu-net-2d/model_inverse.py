import torch.nn as nn
from functions import InverseLeakySoftplus, InverseLowerLayer, InverseUpperLayer

class InverseLUNet(nn.Module):
    def __init__(self, num_lu_blocks=1, params=None):
        """init inverted LuNet with given numer of blocks of LU layers"""
        super(InverseLUNet, self).__init__()
        
        """initialize the weights and bias"""
        alpha = []
        bias = []
        l_weight = []
        u_weight = []
        
        for name, param in reversed(params.items()):
            if "alpha" in name: # slope parameter
                alpha.append(param)
            elif "bias" in name: # bias
                bias.append(param)
            elif len(param.shape) == 2 and param[1,0] == 0: # U weight
                u_weight.append(param)
            else: # L weight
                l_weight.append(param)
        
        self.inverse_lu_blocks = nn.ModuleList()
        self.inverse_lu_blocks.append(InverseLowerLayer(l_weight[0], bias[0]))
        self.inverse_lu_blocks.append(InverseUpperLayer(u_weight[0]))
        for i in range(1, num_lu_blocks + 1):
            self.inverse_lu_blocks.append(InverseLeakySoftplus(alpha[i-1]))
            self.inverse_lu_blocks.append(InverseLowerLayer(l_weight[i], bias[i]))
            self.inverse_lu_blocks.append(InverseUpperLayer(u_weight[i]))
        
        print("... initialized inverse LUNet")
            
    def forward(self, x, device="cuda:0"):
        x = x.to(device)
        for layer in self.inverse_lu_blocks:
            x = layer(x, device)
        return x
 