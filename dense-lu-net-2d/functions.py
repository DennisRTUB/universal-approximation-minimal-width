from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from scipy import interpolate
from joblib import Parallel, delayed

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

class LeakySoftplus(nn.Module):
    def __init__(self, alpha: float = -2.25, num_features=None, leaky_learn=False) -> None:
        super(LeakySoftplus, self).__init__()
        if leaky_learn:
            self.raw_alpha = nn.Parameter(torch.full((num_features,), torch.tensor(alpha)), requires_grad=True)
        else:
            self.raw_alpha = nn.Parameter(torch.full((num_features,), alpha), requires_grad=False)
        
    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        softplus = torch.log1p(torch.exp(-torch.abs(input))) + torch.maximum(input, torch.tensor(0))
        # alpha = self.raw_alpha
        # alpha = torch.exp(self.raw_alpha)
        # alpha = torch.nn.functional.softplus(self.raw_alpha)
        alpha = 0.1 + 0.4 * torch.sigmoid(self.raw_alpha)
        output = alpha * input + (1 - alpha) * softplus
        return output

def lifted_sigmoid(x, raw_alpha=0.1):
    """derivative of leaky softplus"""
    # alpha = raw_alpha
    # alpha = torch.exp(self.raw_alpha)
    # alpha = torch.nn.functional.softplus(self.raw_alpha)
    alpha = 0.1 + 0.4 * torch.sigmoid(raw_alpha)
    return alpha + (1-alpha) * torch.sigmoid(x)

class InverseLeakySoftplus(nn.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        super(InverseLeakySoftplus,self).__init__()
        self.alpha = alpha
        if alpha.shape[0] != 1: # per-feature alpha
            self.num_features = alpha.shape[0]
        self.grid_x = torch.arange(-1000., 1000.001, 0.001)
    
    def compute_inverse(self, input, alpha):
        '''compute inverse via spline interpolation'''
        activation = LeakySoftplus(alpha=alpha, num_features=1)
        y = activation(self.grid_x)
        tck = interpolate.splrep(y, self.grid_x, s=0)
        yfit = interpolate.splev(input.cpu().detach().numpy(), tck, der=0)
        approx = torch.tensor(yfit, dtype=torch.float32)
        return approx

    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        if self.alpha.shape[0] != 1 and not torch.all(self.alpha == self.alpha[0]):
            '''different alphas require per dimension inverse'''
            approx_list = Parallel(n_jobs=2)(
                delayed(self.compute_inverse)(input[:, i], self.alpha[i]) 
                for i in range(len(self.alpha)))
            approx = torch.stack(approx_list, dim=1)
        else:
            '''fixed alpha can be processed batch-wise'''
            approx = self.compute_inverse(input, self.alpha[0])
        return approx
    
class InverseLowerLayer(nn.Module):
    def __init__(self, inverted_weight = None, inverted_bias = None) -> None:
        super(InverseLowerLayer, self).__init__()
        self.inverted_weight = inverted_weight
        self.inverted_bias = inverted_bias
        
    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        input = input.to(device)
        self.inverted_weight = self.inverted_weight.to(device)
        self.inverted_bias = self.inverted_bias.to(device)
        y_tilde = torch.t(input - self.inverted_bias)
        x_tilde = torch.linalg.solve(self.inverted_weight, y_tilde)
        x = torch.t(x_tilde)
        return x

class InverseUpperLayer(nn.Module):
    def __init__(self, inverted_weight = None) -> None:
        super(InverseUpperLayer, self).__init__()
        self.inverted_weight = inverted_weight
        
    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        input = input.to(device)
        self.inverted_weight = self.inverted_weight.to(device)
        y_tilde = torch.t(input)
        x_tilde = torch.linalg.solve(self.inverted_weight, y_tilde)
        x = torch.t(x_tilde)
        return x

"""loss function"""
def neg_log_likelihood(output, model, layers, device="cuda:0"):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D = output.shape # batch size and single output size
    
    """First summand"""
    constant = torch.from_numpy(np.array(0.5 * D * N * np.log(np.pi))).type(torch.float32)
    
    """Second summand"""
    sum_squared_mappings = torch.square(output)
    sum_squared_mappings = torch.sum(sum_squared_mappings)
    sum_squared_mappings = 0.5 * sum_squared_mappings
    
    """Third summand"""
    raw_alphas = []
    for name, param in model.state_dict().items():
        if "raw_alpha" in name: # slope parameter
            raw_alphas.append(param)

    '''log derivatives of leaky softplus activations'''
    log_derivatives = []
    for raw_alpha, (_, activations) in zip(raw_alphas, layers.items()):
        """layers are outputs of the L network layers"""
        """lifted sigmoid = derivative of leaky softplus"""
        log_derivatives.append(torch.log(torch.abs(lifted_sigmoid(activations, raw_alpha))))

    """log diagonals of U"""
    log_diagonals_triu = []
    for param in model.parameters():
        if len(param.shape) == 2 and param[1,0] == 0: # if upper triangular and matrix
            log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param))))

    '''volume correction summands / log determinant'''
    volume_corr = 0
    for l in range(len(log_diagonals_triu) - 1):
        """lu-blocks 1,...,M-1"""
        summand = torch.zeros(N, D).to(device)
        summand = summand + log_derivatives[l] # element-wise addition
        summand = summand + log_diagonals_triu[l] # broadcasted addition
        volume_corr = volume_corr + torch.sum(summand)
        
    """lu-block M"""
    volume_corr = volume_corr + N * torch.sum(log_diagonals_triu[-1])
    
    return constant + sum_squared_mappings - volume_corr
 