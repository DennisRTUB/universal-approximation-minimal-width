import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

# this is a basic generalized leaky ReLU function
def lrelu(x, alpha):
    return torch.where(x >= 0, x, alpha * x)

# discontinuous identity shifted by c at 0
def id_step(x, d):
    return torch.where(x >= 0, x, x + d)

def id_step_shifted(x, c, d):
    mask = x >= c
    x[mask] = x[mask] + d
    return x
    return torch.where(x >= c, x+d, x)

class IdStepShifted(torch.nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c = c
        self.d = d

    def forward(self, x):
        #return torch.where(x >= self.c, x + self.d, x)
        return id_step_shifted(x, self.c, self.d)

# this class is initialized with parameters, a, b, c, d and a device: 
# It is able to compute the functions sigma_{a,b} and sigma_{a,b}^{c,d} from the paper 
# and does so using only with compositions of affine functions and generalized LReLUs in R
class Sigma_ab_cd(torch.nn.Module):
    def __init__(self, a, b, c, d, dtype=torch.float64, device="cpu"):
        super().__init__()
        assert a != 0, "a must be non-zero"
        assert b != 0, "b must be non-zero"
        assert device in ["cpu", "cuda"], "device must be 'cpu' or 'cuda'"
        self.device = device
        self.a = torch.tensor(a, dtype=dtype, device=device) if not isinstance(a, torch.Tensor) else a.to(copy=True, dtype=dtype, device=device)
        self.b = torch.tensor(b, dtype=dtype, device=device) if not isinstance(b, torch.Tensor) else b.to(copy=True, dtype=dtype, device=device)
        if c is not None:
            self.c = c.to(copy=True, dtype=dtype, device=device) if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=dtype, device=device)
        else:
            self.c = torch.tensor(0.0, dtype=dtype, device=device)
        if d is not None:
            self.d = d.to(copy=True, dtype=dtype, device=device) if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=dtype, device=device)
        else:
            self.d = torch.tensor(0.0, dtype=dtype, device=device)

        self.a_abs = torch.abs(self.a)
        self.a_sign = torch.sign(self.a)
        self.a_abs_inverse = 1.0 / self.a_abs
        self.b_abs = torch.abs(self.b)
        self.b_sign = torch.sign(self.b)
        self.b_abs_geq_a_abs = (self.b_abs >= self.a_abs).item()
        self.b_abs_inverse = 1.0 / self.b_abs
        self.b_abs_div_a_abs = self.b_abs / self.a_abs
        self.a_abs_div_b_abs = self.a_abs / self.b_abs
        self.a_b_sign_prod = self.a_sign * self.b_sign
        self.a_and_b_negative = (self.a < 0 and self.b < 0)
        self.a_and_b_negative_factor = torch.tensor(1.0 if self.a_and_b_negative else -1.0)

    def sigma_ab(self, x):
        if self.b_abs_geq_a_abs:
            return self.b_sign * lrelu(self.b_abs * lrelu(x, self.a_abs_div_b_abs), self.a_b_sign_prod)
        else:
            return -self.a_sign * lrelu(self.a_abs * lrelu(-x, self.b_abs_div_a_abs), self.a_b_sign_prod)

    def forward(self, x):
        val = self.sigma_ab(x - self.c)
        y = val + self.d
        return y
        #return self.sigma_ab(x - self.c) + self.d
    
    """@classmethod
    def call_(cls, x, a, b)"""
    
    def plot_sigma_ab(self, points, dpi=100, show_title=True, save_path=None):
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        values = self.sigma_ab(points)
        # Plot
        plt.figure(figsize=(10, 6), dpi=dpi)
        plt.plot(points.cpu().numpy(), values.cpu().numpy(), linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('sigma_ab(x)', fontsize=12)
        if show_title:
            plt.title(f'Sigma with a={self.a.item()}, b={self.b.item()} Function', fontsize=14)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_sigma(self, points, dpi=100, show_title=True, save_path=None):
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        values = self.forward(points)
        # Plot
        plt.figure(figsize=(10, 6), dpi=dpi)
        plt.plot(points.cpu().numpy(), values.cpu().numpy(), linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        if show_title:
            plt.title(f'Sigma with a={self.a.item()}, b={self.b.item()}, c={self.c.item()}, d={self.d.item()}', fontsize=14)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
    def __repr__(self):
        return f"Sigma_ab_cd(a={self.a.item()}, b={self.b.item()}, c={self.c.item()}, d={self.d.item()})"


class EncoderExact1D(nn.Module):
    def __init__(self, K, dtype=torch.float64, device="cpu"):
        super().__init__()
        assert K >= 1, "K must be greater than 1"
        self.K = K
        self.num_levels = 2 ** K
        self.base_slice = 2.0 ** (-K)
        self.binary_positions = [i * self.base_slice for i in range(self.num_levels)]
        self.dtype = dtype
        self.device = device

    def forward(self, x):
        # Map x to its corresponding binary position
        indices = torch.clamp((x / self.base_slice).long(), 0, self.num_levels - 1)
        encoded_x = torch.tensor([self.binary_positions[i] for i in indices], dtype=self.dtype, device=self.device)
        return encoded_x
    
    def plot(self, points=None, dpi=100, show_title=True, save_path=None):
        if points is None:
            points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
        
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        
        # Set up the plot
        plt.figure(figsize=(10, 6), dpi=dpi)
        
        # Plot each continuous segment (each step interval)
        for i in range(self.num_levels):
            start = i * self.base_slice
            end = (i + 1) * self.base_slice if i < self.num_levels - 1 else 1.0
            
            # Get points in this interval
            segment_mask = (points >= start) & (points < end)
            segment_points = points[segment_mask]
            
            if len(segment_points) > 0:
                segment_values = self.forward(segment_points)
                plt.plot(segment_points.cpu().numpy(), 
                        segment_values.cpu().numpy(), 
                        linewidth=2, 
                        color='C0')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'Exact Quantizer with K={self.K} ({self.num_levels} levels)', fontsize=14)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
        
class EncoderApprox1D(torch.nn.Module):
    def __init__(self, K, gamma, flat_slope=1e-6, dtype=torch.float64, device="cpu"):
        super().__init__()
        assert K >= 1, "K must be greater than 1"
        assert gamma > 0, "gamma must be positive"
        assert flat_slope > 0, "flat_slope must be positive"
        self.K = K
        self.dtype = dtype
        self.device = device
        self.num_levels = 2 ** K
        self.gamma = gamma
        self.flat_slope = flat_slope
        self.base_slice = 2.0 ** (-K)
        assert gamma < self.base_slice, "gamma must be less than 2^(-K)"

        self.flat_slice = self.base_slice - gamma
        self.steep_slice = gamma
        self.binary_positions = [i * self.base_slice for i in range(self.num_levels)]
        self.flat_slice_base_value = self.flat_slope * self.flat_slice

        self.steep_slope = (self.base_slice - self.flat_slice_base_value) / gamma
        self.steep_slope_relative_to_flat = self.steep_slope / self.flat_slope
        self.flat_slope_relative_to_steep = self.flat_slope / self.steep_slope
        if self.steep_slope_relative_to_flat <= 1.0:
            if self.steep_slope_relative_to_flat < 0.0:
                raise ValueError("Negative steep slope relative to flat slope encountered,  which possibly indicates an overflow. Try increasing flat_slope or gamma K.")
            else:
                raise ValueError("steep_slope must be greater than flat_slope. Try decreasing flat_slope or gamma.")

        self.sigmas = [Sigma_ab_cd(a=self.flat_slope, b=self.steep_slope, c=self.flat_slice, d=self.flat_slice_base_value, dtype=dtype, device=device)]
        self.binary_positions_tensor = torch.tensor(self.binary_positions, dtype=dtype, device=device)
        self.shift_down_offsets = []
        for i in range(1, self.num_levels):
            c_flat = self.binary_positions[i]
            d_flat = self.binary_positions[i]
            sigma_flat = Sigma_ab_cd(a=1.0, b=self.flat_slope_relative_to_steep, c=c_flat, d=d_flat, dtype=dtype, device=device)
            #c_steep = self.binary_positions[i] - self.gamma
            c_steep = self.binary_positions[i] + self.flat_slice_base_value
            d_steep = self.binary_positions[i] + self.flat_slice_base_value
            sigma_steep = Sigma_ab_cd(a=1.0, b=self.steep_slope_relative_to_flat, c=c_steep, d=d_steep, dtype=dtype, device=device)
            self.sigmas.append(sigma_flat)
            self.sigmas.append(sigma_steep)


    def forward(self, x):
        for i in range(len(self.sigmas)):
            #shift_down_offset = self.binary_positions_tensor[i // 2] - self.gamma if i % 2 == 0 else self.binary_positions_tensor[i // 2]
            x = self.sigmas[i](x)
        return x

    def plot(self, points=None, dpi=100, show_title=True, save_path=None):
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 500000, dtype=self.dtype, device=self.device)
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        values = self.forward(points)
        # Plot
        plt.figure(figsize=(10, 6), dpi=dpi)
        plt.plot(points.cpu().numpy(), values.cpu().numpy(), linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'Quantizer Approximation with K={self.K}, flat_slope={self.flat_slope:.2e}, gamma={self.gamma:.2e} and steep_slope={self.steep_slope:.2e}', fontsize=14)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_with_exact(self, points=None, dpi=100, show_title=True, save_path=None):
        """Plot approximate encoder together with exact encoder for comparison"""
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 500000, dtype=self.dtype, device=self.device)
        
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        
        # Create exact encoder with same K
        exact_encoder = EncoderExact1D(K=self.K, dtype=self.dtype, device=self.device)
        
        # Get approximate encoder values (continuous)
        approx_values = self.forward(points)
        
        # Calculate error metrics
        exact_values_all = exact_encoder.forward(points)
        mse_error = torch.mean((approx_values - exact_values_all)**2).item()
        max_error = torch.max(torch.abs(approx_values - exact_values_all)).item()
        
        # Create plot
        plt.figure(figsize=(12, 7), dpi=dpi)
        
        # Plot exact encoder piecewise (to avoid lines across discontinuities)
        boundaries = [i * exact_encoder.base_slice for i in range(exact_encoder.num_levels)] + [1.0]
        for i in range(exact_encoder.num_levels):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Get points in this interval
            segment_mask = (points >= start) & (points < end)
            segment_points = points[segment_mask]
            
            if len(segment_points) > 0:
                segment_values = exact_encoder.forward(segment_points)
                if i == 0:
                    plt.plot(segment_points.cpu().numpy(), 
                            segment_values.cpu().numpy(), 
                            'b-', linewidth=2, label='Exact Quantizer', alpha=0.7)
                else:
                    plt.plot(segment_points.cpu().numpy(), 
                            segment_values.cpu().numpy(), 
                            'b-', linewidth=2, alpha=0.7)
        
        # Plot approximate encoder continuously (it's smooth)
        plt.plot(points.cpu().numpy(), approx_values.cpu().numpy(), 
                'r--', linewidth=2, label='Approximate Quantizer', alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'Exact vs Approximate Quantizer (K={self.K}, flat_slope={self.flat_slope:.2e}, gamma={self.gamma:.2e})\n'
                     f'MSE: {mse_error:.2e}, Max Error: {max_error:.2e}', fontsize=14)
        plt.legend(fontsize=11)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
        return mse_error, max_error


class EncoderSupApprox1D(nn.Module):
    def __init__(self, K, slope=1e-6, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device
        assert K >= 1, "K must be greater than 1"
        assert slope > 0, "slope must be positive"
        assert slope <= 1, "slope must be less than or equal to 1"
        self.K = K
        self.num_levels = 2 ** K
        self.slope = slope
        self.base_slice = 2.0 ** (-K)
        self.base_value = self.slope * self.base_slice
        self.base_offset = self.base_slice - self.base_value
        assert self.base_offset >= 0, f"base_offset must be greater or equal than zero, but it is {self.base_offset}"

        self.binary_positions = [i * self.base_slice for i in range(self.num_levels)]

        self.composition = [Sigma_ab_cd(a=self.slope, b=self.slope, c=0.0, d=0.0, dtype=dtype, device=device), IdStepShifted(c=0.0, d=0.0)]
        self.composition += [IdStepShifted(c=self.binary_positions[i] + self.base_value, d=self.base_offset) for i in range(self.num_levels - 1)]
        self.sequential = nn.Sequential(*self.composition)

    def forward(self, x):
        # Override forward method for sup approximation if needed
        for i in range(len(self.composition)):
            x = self.composition[i](x)
        return x
        #return self.sequential(x)
    
    def plot(self, points=None, dpi=100, show_title=True, save_path=None):
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 5000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
        
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        
        # Create boundary points including 0 and 1
        boundaries = [0.0] + sorted(self.binary_positions) + [1.0]
        
        # Set up the plot
        plt.figure(figsize=(10, 6), dpi=dpi)
        
        # Plot each continuous segment
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            
            # Create points for this segment
            segment_mask = (points >= start) & (points <= end)
            segment_points = points[segment_mask]
            
            if len(segment_points) > 0:
                segment_values = self.forward(segment_points)
                plt.plot(segment_points.cpu().numpy(), 
                        segment_values.cpu().numpy(), 
                        linewidth=2, 
                        color='C0')  # Use same color for all segments
        
        """# Mark discontinuities
        for pos in self.binary_positions:
            plt.axvline(x=pos, color='r', linestyle='--', alpha=0.5, linewidth=1)"""
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'Quantizer Sup Approximation with K={self.K}, slope={self.slope}', fontsize=14)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def plot_with_exact(self, points=None, dpi=100, show_title=True, save_path=None):
        """Plot sup approximate encoder together with exact encoder for comparison"""
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 5000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
        
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        
        # Create exact encoder with same K
        exact_encoder = EncoderExact1D(K=self.K, dtype=self.dtype, device=self.device)
        
        # Create boundary points including 0 and 1
        boundaries = [0.0] + sorted(self.binary_positions) + [1.0]
        
        # Calculate error metrics over all points
        approx_values_all = self.forward(points)
        exact_values_all = exact_encoder.forward(points)
        mse_error = torch.mean((approx_values_all - exact_values_all)**2).item()
        max_error = torch.max(torch.abs(approx_values_all - exact_values_all)).item()
        
        # Create plot
        plt.figure(figsize=(12, 7), dpi=dpi)
        
        # Plot each continuous segment separately to avoid discontinuities
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            
            # Create points for this segment
            segment_mask = (points >= start) & (points <= end)
            segment_points = points[segment_mask]
            
            if len(segment_points) > 0:
                # Get values for this segment only
                exact_values = exact_encoder.forward(segment_points)
                approx_values = self.forward(segment_points)
                
                # Plot exact encoder (only label on first segment)
                if i == 0:
                    plt.plot(segment_points.cpu().numpy(), exact_values.cpu().numpy(), 
                            'b-', linewidth=2, label='Exact Quantizer', alpha=0.7)
                else:
                    plt.plot(segment_points.cpu().numpy(), exact_values.cpu().numpy(), 
                            'b-', linewidth=2, alpha=0.7)
                
                # Plot approximate encoder (only label on first segment)
                if i == 0:
                    plt.plot(segment_points.cpu().numpy(), approx_values.cpu().numpy(), 
                            'r--', linewidth=2, label='Sup Approximate Quantizer', alpha=0.8)
                else:
                    plt.plot(segment_points.cpu().numpy(), approx_values.cpu().numpy(), 
                            'r--', linewidth=2, alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'Exact vs Sup Approximate Quantizer (K={self.K}, slope={self.slope:.2e})\n'
                     f'MSE: {mse_error:.2e}, Max Error: {max_error:.2e}', fontsize=14)
        plt.legend(fontsize=11)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
        return mse_error, max_error
    
    
class PLCSMMemorizer(nn.Module):
    def __init__(self, base_function, discontinuities=None, flat_slope=1e-4, points=None, values=None, K=2, M=4, dtype=torch.float64, device="cpu"):
        super().__init__()
        assert K >= 1, "K must be greater than 1"
        assert M >= 1, "K must be greater than 1"
        assert flat_slope > 0 and flat_slope < 0.3, f"flat_slope must be positive and smaller than 0.1, but it was {flat_slope}."
        assert callable(base_function), f"base_function must be callable, but {base_function} was given"
        assert discontinuities is None or isinstance(discontinuities, (list, torch.Tensor)), f"discontinuities need to be None, a list, or a torch.Tensor, but {type(discontinuities)} was given."
        
        # Convert discontinuities to tensor if it's a list
        if discontinuities is not None:
            if isinstance(discontinuities, list):
                self.discontinuities = torch.tensor(discontinuities, dtype=dtype, device=device)
            else:
                self.discontinuities = discontinuities.to(dtype=dtype, device=device)
        else:
            self.discontinuities = None
            
        self.flat_slope = flat_slope
        self.base_function = base_function
        self.K = K
        self.M = M
        self.num_levels = 2 ** K
        self.base_slice = 2.0 ** (-K)
        self.binary_positions = [i * self.base_slice for i in range(self.num_levels)]
        self.binary_positions_tensor = torch.tensor(self.binary_positions, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device
        self.values = self.base_function(self.binary_positions_tensor)
        self.discretize_values()
        self.initialize_plcsm_functions()
        self.plcsm_increasing = nn.Sequential(*self.plcsm_increasing_list)
        self.plcsm_decreasing = nn.Sequential(*self.plcsm_decreasing_list)


    def discretize_values(self, values=None):
        """Discretize values using M bits. If values is None, discretizes self.values in place."""
        base_shift = 2 ** self.M
        if values is None:
            scaled_and_rounded_values = torch.floor(base_shift * self.values)
            self.values = scaled_and_rounded_values / base_shift
        else:
            scaled_and_rounded_values = torch.floor(base_shift * values)
            return scaled_and_rounded_values / base_shift


    def initialize_plcsm_functions(self):
        # build two plcsm functions, one for the increasing part and one for the decreasing part
        # both will be constructed increasing, then the decreasing one will be negated, s.t. their sum is the forward value, i.e.
        # f(x) = f_inc(x) + f_dec(x), i.e. f_dec is actually
        self.plcsm_increasing_list = []
        self.plcsm_decreasing_list = []
        prev_y = 0.0
        for i, (x, y) in enumerate(zip(self.binary_positions_tensor, self.values)):
            if y >= prev_y:
                # the current piece is increasing
                if i==0:
                    prev_x = x
                    prev_y = y
                    plcsm_increasing_value = y.clone()
                    plcsm_decreasing_value = 0.0
                    prev_inc_slope = 1.0
                    prev_dec_slope = 1.0
                else:
                    inc_base_slope = (y - prev_y) / (x - prev_x) + self.flat_slope
                    inc_slope = inc_base_slope / prev_inc_slope
                    dec_slope = self.flat_slope / prev_dec_slope
                    c_inc = plcsm_increasing_value if i > 1 else prev_x 
                    d_inc = plcsm_increasing_value 
                    c_dec = plcsm_decreasing_value if i > 1 else prev_x 
                    d_dec = plcsm_decreasing_value 
                    self.plcsm_increasing_list.append(Sigma_ab_cd(1,inc_slope,c_inc,d_inc))
                    self.plcsm_decreasing_list.append(Sigma_ab_cd(1,dec_slope,c_dec,d_dec))
                    plcsm_increasing_value += (x - prev_x) * inc_base_slope
                    plcsm_decreasing_value += (x - prev_x) * self.flat_slope
                    prev_x = x
                    prev_y = y
                    prev_inc_slope = inc_base_slope
                    prev_dec_slope = self.flat_slope

            else:
                # the current piece is decreasing
                if i==0:
                    raise ValueError(f"For i==0 found negative value {y} of base function. This isn't allowed, the base functions must take values in [0,1].")
                else:
                    dec_base_slope = (prev_y - y) / (x - prev_x) + self.flat_slope
                    dec_slope = dec_base_slope / prev_dec_slope
                    inc_slope = self.flat_slope / prev_inc_slope
                    c_inc = plcsm_increasing_value if i > 1 else prev_x  
                    d_inc = plcsm_increasing_value 
                    c_dec = plcsm_decreasing_value if i > 1 else prev_x  
                    d_dec = plcsm_decreasing_value 
                    self.plcsm_increasing_list.append(Sigma_ab_cd(1,inc_slope,c_inc,d_inc))
                    self.plcsm_decreasing_list.append(Sigma_ab_cd(1,dec_slope,c_dec,d_dec))
                    plcsm_increasing_value += (x - prev_x) * self.flat_slope
                    plcsm_decreasing_value += (x - prev_x) * dec_base_slope
                    prev_x = x
                    prev_y = y
                    prev_inc_slope = self.flat_slope
                    prev_dec_slope = dec_base_slope

        class Negate(nn.Module):
            def forward(self,x):
                return -x
        self.plcsm_decreasing_list.append(Negate()) 


    def forward(self, x):
        #return self.plcsm_increasing(x)
        return self.plcsm_increasing(x) + self.plcsm_decreasing(x)
    
    def plot(self, points=None, dpi=100, show_title=True, save_path=None):
        """Plot the PLCSM memorizer output on [0,1].

        The memorizer may have internal discontinuities at the binary slice positions
        (self.binary_positions). We plot the function on each continuous subinterval
        separately to make any jumps visible.
        """
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 50000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)

        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"

        # Create boundary points including 0 and 1
        boundaries = [0.0] + sorted(self.binary_positions) + [1.0]

        # Evaluate memorizer on the dense grid
        mem_values = self.forward(points)

        plt.figure(figsize=(12, 7), dpi=dpi)

        # Plot each continuous segment separately so jumps are not connected
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            mask = (points >= start) & (points <= end)
            seg_pts = points[mask]
            if len(seg_pts) == 0:
                continue
            seg_vals = self.forward(seg_pts)
            # only label first segment
            if i == 0:
                plt.plot(seg_pts.cpu().numpy(), seg_vals.cpu().numpy(), 'r-', linewidth=2, label='PLCSM Memorizer')
            else:
                plt.plot(seg_pts.cpu().numpy(), seg_vals.cpu().numpy(), 'r-', linewidth=2)

        # Optionally mark the binary slice positions
        for pos in self.binary_positions:
            plt.axvline(x=pos, color='k', linestyle='--', alpha=0.25, linewidth=0.8)

        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'PLCSM Memorizer (K={self.K}, M={self.M})', fontsize=14)
        plt.legend(fontsize=11)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_with_base_function(self, points=None, dpi=100, show_title=True, save_path=None):
        """Plot the memorizer together with the base function.

        If `self.discontinuities` is provided (list or torch.Tensor with values in [0,1])
        the base function will be plotted in segments that start/end at those
        discontinuity locations so the jumps are visible. If no discontinuities are
        provided, the base function is plotted on the full interval.
        """
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 50000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)

        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"

        # Evaluate memorizer values on dense grid
        mem_values = self.forward(points)

        # Prepare discontinuities for base function plotting
        discs = self.discontinuities  # Already a tensor or None from __init__
        
        if discs is not None:
            # sanitize
            discs = torch.unique(torch.sort(discs)[0])
            discs = discs[(discs >= 0.0) & (discs <= 1.0)]

        plt.figure(figsize=(16, 10), dpi=dpi)

        # Plot base function: either split at discontinuities or plot once
        if discs is not None and len(discs) > 0:
            boundaries = [0.0] + discs.cpu().tolist() + [1.0]
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                mask = (points >= start) & (points <= end)
                seg_pts = points[mask]
                if len(seg_pts) == 0:
                    continue
                base_vals = self.base_function(seg_pts)
                # Discretize the base function values
                base_vals = self.discretize_values(base_vals)
                if i == 0:
                    plt.plot(seg_pts.cpu().numpy(), base_vals.cpu().numpy(), 'b-', linewidth=2, label='Base function (discretized)')
                else:
                    plt.plot(seg_pts.cpu().numpy(), base_vals.cpu().numpy(), 'b-', linewidth=2)

            # mark discontinuities
            for d in discs:
                plt.axvline(x=d.item(), color='orange', linestyle='--', linewidth=1.5, alpha=0.8,
                           label='Discontinuity' if d == discs[0] else '')
        else:
            base_vals = self.base_function(points)
            # Discretize the base function values
            base_vals = self.discretize_values(base_vals)
            plt.plot(points.cpu().numpy(), base_vals.cpu().numpy(), 'b-', linewidth=2, label='Base function (discretized)')

        # Plot memorizer output (continuous segments split at binary positions so jumps are not connected)
        boundaries_mem = [0.0] + sorted(self.binary_positions) + [1.0]
        for i in range(len(boundaries_mem) - 1):
            start, end = boundaries_mem[i], boundaries_mem[i + 1]
            mask = (points >= start) & (points <= end)
            seg_pts = points[mask]
            if len(seg_pts) == 0:
                continue
            seg_vals = self.forward(seg_pts)
            if i == 0:
                plt.plot(seg_pts.cpu().numpy(), seg_vals.cpu().numpy(), 'r--', linewidth=2, label='PLCSM Memorizer')
            else:
                plt.plot(seg_pts.cpu().numpy(), seg_vals.cpu().numpy(), 'r--', linewidth=2)

        # Mark the sparse fitting points used to construct the memorizer
        sparse_vals = self.forward(self.binary_positions_tensor)
        plt.plot(self.binary_positions_tensor.cpu().numpy(), sparse_vals.cpu().numpy(), 'o', color='gold', markersize=7, label=f'Fitting points (n={len(self.binary_positions_tensor)})')

        # Compute error metrics over full grid (using discretized base function)
        base_all = self.discretize_values(self.base_function(points))
        mse_error = torch.mean((mem_values - base_all) ** 2).item()
        max_error = torch.max(torch.abs(mem_values - base_all)).item()

        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        title = f'PLCSM Memorizer vs Base Function (K={self.K}, M={self.M})\nMSE: {mse_error:.2e}, Max Error: {max_error:.2e}'
        if discs is not None and len(discs) > 0:
            title += f' | Discontinuities: {len(discs)}'
        if show_title:
            plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def plot_for_parameter_list(self, param_list, points=None, dpi=100):
        """
        Plot PLCSM memorizer outputs for multiple parameter combinations.
        
        Args:
            param_list: List of tuples (K, M, flat_slope) specifying parameter combinations
            points: Optional points for plotting
            dpi: DPI for the plots
        """
        for K, M, flat_slope in param_list:
            # Create new memorizer with these parameters
            memorizer = PLCSMMemorizer(
                self.base_function,
                discontinuities=self.discontinuities,
                flat_slope=flat_slope,
                K=K,
                M=M,
                dtype=self.dtype,
                device=self.device
            )
            memorizer.plot(points=points, dpi=dpi)
    
    def plot_with_base_function_for_parameter_list(self, param_list, points=None, dpi=100):
        """
        Plot PLCSM memorizer outputs compared to base function for multiple parameter combinations.
        
        Args:
            param_list: List of tuples (K, M, flat_slope) specifying parameter combinations
            points: Optional points for plotting
            dpi: DPI for the plots
        """
        for K, M, flat_slope in param_list:
            # Create new memorizer with these parameters
            memorizer = PLCSMMemorizer(
                self.base_function,
                discontinuities=self.discontinuities,
                flat_slope=flat_slope,
                K=K,
                M=M,
                dtype=self.dtype,
                device=self.device
            )
            memorizer.plot_with_base_function(points=points, dpi=dpi)

# given a set of points [x_0,...,x_{n-1}] in [0,1] and values [y_0,...,y_{n-1}]=[f(x_0),...,f(x_{n-1})] in [0,1] for some function f,
# constructs a piecewise linear zig zag function h such that h(x_i) = y_i for all i
class ZigZagMemorizer:
    def __init__(self, base_function, points=None, values=None, K=2, M=4, discontinuities=None, dtype=torch.float64, device="cpu"):
        assert K >= 1 and isinstance(K, int), "K must be a positive integer"
        assert M >= 1 and isinstance(M, int), "M must be a positive integer"
        assert callable(base_function), "base_function must be a callable function"
        
        if points is None:
            points = torch.arange(0.0, 1.0, 2 ** (-K), dtype=dtype, device=device)
        if values is None:
            values = base_function(points)
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        assert isinstance(values, torch.Tensor), "values must be a torch.Tensor"
        assert len(points) == len(values), "points and values must have the same length"
        assert points.dtype == dtype, "points must have the same dtype as specified"
        assert points.device.type == device, "points must be on the specified device"
        assert values.dtype == dtype, "values must have the same dtype as specified"
        assert values.device.type == device, "values must be on the specified device"
        # we need that points are ordered increasingly
        is_sorted_increasingly = torch.all(points[:-1] < points[1:])
        assert is_sorted_increasingly, "points must be sorted increasingly and not contain duplicates"
        
        # Handle discontinuities - convert list to tensor if needed
        if discontinuities is not None:
            if isinstance(discontinuities, list):
                discontinuities = torch.tensor(discontinuities, dtype=dtype, device=device)
            elif isinstance(discontinuities, torch.Tensor):
                discontinuities = discontinuities.to(dtype=dtype, device=device)
            else:
                raise TypeError(f"discontinuities must be None, a list, or a torch.Tensor, got {type(discontinuities)}")
            
            # Sort discontinuities and ensure they are unique
            discontinuities = torch.unique(torch.sort(discontinuities)[0])
            # Ensure discontinuities are within [0, 1]
            assert torch.all((discontinuities >= 0) & (discontinuities <= 1)), "discontinuities must be in [0, 1]"
        
        self.dtype = dtype
        self.device = device
        self.K = K
        self.M = M
        self.base_function = base_function
        self.discontinuities = discontinuities
        self.validation_tol = 1e-5
        
        # calibrate all internals of the class according to the points
        self.calibrate_class_for_new_points(points)
        
            
    def calibrate_class_for_new_points(self, points):
        self.num_points = len(points)
        self.points = points
        self.values = self.base_function(points)
        
        self.discretize_values()
        
        self.a_list = [1.5]
        self.b_list = []
        
        # set base points for the zig zag function
        self.set_zig_zag_base_points()
        # define the piecewise linear function using the base points
        self.sigmas = []
        abs_prev_slope = 1.0
        for i in range(len(self.b_list)):
            if i == 0:
                slope = 1.0 /  (self.b_list[i] - self.a_list[i])
                abs_prev_slope = abs(slope)
                sigma_fold_up = Sigma_ab_cd(a=slope, b=slope, c=self.a_list[i], d=0.0)

            else:
                slope = -1.0 / (self.b_list[i] - self.a_list[i])
                scaled_slope = slope / abs_prev_slope
                abs_prev_slope = abs(slope)
                sigma_fold_up = Sigma_ab_cd(a=scaled_slope, b=1.0, c=0.0, d=0.0)
                
            self.sigmas.append(sigma_fold_up)
            slope = -1.0 / (self.a_list[i + 1] - self.b_list[i])
            scaled_slope = slope / abs_prev_slope
            abs_prev_slope = abs(slope)
            sigma_fold_down = Sigma_ab_cd(a=1.0, b=scaled_slope, c=1.0, d=1.0)
            self.sigmas.append(sigma_fold_down)
        
    def discretize_values(self, values=None):
        """Discretize values using M bits. If values is None, discretizes self.values in place."""
        base_shift = 2 ** self.M
        if values is None:
            scaled_and_rounded_values = torch.floor(base_shift * self.values)
            self.values = scaled_and_rounded_values / base_shift
        else:
            scaled_and_rounded_values = torch.floor(base_shift * values)
            return scaled_and_rounded_values / base_shift
        

    def set_zig_zag_base_points(self):
        for i in range(self.num_points-1, -1, -1):
            if abs(self.values[i].item()) > 1e-8:
                a_slope_min = (self.points[i].item() - self.values[i].item() * self.a_list[0]) / (1.0 - self.values[i].item())
                a_min = max(self.points[i-1].item(), a_slope_min) if i >= 1 else a_slope_min
                a = a_min + (self.points[i].item() - a_min) / 8.0
                b = a + (self.points[i].item() - a) / self.values[i].item()
            else:
                a = self.points[i].item()
                b = a + (self.a_list[0] - a) / 2.0
                
            assert b > a, f"b must be greater than a, but got a={a}, b={b}. However, at iteration i={i}, point={self.points[i].item()}, value={self.values[i].item()}"
            self.a_list.insert(0, a)
            self.b_list.insert(0, b)
        #a_prev = a_prev_min + (self.points[0] - a_prev_min) / 8.0
        #self.a_list.insert(0, a_prev)
        return self.a_list, self.b_list
    
    def set_new_points(self, points):
        self.calibrate_class_for_new_points(points)
    
    def forward(self, x):
        # find the interval that x belongs to
        for i in range(len(self.sigmas)):
            x = self.sigmas[i](x)
        return x
    
    def check_close_to_base_function(self, tol=None):
        return torch.allclose(self.forward(self.points), self.base_function(self.points), atol=self.validation_tol if tol is None else tol)
    
    def validate_close_to_base_function(self, tol=None):
        assert self.check_close_to_base_function(tol), f"ZigZag function is not close to base function within tolerance {self.validation_tol if tol is None else tol}"
    
    def plot(self, plot_points=None, dpi=100, show_title=True, save_path=None):
        """
        Plot the ZigZag memorizer output only (without base function comparison).
        """
        # Use dense points for detailed plotting if not specified
        if plot_points is None:
            if self.K < 4:
                plot_points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                plot_points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)
            else:
                plot_points = torch.linspace(0, 1, 500000, dtype=self.dtype, device=self.device)
        
        # Calculate values for the ZigZag approximation
        forward_values = self.forward(plot_points)
        
        # Create the plot
        plt.figure(figsize=(10, 6), dpi=dpi)
        
        # Plot ZigZag memorizer output
        plt.plot(plot_points.cpu().numpy(), forward_values.cpu().numpy(),
                'r-', linewidth=2, label='ZigZag Memorizer', alpha=0.8)
        
        # Mark the fitting points
        sparse_values = self.forward(self.points)
        plt.plot(self.points.cpu().numpy(), sparse_values.cpu().numpy(), 
                'o', color='yellow', markersize=8, label=f'Fitting points (n={len(self.points)})', alpha=1.0)
        
        # Formatting
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'ZigZag Memorizer (K={self.K}, M={self.M})', fontsize=14)
        plt.legend(fontsize=11)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
    def plot_with_base_function(self, plot_points=None, dpi=100, show_title=True, save_path=None):
        """
        Plot the ZigZag memorizer output compared to the base function.
        Handles discontinuous base functions by splitting the plot at discontinuity locations.
        """
        # Use dense points for detailed plotting if not specified
        if plot_points is None:
            if self.K < 4:
                plot_points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                plot_points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)
            else:
                plot_points = torch.linspace(0, 1, 500000, dtype=self.dtype, device=self.device)
            
        # Calculate values for the ZigZag approximation on the full dense grid
        dense_forward_values = self.forward(plot_points)

        # Create the plot
        plt.figure(figsize=(16, 10), dpi=dpi)
        
        # Plot base function with discontinuities handling
        if self.discontinuities is not None and len(self.discontinuities) > 0:
            # Create boundaries: start with 0, add all discontinuities, end with 1
            boundaries = [0.0] + self.discontinuities.cpu().tolist() + [1.0]
            
            # Plot each continuous segment separately
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                
                # Create a mask for points in this segment
                segment_mask = (plot_points >= start) & (plot_points <= end)
                segment_points = plot_points[segment_mask]
                
                if len(segment_points) > 0:
                    base_function_values = self.base_function(segment_points)
                    # Discretize the base function values
                    base_function_values = self.discretize_values(base_function_values)
                    
                    # Only add label on the first segment
                    if i == 0:
                        plt.plot(segment_points.cpu().numpy(), base_function_values.cpu().numpy(),
                                'b-', linewidth=2, label='Original function (discretized)', alpha=0.8)
                    else:
                        plt.plot(segment_points.cpu().numpy(), base_function_values.cpu().numpy(),
                                'b-', linewidth=2, alpha=0.8)
            
            # Mark discontinuities with vertical dashed lines
            for disc in self.discontinuities:
                plt.axvline(x=disc.item(), color='orange', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label='Discontinuity' if disc == self.discontinuities[0] else '')
        else:
            # No discontinuities - plot as before
            base_function_values = self.base_function(plot_points)
            # Discretize the base function values
            base_function_values = self.discretize_values(base_function_values)
            plt.plot(plot_points.cpu().numpy(), base_function_values.cpu().numpy(),
                    'b-', linewidth=2, label='Original function (discretized)', alpha=0.8)

        # Plot ZigZag memorizer (always continuous, no need to split)
        plt.plot(plot_points.cpu().numpy(), dense_forward_values.cpu().numpy(),
                'r--', linewidth=2, label='ZigZag Memorizer', alpha=0.8)
        
        # Mark the fitting points (sparse points used for training)
        sparse_values = self.forward(self.points)
        plt.plot(self.points.cpu().numpy(), sparse_values.cpu().numpy(), 
                'o', color='yellow', markersize=8, label=f'Fitting points (n={len(self.points)})', alpha=1.0)
        
        # Calculate error metrics over all points (using discretized base function)
        base_function_values_all = self.discretize_values(self.base_function(plot_points))
        mse_error = torch.mean((dense_forward_values - base_function_values_all)**2).item()
        max_error = torch.max(torch.abs(dense_forward_values - base_function_values_all)).item()
        
        # Formatting
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        
        title = f'ZigZag Function Approximation (K={self.K}, M={self.M})\n' \
                f'MSE: {mse_error:.2e}, Max Error: {max_error:.2e}'
        if self.discontinuities is not None and len(self.discontinuities) > 0:
            title += f' | Discontinuities: {len(self.discontinuities)}'
        
        if show_title:
            plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def plot_for_parameter_list(self, param_list, plot_points=None, dpi=100):
        """
        Plot ZigZag memorizer outputs for multiple parameter combinations.
        
        Args:
            param_list: List of tuples (K, M) specifying parameter combinations
            plot_points: Optional points for plotting
            dpi: DPI for the plots
        """
        for K, M in param_list:
            # Create new memorizer with these parameters
            memorizer = ZigZagMemorizer(
                self.base_function,
                K=K,
                M=M,
                discontinuities=self.discontinuities,
                dtype=self.dtype,
                device=self.device
            )
            memorizer.plot(plot_points=plot_points, dpi=dpi)
    
    def plot_with_base_function_for_parameter_list(self, param_list, plot_points=None, dpi=100):
        """
        Plot ZigZag memorizer outputs compared to base function for multiple parameter combinations.
        
        Args:
            param_list: List of tuples (K, M) specifying parameter combinations
            plot_points: Optional points for plotting
            dpi: DPI for the plots
        """
        for K, M in param_list:
            # Create new memorizer with these parameters
            memorizer = ZigZagMemorizer(
                self.base_function,
                K=K,
                M=M,
                discontinuities=self.discontinuities,
                dtype=self.dtype,
                device=self.device
            )
            memorizer.plot_with_base_function(plot_points=plot_points, dpi=dpi)


class ZigZagFunctionSup(ZigZagMemorizer):
    def __init__(self, base_function, points=None, values=None, K=2, M=4, dtype=torch.float64, device="cpu"):
        assert K >= 1 and isinstance(K, int), "K must be a positive integer"
        assert M >= 1 and isinstance(M, int), "M must be a positive integer"
        assert callable(base_function), "base_function must be a callable function"
        self.end_of_interval = 1.0 + 2 ** (-K-1)
        if points is None:
            points = torch.arange(0.0, self.end_of_interval, 2 ** (-K), dtype=dtype, device=device)
        if values is None:
            values = base_function(points)
        assert isinstance(points, torch.Tensor), "points must be a torch.Tensor"
        assert isinstance(values, torch.Tensor), "values must be a torch.Tensor"
        assert len(points) == len(values), "points and values must have the same length"
        assert points.dtype == dtype, "points must have the same dtype as specified"
        assert points.device.type == device, "points must be on the specified device"
        assert values.dtype == dtype, "values must have the same dtype as specified"
        assert values.device.type == device, "values must be on the specified device"
        # we need that points are ordered increasingly
        is_sorted_increasingly = torch.all(points[:-1] < points[1:])
        assert is_sorted_increasingly, "points must be sorted increasingly and not contain duplicates"
        
        self.dtype = dtype
        self.device = device
        self.K = K
        self.M = M
        self.base_function = base_function
        self.validation_tol = 1e-5

class CodingScheme1D:
    def __init__(self, function, discontinuities=None, K=2, M=2, gamma=0.1, flat_slope=1e-2, 
                 memorizer_type="zig-zag", scheme_type="lp", dtype=torch.float64, device="cpu"):
        """
        Initialize a 1D Coding Scheme.
        
        Args:
            function: The base function to approximate
            discontinuities: Optional list or tensor of discontinuity points in [0,1]
            K: Encoder parameter (number of bits)
            M: Memorizer parameter (number of levels)
            beta: Beta parameter for lp encoder
            flat_slope: Flat slope parameter for encoders and PLCSM memorizer
            memorizer_type: Type of memorizer - "zig-zag" or "plcsm"
            scheme_type: Type of approximation scheme - "lp" or "sup"
            dtype: torch data type
            device: torch device
        """
        self.function = function
        
        # Validate memorizer_type and scheme_type
        assert memorizer_type in ["zig-zag", "plcsm"], f"memorizer_type must be 'zig-zag' or 'plcsm', got {memorizer_type}"
        assert scheme_type in ["lp", "sup"], f"scheme_type must be 'lp' or 'sup', got {scheme_type}"
        
        self.memorizer_type = memorizer_type
        self.scheme_type = scheme_type
        
        # Convert discontinuities to tensor if it's a list
        if discontinuities is not None:
            if isinstance(discontinuities, list):
                self.discontinuities = torch.tensor(discontinuities, dtype=dtype, device=device)
            elif isinstance(discontinuities, torch.Tensor):
                self.discontinuities = discontinuities.to(dtype=dtype, device=device)
            else:
                raise TypeError(f"discontinuities must be None, a list, or a torch.Tensor, got {type(discontinuities)}")
        else:
            self.discontinuities = None
            
        self.dtype = dtype
        self.device = device
        self.setup_classes_for_parameters(K, M, gamma, flat_slope)

    def setup_classes_for_parameters(self, K, M, gamma, flat_slope):
        assert K >= 1, "K must be greater than 1"
        assert M >= 1, "M must be greater than 1"
        assert gamma > 0, "gamma must be positive"
        assert flat_slope > 0, "flat_slope must be positive"
        self.K = K
        self.M = M
        self.gamma = gamma
        self.flat_slope = flat_slope
        
        # Setup encoder based on scheme_type
        if self.scheme_type == "sup":
            self.encoder = EncoderSupApprox1D(K=K, slope=flat_slope, dtype=self.dtype, device=self.device)
        else:  # scheme_type == "lp"
            self.encoder = EncoderApprox1D(K=K, gamma=gamma, flat_slope=flat_slope, dtype=self.dtype, device=self.device)
        
        # Create sample points for memorizer setup
        sample_points = torch.linspace(0, 1, 2**K + 1, dtype=self.dtype, device=self.device)
        
        # Setup memorizer based on memorizer_type
        if self.memorizer_type == "zig-zag":
            self.memorizer = ZigZagMemorizer(self.function, sample_points, None, K=K, M=M, 
                                            discontinuities=self.discontinuities, dtype=self.dtype, device=self.device)
        else:  # memorizer_type == "plcsm"
            self.memorizer = PLCSMMemorizer(self.function, discontinuities=self.discontinuities, K=K, M=M, 
                                           flat_slope=flat_slope, points=sample_points, dtype=self.dtype, device=self.device)


    def forward(self, x):
        encoded_x = self.encoder(x)
        zig_zag_y = self.memorizer.forward(encoded_x)
        return zig_zag_y
    
    def approximate_Lp_and_max_error(self, points=None, p=2):
        if points is None:
            points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
        zig_zag_values = self.forward(points)
        base_function_values = self.function(points)
        error = torch.abs(zig_zag_values - base_function_values)
        max_error = torch.max(error).item()
        lp_error = torch.mean(error ** p) ** (1 / p)
        return lp_error.item(), max_error
    

    def print_errors(self, function_repr_str, points=None, p=2):
        lp_error, max_error = self.approximate_Lp_and_max_error(points, p)
        print(f"For {function_repr_str} with parameters K={self.K}, M={self.M}, gamma={self.gamma:.2e}, flat_slope={self.flat_slope:.2e}:\n - L{p} error: {lp_error:.2e}, Max error: {max_error:.2e}")


    def plot(self, points=None, dpi=100, show_title=True, save_path=None):
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 1000000, dtype=self.dtype, device=self.device)
        zig_zag_values = self.forward(points)
        # plot the final output
        plt.figure(figsize=(10, 6), dpi=dpi)
        plt.plot(points.cpu().numpy(), zig_zag_values.cpu().numpy(), linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        if show_title:
            plt.title(f'Coding Scheme Output (K={self.K}, M={self.M}, gamma={self.gamma:.2e}, flat_slope={self.flat_slope:.2e})', fontsize=14)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
    def plot_with_base_function(self, points=None, dpi=100, show_title=True, save_path=None):
        if points is None:
            if self.K < 4:
                points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
            elif self.K <= 6:
                points = torch.linspace(0, 1, 50000, dtype=self.dtype, device=self.device)
            else:
                points = torch.linspace(0, 1, 100000, dtype=self.dtype, device=self.device)
            
        zig_zag_values = self.forward(points)
        base_function_values = self.function(points)
        
        # Calculate error metrics
        mse_error = torch.mean((zig_zag_values - base_function_values)**2).item()
        max_error = torch.max(torch.abs(zig_zag_values - base_function_values)).item()
        
        # Create the plot
        plt.figure(figsize=(16, 10), dpi=dpi)
        
        # Plot base function with discontinuities handling
        if self.discontinuities is not None and len(self.discontinuities) > 0:
            # Create boundaries: start with 0, add all discontinuities, end with 1
            boundaries = [0.0] + self.discontinuities.cpu().tolist() + [1.0]
            
            # Plot each continuous segment separately
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                
                # Create a mask for points in this segment
                segment_mask = (points >= start) & (points <= end)
                segment_points = points[segment_mask]
                
                if len(segment_points) > 0:
                    segment_base_values = self.function(segment_points)
                    
                    # Only add label on the first segment
                    if i == 0:
                        plt.plot(segment_points.cpu().numpy(), segment_base_values.cpu().numpy(),
                                'b-', linewidth=2, label='Base Function', alpha=0.8)
                    else:
                        plt.plot(segment_points.cpu().numpy(), segment_base_values.cpu().numpy(),
                                'b-', linewidth=2, alpha=0.8)
            
            """# Mark discontinuities with vertical dashed lines
            for disc in self.discontinuities:
                plt.axvline(x=disc.item(), color='orange', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label='Discontinuity' if disc == self.discontinuities[0] else '')"""
        else:
            # No discontinuities - plot as usual
            plt.plot(points.cpu().numpy(), base_function_values.cpu().numpy(), 
                    'b-', linewidth=2, label='Base Function', alpha=0.8)
        
        # Plot coding scheme output (always continuous)
        plt.plot(points.cpu().numpy(), zig_zag_values.cpu().numpy(), 
                'r--', linewidth=2, label='Coding Scheme Output', alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        
        title = f'Coding Scheme vs Base Function (K={self.K}, M={self.M}, gamma={self.gamma:.2e}, flat_slope={self.flat_slope:.2e})\n'
        title += f'MSE: {mse_error:.2e}, Max Error: {max_error:.2e}'
        if self.discontinuities is not None and len(self.discontinuities) > 0:
            title += f' | Discontinuities: {len(self.discontinuities)}'
        if show_title:
            plt.title(title, fontsize=14)
        
        plt.axhline(y=0, color='k', linewidth=0.2)
        plt.axvline(x=0, color='k', linewidth=0.2)
        plt.legend()
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def evaluate_for_list_of_parameters(self, parameter_list, function_repr_str, points=None, plot=False, p=2):
        results = []
        for params in parameter_list:
            K, M, gamma, flat_slope = params
            try:
                self.setup_classes_for_parameters(K, M, gamma, flat_slope)
                if plot:
                    self.plot_with_base_function(points)
                lp_error, max_error = self.approximate_Lp_and_max_error(points, p)
                results.append((K, M, gamma, flat_slope, lp_error, max_error))
                self.print_errors(function_repr_str, p=p)
            except Exception as e:
                print(f"Error occurred for parameters {params}: {e}\nSkipping this set of parameters.")
                results.append(None)
        return results

    def plot_Lp_and_max_norms_for_gammas(self, gammas, p=2, points=None, dpi=100, show_title=True, save_path=None):
        assert isinstance(gammas, torch.Tensor), f"gammas must be a torch.Tensor, but was of type {type(gammas)}"
        
        if points is None:
            points = torch.linspace(0, 1, 10000, dtype=self.dtype, device=self.device)
        
        # Store original parameters to restore later
        original_gamma = self.gamma
        original_dtype = self.dtype
        
        # Data storage for plotting
        gammas_list = gammas.cpu().numpy()
        lp_errors_float32 = []
        max_errors_float32 = []
        lp_errors_float64 = []
        max_errors_float64 = []
        
        # Test with torch.float32
        for gamma in gammas:
            gamma_val = gamma.item()
            try:
                # Setup with float32
                self.dtype = torch.float32
                self.setup_classes_for_parameters(self.K, self.M, gamma_val, self.flat_slope)
                
                # Convert points to float32
                points_float32 = points.to(torch.float32)
                lp_error, max_error = self.approximate_Lp_and_max_error(points_float32, p)
                lp_errors_float32.append(lp_error)
                max_errors_float32.append(max_error)
            except Exception as e:
                print(f"Error with gamma={gamma_val} and float32: {e}")
                lp_errors_float32.append(float('nan'))
                max_errors_float32.append(float('nan'))
        
        # Test with torch.float64
        for gamma in gammas:
            gamma_val = gamma.item()
            try:
                # Setup with float64
                self.dtype = torch.float64
                self.setup_classes_for_parameters(self.K, self.M, gamma_val, self.flat_slope)
                
                # Convert points to float64
                points_float64 = points.to(torch.float64)
                lp_error, max_error = self.approximate_Lp_and_max_error(points_float64, p)
                lp_errors_float64.append(lp_error)
                max_errors_float64.append(max_error)
            except Exception as e:
                print(f"Error with gamma={gamma_val} and float64: {e}")
                lp_errors_float64.append(float('nan'))
                max_errors_float64.append(float('nan'))
        
        # Create the plot with gammas on x-axis and errors on y-axis
        plt.figure(figsize=(14, 8), dpi=dpi)
        
        # Plot L_p errors for both dtypes
        plt.plot(gammas_list, lp_errors_float32, 'b-o', linewidth=2, markersize=6, 
                label=f'L{p} Error (torch.float32)', alpha=0.8)
        plt.plot(gammas_list, lp_errors_float64, 'b--s', linewidth=2, markersize=6, 
                label=f'L{p} Error (torch.float64)', alpha=0.8)
        
        # Plot Max errors for both dtypes  
        plt.plot(gammas_list, max_errors_float32, 'r-o', linewidth=2, markersize=6, 
                label='Max Error (torch.float32)', alpha=0.8)
        plt.plot(gammas_list, max_errors_float64, 'r--s', linewidth=2, markersize=6, 
                label='Max Error (torch.float64)', alpha=0.8)
        
        plt.xlabel('Gamma', fontsize=12)
        plt.ylabel('Error', fontsize=12)
        if show_title:
            plt.title(f'L{p} and Max Errors vs Gamma (K={self.K}, M={self.M}, flat_slope={self.flat_slope:.2e})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Use log scale for better visualization of errors
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
        # Restore original parameters
        self.dtype = original_dtype
        self.setup_classes_for_parameters(self.K, self.M, original_gamma, self.flat_slope)
        
        # Return the data for further analysis if needed
        return {
            'gammas': gammas_list,
            'lp_errors_float32': lp_errors_float32,
            'max_errors_float32': max_errors_float32,
            'lp_errors_float64': lp_errors_float64,
            'max_errors_float64': max_errors_float64
        }

    def plot_lp_and_max_norms_for_list_of_parameters(self, p=2):
        pass
