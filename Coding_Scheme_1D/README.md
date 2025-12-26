# 1D Coding Scheme for Neural Network Universal Approximation

This repository contains implementations and visualizations of 1D coding schemes for approximating functions using neural networks with minimal width, as presented in the paper "New Advances in Universal Approximation with Neural Networks of Minimal Width".

## Overview

The coding scheme provides a constructive approach to universal approximation using:
- **Encoder (Quantizer)**: Discretizes input values
- **Memorizer**: Fits target functions on discrete points
- **Decoder**: Identity function in 1D

### Network Width - A Critical Distinction

**All constructions are built purely from $\sigma_{a,b}^{c,d}$ functions - simple width-1 LReLU compositions!**

- **Zig-Zag Memorizer = Width-1 FNN**: Complete coding scheme is width-1
- **PLCSM Memorizer = Width-2 FNN**: Complete coding scheme is width-2

⚠️ **Important**: While PLCSM approximations often look visually better, they use **double the network width**. This is a comparatively very unfair advantage - achieving good approximations with width-1 (zig-zag) is **terrifically harder**. When comparing results, remember that the zig-zag width-1 construction demonstrates true minimal-width universal approximation.

## Files

### Notebooks

**📖 Recommended starting point**: Begin with **`coding_scheme_1D.ipynb`** as it includes more detailed explanations of the $\sigma_{a,b}^{c,d}$ functions and introduces foundational concepts. However, both notebooks are equally important:

- **`coding_scheme_1D.ipynb`**: $L^p$ norm approximation with continuous activations (LReLUs and G-ReLUs). Provides detailed introduction to sigma functions and basic concepts.
- **`coding_scheme_1D_sup.ipynb`**: Supremum norm approximation using discontinuous activations (SG-LReLUs and S-ReLUs). Equally important (or more so) for understanding the full scope of the approach, but assumes familiarity with basics from the first notebook.

### Python Modules

- **`coding_scheme_base_functions.py`**: Core implementation containing:
  - `Sigma_ab_cd`: Shifted and rescaled LReLU variants
  - `EncoderApprox1D`: $L^p$ encoder approximation
  - `EncoderSupApprox1D`: Supremum encoder with discontinuous activations
  - `ZigZagMemorizer`: Piecewise linear memorizer using zig-zag construction
  - `PLCSMMemorizer`: Piecewise Linear Continuous Slope Matching memorizer
  - `CodingScheme1D`: Complete coding scheme combining encoder and memorizer

## Requirements

- Python 3.7+
- PyTorch
- matplotlib
- CUDA-capable GPU (optional, but recommended)

Install dependencies:
```bash
pip install torch matplotlib
```

## Usage

### Quick Start

1. Open either notebook in Jupyter or VS Code
2. Set device and dtype preferences:
   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   dtype = torch.float64  # Recommended for precision
   ```
3. Select a base function to approximate:
   ```python
   base_function = squared_shifted
   base_function_name = "Squared Shifted Function"
   ```
4. Run cells sequentially to see:
   - Sigma function constructions
   - Encoder approximations
   - Memorizer demonstrations
   - Complete coding scheme evaluations

### Parameter Control

#### Convergence Parameters

**Encoder Approximation:**
- **$\gamma \to 0$** (L^p only) and **flat_slope $\to 0$**: Control encoder accuracy
- **slope $\to 0$** (Supremum only): Controls encoder accuracy
- ⚠️ Values below $10^{-6}$ may cause numerical instabilities

**Coding Scheme Accuracy:**
- **$K \to \infty$**: Number of discretization levels (exponential impact!)
- **$M \to \infty$**: Precision of memorizer function values
- ⚠️ **Warning**: $K > 10$ leads to exponential runtime increase ($2^K$ function compositions)
- ⚠️ Large $K$ and $M$ can cause numerical instabilities

#### Recommended Parameter Ranges

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| $K$ | 2-7 | Values >10 cause significant slowdown |
| $M$ | 2-6 | Higher values increase precision |
| $\gamma$ | 0.1-$10^{-5}$ | L^p approximation only |
| flat_slope | $10^{-1}$-$10^{-5}$ | Both approximation types |
| slope | $10^{-1}$-$10^{-8}$ | Supremum approximation only |

### Custom Functions

Define your own test functions:

```python
def my_custom_function(x):
    return torch.sin(4 * torch.pi * x) * 0.3 + 0.5

# Use in approximation
base_function = my_custom_function
base_function_name = "My Custom Function"

# Optionally plot comparison
plot_test_functions([
    (squared_shifted, "Squared shifted"),
    (smooth_gaussian_mixture, "Smooth Gaussian mixture"),
    (my_custom_function, "My custom function")
])
```

## Key Concepts

### L^p vs Supremum Approximation

**L^p Approximation** (`coding_scheme_1D.ipynb`):
- Uses continuous LReLUs and G-ReLUs
- Requires $\gamma$ parameter for transition intervals
- Cannot achieve zero error on entire domain with finite $K$
- Suitable for functions with discontinuities

**Supremum Approximation** (`coding_scheme_1D_sup.ipynb`):
- Uses discontinuous activations (steps at zero)
- No $\gamma$ parameter needed
- Achieves exact fitting on binary numbers $\mathcal{C}_K$
- Best for continuous functions

### 1D Simplifications

- **Encoder = Quantizer**: In 1D, these are equivalent
- **Decoder = Identity**: No decoder construction needed
- **Coding Scheme = Encoder ∘ Memorizer**: Simple composition

## Examples

### Example 1: Basic Approximation

```python
# Create coding scheme with zig-zag memorizer
coding_scheme = CodingScheme1D(
    squared_shifted, 
    memorizer_type="zigzag",
    dtype=torch.float64,
    device="cuda"
)

# Evaluate with different parameters
params = [(2,2,0.1,1e-2), (3,3,0.01,1e-3), (4,4,0.001,1e-4)]
coding_scheme.evaluate_for_list_of_parameters(
    params, 
    "Squared Function",
    plot=True
)
```

### Example 2: Memorizer Comparison

```python
# Compare memorizer behavior across parameters
zigzag_param_list = [(2, 2), (3, 3), (4, 4)]
zig_zag_memorizer = ZigZagMemorizer(base_function, K=2, M=2)
zig_zag_memorizer.plot_with_base_function_for_parameter_list(zigzag_param_list)
```

## Performance Notes

- **Runtime**: Exponential in $K$ (approximately $O(2^K \cdot n)$ for $n$ evaluation points)
- **Memory**: Linear in $K$ and $M$
- **Precision**: Use `torch.float64` for better numerical stability
- **GPU**: Significant speedup for large $K$ values

## Troubleshooting

**Issue**: Numerical instabilities / NaN values
- **Solution**: Increase flat_slope/slope parameters, reduce $K$ or $M$

**Issue**: Very slow execution
- **Solution**: Reduce $K$ (each increment doubles runtime)

**Issue**: Plots look jagged/incorrect
- **Solution**: Increase number of plot points or reduce $\gamma$

## Citation

This code corresponds to the following **unpublished paper**:

```bibtex
@unpublished{rochau2024new,
  title={New Advances in Universal Approximation with Neural Networks of Minimal Width},
  author={Rochau, Dennis and Chan, Robin and Gottschalk, Hanno},
  note={Unpublished manuscript},
  year={2024}
}
```

Until the new paper is published, please cite the related arXiv preprint:

```bibtex
@article{rochau2024universal,
  title={Universal approximation with neural networks of minimal width},
  author={Rochau, Dennis and Chan, Robin and Gottschalk, Hanno},
  journal={arXiv preprint arXiv:2411.08735},
  year={2024},
  url={https://arxiv.org/abs/2411.08735}
}
```

**Note**: The arXiv paper represents an older version of this work.

## Authors

- **Dennis Rochau**
- **Robin Chan**
- **Hanno Gottschalk**

## License

MIT License

Copyright (c) 2024 Dennis Rochau, Robin Chan, Hanno Gottschalk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Version History

- **v1.0** (2025-12): Initial release with L^p and supremum approximation schemes
