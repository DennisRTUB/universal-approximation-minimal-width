# New Advances in Universal Approximation with Neural Networks of Minimal Width

This repository contains code implementations accompanying an **unpublished paper** on:
1. **Universal approximation with neural networks of minimal width**
2. **Distributional universal approximation with LU-Net**

## Getting Started

**We recommend starting with the [Coding_Scheme_1D](./Coding_Scheme_1D/) folder** to build intuition about the coding scheme approach, its components (encoder, memorizer, decoder), and how it approximates different functions in one dimension.

## Repository Structure

### 1. [Coding_Scheme_1D](./Coding_Scheme_1D/)

**Purpose**: Interactive demonstrations of the 1D coding scheme for universal approximation.

This folder provides visualizations and explanations to help understand:
- How the encoder (quantizer) discretizes inputs
- How different memorizers (Zig-Zag and PLCSM) fit functions on discrete points
- How the coding scheme approximates various continuous functions
- Parameter convergence behavior and practical limitations
- **Critical network width distinction**: Zig-zag is width-1, PLCSM is width-2

**Key insight**: All constructions are built purely from $\sigma_{a,b,c,d}$ functions (width-1 LReLU compositions) - nothing else! The zig-zag memorizer achieves universal approximation with **minimal width-1 networks**, while PLCSM uses width-2 for smoother results.

**Contents**:
- `coding_scheme_1D.ipynb`: $L^p$ norm approximation with continuous activations
- `coding_scheme_1D_sup.ipynb`: Supremum norm approximation with discontinuous activations
- `coding_scheme_base_functions.py`: Core implementation of encoders and memorizers

**Start here** to gain intuition before exploring the higher-dimensional implementations.

See the [Coding_Scheme_1D README](./Coding_Scheme_1D/README.md) for detailed usage instructions.

### 2. [LU-Net](./LU-Net/)

**Purpose**: Implementation of the dense LU-Net architecture, a normalizing flow for density estimation and generative modeling.

This folder contains:
- Full implementation of the dense LU-Net normalizing flow
- Training utilities and optimization tools
- Toy Jupyter notebook experiments demonstrating the approach
- Tools for evaluating and visualizing trained models

The LU-Net extends the theoretical coding scheme to practical deep learning applications using invertible neural networks.

See the LU-Net README for detailed documentation on training and usage.

## Requirements

- Python 3.7+
- PyTorch (CUDA-enabled GPU recommended)
- matplotlib
- Additional dependencies listed in each folder's README

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd New-advances-in-universal-approximation-with-neural-networks-of-minimal-width

# Install dependencies
pip install torch matplotlib

# Navigate to specific folders for additional requirements
```

## Workflow

1. **Build Intuition**: Start with [Coding_Scheme_1D](./Coding_Scheme_1D/) notebooks to understand the theoretical foundations
2. **Explore Applications**: Move to [LU-Net](./LU-Net/) for practical implementations and experiments

## Citation

This work is currently unpublished. If you use this code in your research, please contact the authors for the appropriate citation.

```bibtex
@unpublished{rochau2024new,
  title={New Advances in Universal Approximation with Neural Networks of Minimal Width},
  author={[Authors]},
  note={Unpublished manuscript},
  year={2024}
}
```

## License

[Add your license information here]

## Authors

[Add author information here]
