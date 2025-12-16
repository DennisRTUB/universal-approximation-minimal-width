# New Advances in Universal Approximation with Neural Networks of Minimal Width

This repository contains code implementations accompanying an **unpublished paper** on:
1. **Universal approximation with neural networks of minimal width**
2. **Distributional universal approximation with LU-Net**

## Getting Started

This repository presents two complementary approaches:

- **Interested in the coding scheme and its visualization?** → Start with [Coding_Scheme_1D](./Coding_Scheme_1D/) (read the [README](./Coding_Scheme_1D/README.md) inside for detailed usage instructions)
- **Interested in LU-Net implementation and experiments?** → Start with [LU-Net](./LU-Net/)

Both folders are self-contained and can be explored independently based on your research interests.

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
git clone https://github.com/DennisRTUB/universal-approximation-minimal-width.git
cd universal-approximation-minimal-width

# Install dependencies
pip install torch matplotlib

# Navigate to specific folders for additional requirements
```

## Usage

Both implementations are independent and serve different purposes:

- **[Coding_Scheme_1D](./Coding_Scheme_1D/)**: Explore theoretical foundations through interactive visualizations
- **[LU-Net](./LU-Net/)**: Train and experiment with normalizing flows for practical applications

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

**Note**: The arXiv paper (https://arxiv.org/abs/2411.08735) represents an older version of this work. Please check back for updates when the new paper is published.

## Authors

- **Dennis Rochau**
- **Robin Chan**
- **Hanno Gottschalk**

## License

[Add your license information here]
