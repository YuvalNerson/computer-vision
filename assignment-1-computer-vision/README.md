# Assignment 1 — Image Classification

This assignment explores classical and neural image-classification methods, with experiments on the CIFAR-10 dataset. It includes both feature-based models and neural networks implemented with NumPy.

## What I implemented

- k-Nearest Neighbors and softmax / linear classifiers for image classification.
- Image-feature extraction, including Histogram of Oriented Gradients (HOG) and color features, then classifier training on those representations.
- Two-layer and fully connected neural networks with modular forward and backward passes.
- Core training components such as loss/gradient calculations, optimization, regularization, and model selection.

## Notebooks

| Notebook | Focus |
|---|---|
| [k-Nearest Neighbors](./a1_code_submission/knn.ipynb) | Distance-based image classification |
| [Softmax](./a1_code_submission/softmax.ipynb) | Linear classification and softmax loss |
| [Feature extraction](./a1_code_submission/features.ipynb) | HOG and image feature experiments |
| [Two-layer network](./a1_code_submission/two_layer_net.ipynb) | Training a fully connected neural classifier |
| [Fully connected networks](./a1_code_submission/FullyConnectedNets.ipynb) | Deeper networks, normalization, and dropout |

## Code

The implementation package is in `a1_code_submission/icv83551/`. The notebooks document experiments and training. This coursework repository also contains assignment scaffolding; see the notebooks and package for the implementation details.

## Data and setup

The CIFAR-10 data is not included. Download and place the course dataset as expected by the notebook utilities before running the experiments. The notebooks use Python, NumPy, SciPy, and Matplotlib.
