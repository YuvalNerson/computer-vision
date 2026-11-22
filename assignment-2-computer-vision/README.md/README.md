# Assignment 2 — CNNs and Image Captioning

This assignment moves from image classification to deeper visual models. It includes convolutional neural-network exercises and a PyTorch image-captioning pipeline that generates a word sequence from image features.

## What I implemented

- Convolutional-network components and CNN classifiers, including forward/backward operations used in the assignment notebooks.
- Batch normalization and dropout experiments, with training and validation workflows.
- PyTorch recurrent components: word embeddings, vanilla RNN and LSTM sequence processing, temporal scoring, and loss masking.
- An image-captioning model that maps image features to a caption and samples words autoregressively at inference time.

## Notebooks

| Notebook | Focus |
|---|---|
| [Convolutional Networks](./ConvolutionalNetworks.ipynb) | CNN layers and image classification |
| [Batch Normalization](./BatchNormalization.ipynb) | Normalization and training behavior |
| [Dropout](./Dropout.ipynb) | Regularization experiments |
| [PyTorch](./PyTorch.ipynb) | PyTorch implementation exercises |
| [RNN Captioning (PyTorch)](./RNN_Captioning_pytorch.ipynb) | Sequence models and image caption generation |

## Code

The implementation package is in `icv83551/`. The CNN and layer code lives under `icv83551/classifiers/` and `icv83551/layers.py`; recurrent and captioning code is in `icv83551/rnn_layers_pytorch.py` and `icv83551/classifiers/rnn_pytorch.py`.

This coursework repository includes assignment scaffolding as well as implementation work. The notebooks and package are the best places to review the code.

## Data and setup

The image-captioning notebook requires the course data, including COCO images and captions; the data is not bundled in the repository. See the dataset helper scripts under `icv83551/datasets/` and the notebook setup cells before running the experiments.
