# Vision Examples

This folder contains runnable image-model commands. They use the same public trainer API as the
tabular examples, but they exercise image-specific runtime paths: NCHW tensors, convolution,
pooling, patch embeddings, attention blocks, CUDA kernels, and CIFAR-10 loaders.

## Files

- `Cnn.lean`: compact convolutional CIFAR-10 classifier. It uses a small crop so the command is a
  practical CUDA regression target while still exercising real convolution/pooling-style data flow.
- `Vit.lean`: compact ViT-style CIFAR-10 classifier. It uses convolutional patch embedding,
  token reshape, one transformer encoder block, and a linear head.

## Data

Prepare the small real CIFAR-10 subset with:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
```

The loader reads `.npy` arrays under `data/real/cifar10/`. The examples crop the image tensors to
small typed shapes so they run quickly while still crossing the same data boundary as larger image
runs.

## Commands

```bash
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

For runtime profiling or fast kernels:

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

## What To Inspect

These examples own the image-classification training path. Useful outputs are:

- the training loss and accuracy trace;
- the `TrainLog` JSON if `--log PATH` is passed;
- the typed image shapes in the Lean source;
- CUDA parity and regression evidence when changing image kernels.

For 3D detector certificates or projection verification, use `NN/Examples/Verification` and the
Geometry3D workflow. That path exports detector tensors as certificate artifacts, checks the camera
projection/enclosure conditions in Lean, and renders overlays for inspection.
