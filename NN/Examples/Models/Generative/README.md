# Generative Examples

This folder contains the runnable generative-model commands in TorchLean. They cover the runtime
paths that generative models stress: reconstruction losses, latent variables, vector quantization,
adversarial updates, masked image reconstruction, diffusion noise schedules, image artifacts, CUDA
kernels, and training logs.

The examples are runtime producers. They train small models, write logs or images, and keep the
model-zoo path honest across families whose losses and artifacts look very different from ordinary
classification. The mathematical identities behind the objectives live in the theory layer:
variational objectives for VAEs, codebook commitments for VQ-VAE, min-max losses for GAN-style
updates, masking/reconstruction contracts for MAE-style models, and denoising/noise-schedule
contracts for diffusion. If one of the generated artifacts is later used in a formal claim, that
claim should name the checker or theorem that consumes the artifact.

## Files

- `Autoencoder.lean`: compact vector autoencoder over real CIFAR image batches.
- `Vae.lean`: variational autoencoder path with reconstruction and latent regularization.
- `VqVae.lean`: vector-quantized autoencoder path for codebook-style latent reconstruction.
- `Gan.lean`: compact GAN-style training loop with generator/discriminator updates.
- `Mae.lean`: ViT-MAE-style masked autoencoder path: patch masking, transformer tokens, and image
  reconstruction.
- `Diffusion.lean`: compact unconditional diffusion command with noise schedule, denoiser training,
  DDIM-style replay, and optional PPM image artifacts.

## Data

Most commands use CIFAR-10 through the shared real-data loader:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
```

The diffusion command also supports ImageNet-style folders converted to `.npy`:

```bash
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/imagenet/train \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 --labels-from-dirs --limit 800
```

## Commands

Quick CUDA checks:

```bash
lake exe -K cuda=true torchlean autoencoder --cuda --steps 1 --n-total 1
lake exe -K cuda=true torchlean mae --cuda --steps 1 --n-total 1
lake exe -K cuda=true torchlean vae --cuda --steps 1 --n-total 1
lake exe -K cuda=true torchlean vqvae --cuda --steps 1 --n-total 1
lake exe -K cuda=true torchlean gan --cuda --steps 1 --n-total 1
lake exe -K cuda=true torchlean diffusion --cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
```

Longer diffusion run with visual artifacts:

```bash
lake exe -K cuda=true torchlean diffusion --cuda --fast-kernels \
  --dataset cifar10 --n-total 800 --steps 200 --hidden-c 8 --T 100 --beta-end 0.12 \
  --reference-ppm data/model_zoo/diffusion_reference.ppm \
  --noisy-ppm data/model_zoo/diffusion_noisy.ppm \
  --reconstruct-ppm data/model_zoo/diffusion_reconstruct.ppm \
  --sample-ppm data/model_zoo/diffusion_sample.ppm \
  --log data/model_zoo/diffusion_trainlog.json
```

## What To Inspect

The useful artifacts are training logs, generated/reconstructed images, and runtime behavior under
CUDA. A low loss curve or sample image is runtime evidence about the command that produced it. A
verified claim begins at the later boundary where Lean checks a predicate over the exported artifact
or proves a theorem about the objective/specification it represents.
