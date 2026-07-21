---
title: Installation
layout: default
usemathjax: true
redirect_from:
  - /backends/
---

# Installation

Start with the CPU build. It does not require PyTorch, CUDA, or a GPU. The repository pins its Lean
version in `lean-toolchain`, so Elan selects the matching compiler.

## Five-Minute CPU Install

Install [Elan](https://github.com/leanprover/elan) on Linux or macOS:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

Open a new terminal so `elan`, `lean`, and `lake` are on your `PATH`. Then:

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 10
```

`lake exe cache get` downloads prebuilt Lean dependencies when available. It is safe to skip;
`lake build` will compile anything that is missing.

If those commands succeed, TorchLean is installed. Useful checks:

```bash
lake exe torchlean --help
lake exe verify --help
```

| Platform | CPU | NVIDIA GPU | Notes |
| --- | --- | --- | --- |
| Linux | Supported | Native CUDA | Primary development path |
| macOS (Intel or Apple silicon) | Supported | Not applicable | Metal/MPS is not implemented yet |
| Windows with WSL2 | Supported | CUDA on WSL2 | Recommended Windows setup |
| Native Windows | Bring-up | Not validated | Use WSL2 for now |

## Linux Packages

You need Git, `curl`, and a C/C++ compiler. On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

Then follow the five-minute install above. The default build uses the portable CPU runtime and
compiles CUDA stub archives so the Lean project links on machines without a GPU.

## NVIDIA CUDA (Linux / WSL2)

Install a supported NVIDIA driver and CUDA toolkit
([CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)).
Confirm the tools:

```bash
nvidia-smi
nvcc --version
```

Rebuild and run with CUDA:

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 10 --show-backend
```

`-K cuda=true` compiles and links the native CUDA implementation. `--device cuda` asks the
executable to use it. Use `-R` whenever you switch between CPU and CUDA Lake configurations.

More CUDA regression and sanitizer notes live on the [CUDA page]({{ '/cuda/' | relative_url }}).

### Optional LibTorch Attention

Ordinary CPU and CUDA builds do not need LibTorch. TorchLean currently uses it only for an optional
scaled-dot-product-attention bridge. Download a GPU-enabled distribution from the
[LibTorch install page](https://docs.pytorch.org/cppdocs/installing.html), extract it so it contains
`include/` and `lib/`, then:

```bash
lake -R -K cuda=true -K libtorch=true \
  -K libtorch_home=/absolute/path/to/libtorch build
lake -R -K cuda=true -K libtorch=true \
  -K libtorch_home=/absolute/path/to/libtorch exe libtorch_sdpa_test
```

## macOS

```bash
xcode-select --install
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 10
```

There is no NVIDIA CUDA path on modern macOS. `--device metal` is reserved in the device vocabulary
but not implemented; selecting it fails instead of silently falling back to CPU.

## Windows

### Recommended: WSL2

```powershell
wsl --install -d Ubuntu
```

After reboot, open Ubuntu and follow the Linux instructions. For a GPU, use NVIDIA's
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html): install the Windows
NVIDIA driver and the CUDA toolkit inside WSL.

### Native Windows

Native Windows is not a regularly tested release path. Prefer WSL2. If you are experimenting
anyway, install Git, the Visual Studio C++ build tools, the Windows SDK, and Elan:

```powershell
curl -O --location https://elan.lean-lang.org/elan-init.ps1
powershell -ExecutionPolicy Bypass -f elan-init.ps1
del elan-init.ps1
lake exe cache get
lake build
```

## Use TorchLean From Another Lean Project

In the downstream `lakefile.lean`:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then:

```bash
lake update
lake exe cache get
lake build
```

Most model files need only:

```lean
import NN
open TorchLean
```

For a neighboring checkout:

```lean
require TorchLean from "../TorchLean"
```

## Check An Installation

```bash
lake build
lake exe torchlean --help
lake exe verify --help
lake exe torchlean quickstart_mlp --device cpu --steps 10
```

For CUDA, rebuild and smoke-test with `-R -K cuda=true` as above.

## References

- [Elan](https://github.com/leanprover/elan)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Installing LibTorch](https://docs.pytorch.org/cppdocs/installing.html)
- [TorchLean trust boundaries](https://github.com/lean-dojo/TorchLean/blob/main/TRUST_BOUNDARIES.md)
