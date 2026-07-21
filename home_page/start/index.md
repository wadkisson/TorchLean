---
title: Getting Started
layout: default
usemathjax: true
---

# Getting Started

Start with a model small enough that every part can be seen. The first run should feel like ordinary
ML: build the project, train a tiny model, and run a compact verifier. The classifier is
deliberately small, so the structure is visible: the same project contains the executable model, the
graph representation, and the checker that reasons about the graph.

```bash
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 10 --dtype float32 --backend eager
lake exe verify -- torchlean-ibp
```

The first command checks the Lean project. The second command runs a compact supervised model through
the public trainer: it initializes parameters, executes a Float32 forward pass, computes a loss, runs
reverse mode, and applies optimizer updates. The third command lowers a small TorchLean model into
the graph used by the verifier and runs interval bound propagation over an input box.

To see the command-line entry points:

```bash
lake exe torchlean --help
lake exe verify --help
```

The first lists runnable examples: quickstarts, supervised models, text models, diffusion, FNO
Burgers, reinforcement learning, data loaders, PyTorch interop, graph examples, and floating-point
checks. The second lists checker entry points: TorchLean IBP/CROWN paths, LiRPA-style fixtures,
PINN certificates, ODE enclosures, VNN-COMP-style MNIST queries, 3D projection certificates, spline
certificates, and two-stage Lyapunov experiments.

## Where To Go Next

1. [Installation]({{ '/installation/' | relative_url }}) covers Linux, macOS, Windows/WSL, CUDA, and
   optional LibTorch integration.
2. [Building Models]({{ '/blueprint/Building-Models/' | relative_url }}) introduces typed tensors,
   layers, datasets, training, and the public trainer.
3. [Runtime And Interop]({{ '/blueprint/Runtime-And-Interop/' | relative_url }})
   explains execution backends, autograd, and PyTorch interop.
4. [Graphs And Numerics]({{ '/blueprint/Graphs-And-Numerics/' | relative_url }}) covers graph IR,
   floating-point semantics, and native/CUDA boundaries.
5. [Verification]({{ '/blueprint/Verification/' | relative_url }})
   covers bounds, certificates, autograd/approximation proofs, and the theory map.
6. [Applications]({{ '/blueprint/Applications/' | relative_url }}) and
   [Examples]({{ '/examples/' | relative_url }}) collect runnable models, CLI tools, and Bug Zoo
   workflows.

## Common Next Steps

- Train a model: `lake exe torchlean mlp --device cpu --steps 100 --dtype float32`.
- Inspect a scientific ML run: [Scientific ML]({{ '/examples/scientific-ml/' | relative_url }}).
- Check a certificate or bound pass: [Verification Bounds]({{ '/examples/verification/' | relative_url }}).
- Read the public import path: `import NN; open TorchLean`.
- Explore module ownership: [Graphs]({{ '/graphs/' | relative_url }}).
- Understand CUDA assumptions: [Native Boundaries]({{ '/blueprint/Graphs-And-Numerics/Native-Boundaries/' | relative_url }}).

When changing code, keep the edit loop small at first: run one example, inspect the file it writes,
then run the matching checker when the example has one.
