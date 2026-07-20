import VersoManual

open Verso.Genre Manual

#doc (Manual) "Conclusion" =>
%%%
tag := "conclusion"
%%%

TorchLean is a research system for a practical problem: modern ML work mixes learned parameters,
generated code, numerical kernels, exported graphs, verifier artifacts, and scientific claims. A
successful run answers whether those pieces execute together. It does not, by itself, explain what
was computed or justify the claim made from the result.

One design principle runs through the project:

> a model should be runnable as ML code, inspectable as an artifact, and connected to named
> semantics whenever a correctness, gradient, bound, or certificate claim is made.

The same thread runs through the tensor API, runtime, graph IR, floating point models,
verification checkers, examples, and widgets.

The long-term value is especially clear in scientific computing. Neural operators, PINNs,
controllers, 3D perception systems, and RL agents are not isolated classifiers. They sit inside
larger mathematical and engineering arguments. A verified ML system should help those arguments name
their assumptions, check their artifacts, and preserve meaning across runtimes.

The repository therefore contains more than theorem statements or training examples in isolation.
Its scientific ML workflows, Bug Zoo cases, RL environments, generative objectives, widgets, and
command-line tools expose the artifacts on which later claims depend. Each one should state whether
its support comes from execution, a checker, a theorem, or an external assumption.

# Application Lessons

The application chapters point to several recurring lessons.

- *Model examples*: modern architectures are useful because they stress different semantic
  boundaries: masks, patch tokens, recurrent state, spectral transforms, latent variables, and
  environments.
- *Scientific ML*: the equation, domain, discretization, dataset, and residual/bound artifact are
  part of the claim, not background context.
- *BugZoo*: small theorem-sized case studies can expose real bug families more clearly than a large
  benchmark.
- *Widgets*: inspection improves proof engineering, but visual output does not replace a theorem.
- *RL*: transition data is a trust boundary; Lean-native environments and Gymnasium environments
  have different claim strength.
- *Generative models*: runnable image/text artifacts and objective/sampler theorems are both useful,
  but they answer different questions.
- *CLI workflows*: command lines are part of reproducibility. Record flags, data helpers, backend,
  dtype, artifact path, and external producers.

# Research Direction

The project now has several natural next steps:

- broaden model coverage while keeping examples small enough to inspect;
- connect more runtime paths to graph artifacts and proof statements;
- strengthen CUDA/kernel conformance stories for operators used by modern models;
- grow BugZoo with more real incident patterns from compilers, serving systems, scientific ML, and
  perception;
- make verification artifacts easier to produce, replay, and cite;
- improve widgets so large Lean objects stay readable without becoming a second semantic layer;
- build more bridges from external ecosystems such as PyTorch, ONNX, Gymnasium, VNN-COMP, and
  scientific-data formats into checked Lean artifacts.

The next stage is a library in which every serious claim names its object, semantics, supporting
artifact, and remaining assumptions. Different model families will require different local
theorems; they do not need to be hidden behind one monolithic claim about all of machine learning.

# Closing

TorchLean keeps two activities connected: building neural network systems and stating mathematical
claims about them. The code can still look like ML code. The examples can still
train, log, save, load, and call native or external tools. The difference is that the central objects
have names in Lean, and claims about those objects can be tied to checkers, theorems, or
explicit assumptions.

The standard the project should preserve is simple: identify the object, identify the semantics,
and say what has been run, checked, proved, or assumed. If TorchLean makes that habit easier for
neural networks, then it has met its purpose.

# References And Anchors

- George et al., [*TorchLean*](https://arxiv.org/abs/2602.22631), 2026.
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385),
  2015/2016.
- Ho, Jain, and Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239),
  2020.
- Li et al., [*Fourier Neural Operator for Parametric Partial Differential
  Equations*](https://arxiv.org/abs/2010.08895), 2020/2021.
- Mnih et al., [*Human-level control through deep reinforcement
  learning*](https://www.nature.com/articles/nature14236), 2015.
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347),
  2017.
- Gymnasium documentation, [environment API](https://gymnasium.farama.org/api/env/).
- Odena et al., [*TensorFuzz*](https://proceedings.mlr.press/v97/odena19a.html), 2019.
- Liu et al., [*NNSmith*](https://arxiv.org/abs/2207.13066), 2022/2023.
