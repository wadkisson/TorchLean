import VersoManual

open Verso.Genre Manual

#doc (Manual) "Conclusion" =>
%%%
tag := "conclusion"
%%%

TorchLean keeps ML code runnable and claims about it tied to named objects, checkers, theorems, or
external assumptions—especially important for scientific ML, controllers, perception, and RL, where
models sit inside larger arguments. Claim vocabulary is defined in *What TorchLean Is*.

# Levels Of Support

Use the Introduction's ladder (run → inspect → check → theorem → named external assumption). The
levels often stack on one workflow:

```
command output
  -> persisted artifact
  -> widget or report
  -> checker acceptance
  -> theorem about the checker or semantics
  -> named external assumptions
```

Example: a PPO CartPole run may produce a reward log and checked transition records. The log shows
the command ran; the boundary records show observations/actions/rewards/done flags had the declared
shape/range when they entered Lean. Neither proves Gymnasium's implementation nor PPO convergence.

A diffusion run may write a PPM sample and train log while theory proves a formal forward-Gaussian
fact or a Lipschitz sampler step. Those local facts do not prove image quality, FID, or native-kernel
equivalence.

# How To State A TorchLean Result

When writing about TorchLean, the strongest statements have a simple form:

- name the object: model, graph, tensor, tape, certificate, parameter store, dataset, or external
  artifact;
- name the semantics: real, `FP32`, `IEEE32Exec`, graph denotation, verifier abstraction, or runtime
  execution path;
- name the support: command output, widget inspection, checker acceptance, theorem, or assumption;
- state the property: shape agreement, gradient correctness, robustness margin, approximation
  bound, certificate validity, or runtime behavior.

For example, a model example run can establish that a particular training script executes and writes a
loss trace. A verifier command can establish that a graph artifact passes a bound checker. A
soundness theorem can establish that accepted bounds enclose the graph semantics for a supported
fragment. A CUDA or Python path can be used, but the external part remains part of the statement
unless a bridge theorem or replay checker covers it.

Here are better and worse forms of the same result:

| Avoid | Prefer |
|---|---|
| "TorchLean verifies GPT-2." | "`torchlean gpt2` runs a small GPT-2-style causal Transformer example; causal mask and token-bound claims must be cited separately." |
| "The diffusion model is proved correct." | "The diffusion example runs a denoising training path; `forwardGaussian_isGaussian` proves a formal forward-process fact." |
| "RL is verified." | "GridWorld dynamics have Lean-side semantics; Gymnasium PPO runs through a checked transition boundary, while the simulator remains external." |
| "The CUDA backend is proved." | "This command used CUDA; proof support depends on the specific kernel, spec, test, or runtime-conformance assumption cited." |
| "The widget proves the graph." | "The widget renders the graph/checker state; the theorem or checker result carries the proof claim." |

This wording lets a reader reproduce the run, inspect the artifact, and see where a stronger theorem
would attach.

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

# The Extension Rule

The same rule applies when extending the project. A new feature is strongest when it adds one clear
bridge:

- a new model should expose its data boundary, parameter shapes, training loop, and logs;
- a new operator should name its spec and runtime behavior;
- a new graph pass should state what semantics it preserves or approximates;
- a new verifier should identify the abstract domain, checker, and soundness theorem;
- a new external integration should specify the artifact format and producer/checker contract;
- a new widget should render an existing Lean object rather than define a new semantic meaning.

That rule keeps TorchLean extensible without making the system vague. Every addition should make at
least one object easier to run, inspect, check, or prove.

# Reading Claims In The Wild

When reading a TorchLean page, paper paragraph, commit message, or example README, ask five
questions:

1. What object is being discussed?
2. Which semantics is attached to it?
3. Which command, checker, widget, or theorem supports the statement?
4. Which external producer assumptions remain?
5. What would be required to upgrade the statement?

The last question is important. A runtime example can be upgraded by adding an artifact checker. A
checker can be upgraded by proving soundness for the accepted fragment. A theorem over real tensors
can be upgraded by adding an `IEEE32Exec` or runtime float32 bridge. A Gymnasium boundary can be
upgraded by moving an environment into Lean or by strengthening the contract around observations and
rewards. A CUDA fast path can be upgraded by a spec equivalence theorem, deterministic replay
contract, or a narrower named assumption.

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

Identify the object, identify the semantics, and say what has been run, checked, proved, or assumed.

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
