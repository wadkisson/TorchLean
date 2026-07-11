import VersoManual

open Verso.Genre Manual

#doc (Manual) "Example Walkthroughs" =>
%%%
tag := "examples"
%%%

The examples cover typed tensors, training, autograd, graph checking, verification, and BugZoo
contracts. Each group identifies where data enters, which artifact is produced, what is checked, and
what a successful run establishes.

For a first run, choose one row from the table below and keep the resulting artifact in view. A loss
trace answers a different question from a graph, and an accepted certificate answers a different
question from both.

# The Examples At A Glance

The tree is organized around questions:

| Goal | Run or read |
|---|---|
| first tensor example | [TensorBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean) |
| first training example | `lake exe torchlean mlp --device cpu --steps 10` |
| first autograd example | [AutogradBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean) |
| first graph example | `lake exe verify -- torchlean-ibp` |
| first BugZoo example | `NN.Examples.BugZoo.All` |
| first CUDA runtime check | `lake -R -K cuda=true exe torchlean mlp --device cuda --steps 20` |
| first scientific ML example | `lake exe verify -- ode` or the PINN/FNO examples |

The produced artifact determines the meaning of a successful run:

| Artifact | Example family | How to read the result |
|---|---|---|
| printed tensor | quickstart tensor examples | value and shape inspection |
| loss trace | MLP/CNN/ViT/GPT/FNO/generative/RL commands | runtime and optimizer sanity check |
| JSON log | model and RL commands | widget-readable training artifact |
| graph / IR | GraphSpec, compiled runtime, verification commands | verifier/export boundary |
| certificate or bounds | `lake exe verify -- ...` | checker acceptance, not training quality |
| image sample | diffusion/autoencoder-style examples | sampling artifact, not a theorem |
| rollout or policy artifact | PPO examples | environment-boundary and behavior inspection |

Most model examples run through one executable:

```
lake exe torchlean <example> [args...]
```

The umbrella module is `NN.Examples.Zoo`. Building it is the standard health check for the
curated examples:

```
lake build NN.Examples.Zoo
```

# Three Representative Workflows

## 1. A compact model trains

The MLP and CNN examples show the basic pattern:

- the dataset is represented as tensors;
- the model is built through the public `TorchLean.nn` API;
- parameters are explicit;
- the optimizer returns updated state;
- the log records an ordinary loss trace.

These small runs use the same runtime path as the larger examples, so they are good places to learn
the training loop before adding a larger dataset or model family.

```
lake exe torchlean mlp --device cpu --steps 10
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
```

## 2. A graph is checked

The verification examples start from the shared IR. A compact model is lowered into `NN.IR.Graph`, a
payload supplies parameters, and a verifier pass computes a bound or checks a certificate.

```
lake exe verify -- torchlean-ibp
```

The verifier does not guess which Python object was meant; it consumes an `NN.IR.Graph` with named
operations and an explicit payload.

## 3. A real ML failure mode becomes a tiny Lean case study

Bug Zoo examples are compact because each one isolates a failure mode. Examples cover attention
masks, KV caches and RoPE positions, tokenizer boundaries, normalization state, batch invariance,
Float32 assumptions, and autograd boundaries.

The style is:

```
-- State the behavior directly, then make the boundary explicit.
#check Spec.hardMaskedSoftmaxSpec
#check runtimeFloat32_add_rewrites_to_ieee32
```

BugZoo turns each bug pattern into a named, checked statement.

# Anatomy Of An Example

Most runnable examples are ordinary Lean modules with four recognizable pieces:

```
def model : nn.M ...
def data  : Trainer.Dataset ...
def trainCurve ...
def main (args : List String) : IO UInt32 := ...
```

For verifier and BugZoo examples, look for the named contract instead:

```
#check RuntimeFloat32MatchesIEEE32Exec
#check Spec.hardMaskedSoftmaxSpec
#check NN.IR.Graph
```

The names are the point. A model command may show that the training code runs. A `#check` target or
theorem name tells you which semantic object the surrounding prose is allowed to cite.

# Fast Kernels, BugZoo, and Verification

The runtime-to-proof chain has four stages:

1. *Fast kernel*: a runtime path uses an optimized implementation, such as fused CUDA attention.
2. *BugZoo example*: BugZoo states the semantic hazard, such as mask polarity, cache position mismatch, or
   floating point boundary mismatch.
3. *Contract*: the spec names the intended mathematical behavior.
4. *Checker or theorem*: Lean checks an artifact, proves a local theorem, or records the remaining
   producer/runtime assumption.

For attention, the chain looks like this:

- Runtime: CUDA `multi_head_attention` may call fused native `flashAttentionFwd` and fused VJP
  kernels.
- Spec: `Spec.flashAttention` and `Spec.scaledDotProductAttention` name the forward contract.
- Theorem: `flashAttention_eq_scaledDotProductAttention` states that the fused spec denotes SDPA.
- BugZoo example: `BugZoo.AttentionMask` records the exact mask property, including future tokens getting
  zero attention under causal masking.
- Verification fixture: `torchlean-transformer-ibp` pushes a tiny transformer graph through
  the same verifier API.

TorchLean welcomes faster execution when it is paired with a named semantic contract and a visible
proof or test boundary.

The pattern is either equality with the spec, or membership in a safe enclosure around the spec.

If the equality is proved, the fast path becomes an optimization. If it is checked only by tests, we
say that plainly. If the kernel is external CUDA code, the CUDA chapter names the assumption instead
of letting it disappear into the example.

The same reading pattern applies outside attention:

- FNO uses a dense DFT reference path and a cuFFT-backed CUDA path; the example demonstrates the
  training/runtime boundary, while scientific claims about Burgers residuals belong to the
  verification workflows.
- Diffusion writes samples and logs; the executable demonstrates a denoising training path, while
  Gaussian forward-process and sampler-step facts live in the generative theory modules.
- PPO with Gymnasium checks the transition boundary before storing rollouts; claims about the
  external simulator remain producer assumptions.
- PyTorch interop examples show a round trip or emitted code; a semantic-preservation theorem is a
  stronger claim and must be cited separately.

# How The Tree Is Organized

The examples are grouped so each subtree has a job:

- [Quickstart examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Quickstart/)
  Typed-tensor warmups, autograd basics, and the smallest training tutorials.
- [Model examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/)
  The main example set used by `lake exe torchlean`: MLP, KAN, CNN, ViT, RNN/LSTM,
  Transformer, GPT text examples, Mamba examples, diffusion, and RL examples.
- [Deep-dive examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/DeepDives/)
  GraphSpec, float-mode, tensor-bridge, and TorchLean-to-PyTorch IR examples.
- [PyTorch interop examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Interop/PyTorch/)
  PyTorch round trip scripts and generated code interop examples.
- [Verification fixtures](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Verification/)
  Small bundled certificate/checker examples used by `lake exe verify -- ...` (with external producers
  under `scripts/verification/`).
- [Widget gallery](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Widgets.lean)
  Infoview widgets used in the editor rather than through a long training run.
- [RL widget viewers](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/RL/)
  File-backed PPO/GridWorld dashboards colocated with the RL examples they visualize.
- [Bug Zoo](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/)
  Small checked case studies for attention masks, KV caches, tokenizers, normalization state,
  batching, floats, and autograd boundaries.
- [real data paths](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/RealPaths.lean) and
  [model data helpers](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Common/RealData.lean)
  Shared paths and loaders for prepared real datasets under `data/real`.

That structure helps the examples read as one system: tensors first, then runnable models, graph
lowering, and checked properties.

# A Good First Pass Through The Examples

For one short pass that touches the main layers, use this order:

1. Build once:
   `lake build`
2. Build the curated examples umbrella:
   `lake build NN.Examples.Zoo`
3. Typed tensor example:
   `lake env lean --run NN/Examples/Quickstart/TensorBasics.lean`
4. Float-backend comparison example:
   `lake exe torchlean float32_modes`
5. Widgets example, opened in the editor:
   open the [widgets example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Widgets.lean)
6. Autograd example:
   `lake env lean --run NN/Examples/Quickstart/AutogradBasics.lean -- --dtype float`
7. Model training example:
   `lake exe torchlean mlp --device cpu --steps 10`
8. Small complete verification example:
   `lake exe verify -- torchlean-ibp`

# What Success Looks Like

Expected outputs on a healthy run:

- `quickstart_tensors`: small typed tensors print cleanly.
- `float32_modes`: the same numeric expression is evaluated under different float32 semantics.
- `quickstart_autograd`: prints a value and a gradient from the recorded tape.
- `lake exe torchlean mlp --device cpu --steps 10`: prints a short loss trace trending downward.
- `lake -R -K cuda=true exe torchlean gpt_adder --device cuda --steps 1`: runs the minGPT
  addition example on GPU when the CUDA backend has been built.
- `torchlean-ibp`: prints interval bounds for the output node, not raw graph internals.

This framing distinguishes a semantic check from a command that exits successfully.

# Dtype And Backend Flags

Most training and runtime examples are written once and then instantiated by the command line
wrapper with a scalar type and backend chosen on the command line.

## `--dtype`

Many examples accept `--dtype`:

- `--dtype float`
  Lean runtime `Float` (binary64).
- `--dtype float32`
  An executable float32 mode used by several tutorials.
- `--dtype ieee32exec` or `--dtype ieee754exec`
  TorchLean's executable IEEE-754 binary32 model defined in Lean (`IEEE32Exec`).

Proof-only backends such as `FP32` and `NF` are discussed in the floating point chapters.

## `--backend`

Many examples accept `--backend eager|compiled`:

- `--backend eager`
  Build and inspect a tape as the run executes.
- `--backend compiled`
  Build a reusable graph artifact for the same public model.

For debugging compilation to IR or verifier-style passes, `--backend compiled` is the usual default.

When documenting a result, record both choices. "The example ran" is less useful than:

```
lake exe torchlean quickstart_mlp --steps 20 --dtype float32 --backend eager
lake exe torchlean quickstart_mlp --steps 20 --dtype ieee32exec --backend compiled
```

The two commands can exercise the same public model but answer different questions. The first is a
runtime/autograd check over executable float32 values. The second is closer to the finite-precision
and graph artifacts used by proof-oriented pages.

# Model Examples and Real Data

The model API is broad: classical feedforward and conv nets, ViT and residual blocks,
recurrent and attention language models, state space and diffusion examples, and PPO RL
environments. The original MLP/CNN tutorials are now the first step rather than the boundary.
Model examples are driven by:

`lake exe torchlean <example> [flags...]`

The [model examples API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/) contains the implementations, and the
[example runner API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean) defines the public subcommand names.

First run:

`lake exe torchlean --help`

## `torchlean` subcommands

Use these as starting points; each module's header documents the full flag set.

- `mlp`: Auto MPG tabular regression; a good first GPU runtime check after `cuda=true`.
- `kan`: Auto MPG tabular regression with KAN edge-basis functions.
- `cnn`: convolutional training on CIFAR tensors; prepare `data/real` with the download
  script for real data.
- `vit`: patch embedding plus Transformer vision blocks; the usual anchor is Dosovitskiy et al.,
  ViT (2020).
- `rnn`, `lstm`: sequence modeling baselines over recurrent state.
- `transformer`: encoder sequence model; pairs naturally with the attention chapters.
- `gpt2`, `text_gpt2`: decoder language modeling; `text_gpt2` uses file-backed text.
- `mamba`: selective state-space model example.
- `diffusion`: generative diffusion training loop.
- `ppo_gridworld`, `ppo_cartpole`: policy optimization in compact controlled environments.
- `floats_arb_ieee_compare`: compares float semantics modes and pairs with *Floating-Point
  Semantics*.

Runnable via `torchlean`: `lake -R -K cuda=true exe torchlean gpt_adder --device cuda --steps 1` trains the
small addition example through the same runner as the other model commands. This particular
example is CUDA-only.

Example invocations:

```
lake exe torchlean mlp --device cpu --steps 10
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean rnn --device cuda --tiny-shakespeare --steps 1
lake -R -K cuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1
lake -R -K cuda=true exe torchlean gpt_adder --device cuda --steps 1
```

The real data helper prepares the small public corpora and CIFAR shards used by these examples:

`python3 scripts/datasets/download_example_data.py --tiny-shakespeare --tinystories-valid --cifar10 --cifar10-limit-train 200 --cifar10-limit-test 50`

Downloaded data is stored in `data/real` and is ignored by Git. Small checked-in synthetic
data remains in the [example data API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Data/) so the repo can still run runtime-check
tests without network access.

These model examples matter because they show that TorchLean's public API is not limited to one
teaching example:

- MLPs, CNNs, attention blocks, and residual blocks share the same `TorchLean` vocabulary,
- the `zero_grad / backward / step` rhythm can be written directly in Lean,
- and the same scripts can still switch dtype and backend from the CLI.

The larger examples should still be run small first. For example:

```
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
```

Those settings are intentionally small. They are meant to test the path through tokenization,
denoising, or rollout collection before you spend time on a meaningful training run.

# Data Loading, Transforms, And Schedulers

The `NN/API/Data*` layer is richer than a first glance suggests. The current tutorial set covers
three increasingly realistic patterns:

- `csv_loader_train`
  Disk-backed tabular data, `Data.Transforms.Compose`, minibatch loaders, and a learning-rate
  schedule (`Trainer.stepEpochLR`).
- `npy_loader_train`
  NumPy/PyTorch `.npy` interop with typed dataset reconstruction on the Lean side.
- `cifar10_npy_cnn_train`
  A more realistic offline image workflow: load `.npy`, split train/test, batch, train, and
  report accuracy.

Together, they answer a common question from PyTorch users:

> Is TorchLean only comfortable with in-memory tutorial tensors?

The answer is no. The examples cover CSV and `.npy` file readers, typed transforms,
deterministic batching, callbacks, and scheduler-driven training loops.

# GraphSpec And Export Workflows

Two examples highlight newer architecture and interop work beyond the smallest tutorials.

## GraphSpec-authored training example

- Lean example: [GraphSpec tutorial API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/GraphSpec/Tutorial.lean)
- Command:
   `lake exe torchlean graphspec --backend eager`

This example defines an MLP once in GraphSpec, lowers it to the TorchLean training API, and then
trains it with the ordinary public API. It is the fastest way to see GraphSpec as a real user
workflow rather than as a research-only subdirectory.

## TorchLean, IR, and Emitted PyTorch Code

- Lean example: [Torch IR PyTorch export API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/TorchIRPyTorch.lean)
- Commands:
  `lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py`

This example sits at a repository boundary:

- TorchLean model, compiled IR, and generated PyTorch code.

The supported architectures already include more than a single linear layer:

- `linear`, `mlp`, `sum`, `autoencoder`,
- `mha`, `mha-mask`,
- `transformer`.

# BugZoo Case Studies

BugZoo deserves its own reading pass because it explains what kinds of mistakes TorchLean is trying
to make visible. The [BugZoo API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/) contains the checked examples, with the
full motivation in the [BugZoo overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/README.md).

The case studies are compact, but they cover a broad range:

- *Shape and broadcast*: missing axes and wrong broadcasts; TorchLean uses tensors indexed by shape
  and explicit broadcast evidence.
- *Stable loss*: unstable loss formulas and invalid domains; TorchLean names stable logit losses and
  safe-domain ops.
- *Ignored labels*: all-ignored label reductions; TorchLean uses an explicit contribution mask and
  empty-reduction policy.
- *Autograd domain*: masking after undefined division; TorchLean uses epsilon-protected division
  before masking.
- *Attention mask*: mask polarity and fully masked attention; TorchLean states exact causal-mask
  zero-weight properties.
- *Compiler boundary*: optimized graph wrong-code bugs; TorchLean asks for a source/target semantic
  preservation contract.
- *Float boundary*: real-proof versus float-deployment gaps; TorchLean names the bridge from runtime
  float32 to `IEEE32Exec`.
- *Normalization state*: BatchNorm formula/state mismatch; TorchLean makes epsilon placement and
  running statistics explicit.
- *Batch invariance*: serving output depends on batch composition; TorchLean states per-example
  batching behavior.
- *KV cache*: shifted or malformed decode cache; TorchLean names append-last key/value invariants.
- *RoPE position*: position mismatch during decoding; TorchLean names the position schedule.
- *Tokenizer boundary*: vocabulary/config mismatch; TorchLean represents token ids as
  `Fin vocabSize`.

These examples do not solve every production incident. They show a repeatable method: take a bug that
normally sits inside an opaque runtime, name the boundary, and attach a
small checked statement to it.

Useful build target:

```
lake build NN.Examples.BugZoo.All
```

For readers coming from ML systems, BugZoo is often the most concrete entry point: a recognizable
bug class, the exact object TorchLean uses as the contract, and the theorem or boundary that
follows.

# Verification Application Workflows

The verification side has also grown beyond one tiny IBP example. The current example tree covers
several distinct application styles.

## Core TorchLean To IR Workflows

- `torchlean-ibp`
  Tiny MLP, compile to IR, seed an input box, read off output bounds.
- `torchlean-crown-ops`
  Exercises a broader operator set, including softmax and loss-shaped examples.
- `torchlean-transformer-ibp`
  Tiny attention / encoder pipeline pushed through the same verifier path.

## Robustness and Benchmark Workflows

- `torchlean-robustness`
  Margin-certification example over a TorchLean-authored model.
- `digits`
  Certified accuracy robustness example.
- `vnncomp-mnistfc`
  A small VNN-COMP suite checker over exported JSON artifacts.

## PINN and ODE workflows

- `pinn-cert`
  Recompute-and-compare certificate check for a physics-informed neural network residual bound.
- `pinn-cli`
  Interactive PDE residual-bounding workbench with box splitting and backend selection.
- `pinn-dataset-check`
  Dataset-backed containment check against reference points.
- `ode`
  ODE enclosure verification for subsolution and supersolution workflows.

Together, these examples show the range of the project: robustness checks, scientific ML residual
bounds, and dynamical-system reasoning all use the same idea of a model plus a checked semantic
artifact.

# Scientific ML As An Application Thread

Scientific ML examples are not just "models with different data." They change the claim type.

- FNO learns an operator between function samples, so the object of interest is
  `initial condition -> solution snapshot`, not `image -> class`.
- PINN workflows check residual bounds against a PDE expression and a domain box.
- ODE workflows reason about enclosures for dynamical systems.
- Lyapunov/controller examples combine learned functions with stability-style inequalities.

The practical lesson is that the dataset, differential equation, discretization, and bound artifact
are part of the claim. A screenshot of a prediction curve is useful for debugging. A residual
certificate or enclosure checker is the stronger artifact to cite.

For the FNO/Burgers example, a typical workflow is:

```
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv
python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
```

This command is a runtime operator-learning workflow. The `pinn-cert`, `pinn-cli`, `ode`, and
Lyapunov commands produce or check verification artifacts.

# Widgets In Practice

Some of the most revealing examples are not command line programs at all. They are editor files that
render live views in the infoview.

The full widget family includes several views; the short list cited most often is:

- tensor views,
- graph and shape-inference views,
- float32 bit-layout views,
- verification-state views,
- autograd tape and gradient views.

There are also application-specific views:

- `#train_log_file_view` for JSON logs produced by model and RL commands;
- `#gpt2_train_log_file_view` for prompt/sample notes in GPT-style logs;
- `#rl_boundary_rollout_file_view` for Gymnasium transition-boundary diagnostics;
- GridWorld/PPO viewers for policies, paths, and rollout curves;
- `#pytorch_translate_file` for a source-level PyTorch-to-TorchLean sketch used as an editor aid.

These widgets help users inspect artifacts. They do not upgrade a runtime claim into a theorem.

The [widgets example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Widgets.lean) pairs well with the compact map of widget families below.

# Check Tests

After editing an example, prefer the smallest check that covers the touched path:

```
lake env lean NN/Examples/Quickstart/TensorBasics.lean
lake build NN.Examples.BugZoo.All
lake exe torchlean --help
lake exe torchlean mlp --device cpu --steps 1
```

Run broader tests when the touched code changes shared APIs or runtime behavior. For guide-only
edits, checking the edited Verso files directly is usually enough.

# Adding A New Example

Checklist for adding a new example:

- Put it in the right subtree:
  `NN/Examples/Models` for model examples, `NN/Examples/Quickstart` for small
  tutorials, `NN/Examples/DeepDives` for architecture/IR examples, and `NN/Examples/Verification` +
  `scripts/verification` for verifier workflows.
- If it belongs in the public runner, register it in the
  [model runner API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean) and make sure `NN.Examples.Zoo`
  imports it.
- Use the public API where appropriate:
  `import NN`, `open TorchLean`, and stick to `nn`, `Data`, `Trainer`, and `optim` unless
  the example is explicitly about an internal subsystem.
- Add a short module docstring at the top with:
  1. the mathematical or runtime object the example exposes, and
  2. the exact `lake env lean --run ...` or `lake exe torchlean ...` command.
- Parse flags consistently:
  support `--seed` when randomness is involved, and prefer the shared `Runtime.runFloat` /
  public `Trainer.new` route for `--dtype`, `--backend`, and device selection.
- Keep the default run bounded:
  small dataset, compact model, few steps or epochs; heavier runs remain behind flags.

There is a central model runner for model examples (`lake exe torchlean ...`), while
deep dive and tutorial files can still be run directly with `lake env lean --run ...`.

Command-line overview: *CLI Entry Points*. To connect an example to verification, read *Graphs and
IR* first, then *Verification*. For GPU build details, use *Runtime and Autograd*.

# Interpreting Example Claims

Use careful verbs:

- "runs" means the executable completed on the selected backend;
- "logs" means it produced a readable artifact such as JSON, CSV, PPM, or printed metrics;
- "checks" means Lean accepted a checker result or artifact parser;
- "proves" means there is a Lean theorem with the cited statement;
- "assumes" means an external producer, CUDA kernel, simulator, dataset converter, or Python tool
  remains outside the proved fragment.

That vocabulary keeps example prose useful. A one-step GPT, diffusion, or PPO run is valuable
because it exercises a complicated boundary. It is not evidence of model quality. A BugZoo theorem
can be strong about one exact invariant while saying nothing about a production system that has not
been connected to that invariant.

# References and further reading

- George et al., *TorchLean* (2026): project overview. https://arxiv.org/abs/2602.22631
- Vaswani et al., *Attention Is All You Need* (2017): transformer baseline. https://arxiv.org/abs/1706.03762
- He et al., *Deep Residual Learning* (2015): ResNet. https://arxiv.org/abs/1512.03385
- Dosovitskiy et al., *An Image Is Worth 16x16 Words* (2020): ViT. https://arxiv.org/abs/2010.11929
- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models* (2020): DDPM. https://arxiv.org/abs/2006.11239
- Gu & Dao, *Mamba* (2023): selective state spaces. https://arxiv.org/abs/2312.00752
- Dao et al., *FlashAttention* (2022): IO-aware exact attention. https://arxiv.org/abs/2205.14135
- Odena et al., *TensorFuzz* (2019): fuzzing numerical failures. https://arxiv.org/abs/1807.10875
- Liu et al., *NNSmith* (2023): generated valid tests for DL compilers. https://arxiv.org/abs/2207.13066
- Schulman et al., *Proximal Policy Optimization* (2017): PPO. https://arxiv.org/abs/1707.06347
- PyTorch documentation: user-side analogy for `NN.API`. https://pytorch.org/docs/stable/index.html
