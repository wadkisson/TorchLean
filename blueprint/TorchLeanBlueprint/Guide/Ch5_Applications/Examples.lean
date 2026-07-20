import VersoManual

open Verso.Genre Manual

#doc (Manual) "Example Walkthroughs" =>
%%%
tag := "examples"
%%%

Runnable example index and flags. Deep family notes: *Model Examples Deep Dive*.

# The Examples At A Glance

:::table
*
 * Goal
 * Run or read
*
 * first tensor example
 * [TensorBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean)
*
 * first training example
 * `lake exe torchlean mlp --device cpu --steps 10`
*
 * first autograd example
 * [AutogradBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)
*
 * first graph example
 * `lake exe verify -- torchlean-ibp`
*
 * first BugZoo example
 * `NN.Examples.BugZoo.All`
*
 * first CUDA runtime check
 * `lake -R -K cuda=true exe torchlean mlp --device cuda --steps 20`
*
 * first scientific ML example
 * `lake exe verify -- ode` or the PINN/FNO examples
:::

:::table
*
 * Artifact
 * Example family
 * How to read the result
*
 * printed tensor
 * quickstart tensor examples
 * value and shape inspection
*
 * loss trace
 * MLP/CNN/ViT/GPT/FNO/generative/RL commands
 * runtime and optimizer sanity check
*
 * JSON log
 * model and RL commands
 * widget-readable training artifact
*
 * graph / IR
 * GraphSpec, compiled runtime, verification commands
 * verifier/export boundary
*
 * certificate or bounds
 * `lake exe verify -- ...`
 * checker acceptance, not training quality
*
 * image sample
 * diffusion/autoencoder-style examples
 * sampling artifact, not a theorem
*
 * rollout or policy artifact
 * PPO examples
 * environment-boundary and behavior inspection
:::

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
  `import NN.API`, `open TorchLean`, and stick to `nn`, `Data`, `Trainer`, and `optim` unless
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
