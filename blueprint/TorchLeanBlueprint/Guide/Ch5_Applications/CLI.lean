import VersoManual

open Verso.Genre Manual

#doc (Manual) "Command-Line Tools" =>
%%%
tag := "cli"
%%%

The public command line has three jobs: run model examples, run individual Lean example files, and run
verification tools. Everything else should be treated as internal unless a page points to it
explicitly.

TorchLean keeps that public command set focused. Internal scripts stay internal unless a guide page
points to them. The first few commands cover the common cases directly:

- `lake exe torchlean <example> [args...]` for the main model examples,
- `lake env lean --run NN/Examples/.../Foo.lean -- [args...]` for files in the
  [examples tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/),
- `lake exe verify -- ...` for verification workflows and checkers.

Three command families cover most everyday use: run a model example, run a direct example file,
and run a verifier tool.

| Command | Purpose |
|---|---|
| `lake build` | build the project |
| `lake build NN.Examples.Zoo` | build curated examples |
| `lake exe torchlean --help` | list model examples |
| `lake exe torchlean <example>` | run a model example |
| `lake env lean --run NN/Examples/.../Foo.lean` | run a direct example file |
| `lake exe verify -- list` | list verifier tools |
| `lake exe verify -- <tool>` | run a verifier/checker |
| `scripts/docs/build_site.sh` | build the website/docs |

The runner accepts runtime flags either before or after the subcommand. These are equivalent:

```
lake exe torchlean mlp --device cpu --steps 10
lake exe torchlean --device cpu mlp --steps 10
```

Use the first form in prose and scripts because it reads like "run this example with these flags."
Use a leading `--` separator only when another wrapper needs it:

```
lake exe torchlean -- mlp --device cpu --steps 10
```

# Building with CUDA (optional)

GPU-backed examples require a CUDA-enabled build of the Lean project so the native archives in the
[CUDA source tree](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/) link against the toolkit:

```
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1
```

If `cuda=true` is not set, the same symbols resolve to CPU stubs and `--device cuda` may error or fall back
depending on the example. See *GPU and CUDA* for the build/runtime split and each example's module
header for model specific flags. Verification CLI tools (`lake exe verify`) do not need CUDA.

# The Two Commands To Remember

Run one model example:

```
lake exe torchlean <example> [args...]
```

Run one direct example file:

```
lake env lean --run NN/Examples/.../Foo.lean -- [args...]
```

List verification workflows:

```
lake exe verify -- list
```

Run one verification workflow:

```
lake exe verify -- <tool> [args...]
```

That covers most of the public CLI. The usual loop is list, pick a name, run it, then dig into source
only when output or errors warrant it.

Common failure modes are usually simple: CUDA examples need `-K cuda=true`; real-data examples need
the dataset files under `data/real`; some verifier tools need an external artifact; and Python
producer workflows need their Python dependencies installed before Lean can check the exported
artifact.

# Command Families

The `torchlean` runner is intentionally broad, but the names still fall into a few useful groups.
Check `lake exe torchlean --help` for the current list.

| Family | Examples | What the command establishes |
|---|---|---|
| quickstart | `quickstart_tensors`, `quickstart_autograd`, `quickstart_mlp` | public API smoke path |
| supervised / vision | `mlp`, `kan`, `cnn`, `vit` | training loop, data path, optimizer, backend |
| sequence | `rnn`, `lstm`, `transformer`, `gpt2`, `text_gpt2`, `chargpt`, `mamba` | token windows, recurrence, attention, scan state |
| scientific ML | `fno1d_burgers` | prepared scientific data and spectral convolution |
| generative | `autoencoder`, `mae`, `vae`, `vqvae`, `gan`, `diffusion` | reconstruction, latent objectives, denoising, sampling artifacts |
| reinforcement learning | `ppo_gridworld`, `ppo_cartpole`, `ppo_pong_ram`, `dqn_replay` | environment boundary, rollout/replay data, policy/value losses |
| interop / deep dives | `pytorch_roundtrip`, `pytorch_export_check`, `graphspec`, `torch_ir_pytorch` | graph and external artifact workflows |
| numerics | `float32_modes`, `floats_arb_ieee_compare` | finite precision inspection |

This table is a claim-scope table, not a promise that every command proves a theorem. A successful
model command establishes an executable run through the selected runtime path. A verifier command
establishes checker acceptance for a named artifact. A theorem must still be cited by name when the
claim depends on proof support.

# A Good First Check Test

After cloning, a fast runtime check is:

```
lake build
lake build NN.Examples.Zoo
lake env lean --run NN/Examples/Quickstart/TensorBasics.lean
lake exe torchlean mlp --device cpu --steps 10
lake exe verify -- torchlean-ibp
```

Those commands answer four different questions:

- does the project build?
- does the typed-tensor layer work?
- does the public model runner and training API feel reasonable?
- does the verification pipeline complete?

This sequence is a confidence check rather than full coverage: within a minute or two it shows
whether the project builds, whether a small training run works, and whether the verifier command is
present.

# Examples In Practice

Model examples use the `torchlean` runner:

```
lake exe torchlean --help
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
```

Some tutorial and deep dive examples are ordinary Lean `--run` programs under `NN/Examples/*`. Pick
one and run it directly:

```
lake env lean --run NN/Examples/Quickstart/TensorBasics.lean
```

For data-backed model runs, prepare the public example datasets first:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --tinystories-valid --cifar10
```

Some examples have specialized data or artifact helpers:

```
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/images \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 --labels-from-dirs --limit 2000
```

When a command writes artifacts, prefer explicit paths in documentation:

```
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
  --steps 10 --windows 1 --generate 64 --log data/model_zoo/gpt2_trainlog.json

lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 \
  --n-total 1 --steps 1 --hidden-c 1 --T 2 \
  --sample-ppm data/model_zoo/cifar_sample.ppm
```

The path is part of the workflow. It tells readers which object a widget, plotter, or later checker
is meant to inspect.

# Verification In Practice

The verification side has the same small feel: list the registered tools, then run one directly.

- Show registered verification tools:

```
lake exe verify -- list
```

- Run the smallest IR-to-bounds workflow:

```
lake exe verify -- torchlean-ibp
```

- Run the CROWN operator example:

```
lake exe verify -- torchlean-crown-ops
```

These commands are plain, so the verification workflow is easy to invoke without
knowing the internal directory structure of the repository.

# Direct Lean Files

The runner covers the curated examples, but direct files are still useful when a tutorial is meant
to be read beside its source. The command shape is:

```
lake env lean --run NN/Examples/Quickstart/AutogradBasics.lean -- --dtype float
```

Use direct files for small, source-local demonstrations. Use `lake exe torchlean ...` when the
example has a public subcommand and should be cited as part of the model suite.

# Reading Failures

Most failed application commands are informative if read through the right boundary:

- *unknown example*: run `lake exe torchlean --help` and use the registered subcommand name;
- *missing file under `data/real`*: run the documented data helper or pass a smaller synthetic-data flag;
- *CUDA symbol or device failure*: rebuild with `lake -R -K cuda=true build` and keep `-K cuda=true`
  on the executable command;
- *Gymnasium import/server failure*: install the Python dependency and check the external environment name;
- *verifier artifact failure*: distinguish "artifact missing" from "artifact present but rejected";
- *widget file view is empty*: confirm the example wrote the JSON/CSV/PPM path the widget is reading.

These are not merely operational tips. They also mark trust boundaries. A dataset converter, a Python
environment, a CUDA kernel, and an external certificate producer are different kinds of dependencies,
and must be named separately.

# Website Build

The website combines the API reference, this book, and the homepage. The local build command is kept in
the repository:

```
scripts/docs/build_site.sh
```

That script builds the API reference with equation rendering disabled, rebuilds the Verso guide (from
the `blueprint/` package), and installs the homepage bundle.

For more runnable examples, see *Example Walkthroughs*. For what `verify` subcommands do internally, see
*Verification*.

# Checking Guide Edits

Each guide file is a Lean source file. During development, a focused check gives faster feedback:

```
lake env lean blueprint/TorchLeanBlueprint/Guide/Ch5_Applications/CLI.lean
lake env lean blueprint/TorchLeanBlueprint/Guide/Ch5_Applications/Examples.lean
lake env lean blueprint/TorchLeanBlueprint/Guide/Ch6_Conclusion/Conclusion.lean
```

Pages that import executable widget or runtime modules should be checked directly. Command tables
should be compared with `lake exe torchlean --help` so that the published flags match the runner.
