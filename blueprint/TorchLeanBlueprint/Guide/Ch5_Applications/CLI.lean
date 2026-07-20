import VersoManual

open Verso.Genre Manual

#doc (Manual) "Command-Line Tools" =>
%%%
tag := "cli"
%%%

The public command line runs model examples, individual Lean example files, and verification tools:

- `lake exe torchlean <example> [args...]` for the main model examples,
- `lake env lean --run NN/Examples/.../Foo.lean -- [args...]` for files in the
  [examples tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/),
- `lake exe verify -- ...` for verification workflows and checkers.

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

# Building with CUDA

GPU-backed examples require a CUDA-enabled build of the Lean project so the native archives in the
[CUDA source tree](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/) link against the toolkit:

```
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1
```

Without `cuda=true`, CUDA symbols resolve to stubs so CPU builds remain portable. An explicit
`--device cuda` request then fails instead of silently changing the requested device. See *GPU and
CUDA Boundaries* for the build/runtime split. Verification CLI tools (`lake exe verify`) do not
require CUDA unless a particular producer workflow says otherwise.

# Shared Parsers

Command authors can import `NN.API.CLI` without loading the tensor or runtime stack. Its definitions
live in the canonical `TorchLean.CLI` namespace:

```
import NN.API.CLI

open TorchLean

#check CLI.takeFlagValueOnce
#check CLI.takeNatFlagOnce
#check CLI.takeBoolFlagOnce
#check CLI.checkNoArgs
#check CLI.orThrowIO
```

The shared parsers accept both `--key value` and `--key=value`, reject duplicate flags, and return
unconsumed arguments so each command can reject misspellings rather than ignore them.

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

