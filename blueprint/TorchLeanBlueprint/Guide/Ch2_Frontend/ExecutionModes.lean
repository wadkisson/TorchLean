import VersoManual

open Verso.Genre Manual

#doc (Manual) "Execution And Backends" =>
%%%
tag := "execution-modes"
%%%


Start with the same MLP:

$$`F_\theta:[2]\to[1]`

Its type stays `[2] → [1]` whether it runs eagerly on the CPU, through a compiled graph, or with
native CUDA kernels. We can therefore change the runtime one choice at a time without rebuilding
the architecture.

TorchLean separates four choices:

1. scalar semantics;
2. eager or compiled execution;
3. device/provider profile;
4. training or evaluation mode.

The easiest way to understand the choices is to run them.

# Ask The Runner

The example runner documents the current surface:

```
lake exe torchlean --help
```

For one command:

```
lake exe torchlean quickstart_mlp --help
```

The common flags are:

```
--dtype float|ieee754exec
--backend eager|compiled
--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external
--seed N
--show-backend
```

The parser knows more device names than the current runtime implements. CPU and CUDA have maintained
profiles today; the other names reserve a clean place for future providers. Asking for one of them
gets an error rather than a suspiciously successful CPU run.

For an interactive device prompt:

```
lake exe torchlean --choose quickstart_mlp --steps 20
```

The prompt is opt-in so scripts and CI never block waiting for input.

# Experiment 1: CPU Eager

```
lake exe torchlean quickstart_mlp \
  --dtype float \
  --backend eager \
  --device cpu \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

Eager execution creates a session and records operations as the model runs. Every operation asks the
profile for an admissible capsule, executes its provider, and appends a local VJP rule when gradients
are required.

On CPU, the maintained profile selects portable reference capsules. The report lets you verify that
the requested CPU path actually ran.

Use eager mode when:

- operation structure depends on runtime values;
- inspecting a tape or provider selection;
- executing the broadest dynamic frontend;
- using the maintained CUDA runtime.

# Experiment 2: CPU Compiled

```
lake exe torchlean quickstart_mlp \
  --dtype float \
  --backend compiled \
  --device cpu \
  --steps 20 \
  --seed 2026
```

Compiled execution records the fixed scalar-loss program once, including forward, JVP, and VJP
behavior, and replays it with current parameters and data.

Compare the initial loss between eager and compiled using the same seed. It should agree for the
supported deterministic program. Then compare final parameters or predictions, not only
six-decimal loss summaries, because different execution orders can hide small discrepancies.

The current compiled trainer is CPU-only. It does not consume an accepted backend graph plan and it
does not mean CUDA Graph capture. A CUDA plus compiled request fails explicitly.

# Experiment 3: Native CUDA

Build TorchLean with the CUDA Lake configuration and run two training steps on device:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 2 --seed 2026 --show-backend
```

`-R` rebuilds targets affected by the Lake configuration, and `-K cuda=true` selects the CUDA
source and link flags. The command requires a working `nvcc` and CUDA toolkit; on machines without
CUDA, keep Experiments 1–2 on `--device cpu` and treat this section as documentation of the
maintained GPU path. When the build succeeds, the backend report should name native CUDA capsules
for supported operations such as `matmul`.

# Inside The Backend Planner

The previous page selected CPU, CUDA, or an optional provider from the public API. Now we can look
at the less visible question: when a graph asks for matrix multiplication, attention, or a
reduction, how does TorchLean decide which implementation is allowed to answer?

A device name is not enough. One CUDA build may contain a hand-written kernel, a cuBLAS call, and a
LibTorch bridge for different operations. Their layouts, numerical behavior, backward support, and
supporting evidence differ. The backend planner keeps those differences in data and either returns
an accepted plan or explains why it could not make one.

The path is:

$$`
\text{operation}+\text{profile}+\text{available providers}
\longrightarrow \text{capsule}
\longrightarrow \text{audit}
\longrightarrow \text{accepted kernel}.
`

That path, rather than another tour of command-line flags, is the subject of this chapter.

# Kernel Capsules

Suppose a graph reaches scaled dot-product attention. TorchLean currently knows three maintained
ways to compute it: a composed TorchLean expression, a native fused CUDA implementation, and a
LibTorch forward bridge with a TorchLean-owned backward pass. The operation is the same; the
implementation contract is not.

A `KernelCapsule` records those differences:

```
structure KernelCapsule where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  supportsForward : Bool
  vjpMode : VJPMode
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
  numericalPolicy : NumericalPolicy
  notes : String
```

Capsules are declared before the run and registered with the backend. The planner may select one
only when its device, provider, gradient mode, and assurance level fit the requested profile. If no
capsule fits, planning stops with an error.

Capsules are collected in named `CapsuleModule`s. Built-in attention, native CUDA, portable
reference, and optional LibTorch code contribute modules to the same registry. A downstream
provider can prepend another module with `BackendProfile.withCapsuleModules`; it does not add a new
model class or a branch to the graph walker. The model still lowers to ordinary `BackendOp`s, and
the planner either finds an admissible capsule for each operation or reports the missing operation.
Adding a module with an existing name replaces that module. Planning rejects repeated module names
and repeated capsule identities, so provider precedence cannot change through accidental duplicate
registration.

`BackendOp` names semantic operation families such as matrix multiplication, reduction, pooling, or
convolution. Rank, axes, padding, strides, and index tensors remain in the graph payload. This keeps
capability discovery general without erasing the information needed to state the operation
correctly.

```
#check NN.Backend.Registry.CapsuleModule
#check NN.Backend.BackendProfile.withCapsuleModules
```
