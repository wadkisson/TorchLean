import VersoManual

open Verso.Genre Manual

#doc (Manual) "Native Boundaries" =>
%%%
tag := "gpu-cuda"
%%%


When the running MLP evaluates

$$`y=W x+b,`

the mathematical specification sees matrix-vector multiplication and addition. The CUDA runtime
sees something quite different: contiguous buffers, dimensions, an execution stream, a matrix
kernel, a broadcast, and several error checks. TorchLean keeps both views and records the contract
at the boundary between them.

This chapter follows one training step all the way to the GPU. CUDA is the maintained accelerator
today; the capsule and target types are intentionally not CUDA-specific, so a future Metal, ROCm,
TPU, or custom-chip provider can state the same kinds of obligations without pretending that its
implementation already exists.

# Build A Native CUDA Runtime

An ordinary CPU build compiles stub archives for CUDA symbols. The stubs let the package link on a
machine without the NVIDIA toolchain, but they reject CUDA session creation. To compile the native
implementation:

```
lake build -R -K cuda=true
```

Run this command from the repository root. `-R` rebuilds targets affected by the Lake configuration,
and `-K cuda=true` selects the CUDA source and link configuration. The build compiles TorchLean's
CUDA code and links the CUDA runtime, cuBLAS, and cuFFT where those libraries are used.

Now run two optimizer steps and print the selected kernel contracts:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 2 --seed 2026 --show-backend
```

The model reports the same 25-example dataset as the CPU run. The backend report includes:

```
matmul: native_cuda.matmul
  provider=native-cuda trust=checked vjp=backend-vjp
  numeric=[round=nearest-even,
           subnormal=implementation-defined,
           contract=implementation-defined,
           reduce=implementation-defined]

add: native_cuda.add
  provider=native-cuda trust=checked vjp=backend-vjp

relu: native_cuda.relu
  provider=native-cuda trust=checked vjp=backend-vjp

mse_loss: native_cuda.mse_loss
  provider=native-cuda trust=checked vjp=backend-vjp
```

This output is worth reading closely. The public model contains two *linear layers*, but the runtime
decomposes them into reshape, permutation, matrix multiplication, broadcast, and addition. Capsules
describe the operations that actually crossed a backend boundary, not only the layer names in
source code.

# The Path Of One Linear Layer

For an unbatched input `x : [2]` and weight `W : [8,2]`, the eager CUDA path proceeds roughly as:

```
typed Tensor [2]
    ↓ upload / existing CUDA handle
opaque contiguous float32 buffer
    ↓ reshape and matrix-layout preparation
native_cuda.matmul
    ↓ broadcast bias [8] to output shape
native_cuda.add
    ↓
typed runtime Tensor [8]
```

The Lean wrapper knows the logical shapes and element counts. The device buffer is opaque; Lean
does not inspect its contents by reducing a theorem. Before each FFI call, wrappers check the
conditions they can observe:

- the session really targets CUDA;
- the selected capsule implements the requested operation;
- the buffer is live and allocated on the expected device;
- flat lengths match the logical shapes;
- ranks, axes, and operation-specific dimensions are supported.

Those checks prevent many ABI and memory errors. They do not prove that a CUDA thread computes the
correct arithmetic expression.

# What A Kernel Capsule Contains

A `KernelCapsule` is the audit record for one operation-provider pair:

```
structure KernelCapsule where
  name             : String
  op               : BackendOp
  provider         : Provider
  device           : Device
  trustLevel       : TrustLevel
  supportsForward  : Bool
  vjpMode          : VJPMode
  shapeContract    : ContractDescriptor
  layoutContract   : ContractDescriptor
  valueContract    : ContractDescriptor
  vjpContract      : ContractDescriptor
  numericalPolicy  : NumericalPolicy
```

# Crossing Lean's Boundary

TorchLean does not require every useful program to be rewritten in Lean. PyTorch can capture a
model, an interval library can propose a numerical enclosure, and a CUDA kernel can compute a
tensor. The engineering question is not whether external code exists; it is what Lean learns when
that code returns.

There are two main boundaries:

- a *subprocess* exchanges files, standard output, or JSON with another executable;
- an *FFI call* invokes a linked native symbol and may exchange opaque memory handles.

Both can be used safely. They have different failure modes and support different proof stories.

# A Subprocess Is An Untrusted Producer

The common process helper is small:

```
def runJsonStdoutChecked
    (ctx : String)
    (cmd : String)
    (args : Array String)
    (cwd : Option String := some ".") :
    IO Json := do
  let stdout ← runStdoutChecked ctx cmd args cwd
  match Json.parse stdout with
  | .ok value => pure value
  | .error message =>
      throw <| IO.userError
        s!"{ctx}: JSON parse error: {message}\nstdout:\n{stdout}"
```

`runStdoutChecked` starts the process, captures its streams, and rejects a nonzero exit code with the
command, arguments, status, and standard error in the diagnostic. `runJsonStdoutChecked` then
requires all of standard output to be one JSON document.

Suppose Python prints:

```
{"format":"torchlean.bound.v1","lower":0.12,"upper":0.31}
```

Successful parsing establishes only that this text is valid JSON. A useful checker must still:

1. require the exact `format` string;
2. require finite numeric fields;
3. establish `lower ≤ upper`;
4. connect the interval to a particular graph, payload, input set, and output;
5. invoke a sound acceptance theorem.

Changing `upper` to `1e999` is a good boundary test. JSON syntax accepts the number, but converting
it to a machine `Float` can produce infinity. Verification parsers therefore use
`expectFiniteFloatE` or `expectFieldFiniteFloatE`, not the permissive float parser, whenever the
certificate schema promises finite claims.

# A Complete PyTorch Capture

Run:

```
lake exe torchlean pytorch_export_check
```

The command asks Python and `torch.export` to capture a small Add+ReLU module, emit
`torchlean.ir.v1` JSON, parse it in Lean, and run graph validators. Broader modules (linear,
conv, attention) and negative unsupported-op tests are still present in the checker's Python model
file for local debugging, but their exported shapes / rejection paths are not yet stable enough to
gate the documented green path.
