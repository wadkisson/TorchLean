import VersoManual

open Verso.Genre Manual

#doc (Manual) "GPU and CUDA Boundaries" =>
%%%
tag := "gpu-cuda"
%%%

CUDA belongs in TorchLean for a practical reason: serious ML systems use GPUs. If TorchLean only
worked for small CPU examples, it would miss many places where semantic mistakes actually happen:
fused attention, reductions, cuBLAS GEMM, cuFFT, selective scans, spectral convolution, and
device-side randomness.

The design is therefore pragmatic. CUDA accelerates supported Float32 runtime paths, while the
Lean side spec and graph semantics remain the reference objects. Claims about native execution go
through runtime agreement statements, parity tests, sanitizer checks, or future kernel-level proofs.

# Backend Choices

The CUDA design follows a few choices that are easy to miss from the command line alone.

1. *CUDA is opt-in at build time.* A normal `lake build` should work on machines without a GPU or a
   CUDA toolkit. Building with `-K cuda=true` is an explicit decision to link native device code.
2. *CUDA is a runtime backend, not a second semantics.* The Lean model, the spec layer, and the
   verifier IR keep their meaning for supported operations under their stated preconditions. CUDA
   changes where supported float32 tensor work runs.
3. *The backend is focused.* The first target is float32 tensor work used by the model examples:
   matmul, convolution, reductions, attention helpers, FFT/FNO kernels, and related VJP rules.
4. *Runtime agreement is stated explicitly.* TorchLean documents which facts are proved in Lean,
   which behaviors are specified in Lean, and which native behaviors are validated by tests.

A precise CUDA claim has three parts: the Lean side specification, the native implementation path,
and the agreement evidence between them.

# The Short Version

Use CUDA when you want to train or evaluate larger float32 models from Lean:

```
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 20
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003
```

Then run the CUDA test suite:

```
lake -R -K cuda=true exe nn_tests_suite
```

To test the native CUDA boundary, run the same Lean-driven suite under NVIDIA's sanitizer:

```
scripts/checks/cuda_sanitize_tests.sh
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

The flag `-K cuda=true` is a *build* flag. It selects the native CUDA objects in the
[CUDA source tree](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/) and links CUDA libraries such as cuBLAS and cuFFT. The
command-line flag `--device cuda` is a *runtime* flag. For long training runs, model commands also expose:

```
--cuda-mem-watch N
```

That flag samples the CUDA allocator every `N` optimizer updates. It reports live and peak runtime
allocation state and warns if the observed free-memory trend would exhaust the device before the
requested run length. When a long CUDA run does not pass an explicit cadence, the public model
examples choose a small default number of samples. The feature lives in the public runner interface,
not in a separate benchmark script, so MLP, CNN, GPT, ViT, and other model commands can report the
same kind of long-run memory signal.

If either piece is missing, TorchLean should fail loudly instead of silently claiming that GPU
execution happened.

# What Lives In The CUDA Backend

The native backend is bounded enough to audit:

- [`tensor/`](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/tensor/)
  owns raw float32 buffers and elementwise tensor kernels;
- [`blas/`](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/blas/)
  provides cuBLAS-backed matrix multiplication;
- [`conv_pool/`](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/conv_pool/)
  provides convolution and pooling kernels;
- [`kernels/`](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/kernels/)
  contains reductions, broadcasting, gathers/scatters, attention helpers, selective scan, FFT, and
  fused spectral convolution kernels;
- [`common/`](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/common/)
  contains shared shape, RNG, error-checking, and deterministic-reduction helpers.

On non-CUDA builds, corresponding stub files are compiled instead.  Those stubs are not fake
success paths; they are CPU reference implementations used by tests and by machines without a CUDA
toolkit.

# ATen, CUDA Runtime, And Lean Proofs

TorchLean may interoperate with PyTorch artifacts, but the proof boundary is not PyTorch's
dispatcher. PyTorch's ATen layer is the C++ tensor library under the Python API; it dynamically
dispatches operations to CPU, CUDA, and other backends. That makes ATen a useful runtime and import
producer, not a Lean semantic oracle.

Read the boundary as a ledger:

- *ATen / PyTorch*: may produce checkpoints, op names, shapes, and reference runs. Lean accepts only
  the imported subset that is parsed, shape checked, lowered to `NN.IR.Graph`, and supplied with
  payloads.
- *CUDA runtime*: allocates device memory, launches kernels, copies buffers, synchronizes streams,
  and reports errors. Lean sees opaque handles and returned bits, not the device execution trace.
- *TorchLean CUDA kernels*: implement selected float32 tensor operations behind FFI symbols. Their
  Lean-side contracts live in `KernelSpec`, `Float32Contract`, and operation-specific specs.
- *cuBLAS / cuFFT / libdevice*: are vendor libraries. They are engineering dependencies and
  documented runtime assumptions unless a narrower checker or proof discharges a specific contract.
- *Lean proofs*: reason about specs, graphs, scalar reference models, and theorems that transport
  native results back into those models once the named agreement assumptions are supplied.

This means a PyTorch or ATen run can be evidence for a regression test, and a CUDA run can be
evidence that the native path executed. Neither is, by itself, a proof about `Graph.denote`,
`IEEE32Exec`, or `FP32`.

# Implemented CUDA Surface

The CUDA work in TorchLean is more than a single "move tensors to GPU" hook.  It is a layered
runtime path for float32 autograd, with the fast pieces placed behind explicit FFI symbols and the
specification pieces kept in Lean.

The main implemented pieces are:

- *opaque CUDA buffers and a CUDA eager tape*: Lean records the shapes, tape nodes, parent links,
  and VJP wiring; native code owns allocation, device memory, kernel launches, and buffer
  reads/writes;
- *host/device conversion*: Lean `Float` values are uploaded through the float32 contract, device
  results are downloaded through raw binary32 bits, and tests compare both CUDA and stub paths;
- *elementwise tensor kernels*: add, multiply, divide, square root, exp/log-style helpers,
  broadcasting, masks, gathers, scatters, concatenation, slices, and related VJP rules;
- *reductions*: fast reductions for training and deterministic reduction contracts for paths where
  bitwise replay matters;
- *cuBLAS-backed GEMM/BMM*: matrix multiplication and batched matrix multiplication use vendor BLAS,
  while Lean records the row-major shape and indexing contract around the column-major cuBLAS API;
- *convolution and pooling*: 2D and rank-generic convolution, transposed convolution, max pooling,
  average pooling, smooth max pooling, and their backward paths;
- *deterministic RNG kernels*: uniform and Bernoulli/dropout-style masks use the same SplitMix64
  stream convention as the CPU path, keyed by explicit seeds and linear indices;
- *Fused attention*: the runtime can call a fused native attention forward and
  fused VJP instead of materializing scores, mask application, softmax, and value multiplication,
  while the Lean spec proves
  the fused contract equal to ordinary scaled dot product attention;
- *selective scan / Mamba support*: lower layer scan kernels support the Mamba recurrent path,
  while the higher layer keeps a pure CPU/CUDA compatible definition at the model API;
- *FFT and FNO kernels*: the FNO path uses cuFFT plus fused spectral multiplication and explicit
  backward kernels, with a dense CPU DFT reference for comparison;
- *sanitizer and parity harnesses*: the CUDA suite compares native kernels, CPU fallbacks, finite-difference
  gradients, deterministic replay, and model-level examples.

That inventory matters because each item has a different agreement shape. Elementwise kernels can
be tied to pointwise `IEEE32Exec` assumptions. Reductions need an additional order assumption.
cuBLAS and cuFFT need vendor-library assumptions. Fused attention and FNO need both algorithmic
specs and FFI agreement.

The boundary can be read family by family:

- *Elementwise maps*: Lean side meaning is `mapSpec` / `map2Spec`; native agreement is pointwise
  primitive-bit agreement; tests compare CUDA, stubs, and finite scalar cases.
- *Reductions and dot products*: Lean side meaning is a fixed reduction spec such as
  `reduceSumLeftSpec`; native agreement must fix or document accumulation order; tests check
  deterministic reduction paths.
- *GEMM/BMM*: Lean side meaning is `bmmSpec` plus row-major shape conventions; native agreement
  includes cuBLAS layout, accumulation, and FMA behavior; tests compare against CPU references.
- *Convolution and pooling*: Lean side meaning is the tensor-index and VJP contract; native
  agreement covers indexing, padding, and layout; tests cover forward and backward paths.
- *FFT/FNO*: Lean side meaning is the spectral convolution contract; native agreement covers cuFFT
  layout, normalization, and fused spectral multiplication; tests include dense DFT references and
  finite-difference checks.
- *Fused attention*: Lean side meaning is the FlashAttention-style spec equal to SDPA; native
  agreement is that the fused CUDA kernel implements that fused spec; tests compare attention
  forward/backward behavior.

NVIDIA's floating-point notes emphasize the same practical issue: GPU arithmetic may follow
IEEE-style rules for individual operations, but fused multiply-add, parenthesization, thread counts,
library choices, and compute capability still affect a numerical claim. TorchLean keeps those cases
separate because they are separate assumptions.

# The Lean Side Contract

The CUDA boundary is more than "some C++ code was linked." The Lean side gives names to the pieces of
the contract.

```
import NN.Runtime.Autograd.Engine.Cuda.Float32Contract
import NN.Runtime.Autograd.Engine.Cuda.KernelSpec
import NN.Runtime.Autograd.Engine.Cuda.NativeSources

namespace Runtime.Autograd.Cuda.Float32Contract
#check RefScalar
#check NativePrimitiveBits
#check NativePrimitiveAgreement
#check native_add_eq_ieee32
#check native_fma_eq_ieee32
#check native_add_abs_error_of_isFinite
#check native_sqrt_abs_error_of_isFinite
end Runtime.Autograd.Cuda.Float32Contract

namespace Runtime.Autograd.Cuda.KernelSpec
#check FlatBuffer
#check NativeBitsBuffer
#check mapSpec
#check map2Spec
#check reduceSumLeftSpec
#check NativeReduceAgreement
#check bmmSpec
#check NativeBmmAgreement
end Runtime.Autograd.Cuda.KernelSpec
```

The CUDA contract has three named pieces:

- `KernelSpec` says what owned kernels mean as pure finite-index computations.
- `Float32Contract` says how native primitive bits are expected to line up with `IEEE32Exec`.
- `NativeSources` keeps a Lean map from external symbols to source files under `csrc/cuda`.

The CUDA contract stays small enough to inspect: a reader can find the named spec, test the runtime
path, and see which agreement assumption is being used.

# Assumptions, Axioms, And Runtime Agreement

When we say "axiom" for CUDA here, we mean an explicit named assumption, not an unrestricted claim
that any GPU result is correct. If a concrete native result satisfies the named agreement contract,
then Lean theorems can transport that result back into the proved float32/spec layer.

The scalar boundary is:

```
import NN.Runtime.Autograd.Engine.Cuda.Float32Contract

namespace Runtime.Autograd.Cuda.Float32Contract
#check NativePrimitiveAgreement
#check native_mul_abs_error_of_isFinite
end Runtime.Autograd.Cuda.Float32Contract
```

`NativePrimitiveAgreement` is the scalar assumption.  It says that native float32 primitive result
bits for add, multiply, divide, fused multiply-add, and square root match the executable
`IEEE32Exec` reference. Once that assumption holds, Lean can reuse the proved `IEEE32Exec` error
bounds. The theorem has the form: given bit agreement with the reference scalar operation, the
reference scalar operation has the proved binary32 error bound.

Lifting from scalar operations to kernels adds more assumptions:

- *elementwise kernels*: every output element's native bits must match the pointwise Lean spec;
- *fixed-order reductions*: the native accumulation order must match `reduceSumLeftSpec`, or a
  separate documented reduction spec must be used;
- *scatter-add and atomic reductions*: repeated indices and atomics are only bitwise deterministic
  under a fixed accumulation order; otherwise they are training kernels, not replay proofs;
- *BMM/GEMM*: row-major TorchLean buffers must be interpreted consistently around cuBLAS, and the
  accumulation/FMA behavior must match the selected reference contract if bitwise proof transport is
  desired;
- *cuFFT/FNO*: cuFFT normalization, half-spectrum layout, omitted modes, and real/imaginary weight
  layout are part of the native agreement contract;
- *Fused attention*: Lean proves the fused attention spec equals SDPA, while the
  native fused kernel is trusted/validated to implement that fused spec;
- *libdevice/transcendentals*: functions outside IEEE 754's basic arithmetic contract are treated as
  toolchain/library assumptions unless separately specified and tested.

The remaining runtime base is ordinary systems software: Lean's FFI marshalling, the C/CUDA
compiler, CUDA runtime and driver, GPU hardware, cuBLAS, cuFFT, libdevice, build flags, and the path
from source to binary. Tests and sanitizer runs validate that base.

# A Concrete GEMM Boundary

Matrix multiplication is the smallest example where all layers are visible.

At the spec level, the operation is an indexed sum. At the graph level, a `.linear` or BMM node
denotes that indexed computation through `Graph.denote` and the parameter payload. At the CUDA
runtime level, TorchLean stores row-major float32 buffers and calls cuBLAS for batched matrix
multiplication.

The native call is not the theorem. The agreement obligation includes:

- dimensions fit both TorchLean's shape discipline and cuBLAS' `int`-shaped API;
- row-major TorchLean buffers are interpreted correctly around cuBLAS' column-major convention;
- the chosen cuBLAS routine, math mode, toolkit version, device architecture, and stream/handle
  policy produce the result bits promised by the selected runtime contract;
- the accumulation behavior is the one named by the contract, or the claim is weakened to a tested
  runtime result rather than bitwise proof transport;
- finite-path hypotheses hold if the result is being connected to `FP32` real-error theorems.

This is why the source comment in `torchlean_cublas_common.h` matters: it records the row-major /
column-major boundary where a silent transpose bug would otherwise look like an ordinary matrix
multiply. It is also why cuBLAS reproducibility is cited as a library contract rather than assumed
from the phrase "single precision GEMM."

# Boundary Rationale

The CUDA decisions are conservative.

First, Lean owns the pieces it can inspect directly: shapes, indices, pure tensor specs, scalar
reference semantics, graph/IR semantics, and theorems that say "if native bits agree with this spec,
then the proved semantic result follows." Native code enters through the corresponding runtime
agreement.

Second, TorchLean still needs to run real models. Proving every GPU instruction before using CUDA
would make the system unusable for training. The compromise is practical: use CUDA for speed, keep
the mathematical contract in Lean, and require tests, sanitizers, and parity checks at the boundary.

Third, we avoid silent semantic changes. CUDA is a runtime backend, not a new meaning for the
model. The same model API and IR should describe CPU eager, CUDA eager, and compiled execution.
So `--device cuda` stays narrow: it changes where work runs, not what the operation is supposed to mean.

Fourth, we started with float32 because it covers the training examples, has an executable
`IEEE32Exec` bridge, and keeps the native-bit agreement contract tractable. Any additional numeric
mode needs its own equally explicit contract.

Fifth, the fused kernels are correctness first. The attention kernel follows the FlashAttention
shape, but the contract we state is fused SDPA with a fused VJP. A stronger claim about a production
IO-tiled FlashAttention implementation would require a separate native-kernel proof or conformance
story.

# Runtime Shape

At the Lean level, CUDA execution is a specialized eager tape, implemented by the
[CUDA runtime modules](https://github.com/lean-dojo/TorchLean/tree/main/NN/Runtime/Autograd/Engine/Cuda/):

```
Runtime.Autograd.Cuda.Buffer
Runtime.Autograd.Cuda.Tape
Runtime.Autograd.Cuda.Tape.matmul
Runtime.Autograd.Cuda.Tape.spectralConv1dRfft
```

The public training API usually hides those names. A model example command such as

```
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

still looks like an ordinary TorchLean run. Under the hood, tensors are stored in CUDA buffers and
the local VJP rules call CUDA kernels.

CUDA remains an eager runtime backend. Verification passes consume the shared IR described in
*Graphs and IR*; they are not proofs of a particular GPU schedule.

CUDA buffers do not have Lean's ordinary value semantics. A forward and backward step creates tape
values, gradient storage, and scratch buffers for matmul, convolution, normalization, attention,
and spectral convolution. Returned tensors stay alive; scratch storage is released once the kernel
has finished with it. The public runners report allocator counts during long jobs so that a leak
appears as a trend in the terminal instead of, much later, as an unexplained allocation failure.

A bug found during FNO testing shows why this distinction matters. Reverse mode begins with a
one-element seed. When that seed was created by a pure allocation expression, Lean was free to
share it, even though the CUDA tape consumed and released the native buffer. A later backward pass
could therefore reuse dead storage. `Cuda.Buffer.fullIO` makes allocation effectful and guarantees
a fresh seed; sparse backward also uses copy-and-release operations when ownership moves between
gradient maps.

The stress suite repeats this path and rejects growth in live allocations. CPU parity tests check
the numerical results of selected kernels, and NVIDIA Compute Sanitizer looks for invalid native
accesses and leaks. Separately, Lean theorems describe the `IEEE32Exec` and `FP32` semantics to which
native results are related. Each check answers a different question. A successful training run is
useful evidence about the runtime, but it is not a proof of the compiled CUDA kernel.

# Runtime Initialization

Large Float/CUDA modules should not have to construct every initial parameter as a nested Lean
tensor before the runtime can allocate device storage.  TorchLean therefore exposes runtime
Float initializers:

```
TorchLean.RuntimeInit.FloatInit
TorchLean.RuntimeInit.Plan
TorchLean.Module.instantiateFloatWithPlan
TorchLean.Module.instantiateFloatWithInit
```

The typed `Plan` is indexed by the module's parameter-shape list.  That means Lean checks that every
parameter has exactly one initializer.  In CPU Float mode the initializer materializes a normal host
tensor.  In CUDA mode the supported initializers allocate device buffers directly and keep the host
slot as a synchronized mirror for explicit readback.

Runtime initializers do not change the mathematical semantics.  The parameter is still a tensor of the
declared shape.  The improvement is where storage is materialized: zeros, ones, uniform,
Xavier/Glorot, Kaiming/He, and exact flat payloads can be created through the runtime path instead
of first building a huge Lean object only to upload it immediately.

# Randomness And Reproducibility

CUDA and CPU stubs share the same SplitMix64-based deterministic RNG contract.  For operations such
as `rand_uniform` and `bernoulli_mask`, both paths use the same low 32 bits of
`splitmix64(key + i)`.  The tests cover this because toggling CUDA should not silently change a
seeded experiment.

Reductions require a separate note.  Floating-point addition is not associative, so atomics may
produce order-dependent roundoff.  TorchLean therefore separates fast atomic reductions from
deterministic reduction paths where reproducibility matters.  The tests cover exact repeatability
for the deterministic paths.

# cuBLAS And cuFFT

TorchLean uses vendor libraries where that is the right engineering choice:

- cuBLAS for GEMM;
- cuFFT for real FFT and inverse real FFT;
- native kernels for layout, broadcasting, reductions, and fused model specific operations.

The FNO Burgers example is exactly why this distinction matters. The mathematical operation is
spectral convolution, the practical implementation uses cuFFT, and the proof story needs to know
which part is spec and which part is runtime agreement. The generic API contains a dense
multidimensional real-split reference model. CUDA execution uses a fused real-FFT spectral
convolution:

```
Runtime.Autograd.Cuda.Tape.spectralConv1dRfft
```

That op performs:

1. real FFT along the grid axis,
2. complex spectral multiplication with real/imaginary weights,
3. zeroing of omitted modes,
4. normalized inverse real FFT,
5. an explicit backward rule for input and spectral weights.

The tests include finite-difference checks for that backward rule and a tape-level gradient test.

# FlashAttention Style Fused Attention

Attention is the clearest example of why TorchLean separates a mathematical contract from a fast
backend implementation. On the Lean side, the fused attention spec is proved equal to ordinary
scaled dot product attention. On the runtime side, CUDA may call a fused native kernel that avoids
materializing the whole attention matrix. These are different claims: the first is a Lean semantic
equality, the second is native implementation agreement.

The proof contract is in the [FlashAttention API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Layers/FlashAttention.lean):

```
import NN.Spec.Layers.FlashAttention

namespace Spec
#check FlashAttentionConfig
#check flashAttention
#check flashAttentionBackward
#check onlineSoftmaxTiledAttention_eq_scaledDotProductAttention
#check flashAttention_eq_scaledDotProductAttention
#check flashAttentionBackward_eq_scaledDotProductAttentionBackward
#check cudaLoopFlashAttention_eq_scaledDotProductAttention
end Spec
```

Those names identify the contract:

- `FlashAttentionConfig` records tiling metadata as scheduling information.
- `flashAttention` is the fused forward contract.
- `flashAttentionBackward` is the fused VJP contract.
- the equality theorems say that the fused spec denotes ordinary scaled dot product attention.

The native runtime path is separate. The [CUDA kernel API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Cuda/Kernels.lean)
declares the FFI symbols, the
[native CUDA source](https://github.com/lean-dojo/TorchLean/blob/main/csrc/cuda/kernels/torchlean_cuda_kernels.cu) implements them, and the CPU
fallback sits next to it in
[the fallback source](https://github.com/lean-dojo/TorchLean/blob/main/csrc/cuda/kernels/torchlean_cuda_kernels_stub.c).

TorchLean's fused attention kernel is correctness-first: it implements the same masked,
stable scaled dot product attention contract and fused VJP interface. It is a specified fused
operator behind a native boundary, separate from the production IO-tiled Dao-AILab implementation.
The terminology is:

- *Fused attention contract* means the fused operator is specified as equal to SDPA.
- *Native fused attention kernel* means the CUDA backend calls external code through the FFI.
- *Verified CUDA FlashAttention* is the stronger theorem target about the native kernel itself.

The regression test for this boundary is
[NN.Tests.Runtime.Cuda.Attention source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tests/Runtime/Cuda/Attention.lean), which compares
CPU eager and CUDA eager multi-head attention forward/backward behavior and keeps the fused kernels
covered by the CUDA test suite.

# Proved Facts And Validated Behavior

A precise CUDA claim separates Lean proofs, runtime agreement, and validation evidence.

What Lean can prove:

- pure specifications for shapes, indexing, tensor operations, and selected kernel algorithms;
- float32 semantic facts about models defined in Lean, such as `IEEE32Exec` and `FP32`;
- correctness theorems for graph and autograd fragments that are represented inside Lean.

What tests validate:

- native CUDA kernels agree with CPU stubs and small reference cases;
- deterministic paths are repeatable;
- fused kernels have correct local VJP behavior on finite-difference checks;
- model examples learn on real data.

What remains trusted:

- the CUDA compiler, GPU hardware, CUDA runtime, cuBLAS, cuFFT, and libdevice;
- the FFI boundary between Lean and native code;
- toolchain flags and platform behavior.

That split is part of the contract. The boundary should be visible enough that a reader knows where
the theorem ends and the engineering validation begins.

# CUDA Validation Checklist

When changing CUDA code, run at least:

```
lake build NN.Tests.Runtime.Cuda.Fft
lake -R -K cuda=true build NN.Tests.Runtime.Cuda.Fft
lake exe nn_tests_suite
lake -R -K cuda=true exe nn_tests_suite
lake -R -K cuda=true build
```

After touching allocation, indexing, FFT, cuBLAS, convolution, or fused kernels, also run:

```
scripts/checks/cuda_sanitize_tests.sh
```

Use `--all-tools` for a slower pass that includes race, initialization, and synchronization checks.

For model changes, also run the relevant example. For the FNO path, the
[Burgers preparation helper](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/prepare_fno1d_burgers.py) downloads and converts
the data, and the [plot helper](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/plot_fno1d_burgers.py) renders a held-out
prediction:

```
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv
python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
```

# Regression And Sanitizer Scripts

```
scripts/checks/example_regression.sh --cuda
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

# References

- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA, *Floating Point and IEEE 754*:
  https://docs.nvidia.com/cuda/pdf/Floating_Point_on_NVIDIA_GPU.pdf
- NVIDIA cuBLAS documentation: https://docs.nvidia.com/cuda/cublas/
- NVIDIA cuFFT documentation: https://docs.nvidia.com/cuda/cufft/
- PyTorch ATen C++ API documentation: https://docs.pytorch.org/cppdocs/api/aten/index.html
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
  arXiv:2205.14135.
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
  arXiv:2307.08691.
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
  arXiv:2407.08608.
- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019.
