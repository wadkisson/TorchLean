import VersoManual

open Verso.Genre Manual

#doc (Manual) "Floating Point Semantics" =>
%%%
tag := "floats"
%%%

In TorchLean, "Float32" is not one object. It can mean an idealized rounded-real model used in
proofs, an executable IEEE-754 bit model inside Lean, a runtime `Float32` value, or a CUDA `float`
produced by a native kernel. Those meanings are related, but they are not interchangeable.

The numerical objects and their bridges need distinct names.

# The Central Example

A dot product shows the issue immediately. Suppose a model computes:

$$`\sum_i w_i x_i`

Over the reals, this sum has one mathematical value. Over float32, the value also depends on
rounding after each operation, whether multiplication and addition are fused, and the order in which
the sum is evaluated. On a GPU, a parallel reduction may use a different tree from a CPU loop.

A theorem about a floating-point computation has to say which arithmetic it is about. It should
state:

- which semantic scalar is used;
- which reduction or operation order is used;
- which finite/no-overflow side conditions are required;
- and which theorem, checker, or runtime agreement connects the executed value to the proof value.

Those questions define TorchLean's numeric discipline.

# Four Numeric Layers

TorchLean separates four numerical layers:

- *Real specification*: `ℝ` tensors and spec functions. This ideal mathematical model describes
  networks, losses, and verifier inequalities before rounding.
- *Rounded real proof model*: `FP32` / `NF`. This model performs real operations and rounds to the
  binary32 grid, which is the right setting for compositional error proofs.
- *Executable IEEE-style bits*: `IEEE32Exec`. This model stores raw `UInt32` binary32 values and
  includes signed zeros, infinities, NaNs, comparisons, and special rules.
- *Native/runtime execution*: Lean `Float32`, CUDA, cuBLAS, cuFFT, and external tools. These are
  fast producers of values whose agreement with Lean semantics is stated through runtime agreement,
  tests, certificates, or assumptions.

The first three layers are Lean definitions. The fourth layer is the implementation path, so claims
about it require an agreement statement with the Lean side model. TorchLean can then run real
examples while still saying exactly which numerical object a theorem concerns.

# The Generic Representation

The generic theory begins with a radix `β`, where `β ≥ 2`. A `NeuralFloat β` is an integer mantissa
and an integer exponent:

$$`(m,e) \longmapsto m\,\beta^e.`

The Lean representation and its interpretation are:

```
import NN.Floats.NeuralFloat.Core

open TorchLean.Floats

#check NeuralRadix
#check NeuralFloat
#check neuralBpow
#check neuralToReal
```

`NeuralFloat` is an exact representation object. It does not impose a precision bound by itself.
For example, the pairs `(1,0)` and `(β,-1)` can denote the same real number. Precision and exponent
selection enter through a format function

$$`f_{\mathrm{exp}} : \mathbb Z \to \mathbb Z.`

For a nonzero real `x`, `neuralMagnitude β x` is the integer `k` satisfying

$$`\beta^{k-1} \le |x| < \beta^k.`

The theorem `neuralMagnitude_spec` proves these inequalities. The format then chooses the canonical
exponent

$$`e_x = f_{\mathrm{exp}}(\operatorname{mag}_\beta(x))`

and rescales the value to its canonical mantissa coordinate:

$$`s_x = x\,\beta^{-e_x} = \frac{x}{\beta^{e_x}}.`

In Lean these are `neuralCexp` and `neuralScaledMantissa`. The reconstruction theorem says

$$`s_x\,\beta^{e_x}=x.`

This equation is the basic reason the generic development works. Rounding can be performed on the
dimensionless quantity `s_x`, where the candidates are adjacent integers, and then transported back
to the original scale by multiplying by `β^{e_x}`.

# Formats And Representable Grids

A real number belongs to the generic format exactly when its canonical scaled mantissa is an
integer. Equivalently, there is an integer `m` such that

$$`x=m\,\beta^{e_x}.`

The predicate `neuralGenericFormat β fexp x` records this statement. TorchLean supplies three useful
format families:

- `FIXExp emin` always selects `emin`. Its values lie on the fixed grid
  $$`\beta^{e_{\min}}\mathbb Z,`
  which is the basic model for fixed-point and uniform quantization.
- `FLXExp p` selects `k-p` at magnitude `k`. It models precision `p` with an unbounded exponent and
  is convenient when overflow and underflow are intentionally excluded from a proof.
- `FLTExp emin p` selects `max (k-p) emin`. It models precision `p` with a lower exponent bound and
  gradual underflow.

For the rounded-real binary32 model, TorchLean uses radix two, precision twenty-four, and minimum
stored scale `-149`:

$$`f_{32}(k)=\max(k-24,-149).`

This captures normal and subnormal spacing on the finite binary32 grid. It does not impose the
binary32 upper exponent bound, so `FP32` remains a finite rounded-real model rather than a complete
model of overflow, infinity, and NaN. Those behaviors belong to `IEEE32Exec`.

The format modules are separated from rounding because the same grid can be paired with different
policies. This becomes particularly useful for quantization: the code range and scale define a
grid, while nearest, directed, stochastic, or saturating behavior determines how values enter that
grid.

```
import NN.Floats.NeuralFloat.Format

open TorchLean.Floats

#check FIXExp
#check FLXExp
#check FLTExp
#check neuralGenericFormat
#check neuralMagnitude_spec
#check neural_generic_format_iff_scaled_mantissa_int
```

# Rounding On The Grid

Let `rnd : ℝ → ℤ` be an integer rounding rule. Generic floating-point rounding is

$$`
\operatorname{round}_{\beta,f,r}(x)
= r(s_x)\,\beta^{e_x}.
`

This is the definition of `neuralRound`. Floor and ceiling give directed rounding; truncation gives
rounding toward zero; `neuralNearestEven` gives round-to-nearest with ties resolved toward an even
mantissa.

The validity classes state the properties needed by later proofs. `NeuralValidRnd` requires the
integer rounding function to be monotone and to fix integers. `NeuralValidRndToNearest` adds the
nearest-integer half-unit bound. Once the scaled bound is multiplied by the positive radix power,
TorchLean obtains the standard floating-point estimate

$$`
|\operatorname{round}(x)-x|
\le \frac{1}{2}\operatorname{ulp}(x).
`

Here

$$`\operatorname{ulp}(x)=\beta^{e_x}`

away from the special definition at zero. Relative-error theorems then derive bounds of the form

$$`
\operatorname{round}(x)=x(1+\delta),\qquad |\delta|\le u,
`

under the format-specific nonzero and normal-range hypotheses. TorchLean keeps the absolute and
relative statements separate because the relative form is not meaningful at zero and changes near
underflow.

```
import NN.Floats.NeuralFloat.Rounding
import NN.Floats.NeuralFloat.Analysis
import NN.Floats.NeuralFloat.Error

open TorchLean.Floats

#check neuralRound
#check neural_error_bound_ulp
#check neuralUlp
#check neuralRoundDownPoint
#check neuralRoundNearestPoint
```

# Three Representations Of A Finite Value

TorchLean uses three related representations because no single type serves every proof and
execution purpose.

First, `NeuralFloat β` stores the exact pair `(m,e)`. It exposes the integer structure needed by
effective rounding, conversion proofs, and exact residual arguments.

Second, `NF β fexp rnd` stores a semantic real value and equips it with rounded arithmetic. For
example,

$$`
(a+b).\mathrm{val}
=\operatorname{round}_{\beta,f,r}(a.\mathrm{val}+b.\mathrm{val}).
`

The raw `NF` constructor may contain an arbitrary comparison real. Therefore `NF.IsRepresentable`
is the explicit invariant saying that its value lies on the declared grid. `NF.ofReal` and the
primitive rounded operations establish this invariant.

Third, `IEEE32Exec` stores a raw `UInt32`. Decoding interprets its sign, exponent, and fraction bits;
operations also handle signed zero, subnormal values, infinity, and NaN. Unlike `NF`, this layer is
executable inside Lean.

For an ordinary finite binary32 result, the intended chain is

$$`
\texttt{UInt32 bits}
\xrightarrow{\text{IEEE decode}}
\texttt{IEEE32Exec.toReal}
=
\texttt{FP32.toReal}
=
m,2^e.
`

The equal signs in this diagram are theorems with hypotheses, not coercions. The executable bridge
requires the relevant operands and result to stay on the finite path. This is how TorchLean avoids
silently treating NaN or infinity as an ordinary real number.

# From Scalars To Tensors

A tensor does not introduce a new scalar semantics. It lifts the chosen scalar operation over a
shape and fixes the order of aggregate operations. With `NF` or `FP32`, an elementwise addition
rounds independently at every coordinate. A matrix product additionally specifies a sequence or
tree of rounded products and additions.

For a left-associated dot product, the semantic recurrence is

$$`
p_i=\operatorname{round}(w_i x_i),\qquad
s_0=p_0,\qquad
s_{i+1}=\operatorname{round}(s_i+p_{i+1}).
`

A balanced tree, a fused multiply-add implementation, and a cuBLAS kernel may all compute different
bit patterns. The real specification still denotes `Σ i, w_i*x_i`; the finite-precision theorem is
about the selected recurrence or tree. TorchLean's reduction APIs therefore carry an evaluation
tree instead of proving a false associativity law.

This distinction also applies to backpropagation. A real derivative theorem identifies the ideal
gradient. A rounded execution theorem must additionally account for each forward operation, each
VJP operation, and the reduction order used to accumulate parameter gradients.

# Where Quantization Fits

Uniform affine quantization has the same separation between a mathematical grid and an executable
code. Given a positive scale `s`, integer zero point `z`, and code interval `[qmin,qmax]`, a common
quantizer is

$$`
q(x)=\operatorname{clamp}_{[q_{\min},q_{\max}]}
\left(r\left(\frac{x}{s}\right)+z\right),
`

with dequantization

$$`
d(q)=s(q-z).`

Before saturation, the dequantized values lie on the fixed grid `s ℤ`. When `s=β^e`, this is exactly
the grid represented by `FIXExp e`. The generic format and rounding theory can therefore prove the
local rounding part. Saturation adds a separate range condition: inside the representable interval,
one proves a bound such as

$$`
|d(q(x))-x|\le \frac{s}{2}
`

for nearest rounding; outside that interval, the error is governed by clipping distance rather than
half a grid step.

Arbitrary floating-point quantization needs a richer format descriptor but follows the same path:

- the radix, precision, and exponent policy determine representable finite values;
- the rounding policy chooses a representable neighbor;
- overflow policy chooses saturation or infinity;
- underflow policy chooses gradual underflow, flush-to-zero, or another explicit rule;
- encoding maps the mathematical representation to bits or integer codes;
- decoding maps those codes back to the semantic value used in the theorem.

TorchLean already supplies the generic grid, rounding, FTZ, interval, and FP32 bridge components.
The remaining quantization layer should add code ranges, scale and zero-point parameters,
per-tensor and per-channel indexing, and proofs of encode/decode and tensor-error properties. It
should reuse `Format`, `Rounding`, and `Interval.Quantized` rather than define another arithmetic
semantics.

```
import NN.Floats.NeuralFloat.Format
import NN.Floats.NeuralFloat.Rounding
import NN.Floats.Interval.Quantized

open TorchLean.Floats

-- Fixed grids provide the semantic basis for uniform quantization.
#check FIXExp
#check FIXFormat

-- Outward rounders lift a discrete grid to sound interval propagation.
#check TorchLean.Floats.Interval.Rounder
#check TorchLean.Floats.Interval.formatRounder
```

# Why Both `FP32` And `IEEE32Exec` Exist

`FP32` and `IEEE32Exec` answer different questions.

`FP32` is the proof model. It is a rounded-real scalar: compute the real operation, round to the
binary32 format, and keep an explicit error bound. Use this level for layerwise forward error,
verifier-margin transfer, and paper statements such as "the rounded execution stays within ε of the
real specification."

`IEEE32Exec` is the executable bit model. It stores raw binary32 bits and implements IEEE-style
behavior for the core operations, including signed zero, infinities, NaNs, comparisons, and
special-value propagation. Use this level for widgets, examples, edge-case tests, and checking what
a binary32-shaped computation actually does.

The bridge theorems connect the two on the finite path. That means the inputs and result decode to
ordinary finite real values, and the operation-specific side conditions are satisfied. When a
computation leaves that finite path, the total theorems use `Option`: finite values become
`some r`, while NaN and infinity become `none`.

# API Map

These imports are the main landmarks:

```
import NN.Floats.Float32
import NN.Floats.FP32
import NN.Floats.Calc
import NN.Floats.IEEEExec
import NN.Floats.IEEEExec.BridgeFP32
import NN.Floats.IEEEExec.BridgeFP32Total
import NN.Floats.IEEEExec.BridgeInitFloat32

open TorchLean.Floats
open TorchLean.Floats.IEEE754

-- User-facing selector.
#check TorchLean.Floats.Float32Mode
#check TorchLean.Floats.F32

-- Proof model: finite rounded-real binary32 arithmetic.
#check TorchLean.Floats.FP32
#check TorchLean.Floats.FP32.toReal
#check TorchLean.Floats.NF.IsRepresentable
#check TorchLean.Floats.FP32.add_residual_isRepresentable

-- Constructive rounding machinery.
#check TorchLean.Floats.NeuralLocation
#check TorchLean.Floats.neuralRoundedFloat
#check TorchLean.Floats.FP32.round_eq_computed

-- Executable bit-level model: raw UInt32 binary32 values.
#check IEEE32Exec
#check IEEE32Exec.ofBits
#check IEEE32Exec.toBits
#check IEEE32Exec.toReal
#check IEEE32Exec.toReal?

-- Bridges from executable bits to proof semantics.
#check IEEE32Exec.toReal_add_eq_fp32Round
#check IEEE32Exec.toReal_add_eq_computed_of_isFinite
#check IEEE32Exec.toReal?_add_eq_ite

-- Runtime Float32 remains a named assumption boundary.
#check Float32Bridge.RuntimeFloat32MatchesIEEE32Exec
```

Read these names as the dependency graph. `F32` selects the scalar. `FP32` is the scalar used in proofs.
`IEEE32Exec` is the executable scalar. `BridgeFP32` and `BridgeFP32Total` say how executable bits
become rounded reals in the proof model. `BridgeInitFloat32` names what remains if a claim uses Lean's
runtime `Float32`.

# A Single Addition At Four Levels

The same expression, `a + b`, has four different readings:

- `a + b : ℝ` is exact mathematical addition.
- `FP32.add a b` is exact real addition rounded to the binary32 grid.
- `IEEE32Exec.add x y` is executable binary32 bit arithmetic with IEEE-style special cases.
- a runtime or CUDA add is a native implementation whose agreement is tested, checked, or assumed
  against a Lean side contract.

The finite bridge theorem says the `IEEE32Exec` result agrees with the `FP32` rounded-real result
when the finite hypotheses hold:

```
import NN.Floats.IEEEExec.BridgeFP32
import NN.Floats.IEEEExec.BridgeFP32Total

open TorchLean.Floats.IEEE754

-- Finite bridge.
#check IEEE32Exec.toReal_add_eq_fp32Round

-- Total bridge.
#check IEEE32Exec.toReal?_add_eq_ite
```

The layers remain separate and are connected by a
named theorem, checker, or runtime agreement.

# Computing The Rounded Representation

The rounded-real model states what the result means. The modules under `NN.Floats.Calc` expose the
corresponding representation-level calculation. Starting from a radix-scaled integer bracket, a
location records whether the exact value is equal to the lower endpoint or lies below, at, or above
the midpoint. The selected rounding mode then chooses an endpoint, and truncation produces a
mantissa and exponent without changing the represented bracket. This machinery is proof-level when
its input is an arbitrary Lean real; actual runs use `IEEE32Exec` or a named runtime backend and are
connected back by bridge theorems.

For binary32, `FP32.round_eq_computed` connects this calculation to the public proof model:

```
import NN.Floats.FP32
import NN.Floats.Calc

open TorchLean.Floats

#check FP32.round_eq_computed
```

The finite IEEE bridge continues the chain. For example,
`IEEE32Exec.toReal_add_eq_computed_of_isFinite` states that an executable finite addition decodes to
the real value of the computed mantissa/exponent representation. Similar theorems cover subtraction,
multiplication, division, sum-tree nodes, and dot-product nodes. For reductions, the evaluation tree
remains part of the statement because changing that tree can change the rounded result.

An `NF` operation always rounds its result, but the raw `NF` constructor can embed an arbitrary real
for comparison and approximation arguments. The predicate `NF.IsRepresentable` records when a value
actually lies on the declared grid. `NF.ofReal` establishes it, and the predicate is closed under the
primitive arithmetic operations. Results that require representable inputs say so explicitly. In
particular, `FP32.add_residual_isRepresentable` proves that the exact residual left by adding two
representable FP32 operands is itself representable, a structural property used by compensated and
error-free arithmetic arguments.

# Binary32 Is A Format, Not A Whole Execution Story

IEEE 754 binary32 fixes the interchange format: one sign bit, eight exponent bits, twenty-three
stored fraction bits, and an implicit leading bit for normal values. It also names the special
values that make floating point unlike ordinary real arithmetic: signed zeros, subnormals,
infinities, and NaNs. In the default round-to-nearest, ties-to-even mode, a correctly rounded
primitive operation is the exact real result rounded once to that binary32 format.

That sentence is already more precise than "float32", but it is still not the whole story for a
neural-network run. A dot product also has an evaluation tree. A fused multiply-add rounds once,
while a separate multiply followed by add rounds twice. A library call such as `expf` or a CUDA
special-function approximation has a documented accuracy contract, not automatically the same
contract as a primitive IEEE arithmetic operation. A verifier statement therefore cannot stop at
"this is FP32"; it must say which computation produced the FP32 value.

TorchLean's rule is:

- use `IEEE32Exec` when the bit pattern and IEEE-style special cases are the object of study;
- use `FP32` when the proof needs rounded-real error bounds on the finite path;
- use a runtime or CUDA agreement assumption when the value came from Lean `Float32`, C/CUDA
  `float`, ATen, cuBLAS, cuFFT, or libdevice.

The standard supplies a vocabulary for binary32. The proof still has to identify the program,
operation order, library boundary, and finite-path side conditions.

# Reductions Need An Order

Real addition is associative. Floating-point addition is not. Therefore a theorem about a float32
sum, dot product, convolution accumulation, attention score, or CUDA reduction needs an order
contract.

TorchLean states reduction facts around a finite evaluation tree. Different parallel schedules can
produce different rounded results, so the evaluation order is part of the statement:

The [reductions API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/Reductions.lean) contains the long-form theorem
names for sum trees and dot products. The reading rule is shorter than the names: soundness is
stated for a specified finite evaluation tree, not for every possible reordering.

CUDA reductions appear in the native boundary chapter for this numerical reason. A fast kernel may
be a perfectly good training kernel, but a proof-quality statement needs a fixed tree or an explicit
reduction specification.

# A Dot Product Written Two Ways

For real numbers, the following two programs denote the same expression:

```
(((w0 * x0) + (w1 * x1)) + (w2 * x2)) + (w3 * x3)

((w0 * x0) + (w1 * x1)) + ((w2 * x2) + (w3 * x3))
```

For binary32, they can differ. If the implementation uses fused multiply-adds, they can differ again:

```
fma w3 x3 (fma w2 x2 (fma w1 x1 (w0 * x0)))
```

All three are reasonable implementation strategies. They are not the same theorem target. A
TorchLean statement about a dot product should therefore name the accumulation tree or cite the
kernel contract that fixes it. The same principle applies to convolution sums, matrix
multiplication, layer normalization, softmax denominators, and backward reductions.

# Finite Path, Precisely

The finite path is the part of IEEE-style execution where the result can be read as an ordinary real
number.

Examples:

- a normal binary32 value is finite;
- a subnormal binary32 value is still finite, although with reduced precision;
- `+0.0` and `-0.0` are finite bit patterns, but proofs may need to know which operation produced
  them;
- overflow to `+Inf` or `-Inf` leaves the finite path;
- `0/0`, `Inf - Inf`, and `sqrt` of a negative finite value produce NaN paths, not finite paths;
- a reduction is finite only relative to a specified evaluation tree whose intermediate results
  remain finite.

The bridge files reflect this split:

- [BridgeFP32 API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/BridgeFP32.lean) proves finite refinements such as
  `toReal_add_eq_fp32Round`, `toReal_mul_eq_fp32Round`, `toReal_div_eq_fp32Round`,
  `toReal_fma_eq_fp32Round`, and `toReal_sqrt_eq_fp32Round`.
- [BridgeFP32Total API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/BridgeFP32Total.lean) packages total statements
  through `toReal?`, with theorem names such as `toReal?_add_eq_ite`,
  `toReal?_mul_eq_ite`, `toReal?_div_eq_ite`, and `toReal?_sqrt_eq_ite`.
- [SpecialRules API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/SpecialRules.lean) records NaN, infinity,
  signed-zero, and special-case behavior for the executable kernel itself.

The informal theorem shape is:

- execute the operation in `IEEE32Exec`;
- decode the result with `toReal`;
- obtain the same real value as applying the corresponding real operation to the decoded inputs and
  rounding that result with `FP32.fp32Round`.

This reading is valid under the stated finite and operation specific hypotheses.

# Transcendentals And Library Boundaries

The `FP32` proof model defines transcendentals by applying the corresponding real function and
then rounding. Theorems such as `exp_abs_error`, `tanh_abs_error`, and interval membership lemmas
are therefore statements relative to Lean's real functions.

Executable Lean semantics and native library calls are different objects. A CPU `libm`, CUDA
`libdevice`, or vendor library may provide an approximation with documented accuracy. TorchLean
therefore keeps transcendental claims explicit:

- use `FP32` / `NF` for rounded real statements in Lean proofs;
- use `IEEE32Exec` for deterministic executable behavior where the operation is defined in Lean;
- use a runtime or CUDA contract when an external library computes the value.

The floating point verification literature makes the same point: compiler, library, and hardware
choices are part of the semantics unless they are isolated behind a proof or contract.

# Runtime And CUDA Boundaries

Lean's runtime `Float32` operations are external runtime calls. Lean documents `Float` operations as
IEEE-style opaque operations that do not reduce in the kernel and compile to C operators; the same
general concern applies here. Runtime operations can be used, tested, and connected to Lean side
semantics through explicit agreement assumptions.

TorchLean names that assumption explicitly:

```
import NN.Floats.IEEEExec.BridgeInitFloat32

open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.Float32Bridge

#check RuntimeFloat32MatchesIEEE32Exec
#check RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_add
#check RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_mul
#check RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_sqrt
```

The class says, operation by operation, that runtime float32 bits match the `IEEE32Exec` reference
bits. Once that bridge is supplied, downstream proofs can reuse the `IEEE32Exec` and `FP32`
theorems. Without it, a runtime result remains implementation evidence rather than a Lean side
scalar statement.

CUDA uses the same idea at the native boundary:

```
import NN.Runtime.Autograd.Engine.Cuda.Float32Contract
import NN.Runtime.Autograd.Engine.Cuda.KernelSpec

open Runtime.Autograd.Cuda.Float32Contract

#check NativePrimitiveAgreement
#check native_add_abs_error_of_isFinite
#check native_mul_abs_error_of_isFinite
#check native_div_abs_error_of_isFinite
#check native_fma_abs_error_of_isFinite
#check native_sqrt_abs_error_of_isFinite
```

For the engineering details, read *GPU and CUDA Boundaries*. The scalar semantics here fix the
meaning: native arithmetic is connected to proofs through explicit bit-agreement contracts, fixed
reduction specifications, and finite-path hypotheses.

# Concrete Bit Patterns

`IEEE32Exec` makes special values concrete:

```
import NN.Floats.IEEEExec.Exec32

open TorchLean.Floats.IEEE754

def plusZero : IEEE32Exec := IEEE32Exec.ofBits (0x00000000 : UInt32)
def minusZero : IEEE32Exec := IEEE32Exec.ofBits (0x80000000 : UInt32)
def plusInf : IEEE32Exec := IEEE32Exec.ofBits (0x7f800000 : UInt32)
def qNaN : IEEE32Exec := IEEE32Exec.ofBits (0x7fc00000 : UInt32)

-- Round a Float64 literal to binary32 deterministically in Lean.
def third32 : IEEE32Exec := IEEE32Exec.ofFloat (Float.ofBits 0x3fd5555555555555)
```

For a visual inspection path, open the widget examples:
[Widgets source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Widgets.lean), especially `#float32_view` and
`#float32_round_view`.

# What The Claims Mean

Checked by Lean:

- `FP32` rounded-real definitions and local error lemmas;
- `IEEE32Exec` bit-level definitions for the supported core operations;
- finite bridge theorems from `IEEE32Exec` to `FP32`;
- total `toReal?` theorems that expose NaN/Inf paths;
- interval-facing and runtime-approximation lemmas that cite these scalar models.

Runtime agreement paths:

- Lean runtime `Float32` agreement with `IEEE32Exec`;
- CUDA primitive and kernel agreement with the Lean contracts;
- cuBLAS, cuFFT, libdevice, compiler, driver, and GPU behavior;
- correctly-rounded claims for external transcendental libraries;
- alternative rounding modes and exception flags not modeled by the current theorem path.

That split is the claim. Native execution is part of the workflow, and TorchLean makes the agreement
with Lean side semantics small, named, and testable.

# How To Choose The Right Layer

Use this rule:

- Use `ℝ` for ideal networks, losses, and verifier statements.
- Use `FP32` / `NF` for compositional finite-precision error bounds.
- Use `IEEE32Exec` for bit inspection, NaN/Inf behavior, signed zeros, and executable binary32
  examples.
- Use `RuntimeFloat32MatchesIEEE32Exec` when runtime float results must be transported into the
  executable reference semantics.
- Use `Float32Contract` / `KernelSpec` when native CUDA kernels must be connected to the proof
  layer.

The rule is simple: choose the numerical object for the claim, then cite the bridge theorem,
checker, or runtime agreement that connects it to the layer below.

# References

- IEEE Std 754-2019, *Standard for Floating-Point Arithmetic*:
  [IEEE 754-2019](https://standards.ieee.org/standard/754-2019/)
- ISO/IEC/IEEE 60559:2020, the international floating-point arithmetic standard:
  [ISO/IEC/IEEE 60559:2020](https://standards.ieee.org/standard/60559-2020.html)
- David Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*:
  [Oracle-hosted reprint](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- Nicholas Higham, *Accuracy and Stability of Numerical Algorithms*:
  [SIAM book page](https://epubs.siam.org/doi/book/10.1137/1.9780898718027)
- Flocq, the Coq floating-point formalization:
  [Flocq documentation](https://flocq.gitlabpages.inria.fr/)
- FloatSpec, a Lean 4 floating-point formalization inspired by Flocq:
  [FloatSpec package](https://reservoir.lean-lang.org/%40Beneficial-AI-Foundation/FloatSpec)
- Gappa, for certified floating-point bounds and proof generation:
  [Gappa paper](https://arxiv.org/abs/0801.0523)
- CompCert, for verified compilation, including machine floating point models:
  [CompCert commented development](https://compcert.org/doc/)
- NVIDIA, *Floating Point and IEEE 754*:
  [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/pdf/Floating_Point_on_NVIDIA_GPU.pdf)
