# Floats (`NN/Floats`)

This directory contains TorchLean's floating point backends and the theory that connects them.

- `FP32/`: a proof oriented, finite float32 model based on rounding over `ℝ`.
- `NeuralFloat/`: generic rounding over `ℝ` (`NeuralRadix`, `NF`, rounding, ULPs, and error bounds).
- `Calc/`: constructive mantissa/exponent calculations, including brackets, rounding decisions,
  truncation, and exact arithmetic on float representations.
- `IEEEExec/`: an executable IEEE-754 binary32 kernel (`IEEE32Exec`) plus bridge theorems to `FP32`.
- `Interval/`: interval and enclosure utilities, including quantized intervals over `ℝ` and executable endpoint intervals.
- `Arb/`: an external Arb/FLINT oracle backend (python-flint) for ball and interval enclosures.

The generic theory is a native Lean development informed by Flocq's organization and results. It is
not a claim that every Coq module has been translated. TorchLean keeps the parts used by its tensor,
error-analysis, verification, and runtime-refinement developments, and proves executable binary32
behavior separately under `IEEEExec/`.

If you want the whole float chapter, import `NN.Floats`. If you only want a single
float32 name to depend on, import `NN.Floats.Float32`.

Executable examples that exercise this infrastructure live under `NN/Examples/`.

## Which Layer Should I Cite?

| If your claim is about... | Use this layer |
| --- | --- |
| executable binary32 values inside Lean | `IEEEExec` |
| finite float32-as-rounded-real error bounds | `FP32` |
| precision-parametric rounding and ULP facts | `NeuralFloat` / `NF` |
| the mantissa and exponent produced by a rounding operation | `Calc`, specialized through `FP32` when appropriate |
| interval enclosures with directed endpoints | `Interval` |
| high-precision external enclosure evidence | `Arb`, with the oracle boundary named |
| CUDA, ATen/libtorch, or Lean runtime `Float` | a runtime bridge or `TRUST_BOUNDARIES.md` assumption |

This distinction is part of the correctness story. A theorem over `FP32` does not become a CUDA
claim until a runtime bridge or trust-boundary statement connects the executable path to that model.

## The Three Float32 Views

TorchLean uses three complementary notions of float32. They have different
strengths, and the bridge files let us move between them without blurring the
trust boundary.

### 1. `IEEE32Exec` (Executable IEEE-754 Binary32)

Folder: `NN/Floats/IEEEExec/`

This is the executable backend. Values are 32-bit encodings, and operations are implemented as pure
Lean code over those bits.

Why we keep it:
- we can execute models inside Lean without relying on opaque runtime float calls,
- we can state and prove theorems about the kernel itself (NaN/Inf propagation, signed zeros, etc.),
- and it provides a stable target for connecting to the mathematical models below.

Where to look:
- `NN/Floats/IEEEExec/Exec32.lean`: the core executable kernel (`IEEE32Exec`),
- `NN/Floats/IEEEExec/SpecialRules.lean`: NaN/Inf propagation rules,
- `NN/Floats/IEEEExec/Reductions.lean`: reduction semantics (sums/dot products) that match deployment realities.
  (For executable endpoint-interval arithmetic, see `NN/Floats/Interval/IEEEExec32.lean`.)

### 2. `FP32` (Finite Real Rounded Float32 Model)

Folder: `NN/Floats/FP32/`

This is the mathematical view: float32 computation is modeled as real arithmetic followed by a
float32 rounding operator. It is finite only, with no NaNs or infinities, which keeps many error
analysis statements manageable.

Why we keep it:
- most error-bound theorems are naturally stated over reals + rounding,
- it composes cleanly with `NeuralFloat`/`NF` error bounds,
- it gives a clear semantic target when we want to describe what a float32 computation means.

Where to look:
- `NN/Floats/FP32/Core.lean`: the main definitions,
- `NN/Floats/FP32/Error.lean`: error bounds,
- `NN/Floats/Interval/FP32.lean`: interval-style enclosure corollaries.

### 3. `NeuralFloat` / `NF` (Generic Format And Error Analysis)

Folder: `NN/Floats/NeuralFloat/`

This layer abstracts over radix, precision, and exponent range. It is where we develop reusable facts
about rounding, ULPs, and relative or absolute error bounds. `FP32` is the
binary32-specialized instance of this generic layer.

Why we keep it:
- we want one set of theorems that applies to multiple precisions and training phases,
- it lets us reuse the same reasoning patterns for float32 and other formats.

Where to look:
- `NN/Floats/NeuralFloat/Format.lean` for representable grids,
- `NN/Floats/NeuralFloat/Rounding.lean` for rounding semantics,
- `NN/Floats/NeuralFloat/Scalar.lean` for `NF`,
- `NN/Floats/NeuralFloat/Analysis.lean` and `NN/Floats/NeuralFloat/Error.lean` for numerical bounds.

## How The Bridge Layers Fit Together

`IEEE32Exec` is executable; `FP32` and `NF` are aimed at proofs. `Calc` supplies representation-level
arithmetic between them. Given a bracket for an exact value, it records the value's position between
adjacent representable numbers, makes the rounding-mode decision, and returns a `NeuralFloat`
mantissa and exponent. Theorems in `FP32/Core.lean` identify the real value of that representation
with `fp32Round`. Thus `Calc` is not another floating-point semantics: it exposes the integer
calculation underlying the rounding operation used by the proof model. Calculations starting from an
arbitrary Lean real remain noncomputable; executable runs use `IEEE32Exec` or a named runtime backend.

The bridge files then connect the executable bit model to that result:

- `NN/Floats/IEEEExec/BridgeFP32.lean`: core refinement theorems on the **finite/no-overflow** path:
  `toReal (op_exec …) = fp32Round (op_real …)`.
- `NN/Floats/IEEEExec/BridgeFP32Total.lean`: total wrappers that combine NaN/Inf propagation rules
  with the finite refinement theorems, phrased using `toReal?`.
- `NN/Floats/IEEEExec/BridgeFP32Expr.lean`: a compact scalar AST + a whole-expression refinement theorem.
- `NN/Floats/IEEEExec/BridgeERealTotal.lean`: an `EReal`-valued semantics that distinguishes `+∞` and `-∞`.
- `NN/Floats/IEEEExec/BridgeInitFloat32.lean`: an assumption based bridge from Lean's runtime
  `Init.Float32` to `IEEE32Exec`, at the bit level.

`NN/Examples/DeepDives/Floats/EffectiveRounding.lean` follows one shaped tensor addition through the
rounded-real and executable IEEE paths. Its theorems expose the computed mantissa/exponent result on
both sides of the finite bridge.

For native CUDA and ATen/libtorch paths, the bridge is not in this folder by default. Those backends
are runtime providers. A proof layer float claim should say which Lean model it uses and where the
runtime/backend agreement assumption is discharged or documented.

Background references that informed the design:
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
- Higham (2002): *Accuracy and Stability of Numerical Algorithms* (2nd ed.), ISBN 0-89871-521-0
- Flocq (Boldo–Melquiond, 2011): https://doi.org/10.1109/ARITH.2011.40
