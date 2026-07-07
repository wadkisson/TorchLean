# `NN/Floats/IEEEExec`: Executable IEEE-754 Float32 Semantics

This directory contains TorchLean's Lean defined executable model of IEEE-754 binary32. It also holds
bridge theorems that connect runtime execution to proof oriented rounding models over `ℝ`.

## What Lives Here

- `Exec32.lean` and `Exec32/`: the executable kernel (`IEEE32Exec`), bit layout, arithmetic,
  comparison, directed rounding, dyadic conversion, instances, and deterministic transcendental
  wrappers.
- `ERealSemantics.lean`: small shared helpers for interpreting `IEEE32Exec` values in `EReal`
  (including a total `toEReal` used in endpoint/enclosure proofs).
- `RealSemantics.lean`, `MkBitsToReal.lean`, `MkBitsToDyadic.lean`: interpretation of bit patterns
  as real/dyadic values.
- `SpecialRules.lean`: proved rewrite rules for NaN/Inf/±0 propagation, so proofs do not need to
  repeatedly unfold the executable definitions.
- `TranscendentalRules.lean`: deterministic, non IEEE specified transcendental functions
  (`exp`, `log`, `tanh`, …) and their special-value rules.
- `TrigRules.lean`, `TrigBounds.lean`: deterministic trigonometric rules and bounds used by
  examples and interval-facing code.
- `Reductions.lean`: reduction semantics for sums/dot products.
- `ErrorBounds.lean`, `OpSandwich.lean`: reusable executable error/enclosure facts.
- `RoundShiftRightEven.lean`, `RoundDyadicToIEEE32Bounds.lean`, `RoundQuotEvenBounds.lean`,
  `RatScaling.lean`, `NatLemmas.lean`: rounding and arithmetic lemmas used by bridge and interval
  proofs.

## Interval / Directed Rounding Semantics

If your goal is interval arithmetic over float32, this folder provides the directed rounding
kernels and their soundness proofs:

- `DirectedRoundingSoundness.lean`: soundness of `roundDyadicDown/Up` and endpoint ops
  (`addDown/addUp/mulDown/mulUp`) in `EReal` (so overflow to `±∞` is handled cleanly).
- `DivDirectedRoundingSoundness.lean`: analogous soundness for rational rounding and
  `divDown/divUp` (finite / nonzero-divisor regime).
- `MinMaxERealSoundness.lean`: basic order lemmas used by endpoint min/max rules.

The interval *API layer* that uses these results lives in `NN/Floats/Interval/`.

## Bridge To Proof Oriented Float Models

TorchLean keeps two float32 views:

- Executable (`IEEE32Exec`): what we can run inside Lean, with bit level semantics.
- Proof oriented (`FP32`): round on real semantics used for numerical error envelopes.

Bridge files connect these on the finite, no overflow path:

- `BridgeFP32.lean` and `BridgeFP32/`: per-operation refinement lemmas on the finite branch,
  including dyadic/rational rounding infrastructure.
- `BridgeFP32Expr.lean`: expression-level refinement (compose op lemmas once).
- `BridgeFP32Total.lean`: packages finite refinement + proved special-value rules using `toReal?`.
- `BridgeERealTotal.lean`: a slightly richer `EReal` interpretation (`+∞` vs `-∞`, NaN as `none`).
- `BridgeInitFloat32.lean`: an assumption based interface relating Lean runtime `Float32` to
  `IEEE32Exec`. The runtime is opaque to the kernel, so this cannot be proved internally.

## How To Use It

Use `IEEE32Exec` when the object you need is executable binary32 inside Lean. Use `FP32` when the
theorem should be a rounded-real error statement. Use runtime `Float32` or CUDA only with an
explicit bridge or trust-boundary statement.

The useful proof shape is:

```text
IEEE32Exec operation
  -> finite/no-overflow bridge
  -> FP32 rounded-real statement
  -> real-valued error envelope or interval claim
```

## Trust boundary (important)

IEEE-754 does not specify bit level results for transcendentals (`expf`, `logf`, ...), and platform
libm implementations can differ. TorchLean therefore treats:

- core arithmetic (add/mul/div/sqrt, specials) as the proved executable kernel, and
- transcendentals as deterministic but not IEEE specified unless you use a separate rigorous
  backend (e.g. the Arb oracle under `NN/Floats/Arb/` and the interval glue in `NN/Floats/Interval/`).
