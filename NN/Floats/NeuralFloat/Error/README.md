# Rounded-Arithmetic Error Theory

These modules prove properties of the error introduced by one rounded arithmetic operation. They
are radix- and format-parametric where the mathematics permits it; specialized `FP32` results are
derived in `NN/Floats/FP32/`.

## Modules

- `Bounds.lean` contains reusable absolute and relative one-step estimates.
- `Directed.lean` treats downward and upward rounding, where the error has a known sign.
- `Relative.lean` develops FLX/FLT relative-error bounds and the normal-range hypotheses needed by
  lower-bounded formats.
- `Exactness.lean` supplies common-grid representation lemmas shared by arithmetic residual proofs.
- `Addition.lean` proves that the residual of nearest-rounded addition is representable under the
  stated format hypotheses.
- `Multiplication.lean` proves exact representability results for rounded-product residuals.
- `DivisionSqrt.lean` treats division and square-root residuals in FLX.

These are not isolated numerical curiosities. Exact residuals support error-free transformations;
one-step bounds compose into layer and reduction bounds; and signed directed errors justify interval
endpoints. TorchLean uses those results when transferring claims from real-valued specifications to
`FP32` computations and then, on finite paths, to `IEEE32Exec`.

Import the complete family with:

```lean
import NN.Floats.NeuralFloat.Error
```

The principal mathematical organization follows Flocq, while the statements and proofs are native
Lean and use TorchLean's `NeuralFloat`, `NF`, and format classes.
