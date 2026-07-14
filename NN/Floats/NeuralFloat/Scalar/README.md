# Rounded Scalars

`NF` lets TorchLean instantiate tensor and neural-network specifications with a configurable
rounded-real scalar.  `NF.ofReal` performs rounding; the raw constructor remains available for
approximation proofs, so grid-dependent theorems use `NF.IsRepresentable` explicitly.

- `NF.lean` defines the carrier and primitive rounded arithmetic.
- `Representable.lean` proves that rounded constructors and operations land on the declared grid.
- `NNOps.lean` supplies rounded nonlinear operations used by neural-network specifications.
- `Conversion.lean` records conversion metadata and error summaries.

This is a semantic scalar model, not a claim about a CPU instruction or GPU kernel.  Runtime claims
must pass through the executable IEEE bridge or a named backend contract.
