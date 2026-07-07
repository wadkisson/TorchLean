import VersoManual

open Verso.Genre Manual

#doc (Manual) "Matrix Factorizations: Cholesky and QR" =>
%%%
tag := "factorizations-cholesky-qr"
%%%

Classical and scientific ML models often depend on matrix factorizations. Gaussian processes,
kernel ridge regression, PCA, least squares, and many inverse problems all need a way to talk about
Cholesky or QR without treating the factorization as an opaque numerical routine.

TorchLean adds specifications for two finite factorizations: Cholesky, where `A = L·Lᵀ`, and QR by
classical Gram Schmidt, where `A = Q·R`. These are direct algorithms. There is no iteration count or
convergence tolerance in the theorem statement. Under the stated pivot hypotheses, correctness is an
exact identity.

# Specifications

Each factorization is given a proposition level meaning over real matrices, independent of any particular
algorithm:

- `IsCholesky A L`: `L` is lower triangular and `A = L·Lᵀ`;
- `IsQR A Q R`: `Q` has orthonormal columns (`Qᵀ·Q = 1`), `R` is upper triangular, and `A = Q·R`.

The executable specs `choleskySpec` / `qrSpec` (over the readable `Fin n → Fin n → α` function
representation, wrapped back into `Spec.Tensor` at the boundary) are then proved to *produce* objects
satisfying these predicates.

# Exact Cholesky Reconstruction

`choleskyFn` builds `L` one column at a time by a left fold. Two structural facts are proved directly
from that fold. First, lower triangularity: entries strictly above the diagonal are forced to `0` by
construction. The theorem is `choleskyFn_lower_triangular`, lifted to the tensor level as
`choleskySpec_lower_triangular`.

Second, reconstruction: the theorem `isCholesky_of_pos` assumes the algorithm's success condition
directly, namely that every executable pivot is positive:

$$`\forall j,\; 0 < choleskyFn A j j.`

Under that hypothesis, the fold satisfies

$$`A = L\,L^{\top},`

equivalently `IsCholesky A (choleskyFn A)`. The statement is exact over `ℝ`; the only hypothesis is positivity of the
executable pivots, which is exactly the condition under which Cholesky succeeds over `ℝ`. (The
mathematical fact that an SPD `A`, meaning `Matrix.PosDef`, yields positive executable pivots is the expected
sufficient condition, but the reduction `PosDef A → ∀ j, 0 < choleskyFn A j j` is *not* formalized here;
the theorem takes the positive-pivot hypothesis as given.)

# Exact QR Reconstruction And Orthonormality

`qrSpec` runs classical Gram–Schmidt. Bridging the executable column fold to Mathlib's `gramSchmidt`
gives the two QR guarantees, both exact under a full column rank hypothesis (each `R` diagonal entry
positive):

$$`Q^{\top} Q = 1 \qquad\text{and}\qquad A = Q\,R, \quad R \text{ upper-triangular}.`

Orthonormality (`qrSpec_orthonormal` / `QT_mul_Q_eq_one`) follows because each Gram–Schmidt column is the
normalization of a vector orthogonal to the span of its predecessors; reconstruction follows by
re-expanding that span. Together they give `IsQR A Q R`.

# Scope

Everything in this chapter is an exact finite identity. There is no sweep count, no residual, and
no asymptotic limit. Three layers are kept distinct:

- *Proved specs* (over `ℝ`): the predicates `IsCholesky` / `IsQR`, together with reconstruction,
  triangularity, and orthonormality, each derived from the executable column fold.
- *Executable examples* (over `Float`): concrete witnesses with residual checks showing that the
  definitions run and reconstruct at runtime.
- *Trusted runtime hooks*: the strict-array `@[implemented_by]` replacements are runtime substitutions
  used for fast evaluation; equality with the clean proof definitions is a named trusted runtime
  boundary.

The formal Cholesky hypothesis is positivity of the executable pivots, `∀ j, 0 < choleskyFn A j j`.
It is not stated as SPD. `Matrix.PosDef A` is the expected sufficient condition for those pivots to
be positive, but the theorem assumes the executable pivot condition directly. A separate
theorem from SPD to positive pivots would discharge that hypothesis. QR likewise assumes positive executable
`R`-pivots, corresponding to full column rank, rather than a separately proved rank hypothesis.
Under those pivot hypotheses, the chapter's theorem statements are
closed. The triangular- and ridge-solve helpers that ride on the Cholesky factor are shipped as
executable APIs; their correctness theorems belong in a later factorization layer.

# Executable witnesses

`NN.Examples.Factorization.Cholesky` and `…QR` exhibit each factorization on a concrete matrix: a
positive reconstruction check (`‖A − L·Lᵀ‖`, `‖A − Q·R‖`, `‖Qᵀ·Q − I‖` all at machine zero) paired with a
negative control, every check a `#eval` over `Float`, with no unproved goals, green on
`lake build NN.Examples.Factorization`.
