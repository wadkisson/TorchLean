import VersoManual

open Verso.Genre Manual

#doc (Manual) "Float32 Soundness" =>
%%%
tag := "fp32-soundness"
%%%

After naming the numerical objects, the next question is how a claim moves from real arithmetic to
Float32.

The basic situation is common. A verifier or proof establishes a real-valued inequality with some
margin. A floating-point analysis bounds how far the rounded computation can move from the real one.
If the margin is larger than the rounding error budget, the property survives finite precision.

`FP32` and `IEEE32Exec` are defined in *Floating Point and Native Boundaries*. Here the guiding
transfer lemma is:

- the decoded FP32 result is within `ε` of the real specification;
- the real specification is above the target threshold by more than `ε`;
- therefore the decoded FP32 result is still above the target threshold.

A real-valued margin of `10^-4` does not help if the Float32 error budget is `10^-3`. A real-valued
margin of `0.1` may survive the same rounding budget. The theorem has to say which case we are in.

For example, suppose a real-valued verifier proves a margin lower bound of `0.12`. Suppose the FP32
approximation theorem says each relevant logit can move by at most `0.03`. For a two-logit margin,
the margin can shrink by at most `0.06`. The FP32 margin is still at least `0.06`, so the
classification claim survives rounding.

# Finite Path First

TorchLean separates two questions:

1. What happens on the ordinary finite path, where every operation produces a finite float32 value?
2. What happens for every IEEE-754 corner case, including NaN, Inf, subnormal behavior, signed zero,
   and backend-specific library behavior?

The `FP32` model used in proofs is aimed at the first question. It is the right object for margin
transfer theorems, forward error bounds, and verifier enclosure inflation. The executable
`IEEE32Exec` model is the right object for bit level behavior and for checking what the runtime
does. Bridge theorems and explicit assumptions connect the two where the finite side conditions are
available.

The split is a practical mathematical choice. If every robustness theorem had to carry the full IEEE
state machine directly, most theorem statements would become harder to read than the property they
are trying to prove.

The finite path includes normal values and subnormals. It excludes computations whose executable
path produces NaN or infinity. For example, a small underflowed subnormal output may still be a
finite value covered by a finite-path theorem, while `0/0`, overflow to `Inf`, or `sqrt` of a
negative finite value are handled by the executable `IEEE32Exec` layer and total `toReal?` bridge,
where the special cases remain visible.

# Where The Approximation Story Becomes Concrete

APIs: [FP32](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/FP32.lean) and
[IEEE32Exec](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/Exec32.lean). This chapter is about
margin transfer, not re-teaching the layer split.

The repository turns this picture into named theorems in three recurring settings: layerwise error bounds, small network-level bounds for
common MLP families, and verifier side statements saying that real valued enclosures can be widened
enough to remain sound for float32 execution.

Theorem names that tend to matter first in this area:

- `approxT_linear_fp32`: float aware forward error for `y = Wx + b`
- `approxT_tanhMlp3_fp32` / `approxT_reluTwoLayerMlp_float32`: float aware forward error for common MLP
  patterns
- `fp32_le_of_real_le_sub_margin` / `fp32_ge_of_real_ge_add_margin`: margin lemmas for lifting
  real-spec inequalities
- `ibpBound_contains_reluTwoLayerMlp_float32`: inflate a real IBP/CROWN box to cover FP32 execution of the
  two-layer ReLU MLP fragment

Behind those statements sit more generic rounded-runtime lemmas for operators and linear algebra.
Those are the engine room for the more visible theorems:

- they keep the algebra of rounding explicit,
- they make it easy to reuse the same proof pattern across many operators,
- and they let the FP32 semantics guide stay focused on the model rather than every proof detail.

# One Transfer Pattern

Most theorems in this area follow a repeatable proof pattern:

1. Prove or compute a real-spec enclosure or inequality with a margin.
2. Use `approxT_*` to obtain an explicit `eps` such that the float32 semantics is within `eps`.
3. Apply a margin lemma like `fp32_le_of_real_le_sub_margin` or
   `fp32_ge_of_real_ge_add_margin` to transfer the property to FP32 execution.

The kind of theorem TorchLean aims at looks like this:

```
import NN.Proofs.RuntimeApprox.FP32.Layers
import NN.Proofs.RuntimeApprox.FP32.MLP
import NN.Proofs.RuntimeApprox.FP32.CROWN

open NN.Proofs.RuntimeApprox.FP32

-- If the real valued network stays above a threshold by more than `eps`,
-- then the FP32-rounded execution still stays above the threshold.
#check approxT_linear_fp32
#check approxT_reluTwoLayerMlp_float32
#check approxT_tanhMlp3_fp32
#check ibpBound_contains_reluTwoLayerMlp_float32
#check fp32_le_of_real_le_sub_margin
#check fp32_ge_of_real_ge_add_margin
```

Keep this shape in mind when reading the declarations. One theorem gives the approximation error,
and another theorem uses that approximation error to preserve the property one actually cares
about.

The recurring scalar pattern is:

$$`|y^{\mathrm{fp32}} - y^{\mathbb R}| \le \varepsilon`

and

$$`y^{\mathbb R} \ge t + \varepsilon`

therefore

$$`y^{\mathrm{fp32}} \ge t`

For classifier margins, the same idea is applied componentwise: the true logit may move down, the
competing logit may move up, and the certified real margin must pay for both movements.

For a vector output, the same pattern is componentwise:

$$`\forall i,\quad
\left| y^{fp32}_i - y^{real}_i \right| \le \varepsilon_i`

and a classifier margin proof normally spends that budget on the true class and the competing
classes before concluding that the argmax is unchanged.

# Template Statement

The template is a transfer lemma with explicit side conditions:

> Assume a real valued network output stays above a threshold by margin `δ`.
> Assume the corresponding FP32 execution stays within error `ε`.
> If `ε < δ`, then the FP32 execution still stays above the threshold.

TorchLean splits that argument into reusable pieces instead of reproving it from scratch every time:

- one theorem gives the forward approximation error,
- one theorem packages verifier side enclosure inflation,
- and one margin lemma moves the property across the `ε` gap.

This theorem family matters beyond small MLP examples. The same pattern shows up again
in layerwise proofs, in verifier side corollaries, and in the kinds of robustness statements one
would actually want to cite in a paper.

In theorem statements, this usually appears as three named ingredients:

- a real-valued property or enclosure;
- an approximation relation such as `approxT_*`;
- a margin lemma such as `fp32_ge_of_real_ge_add_margin`.

# A Concrete Linear Layer Walkthrough

The linear theorem is the smallest example because it already has the same ingredients as a
network:

1. real tensors and real parameters define the ideal layer `y = Wx + b`;
2. FP32 tensors and FP32 parameters define the rounded layer;
3. `approxT_linear_fp32` supplies the output error budget;
4. a verifier or margin lemma spends that budget against the property we want.

```
import NN.Proofs.RuntimeApprox.FP32.Layers

open NN.Proofs.RuntimeApprox.FP32

#check approxT_linear_fp32
#check fp32_le_of_real_le_sub_margin
#check fp32_ge_of_real_ge_add_margin
```

The theorem is stronger than the prose claim "linear layers are close." It names the input
approximation, parameter approximation, output tolerance, and scalar semantics. Larger MLP and CROWN
statements reuse that shape.

# How The FP32 Theorems Fit The Tensor Implementation

The theorem names above are not isolated scalar lemmas. They sit on top of the same tensor and layer
specifications used elsewhere in TorchLean:

- `LinearSpec ℝ inDim outDim` is the real valued layer specification.
- `LinearSpec FP32.R inDim outDim` is the rounded-runtime layer at the proof scalar.
- `Tensor R s` is still a TorchLean tensor; only the scalar semantics changed.
- `approxT` is the relation saying the interpreted runtime tensor is within a tolerance of the
  real tensor, componentwise in the relevant norm.

The linear theorem is therefore an implementation bridge for a full layer:

```
import NN.Proofs.RuntimeApprox.FP32.Layers

open NN.Proofs.RuntimeApprox.FP32

-- Real linear layer and FP32 linear layer are related by an explicit output budget.
#check approxT_linear_fp32
```

The MLP theorems compose that layer theorem with rounded activations:

```
import NN.Proofs.RuntimeApprox.FP32.MLP

open NN.Proofs.RuntimeApprox.FP32

-- Whole-network approximation statements for common examples.
#check approxT_reluTwoLayerMlp_float32
#check approxT_tanhMlp3_fp32
```

Then the CROWN/IBP theorem widens a real valued verification box so it also contains the interpreted
FP32 output:

```
import NN.Proofs.RuntimeApprox.FP32.CROWN

open NN.Proofs.RuntimeApprox.FP32

#check ibpBound_contains_reluTwoLayerMlp_float32
```

The intended workflow is to train or import a model, lower or specify its tensor computation,
prove/check a real valued enclosure, and then spend an explicit FP32 error budget to keep the
checked claim connected to rounded execution.

# What People Usually Cite

Cite `FP32` for proof-model bounds, `IEEE32Exec` for bit behavior, and `NN/Proofs/RuntimeApprox/*`
for real-to-float transfer. Layer definitions: *Floating-Point Semantics*; tool integration:
*Verification*.

# References

- IEEE 754-2019 standard: https://standards.ieee.org/standard/754-2019.html
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic",
  https://dl.acm.org/doi/10.1145/103162.103163
- Flocq: https://flocq.gitlabpages.inria.fr/
- Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM, 2002), the standard
  numerical analysis reference for forward error and stability arguments of the kind TorchLean
  packages into margin-transfer lemmas.
