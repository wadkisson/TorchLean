import VersoManual

open Verso.Genre Manual

#doc (Manual) "Runtime Approximation" =>
%%%
tag := "runtime-approximation"
%%%

A theorem over the reals and a claim about execution are different statements. Runtime
approximation is the layer that connects them.

The basic claim is not equality. It is a tolerance statement: after interpreting the runtime value
in the spec domain, it lies within an explicit error budget of the ideal value:

$$`\operatorname{toSpec}(y_{\mathrm{run}})\approx_\varepsilon y_{\mathrm{spec}}`

or, for a normed tensor statement,

$$`\left\|\operatorname{toSpec}(Y_{\mathrm{run}})-Y_{\mathrm{spec}}\right\|_\infty
\le \varepsilon.`

We encourage readers to come here after the autograd proof layer. The autograd theorems explain what
the ideal forward and backward maps mean; here we explain how far an executable or rounded path
may differ from that ideal, and how we keep that difference explicit.

# Why Real Proofs Are Not Float Proofs

It is tempting to prove a property over real numbers and then deploy float32 code with the same
sentence in mind. TorchLean does not allow that step to be implicit: a real valued safety theorem
does not become a float32 safety theorem unless there is a bridge theorem that relates the runtime
path to the real specification.

That implication is not free. It needs a bridge theorem. Rounding, cancellation, overflow,
different reduction orders, fused kernels, and domain guards can all change the behavior. A
mathematical softmax is smooth; an implementation may use max-subtraction, exponent approximations,
finite accumulators, and library calls specific to a backend. A real convolution is a sum; a GPU
convolution may choose a different accumulation order or fused algorithm.

TorchLean's answer is to name the bridge:

1. prove the claim over the reals or over the spec;
2. prove an approximation relation between runtime tensors and spec tensors;
3. inflate margins, enclosures, or certificates by the approximation budget;
4. leave native hardware assumptions explicit when they are outside Lean.

This matches standard numerical analysis practice; see Higham's *Accuracy and Stability of
Numerical Algorithms* (https://epubs.siam.org/doi/book/10.1137/1.9780898718027) for the classical
background. It also matches the caution needed in neural network verification, where a tiny
rounding gap can matter if the verified margin is tiny.

# The Core Relation: Spec Tensors And Tolerances

The small vocabulary is defined by
[runtime approximation core](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core.lean),
[spec approximation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/SpecApprox.lean), and
[tolerances](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/Tolerance.lean).

The spec approximation layer defines the basic tensor relation. Given a conversion
`toSpec : α → SpecScalar`, a runtime tensor approximates a spec tensor when every component is close after
conversion to the spec scalar. The key names are:

- `tensorToSpec`: pointwise conversion from runtime tensor values to spec scalars.
- `linfNorm`: max style tensor norm used for error statements.
- `approxWith`: absolute tensor approximation using an explicit error tensor.
- `approxWithTol`: approximation using a tolerance object.
- `approxTTol`: packaged tensor tolerance relation.
- `Witness`: a small record for carrying a runtime value and its error evidence.

The tolerance API defines `ApproxTol`, with absolute, relative, and slack components.
`ApproxTol.absOnly`, `approxBound`, and `approxR` let later proofs move between simple absolute
error statements and more scale aware claims.

For a single scalar, the scale-aware shape is:

$$`|r-s|
\le
\varepsilon_{\mathrm{abs}}
+\varepsilon_{\mathrm{rel}}\,|s|
+\varepsilon_{\mathrm{slack}}.`

That distinction is practical. An absolute tolerance of `1e-6` may be meaningful near zero and
irrelevant near `1e9`. A relative tolerance captures "small compared with the scale of the value."
TorchLean uses both styles, but it makes the choice explicit instead of burying it in test
thresholds.

## Reading A Tolerance Statement As A Contract

The tolerance relation is deliberately stronger than a unit test. A test has observed inputs. A
tolerance theorem has quantified inputs and a stated scalar interpretation:

```
toSpec : RuntimeScalar -> SpecScalar
runtime : Tensor RuntimeScalar s
spec    : Tensor SpecScalar s
eps     : ApproxTol
```

The proposition says every coordinate of `tensorToSpec toSpec runtime` is close to the matching
coordinate of `spec` under `eps`. That makes the trusted boundary easy to locate. If `toSpec`
interprets an executable rounded-real model, the theorem is about that rounded-real model. If the
actual deployment path is CUDA, cuBLAS, PyTorch, or a fused native kernel, a separate agreement
statement is needed before the theorem says anything about that path.

The most common deployment mistake is to skip this last sentence. A real theorem plus a numerical
test is useful evidence, but it is not a theorem about the tested runtime unless the tested runtime
is connected to the theorem's scalar model.

# Forward Graph Approximation

The forward graph theorem is in
[NN.Proofs.RuntimeApprox.Graph.ForwardApprox API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean).
It mirrors the structure of the autograd tape proof, but the invariant is approximation rather than
derivative soundness.

The main objects are:

- `EList`: list of scalar error budgets indexed by shape.
- `approxCtx`: approximation relation for a whole context.
- `FwdNode`: a node with spec forward, runtime forward, bound, and local soundness.
- `FwdGraph`: a snoc list graph of forward nodes that carry approximation evidence.
- `FwdGraph.eval_approx`: theorem for the whole forward graph.

The local theorem on a `FwdNode` says: if each runtime input approximates the corresponding spec
input, then the runtime output approximates the spec output within this node's bound.

`FwdGraph.eval_approx` composes those local statements. When a graph appends a node, the proof uses
the node's local `sound` theorem, appends the new bound to the error context, and continues. It uses
the same architecture as the autograd soundness theorem: local correctness first, then a
global induction over the graph.

A tiny example is multiplication. The spec value is the real product, while the runtime value is the
rounded product computed by the chosen scalar model.

If `x_run` is within `eps_x` of `x_real`, and `y_run` is within `eps_y` of `y_real`, the local
multiplication lemma supplies a bound for `z_run` versus `z_real`. The graph theorem then lets that
new bound feed the next node.

# Backward Graph Approximation

Backward approximation is developed in the
[backward approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/BackwardApprox.lean).
This file is the runtime approximation analogue of reverse mode AD.

The key objects are:

- `RevNode`: a forward node plus spec/runtime VJP functions and a VJP error transformer.
- `RevGraph`: reverse graph with local approximation evidence.
- `RevGraph.eval_approx`: forward approximation inherited from the forward graph.
- `RevGraph.backprop_approx`: theorem for reverse accumulation across the whole graph.

The theorem `RevGraph.backprop_approx` states that the runtime reverse pass approximates the spec
reverse pass, provided the input context, forward tape, and seed cotangents are appropriately
related. Addition during gradient accumulation is not treated as automatic; the theorem takes an
explicit `addBound` and `addSound` describing how accumulation affects error.

Backward passes are often dominated by sums: cotangents from fanout, reductions, convolution
gradients, and parameter gradient accumulation. Floating point addition is not associative, and
accumulation order matters. By making addition soundness an explicit parameter, the theorem states
the arithmetic model being used.

# Link Back To Autograd Algebra

The bridge file
[NN.Proofs.RuntimeApprox.Graph.LinkAutogradAlgebra API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/LinkAutogradAlgebra.lean)
connects the runtime approximation graph shape back to the autograd algebra graph.

This bridge is structural. The runtime approximation layer uses the same
shape-indexed tensor-context idea that the public API exposes as `TensorPack`, so the conversion is
not a semantic reinterpretation. The file defines `toNodeData` and `toGraphData`, then proves:

- `evalRuntime_of_toGraphData`;
- `backpropRuntime_of_toGraphData`.

In words, the runtime approximation graph can be viewed as an autograd algebra graph by forgetting
the approximation evidence and keeping the same forward/VJP structure. That means the two layers
compose cleanly:

autograd algebra: the reverse pass is the correct ideal VJP; runtime approximation: the executable
reverse pass stays close to that ideal VJP.

This claim needs careful wording. TorchLean does not collapse "correct gradient" and
"float gradient close to correct gradient" into one claim. The first is an equality theorem over the
ideal semantics. The second is an approximation theorem with a tolerance.

# NF Operations: Rounded Real Arithmetic

The largest collection of local rules is
[NN.Proofs.RuntimeApprox.NF.Ops API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Ops.lean). `NF` is
TorchLean's rounded real neural float model. It gives arithmetic for proofs that resembles finite
precision without claiming that every hardware corner case has disappeared.

The file includes scalar and tensor approximation lemmas for common operations:

- arithmetic: `approx_add_nf`, `approx_sub_nf`, `approx_mul_nf`, `approx_div_nf_of_lb`;
- unary functions: `approx_exp_nf`, `approx_tanh_nf`, `approx_abs_nf`, `approx_neg_nf`;
- guarded operations: `safeLog`, `safeDiv`, `safe_log`;
- tensor rules: `approxT_add_spec`, `approxT_mul_spec`, `approxT_exp_spec`, `approxT_relu_spec`;
- graph nodes: `addNode`, `mulNode`, `expNode`, `reluNode`, `safeDivNode`, `softmaxNode`, `sumNode`.

Several of these lemmas make the numerical analysis tradeoff visible. Division requires a lower
bound away from zero (`approx_div_nf_of_lb`) or a guarded form (`safeDiv`). Square root and log need
domain protection. Exponential uses a mean value bound. Multiplication propagates both input
errors and a product term. These are the exact places where a proof over real numbers would be too
optimistic if copied directly onto a float implementation.

A safe division example has three pieces: the mathematical value is a guarded division, the runtime
value is computed by `safeDivR eps xR yR`, and the theorem states that the runtime value
approximates the guarded spec value under the declared tolerance.

The theorem is not about arbitrary division at `y = 0`. It uses a guarded operation because the
proof should match the stable programming pattern users ought to deploy.

## Reductions And Softmax: Where Error Budgets Grow

Elementwise operations are only half the story. Reductions and normalization layers create coupled
error terms because many inputs flow into one output. TorchLean has explicit reduction approximation
lemmas in
[NN.Proofs.RuntimeApprox.NF.ReductionOps](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ReductionOps.lean):

```
#check NN.Proofs.RuntimeApprox.NF.approxT_reduce_sum_by_row_2d
#check NN.Proofs.RuntimeApprox.NF.approxT_reduce_mean_by_row_2d
#check NN.Proofs.RuntimeApprox.NF.approxT_reduce_sum_by_column_2d
```

The row-sum theorem is not just "sum is close to sum." It accounts for the number of accumulated
terms and for the same row/column indexing used by the executable reducer. That is the sort of
detail that matters for LayerNorm, attention logits, pooled features, and minibatch losses.

Softmax needs even more care. Scalar logistic-style bounds are not a proof of axis softmax, because
axis softmax couples every coordinate through the denominator. TorchLean keeps that boundary
visible in
[NN.Proofs.RuntimeApprox.NF.SoftmaxAxis](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/SoftmaxAxis.lean):
the file states the intended location for a future vector/axis theorem and explicitly avoids
pretending that scalar bounds cover the coupled denominator proof. That is a useful negative result:
the absence of a theorem is recorded as a boundary, not papered over by an approximate-looking name.

# Convolution Forward And Backward

Convolution stresses the approximation layer because it mixes indexing, padding, nested sums, and
gradients with respect to inputs and parameters. TorchLean has dedicated normal form APIs:

- [NN.Proofs.RuntimeApprox.NF.ConvForward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvForward.lean)
- [NN.Proofs.RuntimeApprox.NF.ConvBackward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvBackward.lean)
- [NN.Proofs.RuntimeApprox.NF.Conv API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Conv.lean)

The convolution forward API proves pointwise and full tensor approximation for convolution forward:

- `approx_conv2d_point` for one output coordinate;
- `approxT_conv2d_spec` for the full output tensor;
- `conv2dNode` to package the theorem as a graph node.

The proof has a lot of indexing work because the spec definition and the runtime replay need to
read the same padded input entries. Lemmas such as `foldl_flatMap`,
`entry_eq_scalar_get_at_or_zero3`, and facts about padded input alignment are essential; they are
what prevents the proof from silently comparing two different convolutions.

The convolution backward API covers the three reverse operators:

- `approxT_conv2d_bias_deriv_spec`;
- `approxT_conv2d_kernel_deriv_spec`;
- `approxT_conv2d_input_deriv_spec`.

It also packages the result as `conv2dRevNode`, so the backward approximation theorem can compose
convolution with the rest of a reverse graph. Here is the local-to-global pattern in its cleanest
form: first prove the hard local operator approximation, then hand it to `RevGraph`.

# Scale Aware Tolerances

The scale layer is split across
[scale bounds](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale.lean),
[scale approximation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/ScaleApprox.lean),
[forward scale propagation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/ForwardScale.lean), and
[backward scale propagation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/BackwardScale.lean).

The scale approximation API defines `BList`, a list of nonnegative scale bounds indexed by shape,
plus helpers such as `scaleT`, `scaleCtx`, and `tolFromEpsScale`. An absolute error budget is
computed from a machine-like epsilon times a local scale bound.

A graph can then carry both "how close" and "at what scale" information. The lemmas
`approxTTol_from_scale` and `approxCtx_get_tolFromEpsScale` connect scale estimates back to the
tolerance API used by graph theorems.

This remains a separate layer because not every proof needs scale aware reasoning. Small examples
and operator proofs written by hand are often clearer with absolute tolerances. Larger deployment
claims usually need scale, because one global absolute epsilon is rarely meaningful across all
activations and gradients.

# FP32 And Verification Margins

The [FP32 layer approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/FP32/Layers.lean),
[FP32 MLP approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/FP32/MLP.lean), and
[FP32 CROWN bridge API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/FP32/CROWN.lean) connect the approximation
style to layerwise and verifier reasoning.

The CROWN connection is where runtime approximation meets certification. A verifier over the reals
may prove a margin, but a float runtime can differ from ideal real arithmetic. To transfer the claim,
the real margin must dominate the runtime approximation budget: the certified lower margin after
subtracting the runtime error still has to be positive.

When that inequality is proved, the runtime prediction is still certified.

The Float32 soundness layer uses the same separation. `FP32` is a proof model, `IEEE32Exec` is an
executable bit oriented model, and native hardware remains a named assumption unless a bridge
theorem covers the path being used.

# A Worked Mental Example

Suppose we have a two layer classifier whose hidden layer is `ReLU (W1 x + b1)` and whose output is
`W2 h + b2`.

The ideal proof might establish that, for all inputs in a box, the margin over the reals for class `0`
over class `1` is at least `0.05`.

A float deployment theorem adds the runtime approximation statement: for every input in the same
box, `runtime_y` approximates `real_y` within `0.01` per relevant logit.

Then the margin proof must be adjusted:

$$`f_0^{\mathrm{run}}(x)-f_1^{\mathrm{run}}(x)
\ge 0.05-0.01-0.01
=0.03>0.`

The calculation is small, but it is the whole point of the layer. The real theorem, the runtime
approximation theorem, and the final runtime claim each spend a named budget.

Only after that bridge do we get a classification claim about runtime behavior. TorchLean keeps
autograd algebra, verifier bounds, FP32 semantics, and runtime approximation as separate proof
layers because they are different mathematical facts that happen to support one deployment
workflow.

# End To End NF Graph Snippets

The normal-form end-to-end file packages the graph story into two declarations:

```
#check NN.Proofs.RuntimeApprox.NF.eval_approx_graphData
#check NN.Proofs.RuntimeApprox.NF.backprop_approx_graphData
```

These are graph-level bridge theorems:

- `eval_approx_graphData` says evaluating the executable forward graph is close to evaluating the
  spec forward graph when the local node approximation obligations have been supplied.
- `backprop_approx_graphData` says the same style of statement for reverse accumulation, with the
  accumulation error model still explicit.

That is the runtime approximation analogue of the autograd proof architecture. Local operator
lemmas are the leaves; graph theorems compose them; deployment claims then combine the graph theorem
with any scalar/backend assumptions.

# Runtime Approximation APIs

The definitions are organized in the same order as the proof: define closeness, prove local operator
bounds, compose them over forward and backward graphs, and finally connect the result to autograd.

- The [tolerance API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/Tolerance.lean) and
  [spec approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/SpecApprox.lean) define the approximation relation.
- The [forward graph approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean) contains
  `FwdGraph.eval_approx`; the [backward graph approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/BackwardApprox.lean)
  contains `RevGraph.backprop_approx`.
- The [normal form operator API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Ops.lean) supplies local obligations, including
  domain-sensitive operations such as division and safe log.
- The [convolution forward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvForward.lean) and
  [convolution backward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvBackward.lean) handle a larger operator family.
- The [scale approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/ScaleApprox.lean) supports scale-aware error bounds.
- The [autograd algebra link API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/LinkAutogradAlgebra.lean)
  connects approximation to the autograd proof layer.

# Runtime Agreement

For supported graph and operator fragments, runtime approximation proves that a runtime or rounded
computation stays within a stated tolerance of a spec computation. CUDA kernels, vendor library
paths, compiler rewrites, and PyTorch-exported graphs need their own agreement statements when a
claim is about those paths.

The intended TorchLean shape is to allow fast code and imported artifacts while keeping the
mathematical promise visible. A deployment claim combines a real proof, an approximation theorem,
and any finite or runtime assumptions that remain outside the theorem.

When the approximation theorem is present, the bridge is proved. When it is not present, the claim
should say which runtime path or external producer supplies the remaining evidence.
