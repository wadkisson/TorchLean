import VersoManual

open Verso.Genre Manual

#doc (Manual) "Autograd And Approximation Proofs" =>
%%%
tag := "autograd-proofs"
%%%


Most ML users learn gradients operationally: run the forward pass, call
`loss.backward()`, inspect a few numbers, and move on. That workflow is powerful, and PyTorch and
JAX made it practical at enormous scale. TorchLean keeps the same programmer intuition and puts an
explicit theorem underneath it.

The difference is not that TorchLean "has autograd" while PyTorch and JAX do not. They do, and they
do it very well. The difference is that ordinary framework use usually relies on an implementation
of reverse mode AD, its derivative registrations, compiler rewrites, and native kernels.
TorchLean's proof tree asks a more explicit question:

> For the supported graph fragment, does the reverse pass compute the adjoint of the derivative of
> the forward denotation?

The proof architecture lives in [NN/Proofs/Autograd](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/Autograd/). We built it in layers so the
auditor can stop at the boundary they care about: algebraic tape soundness, Frechet derivative
statements, proofs for particular operators, theorem entry points for model blocks, and algebra for
training steps.

The proof pipeline is:

```
local op derivative
  -> local JVP/VJP adjointness
  -> graph backprop correctness
  -> Frechet derivative theorem
  -> model block theorem
  -> training step algebra
```

That ordering is the proof structure. A theorem about a Transformer sublayer is assembled from
local derivative rules, graph composition, and the analytic bridge to the derivative of the
denotation.

# From Runtime Confidence To Theorem Obligations

In PyTorch, the default runtime model is approximately:

1. the eager engine records operations into a dynamic tape;
2. each operation has a backward rule, either built in or registered by extension code;
3. the engine traverses the tape in reverse and accumulates gradients;
4. tests, numerical checks, and framework maintenance give confidence that the result is right.

JAX moves the same idea into a functional transformation pipeline: `grad`, `vjp`, `jvp`, `jit`, and
lowering passes cooperate to produce differentiated and compiled programs. See the PyTorch autograd
overview at https://pytorch.org/docs/stable/autograd.html and JAX's autodiff guide at
https://jax.readthedocs.io/en/latest/automatic-differentiation.html for the user facing version of
that workflow.

TorchLean keeps the same operational picture: local derivative rules plus reverse accumulation.
What changes is that runtime confidence is split into named theorem obligations.

- A registered backward rule becomes a local JVP/VJP or Fréchet derivative lemma for the op.
- A tape reversal becomes global tape soundness by induction over the graph.
- A scalar loss training step becomes an explicit theorem about the loss seed and parameter update
  algebra.
- A model block such as attention or an RNN cell becomes a packaged theorem with stated hypotheses.

The resulting style is slower to author, but easier to audit. We can say exactly which part is a
Lean theorem and which part is a runtime, compiler, or finite precision agreement statement.

# Tape Soundness: The Algebraic Core

The central algebraic file is
[NN.Proofs.Autograd.Tape.Algebra.Soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean).
It defines the small tape language used by the rest of the autograd proofs, together with the local
proof data carried by each node.

The objects to track are:

- `TensorPack`: public typed tensor payloads indexed by a list of shapes; internally, the
  autograd algebra still has a small context datatype with the same shape-indexed structure.
- `Idx`: pointer into a context, carrying the needed proof data.
- `NodeData`: forward, JVP, and VJP data for one local operation.
- `Node`: a node plus the local inner product soundness law.
- `GraphData`: executable snoc list graph data.
- `Graph`: graph whose nodes can share earlier values and carry local proof obligations.
- `Graph.backprop_correct`: global dot product soundness theorem.

The theorem `Graph.backprop_correct` is the first big hinge. Informally, it says that reverse
accumulation is the adjoint of forward sensitivity:

$$`\langle \operatorname{jvp}_G(x, dx), seed\rangle = \langle dx, \operatorname{backprop}_G(x, seed)\rangle`

This dot product identity is the algebraic essence of reverse mode AD. A forward sensitivity `dx`
pushed through the graph and then paired with an output cotangent gives the same scalar as pairing
`dx` with the cotangent produced by reverse accumulation.

The dot product statement is the common language between implementation and analysis. Runtime
engineers recognize it as the VJP law. Mathematicians recognize it as the adjoint property. Lean can
prove it compositionally: each node gives the local adjoint law, and graph
soundness follows by induction over `Graph`.

The comparison to PyTorch is most direct here. PyTorch's engine performs a reverse walk over a
dynamic graph and accumulates cotangents into inputs. TorchLean's algebraic graph does the same
conceptual work, but the graph object carries enough structure for Lean to prove that the
accumulation is sound for every input context in the supported fragment.

# From Dot Products To Frechet Derivatives

The algebraic theorem is not the last word. A dot product VJP law still has to be connected to the
function being differentiated. The
[Frechet derivative bridge](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Core/FDeriv.lean) provides that link.

That file vectorizes shaped tensor contexts into Euclidean spaces and connects three views:

# Runtime Approximation

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

# Reading A Tolerance Statement As A Contract

The tolerance relation is deliberately stronger than a unit test. A test has observed inputs. A
tolerance theorem has quantified inputs and a stated scalar interpretation:

```
toSpec : RuntimeScalar -> SpecScalar
runtime : Tensor RuntimeScalar s
spec    : Tensor SpecScalar s
eps     : ApproxTol
```

A proof then discharges `approxTTol toSpec runtime spec eps` (or an equivalent packaging) instead of
treating a handful of printed floats as evidence.
