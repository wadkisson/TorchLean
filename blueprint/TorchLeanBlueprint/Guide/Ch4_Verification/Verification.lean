import VersoManual

open Verso.Genre Manual

#doc (Manual) "Verification" =>
%%%
tag := "verification"
%%%


Testing asks what a network did on the inputs we tried. Verification asks what it must do on every
input in a set. For a classifier `f`, an input region `X`, the intended class `y`, and a competing
class `j`, a typical robustness target is

$$`\forall x\in X,\qquad f_y(x)-f_j(x)>0.`

That single formula hides most of the engineering. We need to know which model `f` denotes, how
`X` was represented, how the output bounds were obtained, and whether the arithmetic was exact or
rounded. TorchLean therefore keeps apart four objects that are often compressed into the word
"certificate":

1. the exact graph semantics and mathematical property;
2. a bound-propagation theorem over those semantics;
3. an executable checker or imported artifact;
4. a bridge proving that accepted executable evidence satisfies the theorem's hypotheses.

Only their composition yields the final claim. We begin with a complete executable run, then open
it layer by layer.

# A Complete Robustness Run

The bundled robustness workflow constructs a two-output network, compiles it to TorchLean's
canonical IR, places an `L∞` box of radius `0.1` around `[1, 1]`, and computes both IBP and CROWN
bounds:

```
lake exe verify -- torchlean-robustness
```

The relevant part of the output is:

```
compiled IR nodes: 4
x0 = [1.000000, 1.000000], eps = 0.100000
[IBP] logits lo = [1.800000, -2.200000]
[IBP] logits hi = [2.200000, -1.800000]
[IBP] margin(lo0 - hi1) = 3.600000
[IBP] certified? true
[CROWN] margin(lo0 - hi1) = 3.600000
[CROWN] certified? true
[CROWN-backward] margin lo = 3.600000
[CROWN-backward] margin hi = 4.400000
[CROWN-backward] certified? true
```

The interval calculation says that class-zero's logit is at least `1.8`, while class-one's logit is
at most `-1.8`. Consequently,

$$`\inf_{x\in X}(f_0(x)-f_1(x))\ge 1.8-(-1.8)=3.6>0.`

The printed `true` is useful, but it is not itself the theorem. To turn the run into a proof, we
must connect the following statements:

1. the compiled graph denotes the source model;
2. the bound propagation encloses the graph denotation on `X`;
3. the positive lower margin implies the classification property;
4. if the claim concerns native Float32 execution, the native path refines the arithmetic used in
   the proof.

This distinction is practical. If the model compiler changes, obligation 1 is the place to look.
If a new activation is added to CROWN, obligation 2 changes. If the deployment claim concerns
CUDA rather than real semantics, obligation 4 cannot be skipped.

# Semantic Target And Graph Boundary

The verifier operates on the canonical `NN.IR.Graph`. An interval or affine form is meaningful only
relative to a denotation of that same graph, parameter store, and input box. A compiler theorem is
therefore part of a source-model claim.

TorchLean has two relevant forward correspondences:

- The typed first-order
  [proved forward fragment](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Proved/Public.lean)
  compiles `NN.Verification.TorchLean.Proved.Program` values. Its constructors cover constants,
  parameters, arithmetic, ReLU, `exp`, `log`, inverse, matrix products, reshapes and permutations,
  last-axis softmax, 2D LayerNorm, linear and convolution layers, and MSE loss.
  `compileForward_wellFormed` proves structural well-formedness, while
  `runForwardIR_eq_evalForward` proves equality with the typed program evaluator.
- `execGraphOfIR_semantics_eq` proves that a successful lowering from canonical IR to Lean's
  executable autograd `ExecGraphData` preserves denotation for every input, under `NoMSELoss` and
  `NoRawLog`.

Both are Lean semantic equalities over an abstract scalar `Context`. They are not statements that a
PyTorch module, CUDA kernel, or vendor library agrees with the graph. General API compilation also
does not inherit the typed-fragment theorem merely because it returns the same IR type.

```
#check NN.Verification.TorchLean.Proved.compileForward_wellFormed
#check NN.Verification.TorchLean.Proved.runForwardIR_eq_evalForward
#check NN.Runtime.Autograd.Compiled.IRExec.Correctness.execGraphOfIR_semantics_eq
```

# IBP

Interval bound propagation assigns each node a box. For an affine layer

$$`y=Wx+b,\qquad x\in[\ell,u],`

the usual sign split gives

$$`\ell_y=W^+\ell+W^-u+b,\qquad
u_y=W^+u+W^-\ell+b,`

where `W^+ = max(W,0)` and `W^- = min(W,0)`. Monotone activations transform endpoints; ReLU maps
`[\ell,u]` to `[max(0,\ell),max(0,u)]`. Elementwise multiplication needs all endpoint products.

The generic real soundness theorem is `cert_encloses_semantics`. It requires:

- `TopoSorted g`;
- `Supported g`;
- exact local certificate consistency `CertLocalOK`;
- exact local value consistency `SemLocalOK`;
- `InputsEnclosed`.

It concludes that each available certificate box encloses the matching semantic value. The current
`Supported` predicate contains input, constant, detach, addition, subtraction, elementwise
multiplication, ReLU, linear, matrix multiplication, tanh, sigmoid, sine, and cosine nodes.

The proof-side real evaluator has the stronger end-to-end theorem
`runIBP?_encloses_evalGraphRec`. It proves that the particular real `runIBP?` construction encloses
`evalGraphRec`, under topological order, supported operations, and enclosed inputs. This is a
theorem about those proof-side definitions; it is not automatically a theorem about every
executable `Graph.runIBP` path.

# Proof Systems Beyond Bounds

Verification in TorchLean means connecting an artifact to the semantics it claims to represent.

IBP and CROWN enclose outputs. Compiler correctness preserves graph meaning. The autograd theorems
connect backward rules to derivatives. Runtime approximation accounts for finite precision. BugZoo
contracts make common ML failure modes precise. These are different proof systems, but they follow
the same discipline: name the object, state the relation, and prove or check the relation for the
supported fragment.

# Proof Obligations As Relations

The common structure is relational. Let `S` be a semantic object, `A` an executable or imported
artifact, and `R S A` the proposition that the artifact represents the semantics correctly. A
compiler proof establishes `R` for the executable graph it produces. A certificate checker parses
an untrusted artifact and returns evidence from which `R` follows. A backend contract records `R`
as an assumption when the implementation remains outside the proved fragment.

For a checker

$$`\operatorname{check}:A\to\operatorname{Except}(\mathrm{Error},W),`

the useful theorem is not that the parser terminates. It has the form

$$`\operatorname{check}(a)=\operatorname{ok}(w)
\quad\Longrightarrow\quad
R(S,\operatorname{decode}(a),w).`

This direction matters. The producer may be Python, CUDA, α,β-CROWN, or a remote solver; none of
those programs must be trusted merely because the checker accepts their output. Trust moves to the
Lean definition of `R`, the checker, and its soundness theorem. When no such theorem exists, the
artifact is evidence or an explicit boundary, not a certificate by vocabulary alone.

# IRExec Correctness: The Big Compiler Theorem

The theorem to care about first is:

```
Runtime.Autograd.Compiled.execGraphOfIR_semantics_eq
```

The declaration is in the
[IR execution correctness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalence.lean).

In plain English:

> If `execGraphOfIR` successfully compiles an `NN.IR.Graph` and payload into executable compiled
> graph data, and the graph is in the named supported fragment, then evaluating the compiled graph
> on any input gives the same value table as the Lean denotational evaluator for the original IR
> graph.

The theorem connects three objects:

- `NN.IR.Graph.denoteAll`: the reference denotational semantics of the tagged op IR.
- `execGraphOfIR`: the compiler from IR to executable compiled graph data.
- `ExecGraphData.denoteAll`: the compiled runtime evaluator.

The theorem shape is: if `execGraphOfIR g payload` returns `ok exec`, and the named fragment
side conditions hold, then for every input `x`, the value table produced by
`ExecGraphData.denoteAll exec x` is the same value table produced by
`NN.IR.Graph.denoteAll g payload x`.
