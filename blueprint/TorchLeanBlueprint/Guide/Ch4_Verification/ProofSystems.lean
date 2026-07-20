import VersoManual

open Verso.Genre Manual

#doc (Manual) "Proof Systems Beyond Bounds" =>
%%%
tag := "proof-systems-beyond-bounds"
%%%

Verification in TorchLean means connecting an artifact to the semantics it claims to represent.

IBP and CROWN enclose outputs. Compiler correctness preserves graph meaning. The autograd theorems
connect backward rules to derivatives. Runtime approximation accounts for finite precision. BugZoo
contracts make common ML failure modes precise. These are different proof systems, but they follow
the same discipline: name the object, state the relation, and prove or check the relation for the
supported fragment.

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

In theorem notation, the supported-fragment statement has the shape:

$$`\operatorname{execGraphOfIR}(G,P)=\operatorname{ok}(E)
\;\land\; \operatorname{NoMSELoss}(G)
\;\land\; \operatorname{NoRawLog}(G)
\quad\Longrightarrow\quad
\forall x,\;
\operatorname{ExecGraphData.denoteAll}(E,x)
=
\operatorname{Graph.denoteAll}(G,P,x)`

For the covered IRExec fragment, this theorem prevents "verified the wrong executable graph."

There are side conditions, and they matter. The graph must pass the structural checks, compilation
must succeed, and the current theorem has explicit fragment predicates:

- `NoMSELoss g`, because that op is outside this whole-graph semantic equivalence proof path.
- `NoRawLog g`, because raw real `log` needs a positivity precondition. The local positive-domain
  branch is present, but the whole-graph theorem needs per-node domain facts to use it. Use the
  epsilon-protected safe-log operation when unconditional execution is intended.

That precision is part of the compiler proof: supported ops get named coverage, and unsupported ops
or ops needing extra domain facts do not get folded into the theorem by vague prose.

What this theorem rules out:

- In PyTorch or XLA style compilation, we usually rely on the compiler and test for regressions.
- In TorchLean's supported IRExec fragment, under the named side conditions, the compiler's forward
  result is tied to the IR denotation by a Lean theorem.
- This directly targets the classic silent wrong code problem: the optimized/executable graph
  should not secretly compute a different mathematical program.

The proof is large because it recursively mirrors the compiler. The workhorse lemma is
`buildFrom_preserves_denotation`: as the compiler walks node ids and extends the compiled graph, the
IR value table and compiled context stay aligned. Each operator branch proves one local
preservation fact, then the recursive theorem stitches the branch into the whole graph.

The current proof is split for auditability:

- The [IRExec common API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/Common.lean) contains shared
  infrastructure for dynamic values, compiled contexts, and finishing a node step.
- The [semantic equivalence common API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalenceCommon.lean)
  contains helper lemmas used by the recursive proof.
- The [semantic equivalence op cases API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalenceOpCases.lean)
  contains heavy named cases such as `.linear` and `.conv2d`.
- `Correctness/Ops/*` contains smaller branches by op family: activations, constants, elementwise,
  linear algebra, normalization, pooling, permutation, random, reductions, structural ops, and unary
  ops.
- The [semantic equivalence theorem API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalence.lean)
  ties the cases together into `execGraphOfIR_semantics_eq`.

Here TorchLean is doing more than a normal ML framework can do with tests. A regression test says
"we tried these examples." The theorem says "for every input, if the supported compiler accepts this
graph and the named fragment side conditions hold, the compiled evaluator and IR denotation agree."

# A Tiny IRExec Example

Imagine an IR graph for:

$$`y=\operatorname{ReLU}(Wx+b)`

There are two ways to evaluate it:

1. Interpret the IR directly:

- input node `0` gives `x`;
- linear node `1` reads `W` and `b` from the payload and computes `Wx+b`;
- ReLU node `2` computes `max(0,node1)`.

2. Compile the IR into `ExecGraphData` and run the compiled graph:

- compiled node `1` has a forward closure for the affine map;
- compiled node `2` has a forward closure for ReLU;
- the compiled evaluator visits the same dependency order as the IR denotation.

The theorem says these two paths produce the same result for every `x`, provided the compiler
accepted the graph and the ops are in the proved fragment. At that boundary, the compiled runtime
stops being "some other implementation" and becomes a proved refinement of the IR semantics.

# Autograd: From Runtime Gradients To VJP Theorems

PyTorch autograd is a very successful engineering system. In ordinary PyTorch use, the engine and
the derivative kernels supply the gradient path. If a custom op registers the wrong backward rule,
PyTorch may execute it perfectly and still return the wrong gradient.

TorchLean splits autograd into proof layers:

- *Local op rules*: each primitive VJP/JVP is the correct adjoint or derivative rule.
- *Tape and graph soundness*: composing locally correct nodes yields globally correct backprop.
- *Analytic bridge*: graph backprop equals the adjoint of the Fréchet derivative.
- *Model bridges*: attention, LayerNorm, Transformer sublayers, recurrent cells, and MLP losses can
  reuse the graph theorem.
- *Training algebra*: scalar loss gradients feed optimizer style updates explicitly.

The local correctness APIs, such as
[real autograd correctness](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Core/RealCorrectness.lean) and
[semiring autograd correctness](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Core/SemiringCorrectness.lean), prove
primitive derivative rules. The global tape APIs, such as
[tape algebra soundness](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean) and
[Fréchet derivative soundness](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Core/FDeriv.lean), lift those rules
to whole graphs. Model and training APIs, such as
[MLP/MSE derivative facts](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/MlpMse.lean) and
[training step algebra](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Training/StepAlgebra.lean), specialize the theorem
to concrete losses and optimizer style updates.

A helpful way to say the contrast is:

> PyTorch gives us `loss.backward()`. TorchLean proves, for supported graph fragments,
> that the reverse pass computes the adjoint of the derivative of the forward denotation.

The conceptual difference is large. The theorem is not "gradients looked plausible on a test
batch." It is a mathematical statement about the whole graph, under the stated operator/domain
conditions.

The central equation is the adjoint law:

$$`\left\langle \operatorname{JVP}_G(x;\dot x),\bar y\right\rangle
=
\left\langle \dot x,\operatorname{VJP}_G(x;\bar y)\right\rangle`

Once this is proved locally for primitive nodes and lifted through the graph, reverse mode becomes a
theorem about the graph denotation rather than only an execution trace.

# Learning Theory: DP, Stability, Robustness

The learning theory pages name definitions that mainstream ML code usually keeps in papers or
README prose. The
[differential privacy core](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/DifferentialPrivacy/Core.lean) defines
`(ε, δ)` differential privacy over events using `ProbabilityMeasure`, then proves monotonicity in
`δ` and closure under post processing. The
[stability core](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Core.lean) defines supervised learning
stability over typed datasets. The
[ridge regression theorem API](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/)
gives a worked stability theorem with real and `IEEE32Exec` versions. The
[robustness spec API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Spec.lean) names perturbation and
robustness predicates, while the
[robustness runtime API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Runtime.lean) gives
deterministic diagnostics without treating sampled attacks as certificates.

This material is closer to mathematical ML theory than to runtime training. Definitions such as
"DP is preserved under post processing" or "this Bellman operator is a contraction" belong in the
theory layer, not buried inside model examples.

# Additional Formulas And Snippets

$$`\operatorname{Graph.denote}(g,payload,input)
=
\operatorname{evalForward}(p,params,input)`

# References

- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [JAX automatic differentiation guide](https://docs.jax.dev/en/latest/automatic-differentiation.html)
- [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [auto LiRPA project](https://github.com/Verified-Intelligence/auto%5FLiRPA)
- [VNN-COMP](https://vnn-comp.github.io/)
