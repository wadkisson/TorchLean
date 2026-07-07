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

# Comparison With ML Frameworks

PyTorch, JAX, TensorFlow, XLA, TVM, CUDA, Gymnasium, and α,β-CROWN are excellent tools. TorchLean is
not treating those tools as a problem. The difference is where the mathematical claim is
stated.

- *PyTorch autograd*: ordinary use relies on the dynamic autograd engine and kernels; TorchLean proves
  local VJP/JVP laws and then global backprop correctness for a supported tape/graph fragment.
- *`torch.compile` and graph lowering*: ordinary use relies on compiler passes plus regression tests;
  TorchLean proves that successful lowering from IR to executable graph preserves denotation for
  supported ops.
- *CUDA kernels*: ordinary use relies on native code, vendor libraries, and tests; TorchLean keeps
  CUDA attached to Lean specs and explicit runtime agreement statements.
- *α,β-CROWN*: ordinary use relies on external optimization/search or checks solver outputs informally;
  TorchLean parses or recomputes selected certificate steps and names the remaining oracle boundary.
- *Gymnasium/RL environments*: ordinary use relies on Python environment records; TorchLean can check
  emitted transition records against a Lean boundary contract.
- *Float32 execution*: ordinary use relies on backend arithmetic; TorchLean connects executable
  binary32 semantics, intervals, and explicit approximation theorems where the bridge is present.

The recurring TorchLean pattern is:

1. name the semantic object;
2. run or import an artifact;
3. check the artifact against the semantic object;
4. prove that the check implies the intended claim;
5. record native or external pieces as explicit assumptions until a checker or theorem discharges
   them.

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

# Autograd Proofs For Model Blocks

The autograd proof tree is not limited to scalar examples. It also contains theorem entry points
for selected model families.

Examples:

- The [softmax derivative API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/Softmax.lean) proves the derivative and adjoint
  identities behind softmax.
- The [log-softmax derivative API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/LogSoftmax.lean) records the log-softmax
  derivative account used by stable losses.
- The [attention proof API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/Models/Attention/) contains attention invariants such as
  weight normalization and causal mask properties.
- The [Transformer post-norm API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Transformer/PostNorm.lean) contains
  post-norm Transformer sublayer VJP theorems and the bridge theorem for two sublayers with post norm
  blocks.
- The [Elman cell API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Recurrent/ElmanCell.lean) gives a
  recurrent theorem for one cell and explains that full BPTT is the next induction over the unroll.

That last phrase separates proved fragments from open work. A theorem for one cell is not the same
as a full theorem for a sequence model. A Transformer sublayer theorem is not the same as all of
GPT-2. TorchLean names those boundaries instead of using one broad "autograd works" phrase for
everything.

# Runtime Approximation: Real Proofs Are Not Float Proofs

A common mistake in ML verification is to prove something over real numbers and then deploy Float32
code as if no bridge were needed. TorchLean has a separate runtime approximation tree because we do
not want to blur that boundary.

The approximation relation is:

$$`runtimeValue \approx_{\varepsilon} specValue`

For tensors, read this as a componentwise or normed error budget:

$$`\left\|\operatorname{toSpec}(y_{\mathrm{runtime}})-y_{\mathrm{spec}}\right\|_\infty
\le \varepsilon`

Then each op gets a rule for how its error budget propagates. Forward graphs, backward graphs,
convolutions, reductions, softmax axis rules, and scale aware tolerances all live in this layer.

The [runtime approximation core](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/RuntimeApprox/Core/) defines tolerances and
spec/runtime approximation relations. The
[forward graph approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean) and
[backward graph approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/BackwardApprox.lean)
propagate those relations through graph execution. The
[normal form operator API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Ops.lean),
[convolution forward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvForward.lean), and
[convolution backward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvBackward.lean) give the operator
cases. The [scale approximation API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/RuntimeApprox/Scale/) adds absolute and
relative tolerance tracking, and the
[FP32 CROWN bridge API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/FP32/CROWN.lean) connects approximation
budgets to verifier margins.

A deployment statement should separate three facts:

- a theorem over the reals says what the ideal spec does;
- a runtime approximation theorem says the executable path stays within a tolerance;
- an FP32 or CUDA boundary says which scalar/kernel assumptions remain.

# Reinforcement Learning: MDPs, PPO, Replay, And Boundaries

Reinforcement learning is another place where the claim matters. Saying "we implemented PPO" is a
runtime statement. TorchLean also has Lean definitions for transition data, rollout boundaries, and
MDP facts, so the executable algorithm can be separated from the mathematical claims attached to it.

Runtime side:

- [NN.Runtime.RL.Core API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Core.lean) defines transition and rollout data.
- [NN/Runtime/RL/PPO](https://github.com/lean-dojo/TorchLean/tree/main/NN/Runtime/RL/PPO/) covers PPO rollout collection.
- [NN/Runtime/RL/PolicyGradient](https://github.com/lean-dojo/TorchLean/tree/main/NN/Runtime/RL/PolicyGradient/) contains policy gradient and
  PPO objective code.
- [NN.Runtime.RL.Replay API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Replay.lean) defines replay buffers.
- [NN/Runtime/RL/Gymnasium](https://github.com/lean-dojo/TorchLean/tree/main/NN/Runtime/RL/Gymnasium/) is the external Python environment
  boundary.

Proof side:

- [NN.Proofs.RL.MDP API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/MDP.lean) and
  the [finite stochastic MDP API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/FiniteStochasticMDP.lean) prove monotonicity,
  contraction, uniqueness, and fixed point/error facts for Bellman operators.
- [NN.Proofs.RL.Envs.GridWorld API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Envs/GridWorld.lean) proves simple
  GridWorld transition facts and bridges deterministic successors to finite stochastic MDP rows.
- [NN.Proofs.RL.Replay API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Replay.lean) proves structural properties of replay
  buffer push/size behavior.
- [NN/Proofs/RL/Floats](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/RL/Floats/) connects discounted backups and TD residuals to
  executable float32 formulas under explicit finiteness checks.
- [NN.Proofs.RL.Boundary API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Boundary.lean) proves facts about checked boundary
  contracts.

Compared with ordinary Gymnasium usage, the difference is again the boundary. A Python environment
can return an observation and reward; TorchLean can additionally check that a transition record has
the expected shape/action/reward contract before treating it as evidence on a path toward a theorem.

# Generative Models: Objectives, Samplers, And Checked Claims

TorchLean has generative model examples and theory declarations for objectives, schedulers, and
sampler equations. Image-quality or distributional claims require additional evaluation assumptions,
but the mathematical pieces used by the models can still be named, inspected, and checked in Lean.

Theory APIs:

- [NN.MLTheory.Generative.Diffusion API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion.lean) collects the
  diffusion theory API.
- The [forward Gaussian API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion/ForwardGaussian.lean) states that
  affine forward noising of a standard Gaussian remains Gaussian.
- The [diffusion samplers API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion/Samplers.lean) records boundary and
  dynamics adapter facts for DDPM/DDIM/probability flow style samplers.
- The [VAE theory API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/VAE.lean) proves ELBO/KL style structural
  facts such as nonnegativity of the diagonal Gaussian KL to a standard normal.
- The [VQ-VAE theory API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/VQVAE.lean) proves nearest code and
  quantization loss facts.
- The [GAN theory API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/GAN.lean) records generator/discriminator
  loss decompositions.

Executable examples:

- [NN.Examples.Models.Generative.Diffusion source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Diffusion.lean)
- [autoencoder example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Autoencoder.lean)
- [VAE example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Vae.lean)
- [VQ-VAE example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/VqVae.lean)
- [GAN example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Gan.lean)

The precise wording is:

> TorchLean can run small generative examples and prove selected objective/sampler facts. The claim is
> not full generative model quality, distributional convergence, or production sampler correctness for
> every native kernel.

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

# BugZoo: Real Bugs As Checked Contracts

BugZoo is one of the most teachable parts of TorchLean because it connects formal methods to bugs
ML engineers already recognize.

The [BugZoo API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/) and
[BugZoo overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/README.md) map each example to real bug families.

- *Attention mask*: causal or future positions receive exactly zero attention weight under hard
  mask semantics.
- *KV cache*: appending a key/value makes that key/value the final cache entry.
- *RoPE position*: appending a decode token assigns the next position, avoiding position mismatch.
- *Tokenizer boundary*: token ids are `Fin vocabSize`, so out-of-vocabulary ids are not
  representable in the checked fragment.
- *Stable loss*: stable losses and safe domains are named instead of relying on backend numerics.
- *Ignored labels*: ignored labels contribute zero, including the case where all labels are ignored.
- *Normalization state*: BatchNorm epsilon placement and eval-time running stats become explicit.
- *Shape and broadcast*: intended axes and broadcasts become named shape operations.
- *Compiler boundary*: accepted backends must match source semantics.
- *Float boundary*: runtime Float32 ops are tied to explicit IEEE style semantics.

A concrete comparison with ordinary frameworks is:

- PyTorch usually catches these through tests, bug reports, runtime warnings, and regression suites.
- TorchLean turns the intended behavior into a small spec or theorem so the failure mode has a
  stable regression target.

# Widgets: Inspecting Proof Objects

Widgets matter because proof engineering and verifier debugging are much easier when the objects are
visible.

The [widgets source](https://github.com/lean-dojo/TorchLean/tree/main/NN/Widgets/) gives readable views of proof and runtime artifacts:

- IR graph views show node ids, shapes, op kinds, and execution traces.
- Runtime autograd widgets show tapes, gradients, and reverse traversal.
- CROWN widgets show bound propagation and tightness.
- Float32 widgets show the semantics of executable numeric values.
- RL widgets show trajectories and PPO/GridWorld artifacts.
- Training widgets show logs and metric traces.

The rule is:

> A widget is an inspection view for the same Lean object that a theorem or checker may later
> consume.

Widgets belong in the proof layer because they are the human interface to artifacts that would
otherwise be unreadable arrays, graph contexts, or certificate JSON.

# Model Examples: Runtime Breadth, Not Theorem Overclaiming

The model examples show breadth:

- MLP, CNN, ViT, plus residual/ResNet block specs;
- RNN, LSTM, Transformer, GPT examples, Mamba;
- diffusion, VAE, VQ-VAE, GAN;
- FNO/Burgers operator learning;
- DQN and PPO examples.

Not every model has a full correctness theorem. The examples share the same API, runtime, graph,
CUDA, data, and verification boundaries as the theorem files. That bridge matters: the formal
abstractions are tested against realistic ML shapes instead of living only in small proof snippets.

# Where To Go Next

The detailed follow-up chapters cover the major proof layers:

- *Graphs and IR* covers IRExec correctness and the compiled graph theorem.
- *Autograd Proofs* follows local VJP laws up to `backpropVec_eq_adjoint_fderiv`.
- *Runtime Approximation* explains tolerance propagation and finite precision assumptions.
- *Reinforcement Learning Stack* covers MDP proofs, PPO runtime objects, Gymnasium boundaries, and
  float32 diagnostics.
- *Generative Models* explains diffusion, VAE, VQ-VAE, GAN, and sampler facts.
- *Learning Theory* covers differential privacy, stability, robustness, and ridge regression.
- *BugZoo Catalog* gives the real bug case studies and their checked TorchLean boundaries.
- widget documentation that pairs each widget with the Lean object it visualizes.

# References

- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [JAX automatic differentiation guide](https://docs.jax.dev/en/latest/automatic-differentiation.html)
- [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [auto LiRPA project](https://github.com/Verified-Intelligence/auto%5FLiRPA)
- [VNN-COMP](https://vnn-comp.github.io/)
