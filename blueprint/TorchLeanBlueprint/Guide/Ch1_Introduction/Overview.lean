import VersoManual

open Verso.Genre Manual

#doc (Manual) "What TorchLean Is" =>
%%%
tag := "overview"
%%%

TorchLean is a Lean 4 framework for building neural networks and making precise claims about them.
Its central requirement is that the model being executed, the graph being inspected, and the object
named in a theorem remain connected by definitions that a reader can follow.

Those connections are hard to keep as a model moves from source to checkpoint, graph, runtime,
verifier input, and paper claim. Each move can change a convention—layout, train/eval mode, scalar
semantics, input bounds—without crashing. TorchLean names the pieces inside one Lean development so
those conventions stay inspectable. PyTorch remains the everyday training ecosystem; TorchLean's
job is preserving meaning across the handoffs.

Model and training code begins with
[NN.API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API.lean). Files that combine
specifications, proofs, verification, and runtime internals can use the complete
[`NN`](https://github.com/lean-dojo/TorchLean/blob/main/NN.lean) umbrella. Subsystem development can
start from narrower entry points such as
[NN.GraphSpec](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec.lean),
[NN.IR](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR.lean), and
[NN.Verification](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification.lean).

# What TorchLean Covers

At a high level, TorchLean has three layers.

First, it has a familiar ML interface: tensors, layers, datasets, optimizers, losses, autograd,
logging, import/export, optional CUDA backed execution, and examples ranging from small MLPs to
CNNs, ViTs, GPT models, Mamba sequence models, diffusion, operator learning,
reinforcement learning, and scientific ML. Residual models and ResNets are represented in the
API, spec, and GraphSpec layers.

Second, it gives those programs a semantic core. Tensor shapes can appear in types. Architectures
can be lowered to a shared graph IR whose nodes name their operations. Scalar meanings can be real
valued, executable float32, float32 as modeled in proofs, intervals, or verifier domains. GraphSpec
lets an architecture be described once and read both as executable structure and as symbolic graph
structure.

Third, it has ways to check claims about the resulting artifacts: graphs that are well formed,
shape inference, IBP/CROWN bounds, JSON certificate replay, robustness margins, PINN
residuals, ODE enclosures, 3D projection certificates, attention mask contracts, and floating point
bridge statements.

These layers do not all have the same proof status. Some are executable infrastructure, some are
checked artifacts, and some are Lean theorems. TorchLean keeps those roles separate because they
support different kinds of claims.

# What "The Same Model" Means

TorchLean does not require every representation to be textually identical. A readable model builder,
a compact graph, and an executable runtime value should have different shapes. The important
question is whether a reader can follow the translations between them.

There are three recurring relations.

First, an *elaboration relation* says that user-facing syntax produced a particular architecture and
parameter interface. This is the ordinary programming layer: `nn.Sequential![...]` elaborates to a
model description with an input shape, an output shape, and a parameter layout.

Second, a *semantic relation* says that two objects denote the same mathematical function, or that
one object soundly approximates the other. A compiler theorem is an equality-style statement. An
interval or CROWN theorem is an enclosure-style statement. These are different claims and should not
be collapsed into one word like "verified."

Third, a *runtime relation* says that some executable path is intended to implement a semantic
object. This relation may be proved for a Lean executable fragment, checked by a regression test for
a native kernel, or assumed at an external boundary. TorchLean's job is to name the relation and
make the proof status visible.

The same model, then, means "connected by explicit definitions, checks, or theorems," not "unchanged
because the filenames look related."

# The Design Choice

Ordinary ML systems are often optimized around fast experimentation, which is the right default for
many projects. The cost is that key facts can remain implicit: tensor shapes, parameter
layouts, training mode, random state, scalar semantics, and the exact graph a verifier checked.
TorchLean makes a different tradeoff. We accept more typed structure at the boundary so later tools
can say precisely what they consumed and what they checked.

For example, a loss that expects logits and one-hot labels of shape `[batch, classes]` should not
silently receive labels of shape `[batch]`. A matrix multiply whose right-hand dimension is `64`
should not be applied to activations whose last dimension is `128`. In a dynamic runtime, those
errors may appear only after a particular batch reaches a particular kernel, or worse, they may be
hidden by broadcasting. In TorchLean's typed tensor APIs, the shape is part of the tensor type, so
the mismatch is rejected before the example is treated as a runnable model. If the intended design
really is sparse labels, a squeeze, or a reshape, we name that design choice explicitly.

The same discipline applies to semantic claims. A statement such as "this classifier is robust on
this input box" is meaningful only when it says which graph, parameters, scalar interpretation, and
checker result support the claim. Lean gives us a place to write that statement as a checked object,
not as a comment beside a script.

# Claim Vocabulary

- A *runtime result* is what an executable path produced on some inputs. It can be useful evidence,
  especially when compared against another backend, but it is not by itself a theorem.
- A *check* is an executable validation step. Shape inference, JSON parsing, certificate replay, and
  tolerance comparisons are checks. A check can reject bad artifacts, and if its implementation has
  a theorem behind it, it may also justify a formal claim.
- A *theorem* is a Lean statement accepted by the kernel. It may be about a model denotation, a
  compiler pass, a derivative, a bound, or a float32 approximation.
- A *trust boundary* is a place where TorchLean names an external producer or runtime rather than
  pretending Lean has proved its internals.

These words prevent three different results from being collapsed into one: a CUDA parity test, a
checker that recomputes a bound, and a theorem that proves an enclosure for a supported graph
fragment.

# The Object We Keep In View

Application code starts from `import NN.API` and `open TorchLean`. Under the hood, the same model
appears as a *spec* meaning, a *graph IR* artifact, and a *runtime* execution path; the relations
above say how those views are connected.

The object we keep returning to is a typed model. Informally, it has this shape:

$$`\mathrm{Model}_{\theta} :
   \mathrm{Tensor}(\alpha,s_{\mathrm{in}})
   \longrightarrow
   \mathrm{Tensor}(\alpha,s_{\mathrm{out}})`

Here `θ` is the parameter payload, `α` is the scalar type (for example reals or float32), and
`s_in` / `s_out` are the input and output shapes. The runtime may choose `Float`, executable
`IEEE32Exec`, or CUDA backed float32 buffers. A proof may instantiate the same architecture over
real numbers, interval domains, or a float32 proof model. The architecture statement should be
independent of the execution substrate.

The whole project follows a repeated theorem pattern:

$$`\begin{aligned}
&H(M,R)\\
&\Longrightarrow\;
  \forall x,\;
  \mathrm{denote}(R)(x)
  \in
  \mathrm{safeSet}(\mathrm{denote}(M)(x)).
\end{aligned}`

The variables are:

- `M` — the mathematical model under discussion (architecture plus parameters under a chosen scalar
  semantics);
- `R` — a runtime or checked representation of that model (for example a graph-plus-payload pair, an
  executable path, or a certificate-backed artifact);
- `H(M,R)` — the hypothesis relating them (elaboration agreement, semantic agreement, a checked
  certificate, or a named trust boundary);
- `x` — an input in the domain of the claim;
- `denote(M)` / `denote(R)` — the functions those objects compute;
- `safeSet(...)` — the set of outputs the claim allows for that input.

For an exact theorem, `safeSet` is a singleton. For a finite precision or verifier theorem, it may
be an interval, box, affine enclosure, or checker backed output region.

Different chapters instantiate that pattern in different ways. Autograd statements relate a reverse
pass to the derivative of a denotation. Runtime approximation statements relate executable values to
real specifications. CROWN and certificate checkers establish bound properties of an IR graph. CUDA
chapters name the native boundary when the implementation lives outside Lean.

# External Context

TorchLean sits between several mature traditions. PyTorch shows why an imperative, Pythonic style is
so effective for day-to-day deep learning engineering; see Paszke et al.,
["PyTorch: An Imperative Style, High-Performance Deep Learning Library"](https://arxiv.org/abs/1912.01703).
Lean supplies the dependent type theory and small-kernel proof-checking discipline described in the
[Lean language reference](https://lean-lang.org/doc/reference/latest/) and in de Moura et al.,
["The Lean Theorem Prover"](https://lean-lang.org/papers/system.pdf). Neural-network verification
contributes the bound-propagation and certificate ideas behind TorchLean's IBP, CROWN, and
LiRPA-style graph relaxations.

Those references are background, not claims that TorchLean inherits their results automatically.
TorchLean reuses ideas from the surrounding ecosystem while keeping its own proof objects, executable
checks, and external assumptions explicit.

# Fast When Needed, Explicit When It Matters

Prototyping can use the host runtime or CUDA float32 paths; formal guarantees use graph denotations,
Float32 models, checkers, and theorems. A typical silent failure is a checkpoint that loads by name
while one weight is transposed—see *Why Verification Matters* for the semantic-gap examples.

# References

- de Moura et al., ["The Lean Theorem Prover"](https://lean-lang.org/papers/system.pdf), CADE 2015.
- de Moura and Ullrich, ["The Lean 4 Theorem Prover and Programming
  Language"](https://lean-lang.org/papers/lean4.pdf), CADE 2021.
- Paszke et al., ["PyTorch: An Imperative Style, High-Performance Deep Learning
  Library"](https://arxiv.org/abs/1912.01703), NeurIPS 2019.
- Gowal et al., ["On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust
  Models"](https://arxiv.org/abs/1810.12715), 2018.
- Zhang et al., ["Efficient Neural Network Robustness Certification with General Activation
  Functions"](https://arxiv.org/abs/1811.00866), NeurIPS 2018.
- Xu et al., ["Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond"](https://arxiv.org/abs/2002.12920), NeurIPS 2020.
