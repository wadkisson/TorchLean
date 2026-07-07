import VersoManual

open Verso.Genre Manual

#doc (Manual) "What TorchLean Is" =>
%%%
tag := "overview"
%%%

TorchLean is a Lean 4 framework for neural networks that can be run as programs and discussed as
mathematical objects. The central design constraint is simple: the model we execute, the graph we
inspect, and the object we mention in a theorem should still be recognizably the same model.

That sentence is easy to say and surprisingly hard to maintain in an ML system. A model starts as
source code. It may become initialized parameters, a trained checkpoint, a graph, a compiled
runtime artifact, a batch of CUDA kernel launches, a verifier input file, and finally a statement in
a paper. Each move can be harmless. Each move can also change a convention: row-major or column-major
layout, logits or probabilities, train mode or evaluation mode, real arithmetic or float32, closed
or open input bounds. TorchLean is interested in the places where those conventions stop being
obvious.

Modern ML systems are larger than a model definition. A serious workflow may include a parameter
file, a tokenizer, a graph export, a runtime mode, a fused kernel, a floating point convention, a
verifier certificate, and a scientific claim. TorchLean gives those pieces names inside one Lean
development, so runnable examples, graph semantics, numerical models, and checked artifacts can be
read together instead of treated as unrelated files.

PyTorch remains the everyday training ecosystem for many projects. TorchLean starts where meaning
has to survive movement: from model code to graph, from graph to runtime, from runtime to
certificate, and from certificate to theorem. Exported graphs, generated kernels, and verifier
artifacts are now easy to produce. The harder question is whether the artifact says what the paper,
experiment, or deployment claim says it does.

The ordinary user import is [`NN`](https://github.com/lean-dojo/TorchLean/blob/main/NN.lean).
The implementation umbrella remains
[NN](https://github.com/lean-dojo/TorchLean/blob/main/NN.lean), and narrower entry points include
[NN.Entrypoint.API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/API.lean),
[NN.Entrypoint.GraphSpec](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/GraphSpec.lean),
[NN.Entrypoint.IR](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/IR.lean), and
[NN.Entrypoint.Verification](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Verification.lean).

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

This guide uses a few words carefully.

- A *runtime result* is what an executable path produced on some inputs. It can be useful evidence,
  especially when compared against another backend, but it is not by itself a theorem.
- A *check* is an executable validation step. Shape inference, JSON parsing, certificate replay, and
  tolerance comparisons are checks. A check can reject bad artifacts, and if its implementation has
  a theorem behind it, it may also justify a formal claim.
- A *theorem* is a Lean statement accepted by the kernel. It may be about a model denotation, a
  compiler pass, a derivative, a bound, or a float32 approximation.
- A *trust boundary* is a place where TorchLean names an external producer or runtime rather than
  pretending Lean has proved its internals.

This vocabulary is deliberately modest. It lets the guide say "the CUDA path agrees with this
reference on these test cases," "the checker recomputed this bound," and "the theorem proves this
enclosure for the supported graph fragment" as three different sentences.

# Three Views Of One Model

At the top level, TorchLean presents a familiar ML API over a shared internal model.
User code starts from `import NN` and `open TorchLean`, then works with familiar concepts:
layers, datasets, optimizers, losses, and training loops.

Under the hood, the same model appears in three representations:

- *Spec layer*: the mathematical meaning of tensors, layers, losses, and model structure.
- *Graph IR*: the DAG with named operations shared by runtime tooling, widgets, export, and verification.
- *Runtime layer*: eager or compiled execution, autograd, optimizers, logging, and optional CUDA.

The goal is not to make every layer look the same. The goal is to connect the model you run, the
model you inspect, and the model you state theorems about by explicit translations rather than by
informal correspondence.

# The Object We Keep In View

The object we keep returning to is a typed model. Informally, it has this shape:

$$`\mathrm{Model}_{\theta} :
   \mathrm{Tensor}(\alpha,s_{\mathrm{in}})
   \longrightarrow
   \mathrm{Tensor}(\alpha,s_{\mathrm{out}})`

The runtime may choose `Float`, executable `IEEE32Exec`, or CUDA backed float32 buffers. A proof may
instantiate the same architecture over real numbers, interval domains, or a float32 proof model. The
architecture statement should be independent of the execution substrate.

The whole project follows a repeated theorem pattern:

$$`\begin{aligned}
&H(M,R)\\
&\Longrightarrow\;
  \forall x,\;
  \mathrm{denote}(R)(x)
  \in
  \mathrm{safeSet}(\mathrm{denote}(M)(x)).
\end{aligned}`

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
contributes the bound-propagation and certificate ideas used later in the guide, including IBP,
CROWN, and LiRPA-style graph relaxations.

Those references are background, not claims that TorchLean inherits their results automatically.
TorchLean reuses ideas from the surrounding ecosystem while keeping its own proof objects, executable
checks, and external assumptions explicit.

# Fast When Needed, Explicit When It Matters

TorchLean separates fast execution from proof, but it does not treat them as unrelated worlds. For
prototyping, examples can use the host runtime or optional CUDA backed float32 paths. For guarantees,
we move to graph denotations, Float32 models, certificate checkers, and Lean theorems. The boundary
is visible: some parts are proved in Lean, and some parts are external systems that must be named
and checked around.

A typical runtime bug is a checkpoint whose parameter names load successfully while one weight has
been transposed to match a different convention. The model may still run if the surrounding
dimensions happen to agree, but a verifier might then certify a graph and payload pair that is not the
deployed computation. TorchLean's design pushes parameter payloads, graph lowering, and shape checks
into explicit data so that this kind of agreement can be inspected and, where the library has the
theorem support, proved.

# How To Read The Rest

The next chapters follow the same model through several forms. First it is ordinary user code. Then
it becomes a parameterized computation, a runtime value, a graph, a floating point computation, and
a verification target. Later examples add the details that real systems need: token ids, masks,
spectral convolutions, optimizer state, CUDA kernels, imported certificates, and scientific
residuals.

You do not need to know all of Lean before reading the ML examples. Keep track of which object is
being discussed. When a chapter writes a theorem, ask what it denotes. When a chapter runs a command,
ask what artifact the command produced. When a chapter crosses into CUDA, PyTorch, Julia, Arb, or an
external verifier, ask which part is checked in Lean and which part is an explicitly named producer.
