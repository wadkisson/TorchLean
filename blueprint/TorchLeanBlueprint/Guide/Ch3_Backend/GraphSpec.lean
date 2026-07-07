import VersoManual

open Verso.Genre Manual

#doc (Manual) "GraphSpec" =>
%%%
tag := "graphspec"
%%%

GraphSpec sits between the public model builders and the verifier IR. The public `nn` API is
the direct way to write ordinary training examples. `NN.IR.Graph` is the graph consumed by verifiers
and graph tools. GraphSpec fills the gap between them: it is a typed
architecture language for models whose parameter layout, sharing structure, and pure semantics
should be explicit before lowering.

# Why GraphSpec Exists

A raw IR graph is excellent for a verifier, but it is not the right language for every
architecture. A public `nn.Sequential` model is direct enough for tutorials, but it does not always expose the
architecture-level facts we want to reason about: ordered parameter shapes, reused intermediates,
residual branches, and paired pure/executable interpretations.

GraphSpec exists for that middle layer.

Read the layers this way:

- `nn.Sequential`: beginner model code and training tutorials.
- GraphSpec sequential `Graph ps σ τ`: typed architectures with an explicit parameter list.
- GraphSpec DAG models: residual branches, sharing, and multiple inputs.
- `NN.IR.Graph`: verification, export, widgets, and bound propagation.

# What GraphSpec Adds

The graph taxonomy belongs in *Graphs and IR*. The additional point here is more specific:
GraphSpec gives architectures a typed language before they are lowered into a runtime program or the
shared IR.

That authoring layer adds three things that a raw IR graph does not try to provide:

- a parameter ABI in the type, so the order and shapes of trainable tensors are part of the model
  interface;
- architecture-level sharing, so residual branches and reused intermediates can be written directly;
- paired interpreters, so the same architecture object has a pure `Spec` meaning and an executable
  TorchLean program.

In short: GraphSpec records architecture facts before the model becomes a runtime or verifier artifact.

# Two Authoring APIs

GraphSpec exposes two related APIs because a linear chain and a general DAG with sharing are
different authoring problems.

## Sequential API: `Graph ps σ τ`

The sequential DSL is defined in [NN.GraphSpec.Core API](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/Core.lean).

- `Graph ps σ τ` means: a typed graph from input shape `σ` to output shape `τ`, with parameter
  tensor shapes `ps`.
- Composition uses `>>>`, so straightforward pipelines read like architecture syntax rather than
  like a manually-assembled IR.
- The core MLP examples live here.

This part of GraphSpec is closest to `nn.Sequential`.

Read the type as:

```
Graph ps σ τ
```

means:

- `σ` is the input shape,
- `τ` is the output shape,
- `ps` is the ordered list of parameter tensor shapes.

For example, a dense layer from `Vec inDim` to `Vec outDim` has parameter shapes:

```
[Mat outDim inDim, Vec outDim]
```

That list is the parameter ABI of the graph. The pure interpreter and the executable compiler both
expect parameters in that order.

## DAG API: `DAG.Model ps ins τ`

The [GraphSpec DAG API](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/DAG.lean) and
[GraphSpec DAG core](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/DAG/Core.lean) define the API for shared branches.

- It uses an SSA/A-normal-form term language with explicit binding and sharing.
- It is the right language for residual blocks, skip connections, and models such as ResNet-18.
- GraphSpec's canonical "general model" type is this DAG `Model`.

For "use this intermediate twice" or "add a shortcut branch," use the DAG layer.

# Parameter ABI

GraphSpec treats parameter layout as part of the architecture. Parameter shapes and ordering live in
the type.

By parameter ABI, we mean the ordered list of parameter tensor shapes that every interpreter,
compiler, importer, or checker agrees to use. In:

```
Graph ps σ τ
```

the model maps input shape `σ` to output shape `τ` and expects parameters with shapes `ps`, in that
order. This ordered list is the model's parameter ABI.

The architecture layer gets two practical benefits:

- parameter ABI is explicit and stable enough to reason about, and
- the same model can be interpreted, lowered, and checked against a single declared parameter order.

Many downstream artifacts assume an exact parameter order, so GraphSpec makes that order part of
the model interface instead of leaving it implicit in helper code.

# Choosing The Right Layer

Use the smallest representation that still says what you need to say.

- Use `nn.Sequential` for ordinary tutorials and model training files.
- Use sequential GraphSpec when the model is still a chain, but the parameter ABI and pure semantics
  should be explicit.
- Use GraphSpec DAG models when the architecture has sharing, residual branches, or multiple inputs.
- Use `NN.IR.Graph` when the consumer is a verifier, exporter, or graph analysis pass.

Each representation stays explicit: architecture authoring stays readable, while verifier code
still receives a graph with named operations after lowering.

# Two Semantics From One Model

GraphSpec avoids an early choice between a clean mathematical object and an executable object.

For sequential graphs, the main names are:

- `NN.GraphSpec.Interp.spec`
- `NN.GraphSpec.Compile.torchProgram`

For DAG models, the corresponding names are:

- `NN.GraphSpec.Model.specFwd`
- `NN.GraphSpec.Model.torchProgram`

The pattern is the same in both cases:

- the Spec semantics supplies a pure forward meaning in Lean,
- the TorchLean compiler supplies an executable program in the runtime layer.

GraphSpec packages this semantic alignment at the architecture level.

The declarations worth recognizing in the infoview are:

```
#check NN.GraphSpec.Interp.spec
#check NN.GraphSpec.Compile.torchProgram
#check NN.GraphSpec.DAG.Model.specFwd
#check NN.GraphSpec.DAG.Model.torchProgram
#check NN.GraphSpec.Model.specFwd
#check NN.GraphSpec.Model.torchProgram
#check NN.GraphSpec.ToTorchLean.toSeq
```

# Worked Micro-Example

The smallest GraphSpec check sequence is the one from the README:

```
import NN.Entrypoint.GraphSpec

open NN
open NN.GraphSpec
open Spec

def g (inDim hidDim outDim : Nat) :=
  GraphSpec.Models.mlp inDim hidDim outDim

-- Pure semantics: parameters to input to output.
#check GraphSpec.Interp.spec (g 4 8 2)

-- Executable TorchLean program.
#check GraphSpec.Compile.torchProgram (g 4 8 2)

-- Optional lowering into the general DAG model representation.
#check GraphSpec.LowerToDAG.Graph.toDAGModelZeroInit (g 4 8 2)
```

One architecture can be read at three levels:

- as typed authoring syntax,
- as pure semantics,
- and as executable runtime code.

# Residual Sharing As A Tiny DAG

A residual block is the first example where the DAG API is more natural than a chain. The
architecture says:

```
y = f(x)
z = y + x
```

The value `x` is used twice. In a purely sequential chain, that sharing is awkward because the input
would have to be carried forward as an explicit side value. In a DAG model, the intermediate names
are part of the authoring language.

Examples such as `Models.residualLinear` and `Models.ResNet18.model` use the DAG API because their
architectures contain skip connections. Shared values should be represented directly, not smuggled
through an artificial sequential chain.

# Lowering Paths

GraphSpec supports several lowering paths. They are not competing APIs; they answer different
questions about the same architecture.

The names below are the bridges to look for when debugging how a GraphSpec model becomes a runnable
TorchLean program or a DAG for verification.

## Sequential Graph To DAG Model

The relevant definitions are [GraphSpec core](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/Core.lean) and
[GraphSpec DAG core](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/DAG/Core.lean).

Important names:

- `LowerToDAG.Graph.toDAGTerm`
- `LowerToDAG.Graph.toDAGModelZeroInit`
- `LowerToDAG.Graph.toDAGModelDetInit?`

This path turns a simple pipeline into GraphSpec's general DAG representation.

## GraphSpec To TorchLean `nn.Sequential`

The lowering API is [GraphSpec to TorchLean](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/ToTorchLean.lean).

Important name:

- `NN.GraphSpec.ToTorchLean.toSeq`

This path is used by the runnable
[GraphSpec tutorial](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/GraphSpec/Tutorial.lean): author once in GraphSpec,
then plug the lowered model into the same `Trainer` API used elsewhere in the guide.

## DAG Model To Runtime Example Wrappers

The runtime examples to open are [GraphSpec ResNet18](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/Models/TorchLean/Resnet18.lean)
and [runtime FNO1D](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Fno1d.lean).

These wrappers show how GraphSpec feeds the runtime examples:

- GraphSpec-backed `resnet18Model`, `resnet18Program`, `resnet18InitParams`
- `fno1d` runtime wrappers for operator learning, which sit beside the GraphSpec-backed
  models in the broader example set

# Model Example Landmarks

For model definitions used in examples, `NN.GraphSpec.Models` is the most convenient
single import.

The current example set includes both sequential and DAG-native models:

- `Models.mlp` for the smallest sequential path,
- `Models.twoConvCnn` and `twoConvCnnDAGModelZeroInit` for "same model, different representation" comparisons,
- `Models.residualLinear` for a small residual/shared example,
- `Models.ResNet18.model` for a substantial DAG-authored architecture.

The GraphSpec README suggests this order:

1. `Models.mlp`
2. `Models.twoConvCnn`
3. `Models.residualLinear`
4. `Models.ResNet18.model`

# Reading Rule

When reading a GraphSpec model, look for four things:

1. the input and output shapes;
2. the parameter-shape list;
3. whether the model is sequential or DAG-native;
4. which lowering or theorem cites it.

GraphSpec contributes this information beyond the public tutorial syntax. It is not another
training API; it is the architecture layer that preserves parameter ABI and sharing before runtime
or verification lowering.

# Proof Alignment Checks

GraphSpec also has proof declarations. The declarations below are compact alignment statements: they
connect GraphSpec syntax to reference specs, deterministic parameter initialization, and the
embedding of sequential primitives into DAG primitives.

The entrypoint `NN.Entrypoint.GraphSpec` re-exports the theorem names readers are most likely
to check first:

```
#check NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward
#check NN.GraphSpec.Models.mlp_detInitParams_eq_torchlean_linear_inits
#check NN.GraphSpec.Primitive.toDAGPrimOp_specFwd_eq
```

These facts show GraphSpec as a representation for theorem statements and illustrate the "extend op
by op" development used in the verified runtime and verification layers.

# Best First Path

One concrete reading order:

1. Read the [GraphSpec README](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/README.md).
2. Open the [GraphSpec tutorial API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/GraphSpec/Tutorial.lean).
3. Run:
   `lake exe torchlean graphspec --backend compiled`
4. Read the [residual linear model API](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/Models/ResidualLinear.lean).
5. Read the [ResNet-18 GraphSpec API](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/Models/Resnet18.lean).

That path moves from "author a typed MLP and lower it into the training API" to "author a real
residual DAG with explicit sharing."

# Where It Fits In The System

GraphSpec refines TorchLean's semantic model at the architecture boundary. In a Python codebase,
architecture authoring and execution are often bundled together. GraphSpec separates them enough to
make semantics and proofs easier, without abandoning runnable models.

The implementation path runs through the [GraphSpec README](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/README.md), then
move to the sequential and DAG cores, then to the model examples. The
[GraphSpec to TorchLean API](https://github.com/lean-dojo/TorchLean/blob/main/NN/GraphSpec/ToTorchLean.lean) gives the lowering path into
TorchLean `Seq`, and the
[GraphSpec TorchLean models](https://github.com/lean-dojo/TorchLean/tree/main/NN/GraphSpec/Models/TorchLean/) expose runtime model programs.

Next: *Runtime and Autograd* (execution), *Graphs and IR* (shared graph IR), *Verification*
(certificates and bounds).

# References

- TorchLean paper: https://arxiv.org/abs/2602.22631
- ResNet reference:
  He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016).
  https://arxiv.org/abs/1512.03385
