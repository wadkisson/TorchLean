import VersoManual

open Verso.Genre Manual

#doc (Manual) "A Running Example" =>
%%%
tag := "running-example"
%%%

A compact example makes the design concrete. We will use an ordinary two-layer classifier. Its job
is not to be impressive as a model. Its job is to let us watch one computation
move through the library: user code, parameters, a graph, a Float32 execution, and a verification
problem.

The example is deliberately small because the bookkeeping is the lesson. A ResNet, transformer, or
scientific surrogate has more operators and more state, but the same questions return: what is the
input shape, where are the parameters, which graph was lowered, which scalar semantics are in use,
and what exactly did the checker establish?

The question to keep asking is simple:

> What is the same model at this layer?

The path looks like this:

```
tinyClassifier
  -> model builder
  -> parameter payload
  -> graph plus payload
  -> Float32 execution
  -> input box plus verifier bounds
  -> semantic claim
```

Later examples use the same pattern for more interesting objects: causal masks with zero future
attention weight, fused attention specs related to ordinary scaled dot product attention, 3D
projection certificates, PINN residual bounds, neural-controller checks, and CROWN margin
certificates.

# The Public Model

The ordinary entry point is `import NN`. A small multilayer perceptron looks like familiar
model construction code, but the input and output shapes are part of the type. Here the model consumes a
feature vector of length `4` and produces two logits:

```
import NN

open TorchLean

def tinyClassifier : nn.M (nn.Sequential (Shape.vec 4) (Shape.vec 2)) :=
  nn.Sequential![
    nn.Linear 4 8,
    nn.ReLU,
    nn.Linear 8 2
  ]

def tinyTask (seed : Nat) :=
  Trainer.new tinyClassifier { task := .classification, seed := seed }
```

At this stage there is no proof obligation; it is model code. The difference from a Python script is
that the shape contract has already become part of the program. A later theorem, exporter, graph
pass, or checker does not have to rediscover that the model expects four input features and returns
two outputs.

The type is already saying the essential input/output contract:

$$`\operatorname{tinyClassifier} :
\operatorname{Model}\bigl(\operatorname{Tensor}(\alpha,[4]),
                           \operatorname{Tensor}(\alpha,[2])\bigr)`

That contract is intentionally modest. It says nothing about accuracy, initialization quality, or
robustness. It says the model is a computation from four features to two logits. Later claims add
more hypotheses: trained parameters, an input box, a label, a scalar interpretation, and a checker
or theorem.

# The Same Model As Data

When the model is built, TorchLean separates architecture from parameters. The first verification and
interop boundary is already present: the architecture says which operations exist and how they are
composed, while the parameter bundle says which trained numbers those operations use.

```
def builtTiny :=
  Trainer.new tinyClassifier { task := .classification, seed := 2026 }

-- The task contains the model structure and initialized parameter bundle.
#check builtTiny
```

PyTorch also has a parameter bundle, usually exposed through `state_dict`. TorchLean makes that idea
central rather than incidental: parameters are passed, saved, loaded, lowered, and checked as data.
That choice is what lets the same trained model move from model authoring to graph inspection to
verification without becoming an untyped blob.

The denotation we keep returning to has this shape:

$$`\operatorname{forward}(architecture,\theta,x) = y`

where the architecture, parameters `θ`, input `x`, and output `y` are all ordinary values that can
be inspected or related by theorems.

# A Concrete Input Convention

For the running example, imagine the four input coordinates are normalized sensor features:

```
def featureNames : List String :=
  ["temperature", "pressure", "velocity", "bias"]
```

TorchLean's tensor type can record that there are four coordinates, but it does not know the human
meaning of coordinate `2` unless we record that convention somewhere. This is a useful limitation.
Types catch shape mistakes; documentation, metadata, loaders, and predicates record domain
conventions.

If a later verifier states a box property, it should be clear whether the box is over raw features
or normalized features:

$$`x_i \in [c_i-\varepsilon_i, c_i+\varepsilon_i]`

Changing that convention changes the verification problem even when the tensor shape stays
`Shape.vec 4`.

# The Same Model As A Graph

The graph chapters explain how TorchLean lowers model code into `NN.IR.Graph`, a DAG whose nodes name
their operations and are shared by runtime inspection and verifier code. User-facing model code stays
readable, while the graph gives compilers, widgets, and verifiers an object they can inspect node by
node.

The discipline is:

- user code should stay readable;
- the lowered graph should stay explicit;
- theorem statements should name the semantics of that graph.

We spend time on both `nn.Sequential` and `NN.IR.Graph` because they are not competing interfaces.
They are two views of the same computation.

For a lowered graph, the semantic question becomes:

$$`\operatorname{NN.IR.Graph.denoteAll}(g,payload,input)`

That expression is the reference meaning that a compiler pass, widget, verifier, or runtime bridge
has to respect. If a pass fuses two operations, changes a layout, or exports a certificate, the
claim is always about preserving or soundly approximating this denotation.

# The Same Model As A Payload Contract

A graph without numbers is still a family of computations. The trained classifier is the graph plus
the payload. Informally:

```
structure TinyPayloadSketch where
  w1 : String
  b1 : String
  w2 : String
  b2 : String
```

The real payload stores tensors, not strings. The sketch shows the audit point: the first linear
layer's weight, the first bias, the second weight, and the second bias have names, shapes, and
places in the graph. An import step can check that the payload it received matches those positions.
A theorem about the graph should not silently use a different set of weights.

# The Same Model Under Float32

A theorem over real numbers and a float32 execution are not the same statement. TorchLean keeps the
scalar semantics visible:

- `ℝ` is the clean mathematical target;
- `FP32` is the proof friendly model based on rounded reals;
- `IEEE32Exec` is the executable IEEE-754 binary32 model;
- native CUDA kernels are accelerated external code behind an explicit boundary.

The Float32 chapters explain how those views are connected, and where a theorem still depends on an
assumption about the external runtime.

For the tiny classifier, a real-valued statement might say:

$$`\forall x\in B,\;
\operatorname{logit}_0(\operatorname{denote}_{\mathbb R}(g,\theta,x))
>
\operatorname{logit}_1(\operatorname{denote}_{\mathbb R}(g,\theta,x))`

A float32 statement is not the same sentence with a different font. It has to say whether the
operations are modeled as rounded real operations, executable `IEEE32Exec`, or a native backend. If
the theorem is real-valued and the deployment path is CUDA float32, a bridge or an explicit
assumption is still part of the story.

# The Same Model As A Verification Problem

Once the model has a graph and a payload, verification tools can ask bounded questions about it. For
the two-logit classifier above, a typical local robustness question is:

> For every input `x` in this box around a reference example, is logit `0` still greater than logit `1`?

The verifier does not need to trust the training script. It receives explicit objects:

- What output interval follows from this input box?
- Does a margin remain positive after bound propagation?
- Which certificate or JSON artifact was checked, and which values came from an external producer?

The core checker can be read schematically as a small Boolean computation over checked bounds:

```
-- Accept when class 0 is still ahead of class 1.
def marginCertificateOK (logit0Lower logit1Upper : Float) : Bool :=
  logit0Lower > logit1Upper
```

The real verifier carries richer data than two floats, but the shape of the argument is the same:
an external analyzer may propose bounds or artifacts, while Lean checks the part of the condition
that is represented in the artifact.

The corresponding semantic statement is stronger than the Boolean check:

$$`\operatorname{marginCertificateOK}(\ell_0,u_1)=\mathrm{true}
\;\Longrightarrow\;
\forall x\in B,\; f_0(x)>f_1(x)`

To prove that implication, the checker needs hypotheses saying that `ℓ₀` is a valid lower bound on
logit `0`, that `u₁` is a valid upper bound on logit `1`, and that both bounds apply to the same
graph, payload, input box, and scalar semantics. The tiny Boolean is the final comparison; the
soundness theorem is about why that comparison is allowed to stand for all inputs in the box.

# A Tiny Checked Shape Example

Here is the smallest version of the idea. Two vectors of the same shape can be added. Two tensors of
different shapes cannot be passed to the same typed binary op without an explicit reshape, cast, or
proof.

```
import NN

open TorchLean

def a : Tensor Float (shape![2]) :=
  tensor! [1.0, 2.0]

def b : Tensor Float (shape![2]) :=
  tensor! [0.25, -0.5]

def good := Spec.Tensor.addSpec a b

def wrongShape : Tensor Float (shape![2, 1]) :=
  tensor! [[1.0], [2.0]]

-- Rejected by the type checker:
-- def shapeMismatch := Spec.Tensor.addSpec a wrongShape
```

That tiny example is the design in miniature. TorchLean does not try to guess which broadcast,
reshape, or deployment convention you meant. A real convention should appear as a named operation in
the code, so the model author, exporter, and checker all see the same transformation.

# What To Watch For

As the examples get larger, keep track of where each object lives:

- *Spec*: the mathematical meaning of tensors, layers, and losses.
- *Runtime*: eager or compiled execution, gradients, optimizers, logging, and devices.
- *IR*: a graph with named operations that can be inspected and verified.
- *Proofs*: theorems about the spec, the graph, or the verifier output.
- *Trust boundaries*: CUDA kernels, PyTorch exporters, external certificate producers, and datasets.

Is this a tensor in the spec layer? A runtime value? A graph node? A theorem about graph denotation?
A certificate imported from outside Lean? Most TorchLean mistakes become easier to diagnose once
that question is clear.

# The Paper Trail For This Example

The same classifier produces several artifacts, and each artifact supports a different kind of
sentence.

- Model source: "this is a two-layer classifier from `Shape.vec 4` to `Shape.vec 2`."
- Parameter payload: "these tensors instantiate that architecture."
- Runtime output: "this backend produced these logits on this input."
- Lowered graph: "these named operations are the graph view of the model."
- Shape check: "the payload and graph agree on dimensions."
- Bound artifact: "these intervals or affine bounds were produced for this graph and input box."
- Lean theorem: "under the theorem hypotheses, accepted bounds imply the stated semantic property."

Not every line above is automatically proved. Keeping the artifacts together makes it possible to
see which object supports each claim and where an additional proof obligation remains.

# References

- PyTorch paper for the contrasting imperative ML style: https://arxiv.org/abs/1912.01703
- CROWN robustness certification: https://arxiv.org/abs/1811.00866
- LiRPA on general computational graphs: https://arxiv.org/abs/2002.12920
- TorchLean graph source: https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean
