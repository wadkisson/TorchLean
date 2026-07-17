import VersoManual

open Verso.Genre Manual

#doc (Manual) "A Running Example" =>
%%%
tag := "running-example"
%%%

A two-layer classifier from four features to two logits, small enough to write architecture,
payload, graph, float32 execution, and a verification claim side by side.

Path:

```
tinyClassifier
  -> model builder
  -> parameter payload
  -> graph plus payload
  -> Float32 execution
  -> input box plus verifier bounds
  -> semantic claim
```

The later chapters apply the same structure to causal attention, fused kernels, PINN residuals,
neural controllers, geometric certificates, and CROWN margin bounds.

# The Public Model

The ordinary application entry point is `import NN.API`. A small multilayer perceptron looks like familiar
model construction code, but the input and output shapes are part of the type. Here the model consumes a
feature vector of length `4` and produces two logits:

```
import NN.API

open TorchLean

def tinyClassifier : nn.M (nn.Sequential (.dim 4 .scalar) (.dim 2 .scalar)) :=
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

where `x_i` is coordinate `i` of the input, `c_i` is the center of the box on that coordinate, and
`ε_i` is the allowed half-width. Changing that convention changes the verification problem even when
the tensor shape stays `.dim 4 .scalar`.

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

where `g` is the IR graph, `payload` is its parameter data, and `input` is the tensor fed to the
graph. That expression is the reference meaning that a compiler pass, widget, verifier, or runtime
bridge has to respect. If a pass fuses two operations, changes a layout, or exports a certificate,
the claim is always about preserving or soundly approximating this denotation.

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

Real specs, `FP32`, `IEEE32Exec`, and native CUDA kernels are different numeric layers (see
*Floating Point and Native Boundaries*). For the tiny classifier, a real-valued statement might say:

$$`\forall x\in B,\;
\operatorname{logit}_0(\operatorname{denote}_{\mathbb R}(g,\theta,x))
>
\operatorname{logit}_1(\operatorname{denote}_{\mathbb R}(g,\theta,x))`

where `g` is the graph, `θ` is the parameter payload, `x` ranges over the input box `B`,
`denote_ℝ` is the real-valued network output, and `logit_0` / `logit_1` are the two class scores.
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

Here `ℓ₀` is a lower bound on logit `0`, `u₁` is an upper bound on logit `1`, `B` is the input box,
and `f_0` / `f_1` are the two logit functions of the checked model. To prove that implication, the
checker needs hypotheses saying that `ℓ₀` and `u₁` are valid for the same graph, payload, input box,
and scalar semantics. The tiny Boolean is the final comparison; the soundness theorem is about why
that comparison is allowed to stand for all inputs in the box.

# A Tiny Checked Shape Example

Here is the smallest version of the idea. Two vectors of the same shape can be added. Two tensors of
different shapes cannot be passed to the same typed binary op without an explicit reshape, cast, or
proof.

```
import NN.API

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

# Artifacts From This Example

This classifier produces model source, a parameter payload, runtime logits, a lowered graph, shape
checks, bound artifacts, and (when available) Lean theorems. Use the claim vocabulary from *What
TorchLean Is*: each artifact supports a different sentence, and not every line is automatically
proved.

# References

- PyTorch paper for the contrasting imperative ML style: https://arxiv.org/abs/1912.01703
- CROWN robustness certification: https://arxiv.org/abs/1811.00866
- LiRPA on general computational graphs: https://arxiv.org/abs/2002.12920
- TorchLean graph source: https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean
