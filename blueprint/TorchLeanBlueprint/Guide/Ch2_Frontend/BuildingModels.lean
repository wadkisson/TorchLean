import VersoManual

open Verso.Genre Manual

#doc (Manual) "Building Models With Layers" =>
%%%
tag := "building-models"
%%%

Once tensors have shapes, a model can be read as a typed map between tensor spaces. TorchLean's
layer API keeps the familiar pieces of neural-network code: linear layers, activations,
convolutions, residual blocks, and attention blocks compose into sequential models. The difference
is that the input and output shapes are visible before the model runs.

```
model : Tensor alpha inputShape -> Tensor alpha outputShape
```

The three names to remember are `nn.M`, `nn.Sequential`, and `Trainer.new`.

- `nn.M` is the model builder. It allocates parameter seeds and assembles layers.
- `nn.Sequential sigma tau` is a model from shape `sigma` to shape `tau`.
- `Trainer.new mkModel { task := .regression, seed := seed }` fixes initialization and chooses the training task.

The result still feels close to PyTorch's `nn.Sequential`, but the input and output shapes are part
of the Lean type.

# A First MLP

The smallest model pattern is a multilayer perceptron:

```
import NN
open TorchLean

def inDim : Nat := 2
def hidden : Nat := 8
def outDim : Nat := 1

def mkModel : nn.M (nn.Sequential (Shape.vec inDim) (Shape.vec outDim)) :=
  nn.Sequential![
    nn.Linear inDim hidden,
    nn.ReLU,
    nn.Linear hidden outDim
  ]

def task (seed : Nat) :=
  Trainer.new mkModel { task := .regression, seed := seed }
```

There are three things to notice.

First, the model type says exactly what the model accepts and returns:

```
nn.Sequential (Shape.vec 2) (Shape.vec 1)
```

Second, `nn.Linear inDim hidden` is more than a runtime operation. It also describes the parameter
shapes for the weight and bias. A layer `nn.Linear 2 8` introduces the usual affine parameters for
mapping two features to eight features under the selected prefix convention. When the trainer is
created with a seed, it builds the initial parameter bundle for that layer.

Third, `Trainer.new` attaches a loss convention to the model. The model definition and the
training task are separate: the same model shape can appear in a regression task, a classification
task, an export path, or a proof statement.

The separation is useful even in a tiny file. You can name the architecture once and build several
tasks around it:

```
def regressionTask :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      seed := 11 }

def compiledTask :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      backend := .compiled
      seed := 11 }
```

Both tasks refer to the same typed architecture. The second one changes the runtime artifact, not
the model family.

The runnable file is
[NN.Examples.Quickstart.SimpleMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean).

# Prefix Shapes: The Batch Axis Story

If you are coming from PyTorch, the prefix shape is the one new idea to slow down for. A linear
layer acts on the last dimension. The dimensions before it are the prefix. This matches the PyTorch
intuition that `Linear(inDim, outDim)` can be applied to inputs shaped `[..., inDim]`. TorchLean asks
you to name that prefix.

For one vector:

```
nn.Linear 2 8
-- Shape.vec 2 -> Shape.vec 8
```

For a minibatch:

```
nn.Linear 2 8
-- Shape.mat batch 2 -> Shape.mat batch 8
```

That prefix is why the minibatch MLP can be written with the same layer vocabulary:

```
def mkBatched {batch : Nat} :
    nn.M (nn.Sequential (Shape.mat batch 2) (Shape.mat batch 1)) :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]
```

The layer still transforms features from `2` to `8` to `1`. The prefix says that the operation is
applied across a batch.

Prefix shapes also keep sequence and image conventions honest:

```
-- One token embedding.
nn.Linear dModel hidden
-- Shape.vec dModel -> Shape.vec hidden

-- A batch of token embeddings.
nn.Linear dModel hidden
-- shape![batch, seqLen, dModel] -> shape![batch, seqLen, hidden]

-- A batch of flattened image features.
nn.Linear featSize classes
-- shape![batch, featSize] -> shape![batch, classes]
```

The linear layer is the same layer in each case. The prefix says which axes are carried along while
the last feature axis changes.

The runnable minibatch example is
[NN.Examples.Quickstart.MinibatchMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/MinibatchMlpTrain.lean).

# KANs: Models Still Stay Task-Agnostic

Kolmogorov-Arnold Networks fit the same pattern. A KAN is a model family: each scalar edge is
expanded through a one-dimensional basis, and the layer learns coefficients on those edge features.
Regression or classification is still selected by the trainer.

```
def edge :=
  nn.models.KANPiecewiseLinear.edgeFamily { gridSize := 8, inputScale := 7 }

def cfg : nn.models.KANConfig :=
  { batch := 4
    inDim := 2
    hidden := [8]
    outDim := 1
    edge := edge
    seedBase := 10 }

def mkKan : nn.M (nn.Sequential (nn.models.kanInShape cfg) (nn.models.kanOutShape cfg)) :=
  nn.models.KAN cfg

def task :=
  Trainer.new mkKan { task := .regression, optimizer := optim.adam { lr := 0.01 } }
```

The edge slot is the abstraction that matters. The built-in triangular basis is piecewise linear, so
interval and branch style reasoning stays explicit. A cubic spline, polynomial, or rational edge
basis should provide another `nn.models.KANEdgeFamily`; the same `Trainer.new` path then decides the
task.

This follows the KAN idea from Liu et al. (2024), where learned univariate edge functions replace
ordinary scalar weights. For the spline background, de Boor's *A Practical Guide to Splines* is the
standard reference.

The runnable example is
[NN.Examples.Models.Supervised.Kan](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Supervised/Kan.lean).

# Images and CNNs

Image models use the same `nn.Sequential!` style, but the input shape is now an image batch:

```
Shape.images batch channels height width
```

A small CNN looks like this:

```
def mkCnn {batch : Nat} :
    nn.M (nn.Sequential (Shape.images batch 1 4 4) (shape![batch, 2])) :=
  let outC : Nat := 3
  let outH : Nat := (4 - 2) / 1 + 1
  let outW : Nat := (4 - 2) / 1 + 1
  let featInner : Shape := Shape.image outC outH outW
  let featSize : Nat := Shape.size featInner
  nn.Sequential![
    nn.Conv2d (n := batch) (inC := 1) (inH := 4) (inW := 4)
      { outC := outC, kH := 2, kW := 2, stride := 1, padding := 0 },
    nn.ReLU,
    nn.FlattenBatch,
    nn.Linear featSize 2
  ]
```

Here the shape bookkeeping is part of the model definition:

- the convolution maps `N x 1 x 4 x 4` to `N x 3 x 3 x 3`;
- `nn.FlattenBatch` keeps the batch axis and flattens the feature axes;
- the final linear layer maps each flattened image to two logits.

The CNN example makes the axes explicit. The type records that a batch of images enters, the
convolution changes the channel and spatial axes, and two logits per image leave. Later chapters use
the same information when they lower the model to graphs or discuss verification conditions.

The runnable CNN tutorial is
[NN.Examples.Quickstart.SimpleCnnTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean).

# Residual Blocks

Residual models force the API to express a shape preserving path:

```
input -> block(input) + skip(input)
```

The public builder for a residual block is `nn.ResNetBasicBlock`. A typical shape is:

```
nn.ResNetBasicBlock (n := batch) (inC := 8) (h := 4) (w := 4)
  { outC := 8, stride := 1 }
```

Read the type as a contract: if the block is used in the no downsample case, the residual path and
the main path have compatible output shapes. If a downsample is requested, the block records the
shape change explicitly.

The block itself lives in the public `TorchLean.nn` API, and the ResNet model constructors live
under [NN/API/Models/Resnet.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Resnet.lean).

# Transformer Shaped Blocks

Sequence models use the same principle. A transformer block is a typed map over a batched sequence:

```
shape![batch, seqLen, dModel]
```

The public constructors include:

- `nn.multiheadAttention`,
- `nn.layerNorm`,
- `nn.TransformerEncoderBlock`,
- and model constructors under [NN/API/Models/Transformer.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Transformer.lean),
  [NN/API/Models/Gpt2.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Gpt2.lean), and
  [NN/API/Models/Vit.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Vit.lean).

A small block reads as:

```
nn.TransformerEncoderBlock
  (batch := batch) (n := seqLen) (dModel := dModel)
  { numHeads := 2, headDim := dModel / 2, ffnHidden := 4 * dModel }
```

The applications chapters give the longer model examples tour. Here the lesson is simpler: MLPs, CNNs,
residual blocks, and transformer blocks all enter through the same typed model construction path. State
the shape, choose the layers, create a trainer with a seed, then train or inspect the resulting
task.

# Parameters Are Explicit

PyTorch stores parameters inside module objects. TorchLean keeps the parameter bundle explicit. That
choice makes the training loop and the verification path much easier to read.

Informally:

```
Trainer.new mkModel { task := .regression, seed := seed }
-- produces a supervised task with initialized parameters
```

The trainer owns the initial parameter bundle for the chosen task. Training updates that bundle.
Prediction evaluates the same structure with the current parameter values.

The structure can look verbose for a two-layer MLP, but it is the same structure that later lets us
lower the model to a graph and check a certificate without rediscovering the parameter shapes.

For a mental model, read a linear layer as contributing two named tensors:

```
-- Informal parameter shape contract for nn.Linear inDim outDim:
weight : Tensor.T α (shape![outDim, inDim])
bias   : Tensor.T α (shape![outDim])
```

A two-layer MLP therefore has a small tree of parameter tensors. The exact public names are owned by
the model builder, but the shape story is the usual neural-network shape story. What changes is that
TorchLean can carry the same structure into a graph, a JSON payload, or a proof statement.

This also explains why TorchLean examples avoid mutating a model object in place. Training returns a
new trained handle:

```
let trained ← trainer.train data { steps := 200, batchSize := 16 }
let yhat ← trained.predict xHeldout
```

The handle owns the updated parameter bundle and runtime state. The architecture `mkModel` remains
the reusable definition.

# Choosing The Trainer

After a model is built, choose the public trainer that matches the target:

- use `Trainer.new model { task := .regression }` for mean squared error style vector targets;
- use `Trainer.new model { task := .classification }` for one-hot classification targets;
- use the model command layer when a family has a specialized objective or data shape.

The public trainer records the same shape contract as the model and target data. Runtime-internal
code can still look at the checked runtime task object directly. The public path normally stops one
layer higher: build the model, choose the trainer with `Trainer.new`, then pass a dataset and
`Trainer.TrainOptions`.

# Model Code Versus Proof Code

The same architecture can appear in three different places:

- tutorial code, where it is trained or used for prediction;
- runtime code, where it is instantiated with buffers, backend options, and logs;
- proof code, where a theorem states what a graph, evaluator, loss, or derivative means.

The shape in the model type is the thread that ties those places together. A successful training run
does not prove the model is robust. A graph theorem does not say a native kernel is correct unless
the theorem names that native boundary. A certificate check does not retrain the model. Those are
separate claims, and TorchLean's model API is organized so they can refer to the same architecture
without pretending to be the same activity.
