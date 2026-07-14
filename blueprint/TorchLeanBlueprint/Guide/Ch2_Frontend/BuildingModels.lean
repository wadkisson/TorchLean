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
import NN.API
open TorchLean

def inDim : Nat := 2
def hidden : Nat := 8
def outDim : Nat := 1

def mkModel : nn.M (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar)) :=
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
nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)
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
-- .dim 2 .scalar -> .dim 8 .scalar
```

For a minibatch:

```
nn.Linear 2 8
-- .dim batch (.dim 2 .scalar) -> .dim batch (.dim 8 .scalar)
```

That prefix is why the minibatch MLP can be written with the same layer vocabulary:

```
def mkBatched {batch : Nat} :
    nn.M (nn.Sequential (.dim batch (.dim 2 .scalar)) (.dim batch (.dim 1 .scalar))) :=
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
-- .dim dModel .scalar -> .dim hidden .scalar

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

# Convolutional Models

Convolution uses ordinary tensors. A two-dimensional dataset conventionally supplies a tensor of
shape:

```
.dim batch (.dim channels (.dim height (.dim width .scalar)))
```

A compact two-dimensional classifier is one instance of the rank-generic CNN builder:

```
def mkCnn {batch : Nat} :
    nn.M (nn.Sequential (.dim batch (.dim 1 (.dim 4 (.dim 4 .scalar)))) (shape![batch, 2])) :=
  let cfg : nn.models.CnnConfig 2 :=
    { batch := batch
      inChannels := 1
      spatial := #v[4, 4]
      outDim := 2
      conv :=
        { outChannels := 3
          kernel := #v[2, 2]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide }
      pool :=
        { kernel := #v[1, 1]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide } }
  by
    simpa [cfg, nn.models.cnnInShape, nn.models.cnnOutShape, Spec.Shape.ofList] using
      nn.models.cnn cfg (hInChannels := by simp [cfg])
```

Here `CnnConfig 2` means there are two spatial axes. The convolution maps `N x 1 x 4 x 4` to
`N x 3 x 3 x 3`; the pool preserves that shape because its window is `1 x 1`; flattening then
feeds 27 features to the two-logit classifier. Changing the vector length gives a signal, volume,
or higher-rank spatial model without introducing separate tensor types.

The runnable CNN tutorial is
[NN.Examples.Quickstart.SimpleCnnTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean).

# Residual Blocks

Residual models force the API to express a shape preserving path:

```
input -> block(input) + skip(input)
```

The public ResNet builder uses the same rank-generic convolution and pooling records as the CNN.
For a two-dimensional input, a compact configuration is:

```
let cfg : nn.models.ResNetConfig 2 :=
  { batch := batch
    inChannels := 3
    spatial := #v[32, 32]
    spatialNonzero := by intro i; fin_cases i <;> decide
    hiddenChannels := 32
    numClasses := 10 }

nn.models.resnet cfg
```

The residual branches are built only after both paths have the same typed shape. The same
constructor also works for one-dimensional signals and higher-dimensional spatial data by changing
the length of `spatial`; it does not introduce separate image or volume tensor types.

The implementation lives in
[NN/API/Models/ResNet.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/ResNet.lean).

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
