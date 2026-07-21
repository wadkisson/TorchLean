import VersoManual

open Verso.Genre Manual

#doc (Manual) "Tensors And Models" =>
%%%
tag := "tensors-shapes"
%%%


Most tensor libraries carry a shape beside a buffer and check compatibility when an operation
runs. TorchLean also carries the shape in the Lean type. A function that accepts a length-four
vector cannot accidentally receive a `2 × 2` matrix, even though both contain four scalar values.

That choice is the foundation for the rest of the library. Layers become checked maps between
shapes, parameter packs remember the layout expected by a model, and theorem statements do not
need a side condition saying that every intermediate tensor happened to have the right dimensions.

# Run The First Tensor Program

From the repository root, run:

```
lake exe torchlean quickstart_tensors
```

The output is:

```
== Quickstart: tensor basics ==
[Float] [0.100000, 0.200000, 0.300000, 0.400000]
[ℚ] [1/10, 1/5, 3/10, 2/5]
[Int] [1, 2, 3, 4]
[IEEE32Exec] [0.100000, 0.200000, 0.300000, 0.400000]
[Float] [[[1.000000, 2.000000], [3.000000, 4.000000]],
         [[5.000000, 6.000000], [7.000000, 8.000000]]]
Expected failure printing Tensor ℝ:
  Refusing to print `Tensor ℝ` (proof-level);
  cast to `Float`/`IEEE32Exec`/`ℚ` to display.
```

The first four lines have the same shape but different scalar meanings. `ℚ` displays exact
fractions. `IEEE32Exec` stores and executes explicit binary32 bit patterns. `Float` uses Lean's host
runtime. The final attempted tensor over `ℝ` is a mathematical object; arbitrary real numbers are
not executable data, so printing it is rejected rather than pretending to approximate it.

The source is
[`TensorBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean).
Keep it open while reading this chapter: every definition below is a small variation of that file.

# One Tensor Type

The canonical specification type is:

$$`\operatorname{Spec.Tensor}\;\alpha\;s`.

The public spelling is `Tensor.T α s`. The first parameter is the scalar type and the second is the
shape. A shape is recursively built from:

```
inductive Shape
  | scalar
  | dim (size : Nat) (rest : Shape)
```

Thus a matrix of shape `[3,2]` is represented by:

$$`
\operatorname{dim}(3,\operatorname{dim}(2,\operatorname{scalar})).
`

The `shape!` macro gives the familiar notation:

```
import NN.API
open TorchLean

def scalarShape : Shape := .scalar
def vectorShape : Shape := shape![4]
def matrixShape : Shape := shape![3, 2]
def rankFourShape : Shape := shape![8, 3, 32, 32]
```

`rankFourShape` is not an “image shape” at the type level. It is an ordinary rank-four tensor.
Layout conventions belong to operations and models, not to separate tensor datatypes. In vision
code those conventions are often abbreviated:

- _NCHW_: batch `N`, channels `C`, height `H`, width `W` — so `shape![8, 3, 32, 32]` is eight RGB
  images of size `32×32` with channels first;
- _NHWC_: the same axes with channels last — `shape![8, 32, 32, 3]`.

Sequence models instead treat axes as batch × time × features; scientific grids may use batch ×
channels × spatial axes of any rank. TorchLean stores only the shape index; the meaning of each
axis is fixed by the layer or operator that consumes the tensor. That is why the same tensor core
can represent language tokens, PDE grids, volumetric data, batched matrices, or an unusual
scientific coordinate system.

# Literals Prove Their Own Shape

The `tensor!` macro reads a rectangular nested literal and infers its shape:

```
def x : Tensor.T Float (shape![2, 2]) :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]

def y : Tensor.T Float (shape![2, 2]) :=
  tensor! [[0.2, -0.1], [0.0, 0.3]]

def z : Tensor.T Float (shape![2, 2]) :=
  x + y
```

The literal is flattened in row-major order: the last index changes fastest. In this example the
flat order is `1, 2, 3, 4`.

Try either of these deliberate mistakes in a scratch Lean file:

```
def ragged :=
  tensor! [[1.0, 2.0], [3.0]]

def wrongAnnotation : Tensor.T Float (shape![4]) :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]
```

The first literal is not rectangular. The second has four values but the wrong structure. Both are
rejected while Lean elaborates the file. The total element count alone is not enough to identify a
shape.

When the scalar type is ambiguous, make it explicit:

# Building A Model

A TorchLean model begins as a map between tensor shapes. Initialization, loss functions, optimizer
state, and execution devices are attached later. This separation is useful even before proving a
theorem: the architecture can be inspected without allocating parameters, and the same model can
be initialized twice with different seeds or interpreted by different runtimes.

We will sketch three public-API models that share the same composition mechanism. The running
example through later chapters is the MLP; the CNN and transformer show that vision and sequence
layouts are ordinary shape-indexed `nn.Sequential` values, not separate frameworks.

# A Layer Is A Checked Shape Map

The public layer type is:

```
nn.LayerDef inputShape outputShape
```

A sequential model has:

```
nn.Sequential inputShape outputShape
```

The input and output shapes are indices of the type, so composition requires equality at the
boundary. If:

$$`f:s_0\to s_1`

and

$$`g:s_1\to s_2`,

then `g` may follow `f`. A layer expecting `s_3` cannot be inserted there merely because the two
shapes contain the same number of values.

Model builders use:

```
nn.M A
```

which is a seeded construction of `A`. It allocates deterministic seeds for parameterized layers
but does not run a forward pass or optimizer update.

# The Running MLP

Here is the model used throughout the introduction:

```
import NN.API
open TorchLean

def inDim : Nat := 2
def hidden : Nat := 8
def outDim : Nat := 1

def model :
    nn.M (nn.Sequential (shape![inDim]) (shape![outDim])) :=
  nn.Sequential![
    nn.linear inDim hidden,
    nn.relu,
    nn.linear hidden outDim
  ]
```

Read the macro from top to bottom:

```
input [2]
  -> linear 2 8
hidden [8]
  -> relu
hidden [8]
  -> linear 8 1
output [1]
```

`ReLU` is shape-preserving. The two linear layers each change the final feature axis. The macro
checks that the chain is composable and returns one `Sequential [2] [1]`.

The parameter layout follows PyTorch's linear-layer convention:

$$`
W:\operatorname{Tensor}\;\alpha\;[\mathrm{out},\mathrm{in}],
\qquad
b:\operatorname{Tensor}\;\alpha\;[\mathrm{out}].
`

# A Small Convolutional Classifier

For images, the public helper `nn.models.cnn` builds a convolutional classifier over an NCHW
minibatch. The compact CIFAR example uses batch `1`, three input channels, an `8×8` crop, and ten
output classes; the full configuration lives in
[`NN/Examples/Models/Vision/Cnn.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Cnn.lean).
Download the CIFAR arrays first (about 170 MB; the Toronto host can be slow), then run the model:

```
python3 scripts/datasets/download_example_data.py --cifar10
lake exe torchlean cnn --device cpu --n-total 1 --steps 1
```

The important point for this chapter is architectural: convolution, pooling, and the linear head
are still ordinary layers composed into one `nn.Sequential`, indexed by the input and output shapes
of that pipeline.

# A Transformer Encoder Block

Sequence models use the same composition surface. A single encoder block is:

```
def block :
    nn.M (nn.Sequential (shape![1, 16, 32]) (shape![1, 16, 32])) :=
  nn.models.transformerEncoder
    { batch := 1
      seqLen := 16
      dModel := 32
      numHeads := 4
      headDim := 8
      ffnHidden := 64 }
```

The shape `1 × 16 × 32` is batch × sequence length × model width. Attention and the feed-forward
sublayer preserve that shape. The runnable text example is
`lake exe torchlean transformer --device cpu --tiny-shakespeare --steps 1`; see
[`NN/Examples/Models/Sequence/Transformer.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Transformer.lean).
