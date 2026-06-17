import VersoManual

open Verso.Genre Manual

#doc (Manual) "Tensors, Shapes, and DTypes" =>
%%%
tag := "tensors-shapes"
%%%

The first thing to understand in TorchLean is that a tensor's shape is part of the object, not only
runtime metadata. This is a small change with a large effect. If a model returns predictions of
shape `[batch, 1]` and the target has shape `[batch]`, TorchLean does not silently guess the
intended convention. Maybe the target should be reshaped. Maybe the model head is wrong. Maybe the
loss should be different. The choice belongs in the code.

The basic type is:

$$`\mathrm{Tensor}(\alpha,s)`

Read this as: a tensor with scalar type `α` and shape `s`. The scalar type is the dtype, and the
shape is part of the Lean type.

The proof layer then relates this shaped view to flat vectors, pointwise algebra, dot products, and
runtime payloads.

# The Canonical Tensor Type

The spec layer defines the canonical tensor type:

- `Spec.Tensor α s` is “a tensor with scalar type `α` and shape `s`”.

The shape grammar is inductive:

$$`\mathrm{Shape} ::= \mathrm{scalar}\;|\;\mathrm{dim}(n,\mathrm{Shape})`

So a matrix is not a tensor plus a loose runtime rank field. It is a value whose type remembers that
it has two dimensions.

User-facing code does not write `Spec.Tensor` explicitly. Instead, examples usually import
`TorchLean` and use the public `Tensor`, `Shape`, and constructor macros from the `NN` umbrella.

This is still a tensor in the spec layer, with a precise meaning. The `NN.Tensor` layer gives you
good constructors and printing without changing the math.

## PyTorch Similarity

The closest PyTorch mental model is:

```
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[0.2, -0.1], [0.0, 0.3]])
z = x + y
```

In TorchLean, the same idea is written with shape information in the type:

```
import NN
open TorchLean

def x : Tensor.T Float (shape![2, 2]) :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]

def y : Tensor.T Float (shape![2, 2]) :=
  tensor! [[0.2, -0.1], [0.0, 0.3]]

def z := x + y
```

That is the basic pattern throughout TorchLean: the syntax feels familiar, but the shape is
checked by Lean instead of being discovered only at runtime.

# Shapes: Values (and Often Types)

A tensor shape is a `TorchLean.Shape`. In examples, the most convenient way to write a shape is the
`shape![...]` macro:

```
import NN
open TorchLean

def sVec : TorchLean.Shape := shape![4]
def sMat : TorchLean.Shape := shape![3, 2]
```

For common ML shapes, TorchLean also provides readable constructors:

- `Shape.vec n` for length-`n` vectors,
- `Shape.mat rows cols` for matrices,
- `Shape.images n c h w` for NCHW image batches,
- plus `Shape.CHW`, `Shape.nchw`, `Shape.NHWC`, and others.

The full list is generated in the tensor reference, but the convention is simple: names such as
`vec`, `mat`, `nchw`, and `images` are readable names for repeated `Shape.dim` constructors.

# DTypes: Just Lean Types

TorchLean uses the scalar element type as the “dtype”. For example:

- `Tensor Float s` is a runnable floating tensor (convenient for examples),
- `Tensor ℚ s` is a rational tensor (useful for exact arithmetic),
- `Tensor ℝ s` is a proof-side mathematical tensor,
- `Tensor TorchLean.Floats.IEEE32Exec s` is an executable IEEE-754 binary32 tensor.
- `Tensor Interval s` is the shape of a verifier or bound-propagation tensor when the scalar domain
  carries intervals or relaxations.

That means ordinary Lean abstractions, including typeclasses and polymorphic definitions, can be
used when code should run on multiple scalar backends.

One architecture can therefore be read as a family of functions:

$$`f_\alpha :
\mathrm{Tensor}(\alpha,s_{\mathrm{in}})
\longrightarrow
\mathrm{Tensor}(\alpha,s_{\mathrm{out}})`

The subscript changes when we choose `Float`, `IEEE32Exec`, `ℝ`, intervals, or another scalar
domain. The architecture and shapes remain the same.

In the codebase, the scalar layer splits into a few pieces:

- `Spec.SpecScalar` is fixed to `ℝ` for the mathematical spec layer.
- public trainers choose the runtime scalar through the `dtype` field passed to `Trainer.new`,
  while proof and runtime
  internals still quantify over scalar interfaces directly.
- `tensorF!` and `tensorF321d` let you author examples once in `Float` and then cast to a runtime
  scalar or executable FP32 backend.

For readers in a theorem file, `α` often means “the scalar the theorem is polymorphic over”. In a
training tutorial, the beginner path usually avoids exposing `α` at all: the trainer picks the
runtime scalar from the `dtype` field in the `Trainer.new` config. Advanced runtime files may still
spell out the scalar parameter because the same model can be instantiated over `Float`,
`IEEE32Exec`, or `ℝ` depending on whether you are executing, validating, or proving.

# The Bug This Prevents

Here is the shape mismatch from the opening, written as a TorchLean object:

```
import NN
open TorchLean

def pred : Tensor.T Float (shape![32, 1]) :=
  tensorND! [32, 1] (List.replicate 32 0.0)

def label : Tensor.T Float (shape![32]) :=
  tensorND! [32] (List.replicate 32 0.0)
```

TorchLean does not decide whether this should be a squeeze, a reshape, a one-hot encoding, or a
different loss. The model author has to name the intended conversion. That small requirement is what
keeps later graph and verification code from inheriting a silent broadcasting convention.

# Constructing Tensors (The Ergonomic Layer)

For compact examples, the most common constructors are:

- `tensor1d` / `tensor2d` for lists of numbers,
- `tensorND` for runtime dims plus a flat row-major list,
- `tensorND!` for constants when the length proof should be solved automatically,
- `tensor!` for nested bracket literals (often the nicest for handwritten tensors).

Example (nested brackets, like nested Python lists):

```
import NN
open TorchLean

-- A 2×2 tensor (shape inferred from the nesting)
def x : Tensor.T Float (shape![2, 2]) :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]
```

When you are writing code that should work with several backends, `tensorF!` is a useful trick: write Float
literals once, then cast elementwise into a runtime scalar `α`.

```
-- `cast : Float → α` comes from the runtime runner (see the training tutorials).
def w {α : Type} (cast : Float → α) : Tensor.T α (shape![3, 2]) :=
  tensorF! cast [3, 2] [0.2, -0.1, 0.0, 0.3, -0.4, 0.1]
```

For runtime examples, the same idea shows up as a cast from `Float` constants into the active
backend scalar in the model code. The [runtime module API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Module.lean)
contains tensor-casting operations; at the public API boundary these values are exposed as
`TensorPack`/`ParamTensors` rather than as the runtime heterogeneous-list representation.

# Four Everyday Shapes

Most model code uses the same few shapes over and over. It helps to learn them as reading patterns,
not as abstract syntax.

## A feature vector

```
def x : Tensor.T Float (Shape.vec 3) :=
  tensor! [0.2, -0.1, 0.7]
```

Read this as one sample with three features.

## A matrix batch

```
def xs : Tensor.T Float (Shape.mat 4 3) :=
  tensor! [
    [0.0, 0.1, 0.2],
    [0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1]
  ]
```

Read this as four samples, each with three features. A linear layer with
`pfx := Shape.vec 4` will act on the last dimension and preserve the batch axis.

## A CHW image

```
def img : Tensor.T Float (Shape.CHW 1 4 4) :=
  tensorND! [1, 4, 4] [
    0, 0, 1, 1,
    0, 1, 1, 0,
    1, 1, 0, 0,
    1, 0, 0, 1
  ]
```

Read this as one channel, height four, width four. The CNN tutorials use this convention because it
matches the usual PyTorch `NCHW` layout once a batch axis is added.

## An image batch

```
Shape.images batch channels height width
```

This is the shape expected by the public `nn.conv` builder. The batch dimension is part of the model
type, so a model trained with batch size `8` and a model trained with batch size `16` have different
types unless the file abstracts over `batch`.

# Reading Tensor Types In Model Code

When you see a model type such as:

```
nn.Sequential (Shape.mat batch 2) (Shape.mat batch 1)
```

read it from right to left as a contract:

- the input is a batch of two-feature samples;
- the output is a batch of one-value predictions;
- every layer in between must preserve or transform shapes so the sequence composes.

When you see:

```
nn.Sequential (Shape.images batch 3 32 32) (shape![batch, 10])
```

read it as an image classifier: a batch of RGB images goes to ten logits per image.

This habit is more useful than memorizing every constructor. The shape tells you what the model
claims to do before you read the body.

# Shape Design Decisions

TorchLean's tensor layer makes a few choices that are worth understanding early.

1. *Shapes are part of the specification.* A tensor is not just a pointer plus runtime metadata; in
   the core APIs, its shape appears in the Lean type.
2. *Dynamic inputs are allowed at the boundary.* File loaders, JSON artifacts, and CLI data often
   arrive with runtime dimensions. TorchLean checks those dimensions as data and then packages them
   into typed or dynamic tensors.
3. *Broadcasting is explicit at the operator layer.* The project avoids implicit promotion rules
   when those rules would make proof obligations harder to read.
4. *Flattened and shaped views are connected by lemmas.* Runtime and verifier code often wants flat
   buffers; proof code often wants shaped tensors. The tensor library keeps those views related.

This is why a tutorial tensor can feel close to PyTorch while the theorem declarations still have
enough structure to prove algebraic facts about the same operations.

For example, pointwise addition is not "add two buffers and hope their metadata agrees." Its
informal type is:

$$`\mathrm{add} :
\mathrm{Tensor}(\alpha,s)
\times
\mathrm{Tensor}(\alpha,s)
\longrightarrow
\mathrm{Tensor}(\alpha,s)`

That one shared `s` is doing real work. It is why the verifier and proof layer can later talk about
the same operation without rediscovering the layout from a runtime trace.

# A Tiny Shape Bug Walkthrough

Suppose a model expects a batch of two length-three inputs and a weight matrix from three features to
four outputs. In ordinary Python code it is easy to accidentally transpose the weight and discover
the mistake only when `matmul` runs.

In TorchLean, the intended shapes are visible at the definition site:

```
import NN
open TorchLean

def xs : Tensor.T Float (shape![2, 3]) :=
  tensor! [[1, 2, 3], [4, 5, 6]]

def wGood : Tensor.T Float (shape![3, 4]) :=
  tensor! [[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]]
```

A transposed weight has shape `shape![4, 3]`, which is a different type. That does not prove the
model is correct, but it prevents a large class of accidental wiring mistakes from reaching the
runtime or verifier.

# Dynamic Shapes

Sometimes you truly do not know the shape statically (CLI inputs, JSON, file formats). For those
cases TorchLean provides:

- `tensorND : List Nat → List α → Except String (Tensor α (shapeOfDims dims))`, with a runtime
  length check;
- `DynTensor α`, which stores the `Shape` as data alongside the tensor.

This lets tooling remain shape-aware even when the type cannot carry the full shape index.

If you want the most permissive constructors, the API also includes:

- `tensor2d?` and `tensor2d` for nested lists,
- `tensor2dPadTo` and `tensor2dPadRight` for ragged inputs that you intend to pad,
- `tensorDynND` when you want a dynamic container rather than a static shape index.

The relevant design note is that TorchLean does not force every input format to be perfectly typed at
the boundary. Instead, it lets you choose between:

- shapes with explicit proofs (`tensorNDOfLenEq`),
- shapes checked at runtime (`tensorND`, `tensorDynND`), and
- permissive padding behavior for sequence data (`tensor2dPadTo`, `tensor2dPadRight`).

That is the shape of a mixed proof/runtime ML stack.

## How TorchLean Differs from Plain PyTorch

The PyTorch version checks shapes when you run the code. TorchLean checks many of them when the file
elaborates, so a shape mismatch becomes a compile error instead of a late failure.

For example, a common bug in numerical code is accidentally mixing a vector and a matrix with the
wrong axis order or layout. In PyTorch you often discover that after a runtime error; in TorchLean the type
forces the mismatch to be visible up front.

```
-- PyTorch style idea:
--   y = x @ w
--
-- TorchLean-shaped idea:
def xMat : Tensor.T Float (shape![2, 3]) := tensor! [[1, 2, 3], [4, 5, 6]]
def wMat : Tensor.T Float (shape![3, 4]) := tensor! [[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]]
-- result shape is checked by the matmul API / layer shape rules, not by a hidden runtime guard.
```

# A Runnable Starting Point

If you want a small file that you can run immediately and stay focused on tensors,
start here:

- [NN.Examples.Quickstart.TensorBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean)
- [NN.Examples.Advanced.Tensors.Basic API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Tensors/Basic.lean)

It demonstrates:

- dtype-as-type (`Float`, `ℚ`, `Int`),
- `tensor!` nesting + row-major flattening,
- `tensorF321d` for executable float32,
- `tensorND` / `tensorDynND` for runtime-shaped inputs,
- and the fact that printing is disabled by design for proof-only scalar backends like `ℝ`.

# What the Tensor Layer Gives You

TorchLean’s tensor layer is part of the specification, not only an ergonomic layer. It is strong because the shape index is
part of the tensor’s type, and the proof layer already knows how to move between typed views,
flattened views, and algebraic laws.

Representative theorems and definitions:

- `Spec.flattenR` / `Spec.unflattenR` round trip the tensor view used in proofs.
- `Spec.toVec_add_spec` and `Spec.toVec_scale_spec` connect tensor operations to pointwise algebra.
- `Spec.mul_spec_assoc` and `Spec.mul_spec_comm` give the usual algebraic laws for pointwise tensor
  multiplication.
- `Spec.add_spec_comm` and `Spec.mul_spec_fill_zero` show the expected behavior of tensor arithmetic
  on special cases.
- `Spec.dot` packages the familiar “elementwise multiply, then sum” pattern.

Those facts matter because they let later chapters talk about models, losses, and verification using
ordinary algebra instead of ad hoc shape bookkeeping. The same tensor library can be used for:

- exact reasoning in `ℚ` or `ℝ`,
- executable float32 examples,
- and shape-safe data / model pipelines.

If you want the proof companion, start from the
[tensor proof API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic.lean).
