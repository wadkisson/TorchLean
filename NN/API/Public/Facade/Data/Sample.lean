/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer

/-!
# TorchLean Public Samples

Public sample types and constructors for supervised datasets.
-/

@[expose] public section

namespace TorchLean

namespace Sample

/-- A supervised sample `(x, y)` with input shape `σ` and target shape `τ`. -/
abbrev Supervised := NN.API.sample.Supervised

/-- A fixed-size minibatch of supervised samples. -/
abbrev Batch := NN.API.sample.Batch

/-- Build a supervised sample `(x, y)`. -/
def mk {α : Type} {σ τ : Shape} (x : Tensor.T α σ) (y : Tensor.T α τ) :
    Supervised α σ τ :=
  NN.API.sample.mk x y

/-- Build a batched supervised sample `(xBatch, yBatch)`. -/
def batch {α : Type} {n : Nat} {σ τ : Shape}
    (x : Tensor.T α (.dim n σ)) (y : Tensor.T α (.dim n τ)) :
    Batch α n σ τ :=
  NN.API.sample.batch x y

/-- Extract the input tensor `x` from a supervised sample. -/
def x {α : Type} {σ τ : Shape} (s : Supervised α σ τ) : Tensor.T α σ :=
  NN.API.sample.x s

/-- Extract the target tensor `y` from a supervised sample. -/
def y {α : Type} {σ τ : Shape} (s : Supervised α σ τ) : Tensor.T α τ :=
  NN.API.sample.y s

/-- Unpack a supervised sample as the ordinary pair `(x, y)`. -/
def toPair {α : Type} {σ τ : Shape} (s : Supervised α σ τ) :
    Tensor.T α σ × Tensor.T α τ :=
  NN.API.sample.toPair s

/-- Map a function over the input tensor `x`, leaving the target `y` unchanged. -/
def mapX {α : Type} {σ τ : Shape}
    (f : Tensor.T α σ → Tensor.T α σ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  NN.API.sample.mapX f s

/-- Map a function over the target tensor `y`, leaving the input `x` unchanged. -/
def mapY {α : Type} {σ τ : Shape}
    (f : Tensor.T α τ → Tensor.T α τ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  NN.API.sample.mapY f s

/-- Map functions over both `x` and `y` in a supervised sample. -/
def mapXY {α : Type} {σ τ : Shape}
    (fx : Tensor.T α σ → Tensor.T α σ)
    (fy : Tensor.T α τ → Tensor.T α τ)
    (s : Supervised α σ τ) :
    Supervised α σ τ :=
  NN.API.sample.mapXY fx fy s

/--
Repeat one supervised sample across a fixed batch axis.

Use this for examples that naturally produce one `(x, y)` pair but need a model whose input/output
shapes already include a batch dimension.
-/
def repeatBatch {α : Type} {σ τ : Shape} (batch : Nat)
    (s : Supervised α σ τ) : Supervised α (.dim batch σ) (.dim batch τ) :=
  mk (_root_.Spec.Tensor.dim (fun _ => x s)) (_root_.Spec.Tensor.dim (fun _ => y s))

end Sample

namespace Samples

export NN.API.Samples
  (vec1Float vec2Float grid2Square affine2 regression2to1Float)

/-- A length-1 tensor, casting host `Float` into the selected runtime scalar. -/
def vec1 {α : Type} [Runtime.TensorScalar α] [Runtime.Scalar α] (y : Float) :
    Tensor.T α (Shape.vec 1) :=
  NN.API.Samples.vec1 NN.API.Runtime.ofFloat y

/-- A length-2 tensor, casting host `Float` coordinates into the selected runtime scalar. -/
def vec2 {α : Type} [Runtime.TensorScalar α] [Runtime.Scalar α] (x1 x2 : Float) :
    Tensor.T α (Shape.vec 2) :=
  NN.API.Samples.vec2 NN.API.Runtime.ofFloat x1 x2

end Samples

end TorchLean
