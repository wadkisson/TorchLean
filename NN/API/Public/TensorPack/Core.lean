/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime

/-!
# Public TorchLean.TensorPack Core

Shape-indexed tensor packs and supervised-sample helpers used by the public API.

Main declarations:
- `NN.API.TorchLean.TensorPack`: the runtime typed tuple of tensors.
- `NN.API.tensorpack.*`: typed-pack constructors, projections, mapping, zipping, append, and split.
- `TorchLean.Sample.*`: supervised `(x, y)` samples and minibatch wrappers.

The operations here are small wrappers around the checked tensor-pack representation; they do not
introduce a second semantics for tensors or samples.
-/

@[expose] public section

namespace NN
namespace API

namespace tensorpack

/-- Construct a one-element tensor pack. -/
abbrev singleton {α : Type} {s : Spec.Shape} (x : Spec.Tensor α s) : TorchLean.TensorPack α [s] :=
  TorchLean.tensorpackSingleton x

/-- Construct a two-element tensor pack. -/
abbrev pair {α : Type} {s₁ s₂ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) :
    TorchLean.TensorPack α [s₁, s₂] :=
  TorchLean.tensorpackPair x₁ x₂

/-- Construct a three-element tensor pack. -/
abbrev triple {α : Type} {s₁ s₂ s₃ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) (x₃ : Spec.Tensor α s₃) :
    TorchLean.TensorPack α [s₁, s₂, s₃] :=
  TorchLean.tensorpackTriple x₁ x₂ x₃

/-- Construct a four-element tensor pack. -/
abbrev quad {α : Type} {s₁ s₂ s₃ s₄ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂)
    (x₃ : Spec.Tensor α s₃) (x₄ : Spec.Tensor α s₄) :
    TorchLean.TensorPack α [s₁, s₂, s₃, s₄] :=
  TorchLean.tensorpackQuad x₁ x₂ x₃ x₄

/-- Map each tensor entry (shape-preserving). -/
def map {α β : Type} (f : ∀ {s : Spec.Shape}, Spec.Tensor α s → Spec.Tensor β s) :
    {ss : List Spec.Shape} → TorchLean.TensorPack α ss → TorchLean.TensorPack β ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs => .cons (f x) (map (f := f) (ss := ss) xs)

/-- Zip two tensor packs pointwise (shape-preserving). -/
def zipWith {α β γ : Type}
    (f : ∀ {s : Spec.Shape}, Spec.Tensor α s → Spec.Tensor β s → Spec.Tensor γ s) :
    {ss : List Spec.Shape} → TorchLean.TensorPack α ss → TorchLean.TensorPack β ss → TorchLean.TensorPack γ ss
  | [], .nil, .nil => .nil
  | _s :: ss, .cons x xs, .cons y ys =>
      .cons (f x y) (zipWith (f := f) (ss := ss) xs ys)

/-- Append two tensor packs. -/
def append {α : Type} :
    {ss₁ ss₂ : List Spec.Shape} → TorchLean.TensorPack α ss₁ → TorchLean.TensorPack α ss₂ → TorchLean.TensorPack α (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a tensor pack into its prefix and suffix. -/
def split {α : Type} :
    {ss₁ ss₂ : List Spec.Shape} → TorchLean.TensorPack α (ss₁ ++ ss₂) → TorchLean.TensorPack α ss₁ × TorchLean.TensorPack α ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (xs₁, xs₂) := split (α := α) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xs₁, xs₂)

/-- First element of a non-empty tensor pack. -/
def first {α : Type} {s : Spec.Shape} {ss : List Spec.Shape} :
    TorchLean.TensorPack α (s :: ss) → Spec.Tensor α s
  | .cons x _ => x

/-- Second element of a tensor pack with at least two entries. -/
def second {α : Type} {s₀ s₁ : Spec.Shape} {ss : List Spec.Shape} :
    TorchLean.TensorPack α (s₀ :: s₁ :: ss) → Spec.Tensor α s₁
  | .cons _ (.cons x _) => x

/-- Third element of a tensor pack with at least three entries. -/
def third {α : Type} {s₀ s₁ s₂ : Spec.Shape} {ss : List Spec.Shape} :
    TorchLean.TensorPack α (s₀ :: s₁ :: s₂ :: ss) → Spec.Tensor α s₂
  | .cons _ (.cons _ (.cons x _)) => x

/-- Fourth element of a tensor pack with at least four entries. -/
def fourth {α : Type} {s₀ s₁ s₂ s₃ : Spec.Shape} {ss : List Spec.Shape} :
    TorchLean.TensorPack α (s₀ :: s₁ :: s₂ :: s₃ :: ss) → Spec.Tensor α s₃
  | .cons _ (.cons _ (.cons _ (.cons x _))) => x

/-- Unpack a one-element tensor pack. -/
def unpackSingleton {α : Type} {s : Spec.Shape} :
    TorchLean.TensorPack α [s] → Spec.Tensor α s
  | .cons x .nil => x

/-- Unpack a two-element tensor pack into a Lean pair. -/
def unpackPair {α : Type} {s₁ s₂ : Spec.Shape} :
    TorchLean.TensorPack α [s₁, s₂] → (Spec.Tensor α s₁ × Spec.Tensor α s₂)
  | .cons x₁ (.cons x₂ .nil) => (x₁, x₂)

/-- Unpack a three-element tensor pack into a Lean triple. -/
def unpackTriple {α : Type} {s₁ s₂ s₃ : Spec.Shape} :
    TorchLean.TensorPack α [s₁, s₂, s₃] → (Spec.Tensor α s₁ × Spec.Tensor α s₂ × Spec.Tensor α s₃)
  | .cons x₁ (.cons x₂ (.cons x₃ .nil)) => (x₁, x₂, x₃)

/-- Unpack a four-element tensor pack into a Lean tuple. -/
def unpackQuad {α : Type} {s₁ s₂ s₃ s₄ : Spec.Shape} :
    TorchLean.TensorPack α [s₁, s₂, s₃, s₄] →
      (Spec.Tensor α s₁ × Spec.Tensor α s₂ × Spec.Tensor α s₃ × Spec.Tensor α s₄)
  | .cons x₁ (.cons x₂ (.cons x₃ (.cons x₄ .nil))) => (x₁, x₂, x₃, x₄)

end tensorpack

end API
end NN

namespace TorchLean.Sample

/-- A supervised sample `(x, y)` with input shape `σ` and target shape `τ`. -/
abbrev Supervised (α : Type) (σ τ : Spec.Shape) :=
  _root_.NN.API.TorchLean.TensorPack α [σ, τ]

/-- A fixed-size minibatch of supervised samples. -/
abbrev Batch (α : Type) (n : Nat) (σ τ : Spec.Shape) :=
  Supervised α (.dim n σ) (.dim n τ)

/-- Build a supervised sample `(x, y)` as a two-tensor pack. -/
def mk {α : Type} {σ τ : Spec.Shape} (x : Spec.Tensor α σ) (y : Spec.Tensor α τ) :
    Supervised α σ τ :=
  _root_.NN.API.tensorpack.pair x y

/-- Build a batched supervised sample `(xBatch, yBatch)`. -/
def batch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (x : Spec.Tensor α (.dim n σ)) (y : Spec.Tensor α (.dim n τ)) :
    Batch α n σ τ :=
  mk x y

/-- Extract the input tensor `x` from a supervised sample. -/
def x {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) : Spec.Tensor α σ :=
  _root_.NN.API.tensorpack.first s

/-- Extract the target tensor `y` from a supervised sample. -/
def y {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) : Spec.Tensor α τ :=
  _root_.NN.API.tensorpack.second s

/-- Unpack a supervised sample as the ordinary pair `(x, y)`. -/
def toPair {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) :
    Spec.Tensor α σ × Spec.Tensor α τ :=
  _root_.NN.API.tensorpack.unpackPair s

/-- `x` of a constructed supervised sample `mk x y` is `x`. -/
@[simp] theorem x_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    x (mk (α := α) (σ := σ) (τ := τ) xT yT) = xT := by
  rfl

/-- `y` of a constructed supervised sample `mk x y` is `y`. -/
@[simp] theorem y_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    y (mk (α := α) (σ := σ) (τ := τ) xT yT) = yT := by
  rfl

@[simp] theorem toPair_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    toPair (mk (α := α) (σ := σ) (τ := τ) xT yT) = (xT, yT) := by
  rfl

/-- Map a function over the input tensor `x`, leaving the target `y` unchanged. -/
def mapX {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α σ → Spec.Tensor α σ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (f (x s)) (y s)

/-- Map a function over the target tensor `y`, leaving the input `x` unchanged. -/
def mapY {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α τ → Spec.Tensor α τ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (x s) (f (y s))

/-- Map functions over both `x` and `y` in a supervised sample. -/
def mapXY {α : Type} {σ τ : Spec.Shape}
    (fx : Spec.Tensor α σ → Spec.Tensor α σ)
    (fy : Spec.Tensor α τ → Spec.Tensor α τ)
    (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (fx (x s)) (fy (y s))

end TorchLean.Sample
