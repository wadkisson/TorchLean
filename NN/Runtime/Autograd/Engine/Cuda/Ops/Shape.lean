/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Elementwise

/-!
# CUDA Tape Operations: Shape and Reduction Nodes
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Tape

/-!
## Reductions / views
-/

/-- Reduce-sum of all entries, producing a scalar. -/
def sum {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let x ← requireValue (t := t) xId s
  let y := Buffer.reduceSum x
  let node : Node :=
    { name := some "sum"
      value := { s := Shape.scalar, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let dx := broadcastScalarToShape dLdy.buf s
        pure [(xId, { s := s, buf := dx })] }
  pure (t.addNode node)

/-- Flatten `s` into a 1D vector of length `Spec.Shape.size s`. -/
def flatten {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "flatten" xId s (.dim (Spec.Shape.size s) .scalar)
    (forward := fun x => x)
    (backward := fun _x dLdy => Buffer.copy dLdy)

/--
Reshape a buffer while preserving number of elements.

This is a no-copy view operation: it reuses the same contiguous buffer.
-/
def reshape {s₁ s₂ : Shape} (t : Tape) (xId : Nat) (_h : Spec.Shape.size s₁ = Spec.Shape.size s₂) :
    Result (Tape × Nat) :=
  unary (t := t) "reshape" xId s₁ s₂
    (forward := fun x => x)
    (backward := fun _x dLdy => Buffer.copy dLdy)

/-- Transpose an `m × n` CUDA buffer and register the matching transpose rule for backpropagation. -/
def transpose2d {m n : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let m32 ← u32 m
  let n32 ← u32 n
  unary (t := t) "transpose2d" xId (.dim m (.dim n .scalar)) (.dim n (.dim m .scalar))
    (forward := fun x => Buffer.transpose2d x m32 n32)
    (backward := fun _x dLdy => Buffer.transpose2d dLdy n32 m32)

/--
Swap adjacent axes at a given depth in an N-D buffer.

If `depth` is out of range, this is treated as the identity (matches the spec-layer helper).
-/
def swapAdjacentAtDepth {s : Shape} (t : Tape) (depth : Nat) (xId : Nat) : Result (Tape × Nat) := do
  let depth32 ← u32 depth
  let dimsIn : Array Nat := Shape.toArray s
  let outShape : Shape := s.swapAdjacentAtDepth depth
  let dimsOut : Array Nat := Shape.toArray outShape
  let validDepth := depth + 1 < Spec.Shape.rank s
  unary (t := t) "swapAdjacentAtDepth" xId s outShape
    (forward := fun x =>
      if validDepth then
        Buffer.swapAdjacentAtDepth x dimsIn depth32
      else
        Buffer.copy x)
    (backward := fun _x dLdy =>
      if validDepth then
        Buffer.swapAdjacentAtDepth dLdy dimsOut depth32
      else
        Buffer.copy dLdy)

/-- Permute a 3D tensor `(a,b,c) → (b,c,a)`. -/
def transpose3dFirstToLast {a b c : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let s₀ : Shape := .dim a (.dim b (.dim c .scalar))
  let (t1, id1) ← swapAdjacentAtDepth (t := t) (s := s₀) 0 xId
  let s₁ : Shape := .dim b (.dim a (.dim c .scalar))
  swapAdjacentAtDepth (t := t1) (s := s₁) 1 id1

/-- Permute a 3D tensor `(a,b,c) → (c,a,b)`. -/
def transpose3dLastToFirst {a b c : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let s₀ : Shape := .dim a (.dim b (.dim c .scalar))
  let (t1, id1) ← swapAdjacentAtDepth (t := t) (s := s₀) 1 xId
  let s₁ : Shape := .dim a (.dim c (.dim b .scalar))
  swapAdjacentAtDepth (t := t1) (s := s₁) 0 id1

/-- Swap the last two axes of a 3D tensor `(a,b,c) → (a,c,b)`. -/
def transpose3dLastTwo {a b c : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let s : Shape := .dim a (.dim b (.dim c .scalar))
  swapAdjacentAtDepth (t := t) (s := s) 1 xId

/--
Broadcast `x : s₁` to `s₂`.

Forward: `broadcastTo`.
Backward: sum-reduce broadcasted axes (`reduceFromBroadcastTo`).
-/
def broadcastTo {s₁ s₂ : Shape} (t : Tape) (cb : Shape.CanBroadcastTo s₁ s₂) (xId : Nat) :
    Result (Tape × Nat) := do
  let inDims := Shape.toArray s₁
  let outDims := Shape.toArray s₂
  let axisMap := Broadcast.axisMap cb
  unary (t := t) "broadcastTo" xId s₁ s₂
    (forward := fun x => Buffer.broadcastTo x inDims outDims axisMap)
    (backward := fun _x dLdy => Buffer.reduceFromBroadcastTo dLdy inDims outDims axisMap)

/-- Reduce-sum along `axis`. -/
def reduceSum {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let axis32 ← u32 axis
  let dims : Array Nat := Shape.toArray s
  let outShape : Shape := shapeAfterSum s axis
  unary (t := t) s!"reduce_sum(axis={axis})" xId s outShape
    (forward := fun x => Buffer.reduceSumAxis x dims axis32)
    (backward := fun _x dLdy =>
      let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
      let (inDims, outDims, axisMap) := Broadcast.broadcastArgs cb
      Buffer.broadcastTo dLdy inDims outDims axisMap)

/-- Reduce-mean along `axis`. -/
def reduceMean {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let axis32 ← u32 axis
  let dims : Array Nat := Shape.toArray s
  let outShape : Shape := shapeAfterSum s axis
  unary (t := t) s!"reduce_mean(axis={axis})" xId s outShape
    (forward := fun x =>
      let sum := Buffer.reduceSumAxis x dims axis32
      let denomNat :=
        match getDimSize s axis with
        | some n => n
        | none => 1
      Buffer.scale sum (1.0 / (Float.ofNat denomNat)))
    (backward := fun _x dLdy =>
      let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
      let (inDims, outDims, axisMap) := Broadcast.broadcastArgs cb
      let dLdx := Buffer.broadcastTo dLdy inDims outDims axisMap
      let denomNat :=
        match getDimSize s axis with
        | some n => n
        | none => 1
      Buffer.releaseThen dLdx <| Buffer.scale dLdx (1.0 / (Float.ofNat denomNat)))
end Tape

end Cuda
end Autograd
end Runtime
