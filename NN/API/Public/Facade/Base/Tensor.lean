/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Base.Runtime

/-!
# TorchLean Tensor Names

Ops, random, shape, tensor, and semantic names exposed by the `NN` umbrella.
-/

@[expose] public section

namespace TorchLean

namespace Ops

/-!
Low-level executable ops for verification and compiler-facing examples.

Most model code should use `nn.*` and `Trainer.*`. Use `Ops.*` when writing an explicit
TorchLean executable program directly, for example before compiling a hand-built fragment to
`NN.IR.Graph`.
-/

export NN.API.TorchLean
  (RefTy const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape swapAdjacentAtDepth reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec scatterAddRow
   matmul concatVectors
   maxPool avgPool smoothMaxPool
   relu silu gelu sigmoid tanh softmax softplus exp log inv safeLog logSoftmax
   sum flatten linear mseLoss layerNorm multiHeadAttention conv convTranspose)

end Ops

namespace rand

export NN.API.rand
  (keyOf uniform mask uniformND maskND randND
   nextSeed nextSeedGlobal nextSeedsGlobal)

end rand

namespace Shape

@[inherit_doc Spec.Shape.scalar]
abbrev scalar := Spec.Shape.scalar

@[inherit_doc Spec.Shape.size]
abbrev size := Spec.Shape.size

@[inherit_doc Spec.Shape.toList]
abbrev toList := Spec.Shape.toList

@[inherit_doc NN.Tensor.shapeOfDims]
abbrev ofDims := NN.Tensor.shapeOfDims

end Shape

namespace Tensor

@[inherit_doc NN.Tensor.vector]
abbrev vector := NN.Tensor.vector

@[inherit_doc NN.Tensor.ofList]
abbrev ofList {α : Type} (dims : List Nat) (xs : List α) :
    Except String (Spec.Tensor α (Shape.ofDims dims)) :=
  NN.Tensor.ofList dims xs

@[inherit_doc NN.Tensor.float32Vector]
abbrev float32Vector := NN.Tensor.float32Vector

@[inherit_doc NN.Tensor.print]
abbrev print {α : Type} [NN.Tensor.DTypeName α] [NN.Tensor.TensorPrintable α]
    {s : Shape} (t : Spec.Tensor α s) : IO Unit :=
  NN.Tensor.print t

@[inherit_doc Spec.vectorFromList]
abbrev vectorFromList {α : Type} (xs : List α) :
    Spec.Tensor α (.dim xs.length .scalar) :=
  Spec.vectorFromList xs

@[inherit_doc Spec.matrixFromRows]
abbrev matrixFromRows {α : Type} [Inhabited α] (xss : List (List α)) :=
  Spec.matrixFromRows xss

@[inherit_doc Spec.fill]
abbrev fill {α : Type} (value : α) (s : Shape) : Spec.Tensor α s :=
  Spec.fill value s

@[inherit_doc Spec.pretty]
abbrev pretty {α : Type} [ToString α] {s : Shape} (t : Spec.Tensor α s) : String :=
  Spec.pretty t

@[inherit_doc Spec.Tensor.vecGet]
abbrev vecGet {α : Type} {n : Nat} (x : Spec.Tensor α (.dim n .scalar)) (i : Fin n) : α :=
  Spec.Tensor.vecGet x i

@[inherit_doc Spec.Tensor.toScalar]
abbrev toScalar {α : Type} (t : Spec.Tensor α .scalar) : α :=
  Spec.Tensor.toScalar t

@[inherit_doc Spec.Tensor]
abbrev T := Spec.Tensor

@[inherit_doc Spec.mapTensor]
abbrev map {α β : Type} {s : Shape} (f : α → β) (x : Tensor.T α s) : Tensor.T β s :=
  Spec.mapTensor f x

@[inherit_doc NN.API.Common.castTensor]
abbrev castFloat {α : Type} (cast : Float → α) {s : Shape} (t : Tensor.T Float s) :
    Tensor.T α s :=
  NN.API.Common.castTensor cast t

/--
Repeat one tensor across a fixed batch axis.

Use this for classifier demos whose checked model consumes a whole batch, while the example wants to
inspect one ordinary input.
-/
def repeatBatch {α : Type} {s : Shape} (batch : Nat) (x : Tensor.T α s) :
    Tensor.T α (.dim batch s) :=
  Spec.Tensor.dim (fun _ => x)

/--
Convert a runtime tensor back to a `Float` tensor inside `IO`.

Trainer prediction handles use this so examples can train under executable IEEE32 or another scalar
backend, then inspect ordinary `Float` tensors afterward.
-/
def toFloatIO {α : Type} [Runtime.TensorScalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
    ∀ {s : Shape}, Tensor.T α s → IO (Tensor.T Float s)
  | .scalar, .scalar x => do
      pure <| .scalar (← _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv.toFloat
        (α := α) x)
  | .dim n s, .dim f => do
      let xs ← Array.mapM' (fun t => toFloatIO (s := s) t) (Array.ofFn f)
      have hSize : xs.val.size = n := by
        simpa using xs.property
      pure <| .dim (fun i =>
        xs.val[i.val]'(by
          rw [hSize]
          exact i.isLt))

export NN.API.Common
  (tensorF tensorFGen tensorFGen! tensorFGenShape!)

end Tensor

namespace Semantics

@[inherit_doc NN.API.Semantics.relu]
abbrev relu {α : Type} [Zero α] [Max α] (x : α) : α :=
  NN.API.Semantics.relu x

end Semantics

/-!
## Public Namespaces

These are the names users usually type for model construction, tensor utilities, training, runtime
selection, and verification examples. The definitions below forward to the checked implementation;
the semantics are not copied or forked.
-/

end TorchLean
