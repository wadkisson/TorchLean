/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer

/-!
# TorchLean Sample Utilities

Additional sample and synthetic-data helpers. The canonical supervised sample representation and
its constructors are declared directly in `TorchLean.Sample` by `NN.API.Public.TensorPack`.
-/

@[expose] public section

namespace TorchLean

namespace Sample

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
  (singletonVectorFloat pointVectorFloat squareGrid affinePlane regressionTargetsFloat)

/-- A length-1 tensor, casting host `Float` into the selected runtime scalar. -/
def singletonVector {α : Type} [Runtime.TensorScalar α] [Runtime.Scalar α] (y : Float) :
    Tensor.T α (.dim 1 .scalar) :=
  NN.API.Samples.singletonVector NN.API.Runtime.ofFloat y

/-- A length-2 tensor, casting host `Float` coordinates into the selected runtime scalar. -/
def pointVector {α : Type} [Runtime.TensorScalar α] [Runtime.Scalar α] (x y : Float) :
    Tensor.T α (.dim 2 .scalar) :=
  NN.API.Samples.pointVector NN.API.Runtime.ofFloat x y

end Samples

end TorchLean
