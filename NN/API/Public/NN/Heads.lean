/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Public.NN.Transformer

/-!
Task heads for public neural-network models.

The definitions here package classifier, regression, and language-model heads that sit on top of
the reusable block and Transformer APIs.
-/

@[expose] public section

namespace NN
namespace API
namespace nn
namespace pure
namespace heads


/--
Classification head: `Flatten -> Linear`.

Named head constructor built from `nn.flattenLinear`.
-/
def classifier {s : Spec.Shape} (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential s (NN.Tensor.Shape.Vec classes) :=
  flattenLinear (s := s) classes seedW seedB

/-- Regression head: `Flatten -> Linear` with `outDim` outputs. -/
def regressor {s : Spec.Shape} (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential s (NN.Tensor.Shape.Vec outDim) :=
  flattenLinear (s := s) outDim seedW seedB

/--
`Flatten(start_dim=1) -> Linear` head for batched tensors.

Input:  `N × σ`
Output: `Mat N classes`
-/
def classifierBatch {n : Nat} {s : Spec.Shape} (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential (.dim n s) (NN.Tensor.Shape.Mat n classes) :=
  seq!
    flattenBatch (n := n) (s := s),
    linear (Spec.Shape.size s) classes (seedW := seedW) (seedB := seedB) (pfx :=
      NN.Tensor.Shape.Vec n)

/-- Batched regression head: `Flatten(start_dim=1) -> Linear(_, outDim)` producing `Mat N outDim`.
  -/
def regressorBatch {n : Nat} {s : Spec.Shape} (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential (.dim n s) (NN.Tensor.Shape.Mat n outDim) :=
  seq!
    flattenBatch (n := n) (s := s),
    linear (Spec.Shape.size s) outDim (seedW := seedW) (seedB := seedB) (pfx :=
      NN.Tensor.Shape.Vec n)

end heads

end pure

end nn

end API
end NN
