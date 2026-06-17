/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.ConvPool

/-!
# TorchLean NN: Attention and Sequence Syntax
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/--
Multi-head self-attention layer for a sequence `(n × dModel) → (n × dModel)`.

This layer packs the four projection matrices `(Wq, Wk, Wv, Wo)` and calls the TorchLean attention
primitive. An optional boolean mask of shape `(n × n)` can be provided (e.g. causal masking).

PyTorch analogy: `torch.nn.MultiheadAttention(embed_dim=dModel, num_heads=numHeads)` in
  self-attention
mode (shape conventions differ; TorchLean uses explicit `n × dModel` tensors).
-/
def multiHeadAttention
    (batch n dModel numHeads headDim : Nat)
    {h1 : n ≠ 0}
    (seedW : Nat := 0)
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
    LayerDef (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  let projDim := numHeads * headDim
  let wProjShape : Shape := .dim dModel (.dim projDim .scalar)
  let wOShape : Shape := .dim projDim (.dim dModel .scalar)
  let wq0 : Tensor Float wProjShape := Torch.Init.xavierW (outDim := dModel) (inDim := projDim)
    (seed := seedW)
  let wk0 : Tensor Float wProjShape := Torch.Init.xavierW (outDim := dModel) (inDim := projDim)
    (seed := seedW + 1)
  let wv0 : Tensor Float wProjShape := Torch.Init.xavierW (outDim := dModel) (inDim := projDim)
    (seed := seedW + 2)
  let wo0 : Tensor Float wOShape := Torch.Init.xavierW (outDim := projDim) (inDim := dModel) (seed
    := seedW + 3)
  { kind := s!"MultiHeadAttention(heads={numHeads}, headDim={headDim})"
    paramShapes := [wProjShape, wProjShape, wProjShape, wOShape]
    initParams := Torch.tlist4 wq0 wk0 wv0 wo0
    paramRequiresGrad := [true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wq wk wv wo x =>
          TorchLean.multiHeadAttention (m := m) (α := α)
            (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
            (h1 := h1)
            wq wk wv wo x (mask := mask)
  }

/-- Lift a single layer into a 1-layer sequential model. -/
def seq1 {σ τ : Shape} (l : LayerDef σ τ) : Seq σ τ :=
  .cons l (.id τ)

/-!
## Sequential model literal (`tlseq[...]`)

When writing compact models, chaining `seq1` with `>>>` is a bit verbose:

```lean
TorchLean.NN.seq1 (TorchLean.NN.linear inDim hidDim) >>>
TorchLean.NN.seq1 TorchLean.NN.tanh >>>
TorchLean.NN.seq1 (TorchLean.NN.linear hidDim outDim)
```

This macro provides a compact, explicit alternative:

```lean
tlseq[
  TorchLean.NN.linear inDim hidDim,
  TorchLean.NN.tanh,
  TorchLean.NN.linear hidDim outDim
]
```

It expands to `seq1 ... >>> seq1 ... >>> ...`.
The syntax is namespaced to avoid colliding with other libraries.
-/

syntax (name := torchLeanSeqLit) "tlseq" "[" term,+ "]" : term

macro_rules
  | `(tlseq[$l]) => `(TorchLean.NN.seq1 $l)
  | `(tlseq[$l, $ls,*]) => `(TorchLean.NN.seq1 $l >>> tlseq[$ls,*])

end NN

end TorchLean
end Autograd
end Runtime
