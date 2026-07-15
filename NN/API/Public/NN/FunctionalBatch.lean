/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Public.NN.Core

@[expose] public section

namespace NN
namespace API
namespace nn
namespace Internal

/-!
`nn.functional` mirrors `torch.nn.functional`: pure, stateless building blocks.

In TorchLean these are derived ops over the small primitive `Ops` API, so the same code works on
both the eager backend and the compiled backend.
-/
namespace functional

/-!
PyTorch references:
- `torch.nn.functional`: `https://pytorch.org/docs/stable/nn.functional.html`
-/

export TorchLean.F
  (square checkpoint
   detach
   addB mulB
   embedding embeddingRowsNat embeddingBatchSeqNat mean
   dropoutSeeded)

-- Elementwise transcendentals + scalar-affine for scientific forward models
-- (fully qualified to disambiguate the `exp`/`log`/`scale` identifiers, which
-- also name primitives in scope).
export _root_.Runtime.Autograd.TorchLean.F
  (exp log scale shift affine)

end functional

/-!
## Leading-Dimension Mapping

`mapLeading leading model` applies a model independently across every index in an arbitrary leading
shape. A conventional batch is the special case `leading = .dim batch .scalar`; multiple leading
axes work without introducing another tensor or model type.

Correctness-first batch lift for exposing PyTorch-like `N×...` APIs even when a primitive only
exists for the unbatched shape.
-/

namespace Implementation

/--
Expose a runtime layer whose outer axis is a flat batch as a layer over an arbitrary leading
shape. The adapter changes only the view of the input and output; parameters, buffer updates, and
the underlying forward program are preserved.
-/
def adaptFlatBatch (leading : Spec.Shape) {σ τ : Spec.Shape}
    (l : LayerDef (.dim (Spec.Shape.size leading) σ) (.dim (Spec.Shape.size leading) τ)) :
    LayerDef (leading.concat σ) (leading.concat τ) :=
  { kind := l.kind
    paramShapes := l.paramShapes
    initParams := l.initParams
    runtimeInit := l.runtimeInit
    paramRequiresGrad := l.paramRequiresGrad
    updateBuffers := l.updateBuffers.map fun update mode {α} _ _ ps x =>
      update mode ps <| Spec.Tensor.reshapeSpec x (by
        simp [Spec.Shape.size_concat, Spec.Shape.size])
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := l.paramShapes ++ [leading.concat σ])
          (β := m (TorchLean.RefTy (m := m) (α := α) (leading.concat τ)))
          (fun args => do
            let (ps, x) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes) (τ := leading.concat σ) args
            let xBatch ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := leading.concat σ) (s₂ := .dim (Spec.Shape.size leading) σ)
                x (by simp [Spec.Shape.size_concat, Spec.Shape.size])
            let yBatch ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes ++ [.dim (Spec.Shape.size leading) σ])
                (β := m (TorchLean.RefTy (m := m) (α := α)
                  (.dim (Spec.Shape.size leading) τ)))
                (l.forward mode (α := α) (m := m))
                (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons xBatch .nil))
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim (Spec.Shape.size leading) τ) (s₂ := leading.concat τ)
              yBatch (by simp [Spec.Shape.size_concat, Spec.Shape.size]))
  }

/--
Lift a single-example `LayerDef σ τ` to operate on a leading batch axis.

This is a correctness-first batch lift: it runs the underlying layer independently on each batch
element. Prefer a primitive batched layer when one exists.
-/
def mapLayerOverAxis (n : Nat) {σ τ : Spec.Shape} (l : LayerDef σ τ) :
    LayerDef (.dim n σ) (.dim n τ) :=
  let inSize : Nat := Spec.Shape.size σ
  let outSize : Nat := Spec.Shape.size τ
  { kind := l.kind
    paramShapes := l.paramShapes
    initParams := l.initParams
    runtimeInit := l.runtimeInit
    paramRequiresGrad := l.paramRequiresGrad
    updateBuffers := l.updateBuffers.map fun update mode {_α} _ _ ps x =>
      match x with
      | Spec.Tensor.dim rows =>
          (List.finRange n).foldlM (init := ps) fun state i => update mode state (rows i)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := l.paramShapes ++ [.dim n σ])
          (β := m (TorchLean.RefTy (m := m) (α := α) (.dim n τ)))
          (fun args => do
            let (ps, xBatch) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes) (τ := .dim n σ) args
            let xMat ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := .dim n σ) (s₂ := .dim n (.dim inSize .scalar))
                xBatch (by simp [Spec.Shape.size, inSize])
            let zeros : Spec.Tensor α (.dim n (.dim outSize .scalar)) :=
              _root_.Spec.Tensor.dim (fun _ =>
                _root_.Spec.Tensor.dim (fun _ =>
                  _root_.Spec.Tensor.scalar (0 : α)))
            let out0 ← TorchLean.const (m := m) (α := α) (s := .dim n (.dim outSize .scalar)) zeros
            let outMat ← (List.finRange n).foldlM (init := out0) (fun acc i => do
              let xRow ← TorchLean.gatherRow (m := m) (α := α) (rows := n) (cols := inSize) xMat i
              let xSample ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := .dim inSize .scalar) (s₂ := σ)
                  xRow (by simp [Spec.Shape.size, inSize])
              let ySample ←
                _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                  (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                  (ss := l.paramShapes ++ [σ])
                  (β := m (TorchLean.RefTy (m := m) (α := α) τ))
                  (l.forward mode (α := α) (m := m))
                  (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons xSample .nil))
              let yRow ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := τ) (s₂ := .dim outSize .scalar)
                  ySample (by simp [Spec.Shape.size, outSize])
              TorchLean.scatterAddRow (m := m) (α := α) (rows := n) (cols := outSize) acc yRow i)
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim n (.dim outSize .scalar)) (s₂ := .dim n τ)
              outMat (by simp [Spec.Shape.size, outSize]))
  }

/-- Lift a sequential model to act pointwise on a leading batch axis. -/
def mapModelOverAxis (n : Nat) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (.dim n σ) (.dim n τ)
  | .id s => .id (.dim n s)
  | .cons l rest => .cons (mapLayerOverAxis n l) (mapModelOverAxis n rest)

end Implementation

/-- Apply a model pointwise over an arbitrary collection of leading dimensions. -/
def mapLeading (leading : Spec.Shape) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (leading.concat σ) (leading.concat τ)
  | model =>
      match leading with
      | .scalar => model
      | .dim n rest => Implementation.mapModelOverAxis n (mapLeading rest model)

/-!
Note: some low-level TorchLean layers (notably conv/pool/norm) have Nat-side well-formedness
proof arguments (e.g. `kH ≠ 0`).

The public path is *record-based specs* that hide those proofs via typeclasses like `NeZero`,
so examples can stay PyTorch-like without relying on positional macros.
-/
