/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Recurrent

/-!
# TorchLean NN: Activation and Shape Layers
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
ReLU activation layer (no parameters).

PyTorch analogy: `torch.nn.relu` / `torch.nn.functional.relu`.
-/
def relu {s : Shape} : LayerDef s s :=
  { kind := "ReLU"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.relu (m := m) (α := α) (s := s) x
  }

/--
SiLU (a.k.a. swish) activation layer (no parameters).

PyTorch analogy: `torch.nn.silu` / `torch.nn.functional.silu`.
-/
def silu {s : Shape} : LayerDef s s :=
  { kind := "SiLU"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => _root_.Runtime.Autograd.Torch.silu (m := m) (α := α) (s := s) x
  }

/--
GELU activation layer (no parameters).

PyTorch analogy: `torch.nn.gelu` / `torch.nn.functional.gelu`.
-/
def gelu {s : Shape} : LayerDef s s :=
  { kind := "GELU"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => _root_.Runtime.Autograd.Torch.gelu (m := m) (α := α) (s := s) x
  }

/--
Sigmoid activation layer (no parameters).

PyTorch analogy: `torch.sigmoid`.
-/
def sigmoid {s : Shape} : LayerDef s s :=
  { kind := "Sigmoid"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.sigmoid (m := m) (α := α) (s := s) x
  }

/--
Hyperbolic tangent activation layer (no parameters).

PyTorch analogy: `torch.tanh`.
-/
def tanh {s : Shape} : LayerDef s s :=
  { kind := "Tanh"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.tanh (m := m) (α := α) (s := s) x
  }

/--
Softmax layer along the last axis (shape-preserving, no parameters).

PyTorch analogy: `torch.softmax(x, dim=-1)`.
-/
def softmax {s : Shape} : LayerDef s s :=
  { kind := "Softmax"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.softmax (m := m) (α := α) (s := s) x
  }

/--
Pointwise square `x ↦ x^2` (no parameters).

PyTorch analogy: `torch.square(x)` / `x.square()`.
-/
def square {s : Shape} : LayerDef s s :=
  { kind := "Square"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.F.square (m := m) (α := α) (s := s) x
  }

/--
Sum-reduce all elements of the input to a scalar (no parameters).

PyTorch analogy: `x.sum()`.
-/
def sum {s : Shape} : LayerDef s Shape.scalar :=
  { kind := "Sum"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.sum (m := m) (α := α) (s := s) x
  }

/--
Flatten any tensor to a 1D vector of length `Spec.Shape.size s` (no parameters).

PyTorch analogy: `torch.flatten(x)` or `x.reshape(-1)`.
-/
def flatten {s : Shape} : LayerDef s (.dim (Spec.Shape.size s) .scalar) :=
  { kind := "Flatten"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.flatten (m := m) (α := α) (s := s) x
  }

/--
Flatten everything except the leading batch axis.

Input shape: `N × s`. Output shape: `N × (size s)`.

PyTorch analogy: `torch.flatten(x, start_dim=1)` for an `N×…` tensor.
-/
def flattenKeep0 {batch : Nat} {s : Shape} :
    LayerDef (.dim batch s) (.dim batch (.dim (Spec.Shape.size s) .scalar)) :=
  { kind := "FlattenBatch"
    paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.flattenKeep0 (m := m) (α := α) (batch := batch) (s := s) x
  }

/--
Dropout layer controlled by `Mode`.

- In `Mode.train`, randomly zeroes entries with probability `p`.
- In `Mode.eval`, it is the identity.

We store `p` as a scalar parameter tensor (with `requires_grad := false`) so it can be threaded
through the unified parameter list without being optimized.

PyTorch analogy: `torch.nn.Dropout(p)` / `torch.nn.functional.dropout(x, p, training=...)`.
-/
def dropout {s : Shape} (p : Float) (seed : Nat := 0) : LayerDef s s :=
  let pShape : Shape := Shape.scalar
  let p0 : Tensor Float pShape := Tensor.scalar p
  { kind := s!"Dropout(p={p})"
    paramShapes := [pShape]
    initParams := Torch.tlistSingleton p0
    runtimeInit := some (.cons (.flat (FloatArray.mk #[p])) .nil)
    paramRequiresGrad := [false]
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun pRef x =>
          _root_.Runtime.Autograd.TorchLean.F.dropoutRefSeeded (m := m) (α := α) (s := s) x pRef
            seed
            (training := mode == .train)
  }
end NN

end TorchLean
end Autograd
end Runtime
