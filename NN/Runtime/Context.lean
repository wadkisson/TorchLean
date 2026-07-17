/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Utils

/-!
# Runtime context

Shape-erased tensor packing used by runtime registries and autograd tapes.

Tensors are dependently typed by `Shape`, but runtime maps need a heterogeneous container.
`AnyTensor` carries the shape alongside the value; typed lookups cast using an equality proof.
`RuntimeContext` is a simple named registry record used by examples and widgets.
-/

@[expose] public section


namespace Runtime

open Spec
open Tensor

/--
Existential wrapper for tensors of arbitrary shape.

This is the core trick that lets a runtime registry store tensors of different shapes in one
container.
-/
structure AnyTensor (α : Type) where
  /-- The shape carried alongside the tensor value. -/
  s : Shape
  /-- The tensor value, indexed by `s`. -/
  t : Tensor α s

/--
Runtime context for tracking named variables and their gradients.

We keep two registries:
- `var_registry` for values, and
- `gradients` for accumulated gradients (typically produced by backprop).
-/
structure RuntimeContext (α : Type) where
  /--
  Registry of named variable values.

  The registry is ordered by shadowing priority: lookup returns the first matching name.
  -/
  var_registry : List (String × AnyTensor α)
  /--
  Registry of named gradients.

  The gradient registry uses the same ordering convention as `var_registry`. Higher-level training
  code decides whether a gradient should shadow, replace, or accumulate with an existing entry.
  -/
  gradients : List (String × AnyTensor α)
  /--
  Fresh id counter for allocating new variables/parameters.
  -/
  next_id : Nat

end Runtime
