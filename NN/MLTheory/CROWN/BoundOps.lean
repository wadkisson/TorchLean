/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# Directed-rounding primitives for interval propagation

TorchLean’s IBP/CROWN code represents bounds as endpoint pairs (`lo`/`hi`) inside `Box`/`FlatBox`.
To make those bounds meaningful under different numeric semantics, we abstract the *primitive*
endpoint operations (directed rounding).

Intuition:
- For pure real/interval backends, using ordinary `+`/`*` is already enclosure-safe (because the
  scalar itself is an interval type with outward rounding).
- For finite-precision backends with discrete grids (e.g. `IEEE32Exec`), we want *directed rounding*
  primitives like `addDown/addUp` and `mulDown/mulUp` so that interval propagation encloses the
  corresponding exact real operation.

This file defines a small typeclass `BoundOps` and some helper combinators. There is intentionally
no generic fallback instance: ordinary finite-precision arithmetic is not directed rounding and
must not silently enter a sound bound-propagation path.

## Integration points in the current codebase

The intended usage is:

- Keep graphs/layers scalar-polymorphic over `[Context α]`.
- When a routine *propagates bounds* (IBP/affine/CROWN), also require `[BoundOps α]` and use
  `addDown/addUp/subDown/subUp/mulDown/mulUp` at the endpoints.

Concretely:

- `NN/MLTheory/CROWN/Core.lean`
  - `AffineVec.eval_on_box`: min/max over products and accumulation use `BoundOps`.
  - `IBP.linear`: interval linear layer propagation uses `BoundOps`.
- `NN.MLTheory.CROWN.Graph`
  - `box_add`, `box_sub`, `box_mul_elem`: endpoint propagation uses `BoundOps`.

To extend coverage, the next candidates are `box_inv`, `box_sqrt`, and the various nonlinearity
propagation rules in `NN.MLTheory.CROWN.Runtime.Ops` (these may need additional directed math
primitives or an Arb-backed enclosure oracle).
-/

@[expose] public section


namespace NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-!
## `BoundOps α`

`BoundOps` supplies directed-rounding versions of the arithmetic
primitives that appear in IBP for affine/linear layers and basic arithmetic nodes.

If you want to swap in a quantized backend, the key is to provide an instance of `BoundOps` for
your scalar type.
-/
class BoundOps (α : Type) [Context α] where
  addDown : α → α → α
  addUp   : α → α → α
  subDown : α → α → α
  subUp   : α → α → α
  mulDown : α → α → α
  mulUp   : α → α → α

/-!
Exact real arithmetic needs no rounding, so its lower and upper operations coincide.
-/
noncomputable instance instBoundOpsReal : BoundOps ℝ where
  addDown := (· + ·)
  addUp   := (· + ·)
  subDown := (· - ·)
  subUp   := (· - ·)
  mulDown := (· * ·)
  mulUp   := (· * ·)

/-!
## Host binary64 endpoints

Lean's `Float` operations round to nearest on the host binary64 format. For executable checking we
widen every finite result by one adjacent representable value. This is deliberately an explicit
instance rather than a generic fallback: its soundness depends on the host IEEE-754 arithmetic
boundary documented by Lean, whereas `instBoundOpsReal` is exact and the `IEEE32Exec` instance is
connected to TorchLean's bit-level binary32 proofs.
-/

namespace HostFloat

def signMask : UInt64 := 0x8000000000000000
def posInfBits : UInt64 := 0x7ff0000000000000
def negInfBits : UInt64 := 0xfff0000000000000

/-- Adjacent binary64 value above `x`, with the usual IEEE behavior at infinities and zeros. -/
def nextUp (x : Float) : Float :=
  let bits := x.toBits
  if x.isNaN || bits = posInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float.ofBits 1
  else if bits &&& signMask = 0 then
    Float.ofBits (bits + 1)
  else
    Float.ofBits (bits - 1)

/-- Adjacent binary64 value below `x`, with the usual IEEE behavior at infinities and zeros. -/
def nextDown (x : Float) : Float :=
  let bits := x.toBits
  if x.isNaN || bits = negInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float.ofBits (signMask + 1)
  else if bits &&& signMask = 0 then
    Float.ofBits (bits - 1)
  else
    Float.ofBits (bits + 1)

end HostFloat

/--
Outward-widened host binary64 operations.

This instance is suitable for executable certificate replay under the trusted host-Float boundary.
Use `IEEE32Exec` when the binary32 endpoint calculation itself must be connected to Lean proofs.
-/
instance instBoundOpsFloat : BoundOps Float where
  addDown a b := HostFloat.nextDown (a + b)
  addUp a b := HostFloat.nextUp (a + b)
  subDown a b := HostFloat.nextDown (a - b)
  subUp a b := HostFloat.nextUp (a - b)
  mulDown a b := HostFloat.nextDown (a * b)
  mulUp a b := HostFloat.nextUp (a * b)

namespace BoundOps

/-! Small helpers built only from `Context.decidable_gt`. -/

@[inline] def min2 (a b : α) : α :=
  if decide (a > b) then b else a

/-- Maximum of two scalars, using only `Context.decidable_gt` via `decide (a > b)`. -/
@[inline] def max2 (a b : α) : α :=
  if decide (a > b) then a else b

end BoundOps

end NN.MLTheory.CROWN
