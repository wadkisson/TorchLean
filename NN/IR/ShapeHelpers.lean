/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape

/-!
# IR shape helpers

Small runtime witnesses and list utilities shared by IR evaluation, CROWN, and permute lowering.

Kept dependency-light (Shape only) so verification and runtime paths can share one implementation.
-/

@[expose] public section

namespace NN.IR
namespace Graph

open _root_.Spec

/-- Build a witness that `axis` is a valid axis for shape `s`. Returns `none` when out of bounds. -/
def mkValidAxis? (axis : Nat) : (s : Shape) → Option (PLift (Shape.valid_axis axis s))
  | .scalar => none
  | .dim n rest =>
      match axis, n with
      | 0, Nat.succ k => some ⟨Shape.valid_axis.valid_zero (n := k) (s := rest)⟩
      | 0, 0 => none
      | Nat.succ a, Nat.succ k =>
          (mkValidAxis? a rest).map (fun h =>
            ⟨Shape.valid_axis.valid_succ (n := k) (s := rest) (k := a) h.down⟩)
      | Nat.succ _, 0 => none

/-- Return the index of the first occurrence of `x` in `xs` (or `none` if absent). -/
def findIndex? {α : Type} [BEq α] (xs : List α) (x : α) : Option Nat :=
  let rec go (i : Nat) : List α → Option Nat
    | [] => none
    | y :: ys => if y == x then some i else go (i + 1) ys
  go 0 xs

/-- Safe list indexing: `listGet? xs n` returns `some xs[n]` when in bounds. -/
def listGet? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n + 1 => listGet? xs n

/-- Swap the adjacent entries at positions `d` and `d+1` (no-op when out of range). -/
def swapAt {α : Type} (xs : List α) (d : Nat) : List α :=
  match xs, d with
  | [], _ => []
  | [x], _ => [x]
  | x :: y :: rest, 0 => y :: x :: rest
  | x :: rest, d + 1 => x :: swapAt rest d

/--
Compute a sequence of adjacent swaps that realizes a target permutation.

Returns `none` when `perm` is not a permutation of `range r`.
-/
def swapDepthsForPerm? (perm : List Nat) (r : Nat) : Option (List Nat) :=
  let rec bubbleLeft (cur : List Nat) (swapsRev : List Nat) (i j : Nat) : List Nat × List Nat :=
    if j ≤ i then
      (cur, swapsRev)
    else
      bubbleLeft (swapAt cur (j - 1)) ((j - 1) :: swapsRev) i (j - 1)
  if perm.length = r && perm.all (fun d => d < r) then
    let rec go (i : Nat) (targets : List Nat) (cur : List Nat) (swapsRev : List Nat) :
        Option (List Nat) :=
      match targets with
      | [] => some swapsRev.reverse
      | target :: targets' =>
          match findIndex? cur target with
          | none => none
          | some j =>
              let (cur', swapsRev') := bubbleLeft cur swapsRev i j
              go (i + 1) targets' cur' swapsRev'
    go 0 perm (List.range r) []
  else
    none

/--
Except-wrapped permutation lowering used by IR evaluation.

Same algorithm as `swapDepthsForPerm?`, with readable error messages on failure.
-/
def swapDepthsForPerm (perm : List Nat) (r : Nat) : Except String (List Nat) :=
  match swapDepthsForPerm? perm r with
  | some swaps => .ok swaps
  | none =>
      if perm.length ≠ r then
        .error s!"permute: expected length {r}, got {perm.length}"
      else if !(perm.all (fun d => d < r)) then
        .error s!"permute: axis out of range in {repr perm} for rank {r}"
      else
        .error s!"permute: invalid permutation {repr perm} for rank {r}"

end Graph
end NN.IR
