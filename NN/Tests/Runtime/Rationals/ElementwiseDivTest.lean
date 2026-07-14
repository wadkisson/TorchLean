/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Utils
public import NN.Tensor

/-!
# ElementwiseDivTest

Regression test for the CPU elementwise division node (`Tape.div` / `TapeM.div`) over `ℚ`.

Exact rational arithmetic (no floating-point roundoff) lets us assert both the forward value
`a / b` *and* the quotient-rule backward by exact equality:

  `∂(a/b)/∂a = 1/b`,  `∂(a/b)/∂b = −a/b²`.

With `a = [6, 8]`, `b = [2, 4]`, and upstream gradient `dLdy = [1, 1]`:

  `y  = [3, 2]`,  `∂L/∂a = [1/2, 1/4]`,  `∂L/∂b = [−3/2, −1/2]`.
-/

open scoped NN.Spec.RationalAlgebraic

@[expose] public section

open Spec
open Tensor

namespace Tests
namespace Rationals
namespace ElementwiseDiv

open Runtime.Autograd

/-- Tag used for readable error messages. -/
abbrev tag : String := "elementwise_div_test (Rat)"

/-- Two-element vector shape. -/
abbrev s2 : Shape := .dim 2 .scalar

/-- Numerator `a`. -/
def a : Tensor ℚ s2 := tensorOfList! [2] [6.0, 8.0]
/-- Denominator `b`. -/
def b : Tensor ℚ s2 := tensorOfList! [2] [2.0, 4.0]
/-- Upstream gradient `∂L/∂y`. -/
def dLdy : Tensor ℚ s2 := tensorOfList! [2] [1.0, 1.0]

/-- Expected forward `a / b = [3, 2]`. -/
def yExp : Tensor ℚ s2 := tensorOfList! [2] [3.0, 2.0]
/-- Expected `∂L/∂a = dLdy / b = [1/2, 1/4]`. -/
def daExp : Tensor ℚ s2 := tensorOfList! [2] [0.5, 0.25]
/-- Expected `∂L/∂b = −dLdy·a/b² = [−3/2, −1/2]` (built by negating `[3/2, 1/2]`). -/
def dbExp : Tensor ℚ s2 := - tensorOfList! [2] [1.5, 0.5]

/-- Build a tape `y = a / b`, run the backward pass, and check the forward value and both
input gradients against the exact rational references. -/
def checkDiv : Runtime.Autograd.Result Bool := do
  let t0 : Tape ℚ := Tape.empty
  let m : TapeM ℚ _ := do
    let aId ← TapeM.leaf a (name := some "a") (requires_grad := true)
    let bId ← TapeM.leaf b (name := some "b") (requires_grad := true)
    let yId ← TapeM.div (s := s2) aId bId
    let t ← TapeM.getTape
    let yVal ← liftM (Tape.requireValue (α := ℚ) (t := t) (s := s2) yId)
    let grads ← liftM (Tape.backward (t := t) yId (Runtime.Autograd.AnyTensor.mk dLdy))
    pure (aId, bId, yVal, grads)
  let ((aId, bId, yVal, grads), _) ← TapeM.run t0 m
  let da ← Train.requireGradTensor (tag := tag) (s := s2) grads aId
  let db ← Train.requireGradTensor (tag := tag) (s := s2) grads bId
  let okY  := decide (pretty yVal = pretty yExp)
  let okDa := decide (pretty da = pretty daExp)
  let okDb := decide (pretty db = pretty dbExp)
  pure (okY && okDa && okDb)

/-- Entrypoint (called by `NN/Tests/Runtime/Rationals/Suite.lean`). -/
def run : IO Unit := do
  match checkDiv with
  | .ok true => IO.println "elementwise_div_test (Rat): OK"
  | .ok false => throw <| IO.userError "elementwise_div_test (Rat): FAILED"
  | .error msg => throw <| IO.userError s!"elementwise_div_test (Rat): {msg}"

end ElementwiseDiv
end Rationals
end Tests
