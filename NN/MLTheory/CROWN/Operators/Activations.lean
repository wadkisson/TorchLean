/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core

/-!
# Leaky-ReLU Bounds

Interval, affine, and derivative transfer rules for
`leakyRelu negSlope x = if x > 0 then x else negSlope * x`.

The formulas cover positive, zero, and negative branch slopes. In particular, when a non-positive
slope crosses zero, the interval rule includes the value at the kink instead of considering only
the two endpoints.

Reference: Zhang et al., "Efficient Neural Network Robustness Certification with General
Activation Functions", NeurIPS 2018, arXiv:1811.00866.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators.Activations

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Leaky ReLU with slope `negSlope` on the non-positive branch. -/
def leakyRelu (negSlope : α) (x : α) : α :=
  if x > Numbers.zero then x else negSlope * x

/-- Minimum of two values using the executable scalar order. -/
def min2 (x y : α) : α :=
  if x < y then x else y

/-- Maximum of two values using the executable scalar order. -/
def max2 (x y : α) : α :=
  if x > y then x else y

/-- Exact endpoint-and-kink interval propagation for scalar Leaky ReLU. -/
def ibpLeakyReluScalar (negSlope : α) (lo hi : α) : α × α :=
  let flo := leakyRelu negSlope lo
  let fhi := leakyRelu negSlope hi
  if negSlope > Numbers.zero then
    (flo, fhi)
  else if (!(lo > Numbers.zero)) && (!(hi < Numbers.zero)) then
    (min2 (min2 flo fhi) Numbers.zero, max2 (max2 flo fhi) Numbers.zero)
  else
    (min2 flo fhi, max2 flo fhi)

/-- Apply `ibpLeakyReluScalar` coordinatewise to a vector box. -/
def ibpLeakyRelu (n : Nat) (negSlope : α) (box : Box α (.dim n .scalar)) :
    Box α (.dim n .scalar) :=
  match box.lo, box.hi with
  | .dim lo, .dim hi =>
    { lo := Tensor.dim fun i =>
        match lo i, hi i with
        | .scalar l, .scalar u => Tensor.scalar (ibpLeakyReluScalar negSlope l u).1
      hi := Tensor.dim fun i =>
        match lo i, hi i with
        | .scalar l, .scalar u => Tensor.scalar (ibpLeakyReluScalar negSlope l u).2 }

/--
Lower and upper affine forms for Leaky ReLU on `[lo, hi]`, returned as
`(lowerSlope, lowerBias, upperSlope, upperBias)`.

For a crossing interval the function is convex when `negSlope ≤ 1` and concave when
`negSlope > 1`; the secant and branch support exchange roles accordingly.
-/
def affLeakyRelu (negSlope : α) (lo hi : α) : α × α × α × α :=
  if lo > Numbers.zero then
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  else if hi < Numbers.zero then
    (negSlope, Numbers.zero, negSlope, Numbers.zero)
  else if !(hi > lo) then
    (Numbers.zero, Numbers.zero, Numbers.zero, Numbers.zero)
  else
    let secantSlope := (hi - negSlope * lo) / (hi - lo)
    let secantBias := hi * (Numbers.one - secantSlope)
    if negSlope > Numbers.one then
      (secantSlope, secantBias, negSlope, Numbers.zero)
    else
      (negSlope, Numbers.zero, secantSlope, secantBias)

/-- Range of the two branch derivatives over an interval. -/
def derivLeakyRelu (negSlope : α) (lo hi : α) : α × α :=
  if lo > Numbers.zero then
    (Numbers.one, Numbers.one)
  else if hi < Numbers.zero then
    (negSlope, negSlope)
  else
    (if negSlope < Numbers.one then negSlope else Numbers.one,
     if negSlope > Numbers.one then negSlope else Numbers.one)

end NN.MLTheory.CROWN.Operators.Activations
