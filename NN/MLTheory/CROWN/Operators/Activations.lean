/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor

/-!
# Supplemental Activation Operators

This file defines additional activation functions and transfer rules intended for CROWN/IBP-style
bound propagation:

- Leaky ReLU: `f(x) = max(αx, x)` (typically `0 < α < 1`)
- ELU: `f(x) = x` if `x > 0`, else `α (exp(x) - 1)`
- Softplus: `f(x) = log(1 + exp(x))`
- Mish: `f(x) = x * tanh(softplus(x))`
- SiLU/Swish: `f(x) = x * sigmoid(x)`

## Soundness status

A few helpers in this file use low-order Taylor approximations (`exp_approx`, `log_approx`,
`tanh_approx`, `sigmoid_approx`). These are executable approximations, not theorem-backed
enclosures of the true transcendental functions. Bounds derived from them should only be used under
an explicit checker/oracle assumption, not as part of the unconditional CROWN soundness core.

If you need enclosure-sound bounds for transcendentals, prefer validated enclosure backends (e.g.
Arb-backed intervals in `NN/Floats`) or restrict to the operator subset covered by the main
soundness theorems.

## References

Activations:
- ELU: Clevert, Unterthiner, Hochreiter, "Fast and Accurate Deep Network Learning by Exponential
  Linear Units (ELUs)", ICLR 2016: https://arxiv.org/abs/1511.07289
- Swish/SiLU: Ramachandran, Zoph, Le, "Searching for Activation Functions", 2017:
  https://arxiv.org/abs/1710.05941
- Mish: Misra, "Mish: A Self Regularized Non-Monotonic Activation Function", 2019:
  https://arxiv.org/abs/1908.08681

Bound propagation context:
- CROWN: Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions", 2018: https://arxiv.org/abs/1811.00866
- auto_LiRPA: Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond", NeurIPS 2020: https://arxiv.org/abs/2002.12920
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.Activations

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-! ## Approximation Helpers -/

/-- Low-order Taylor approximation of `exp`. This is executable, not an enclosure theorem. -/
def expApprox (x : α) : α :=
  -- Taylor: exp(x) ≈ 1 + x + x²/2 + x³/6
  let x2 := x * x
  let x3 := x2 * x
  Numbers.one + x + x2 * Numbers.pointfive + x3 / (Numbers.one + Numbers.one + Numbers.one +
    Numbers.one + Numbers.one + Numbers.one)

/-- Low-order approximation of `log(1 + y)` around `y = 0`; not an enclosure theorem. -/
def logApprox (x : α) : α :=
  -- log(1 + y) ≈ y - y²/2 + y³/3 for small y
  let y := x - Numbers.one
  y - y * y * Numbers.pointfive + y * y * y / Numbers.three

/-- Executable `softplus` approximation computed via `logApprox (1 + expApprox x)`. -/
def softplusApprox (x : α) : α :=
  -- softplus(x) ≈ log(1 + exp(x))
  let ex := expApprox x
  logApprox (Numbers.one + ex)

/-- Executable `sigmoid` approximation computed via `expApprox`. -/
def sigmoidApprox (x : α) : α :=
  Numbers.one / (Numbers.one + expApprox (-x))

/-- Low-order approximation of `tanh`; not an enclosure theorem. -/
def tanhApprox (x : α) : α :=
  let x2 := x * x
  let x3 := x2 * x
  x - x3 / Numbers.three

/-! ### Leaky ReLU -/

/-- Leaky ReLU: f(x) = x if x ≥ 0, αx if x < 0. -/
def leakyRelu (negSlope : α) (x : α) : α :=
  if x > Numbers.zero then x else negSlope * x

/-- Minimum of two scalar values using the executable order from the scalar context. -/
def min2 (x y : α) : α :=
  if x < y then x else y

/-- Maximum of two scalar values using the executable order from the scalar context. -/
def max2 (x y : α) : α :=
  if x > y then x else y

/-- IBP for Leaky ReLU on scalars. -/
def ibpLeakyReluScalar (negSlope : α) (l u : α) : α × α :=
  let fl := leakyRelu negSlope l
  let fu := leakyRelu negSlope u
  -- For the usual positive slope, the function is monotone increasing.
  if negSlope > Numbers.zero then
    (fl, fu)
  else
    -- For non-positive negative-branch slopes, a crossing interval has a kink value at zero.
    -- Endpoint-only min/max is unsound, e.g. α=-1 on [-1,2] has minimum 0 at the kink.
    if (!(l > Numbers.zero)) && (!(u < Numbers.zero)) then
      (min2 (min2 fl fu) Numbers.zero, max2 (max2 fl fu) Numbers.zero)
    else
      (min2 fl fu, max2 fl fu)

/-- IBP for Leaky ReLU on boxes. -/
def ibpLeakyRelu (n : Nat) (negSlope : α) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (bl, _) := ibpLeakyReluScalar negSlope l u
        Tensor.scalar bl)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (_, bh) := ibpLeakyReluScalar negSlope l u
        Tensor.scalar bh)
    { lo := outLo, hi := outHi }

/-- Affine bounds for Leaky ReLU.
    Unlike ReLU, both branches are linear so relaxation is easier. -/
def affLeakyRelu (negSlope : α) (l u : α) : α × α × α × α :=
  if l > Numbers.zero then
    -- Positive region: y = x
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  else if u < Numbers.zero then
    -- Negative region: y = αx
    (negSlope, Numbers.zero, negSlope, Numbers.zero)
  else
    if !(u > l) then
      -- The only valid degenerate crossing interval is `[0,0]`; use its exact zero map and avoid
      -- the secant's `0/0` denominator.
      (Numbers.zero, Numbers.zero, Numbers.zero, Numbers.zero)
    else
      -- The secant joins `(l, αl)` to `(u, u)`. For α ≤ 1 the kink is convex, so the
      -- negative-branch line is a lower support and the secant is upper. For α > 1 the kink is
      -- concave and those roles reverse.
      let secantSlope := (u - negSlope * l) / (u - l)
      let secantBias := u * (Numbers.one - secantSlope)
      if negSlope > Numbers.one then
        (secantSlope, secantBias, negSlope, Numbers.zero)
      else
        (negSlope, Numbers.zero, secantSlope, secantBias)

/-- Derivative bounds for Leaky ReLU (for Lyapunov). -/
def derivLeakyRelu (negSlope : α) (l u : α) : α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.one)
  else if u < Numbers.zero then
    (negSlope, negSlope)
  else
    -- Crossing: derivative is either α or 1
    (if negSlope < Numbers.one then negSlope else Numbers.one,
     if negSlope > Numbers.one then negSlope else Numbers.one)

/-! ### ELU (Exponential Linear Unit) -/

/-- Executable ELU approximation using `expApprox` on the negative branch. -/
def eluApprox (scale : α) (x : α) : α :=
  if x > Numbers.zero then x
  else scale * (expApprox x - Numbers.one)

/-- Approximate scalar interval rule for ELU over an input interval `[l,u]`. -/
def ibpEluApproxScalar (scale : α) (l u : α) : α × α :=
  if l > Numbers.zero then
    -- Pure positive region
    (l, u)
  else if u < Numbers.zero then
    -- Pure negative region: α(exp(x) - 1), monotone
    (scale * (expApprox l - Numbers.one), scale * (expApprox u - Numbers.one))
  else
    -- Crossing: min at most negative, max at most positive
    let negMin := scale * (expApprox l - Numbers.one)
    (negMin, u)

/-- Approximate interval propagation for ELU on a vector box. -/
def ibpEluApprox (n : Nat) (scale : α) (B : Box α (.dim n .scalar)) :
    Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpEluApproxScalar scale l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpEluApproxScalar scale l u).2)
    { lo := outLo, hi := outHi }

/-- Approximate affine ELU candidates; no enclosure theorem is claimed. -/
def affEluApprox (scale : α) (l u : α) : α × α × α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  else if u < Numbers.zero then
    -- Negative region: use secant for bounds
    let fl := scale * (expApprox l - Numbers.one)
    let fu := scale * (expApprox u - Numbers.one)
    let slope := (fu - fl) / (u - l)
    (slope, fl - slope * l, slope, fl - slope * l)
  else
    -- Crossing region
    let fl := scale * (expApprox l - Numbers.one)
    -- Upper: linear for positive, secant for negative part
    let slope_upper := (u - fl) / (u - l)
    let bias_upper := u - slope_upper * u
    -- Lower: conservative
    let slope_lower := scale * expApprox l  -- derivative at l
    (slope_lower, fl - slope_lower * l, slope_upper, bias_upper)

/-- Approximate ELU derivative range computed with `expApprox`. -/
def derivEluApprox (scale : α) (l u : α) : α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.one)
  else if u < Numbers.zero then
    -- Derivative: α·exp(x), monotone increasing
    (scale * expApprox l, scale * expApprox u)
  else
    -- Crossing: min derivative in negative region, max is 1
    (scale * expApprox l, Numbers.one)

/-! ### Softplus -/

/-- Approximate IBP for Softplus. Softplus is monotone increasing for the intended function. -/
def ibpSoftplusApproxScalar (l u : α) : α × α :=
  (softplusApprox l, softplusApprox u)

/-- Approximate IBP for Softplus on boxes. -/
def ibpSoftplusApprox (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l => Tensor.scalar (softplusApprox l))
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u => Tensor.scalar (softplusApprox u))
    { lo := outLo, hi := outHi }

/-! ### SiLU/Swish -/

/-- Executable SiLU approximation using `sigmoidApprox`. -/
def siluApprox (x : α) : α :=
  x * sigmoidApprox x

/-- Approximate IBP for SiLU. Not monotone, has global minimum around x ≈ -1.28. -/
def ibpSiluApproxScalar (l u : α) : α × α :=
  let fl := siluApprox l
  let fu := siluApprox u
  -- SiLU minimum is at x ≈ -1.28, value ≈ -0.28
  let critX := -(Numbers.one + Numbers.pointfive * Numbers.pointfive)  -- approx -1.25
  let critVal := critX * sigmoidApprox critX
  if l > critX then
    -- Right of minimum, monotone increasing
    (fl, fu)
  else if u < critX then
    -- Left of minimum, monotone decreasing
    (fu, fl)
  else
    -- Spans minimum
    (critVal, if fl > fu then fl else fu)

/-- Apply the approximate scalar SiLU interval rule coordinatewise to a vector box. -/
def ibpSiluApprox (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSiluApproxScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSiluApproxScalar l u).2)
    { lo := outLo, hi := outHi }

/-! ### Mish -/

/-- Executable Mish approximation using the local tanh and softplus approximations. -/
def mishApprox (x : α) : α :=
  x * tanhApprox (softplusApprox x)

/-- Approximate IBP for Mish. Similar behavior to SiLU. -/
def ibpMishApproxScalar (l u : α) : α × α :=
  let fl := mishApprox l
  let fu := mishApprox u
  -- Mish minimum is around x ≈ -1.22
  let critX := -(Numbers.one + Numbers.pointfive * Numbers.pointfive)
  let critVal := mishApprox critX
  if l > critX then
    (fl, fu)
  else if u < critX then
    (fu, fl)
  else
    (critVal, if fl > fu then fl else fu)

/-- Apply the approximate scalar Mish interval rule coordinatewise to a vector box. -/
def ibpMishApprox (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpMishApproxScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpMishApproxScalar l u).2)
    { lo := outLo, hi := outHi }

end NN.MLTheory.CROWN.Operators.Activations
