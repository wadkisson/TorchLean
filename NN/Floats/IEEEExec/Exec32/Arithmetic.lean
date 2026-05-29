/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Dyadic

/-!
# IEEE32 Executable Arithmetic

This file defines executable binary32 arithmetic for the core operations: addition, subtraction,
multiplication, division, square root, and fused multiply-add. The implementation follows IEEE-style
case splits for NaN/Inf/zero and delegates finite rounding to the dyadic rounding kernel.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-- IEEE754 addition (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
def add (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then
          if signBit x == signBit y then x else canonicalNaN
        else
          x
      else if isInf y then
        y
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy => roundDyadicToIEEE32 (addDyadic dx dy)
        | _, _ => canonicalNaN

/-- IEEE754 subtraction (defined as addition with sign-flip). -/
@[inline] def sub (x y : IEEE32Exec) : IEEE32Exec :=
  add x (neg y)

/-- IEEE754 multiplication (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
def mul (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isZero y then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        if isZero x then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            let s := Bool.xor dx.sign dy.sign
            if dx.mant == 0 || dy.mant == 0 then
              if s then negZero else posZero
            else
              roundDyadicToIEEE32 { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
        | _, _ => canonicalNaN

/-- IEEE754 division (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
def div (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then canonicalNaN
        else
          -- ±Inf / finite (including ±0) = ±Inf
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        -- finite / ±Inf = signed zero
        if signBit x != signBit y then negZero else posZero
      else if isZero y then
        if isZero x then canonicalNaN
        else
          -- finite nonzero / ±0 = ±Inf
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            -- Exact quotient: (mx * 2^ex) / (my * 2^ey) = (mx/my) * 2^(ex-ey).
            let sign := Bool.xor dx.sign dy.sign
            if dx.mant == 0 then
              if sign then negZero else posZero
            else
              let eDiff : Int := dx.exp - dy.exp
              let (num, den) :=
                match eDiff with
                | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
                | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
              roundRatToIEEE32 sign num den
        | _, _ => canonicalNaN

/-- IEEE754 fused multiply-add: compute `x*y+z` and round once (ties-to-even). -/
def fma (x y z : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN3 x y z with
  | some nan => nan
  | none =>
      if isInf x || isInf y then
        if isZero x || isZero y then
          canonicalNaN
        else
          let prodSign := Bool.xor (signBit x) (signBit y)
          let prodInf := if prodSign then negInf else posInf
          if isInf z then
            if signBit z != prodSign then canonicalNaN else prodInf
          else
            prodInf
      else if isInf z then
        z
      else
        match toDyadic? x, toDyadic? y, toDyadic? z with
        | some dx, some dy, some dz =>
            let prod : Dyadic :=
              { sign := Bool.xor dx.sign dy.sign
                mant := dx.mant * dy.mant
                exp := dx.exp + dy.exp }
            roundDyadicToIEEE32 (addDyadic prod dz)
        | _, _, _ => canonicalNaN

/--
If both inputs decode to dyadics, `add` is “exact dyadic add, then round once”.

Informal: for finite `x,y`, we compute the exact dyadic value `dx + dy` and apply IEEE
round-to-nearest-even.
-/
theorem add_eq_roundDyadicToIEEE32_of_toDyadic? {x y : IEEE32Exec} {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    add x y = roundDyadicToIEEE32 (addDyadic dx dy) := by
  unfold add
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  simp [hchoose, hxInf, hyInf, hx, hy]

/--
If both inputs decode to dyadics, `mul` is “exact dyadic multiply, then round once”.

Informal: for finite `x,y`, the exact product is a dyadic with mantissa `dx.mant * dy.mant` and
exponent `dx.exp + dy.exp`, and we round that back to binary32.
-/
theorem mul_eq_roundDyadicToIEEE32_of_toDyadic? {x y : IEEE32Exec} {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    mul x y =
      roundDyadicToIEEE32
        { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp } :=
          by
  unfold mul
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  simp (config := { zeta := true }) [hchoose, hxInf, hyInf, hx, hy]
  -- Remaining goal: the explicit zero short-circuit agrees with rounding a dyadic with mantissa 0.
  cases h0 : (dx.mant == 0 || dy.mant == 0)
  · -- Nonzero mantissas: the implication premise is impossible.
    intro h
    have hboth : (dx.mant == 0) = false ∧ (dy.mant == 0) = false :=
      (Bool.or_eq_false_iff (x := dx.mant == 0) (y := dy.mant == 0)).1 h0
    cases h with
    | inl hx0 =>
        have hx0b : (dx.mant == 0) = true := (beq_iff_eq).2 hx0
        have : false = true := by
          simp [hboth.1] at hx0b
        cases this
    | inr hy0 =>
        have hy0b : (dy.mant == 0) = true := (beq_iff_eq).2 hy0
        have : false = true := by
          simp [hboth.2] at hy0b
        cases this
  · have h0' : (dx.mant == 0 || dy.mant == 0) = true := by simpa using h0
    have hor : (dx.mant == 0) = true ∨ (dy.mant == 0) = true := by
      have h := h0'
      rw [Bool.or_eq_true (a := dx.mant == 0) (b := dy.mant == 0)] at h
      exact h
    have hprod : ((dx.mant * dy.mant) == 0) = true := by
      cases hor with
      | inl hx0 =>
          have hx0' : dx.mant = 0 := (beq_iff_eq).1 hx0
          simp [hx0']
      | inr hy0 =>
          have hy0' : dy.mant = 0 := (beq_iff_eq).1 hy0
          simp [hy0']
    simp [roundDyadicToIEEE32, hprod]

/--
If all inputs decode to dyadics, `fma x y z` is “exact dyadic `(x*y) + z`, then round once”.

Informal: for finite inputs, we compute the exact dyadic product `dx*dy`, add `dz`, and finally
apply IEEE round-to-nearest-even.
-/
theorem fma_eq_roundDyadicToIEEE32_of_toDyadic? {x y z : IEEE32Exec} {dx dy dz : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hz : toDyadic? z = some dz) :
    fma x y z =
      roundDyadicToIEEE32
        (addDyadic
          { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
            dz) := by
  unfold fma
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hzNaN : isNaN z = false := isNaN_eq_false_of_toDyadic?_some (hx := hz)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hzInf : isInf z = false := isInf_eq_false_of_toDyadic?_some (hx := hz)
  have hchoose : chooseNaN3 x y z = none := by
    simp [chooseNaN3, isSNaN, hxNaN, hyNaN, hzNaN]
  simp (config := { zeta := true }) [hchoose, hxInf, hyInf, hzInf, hx, hy, hz]

/--
If `x` and `y` decode to dyadics and the denominator mantissa is nonzero, `div x y` is obtained by
forming the exact rational quotient and rounding once.

Informal: for finite nonzero `y`, we compute the exact value `(dx.mant * 2^dx.exp) / (dy.mant *
  2^dy.exp)`
as a rational `num/den` with an exponent adjustment, then apply IEEE round-to-nearest-even.
-/
theorem div_eq_roundRatToIEEE32_of_toDyadic? {x y : IEEE32Exec} {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hy0 : dy.mant ≠ 0) :
    div x y =
      let sign : Bool := Bool.xor dx.sign dy.sign
      let eDiff : Int := dx.exp - dy.exp
      let (num, den) :=
        match eDiff with
        | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
        | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
      roundRatToIEEE32 sign num den := by
  classical
  unfold div
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]

  have hyZero : isZero y = false := by
    cases hzy : isZero y with
    | false => rfl
    | true =>
        unfold isZero at hzy
        have hfields : (expField y == 0) = true ∧ (fracField y == 0) = true := by
          simpa [Bool.and_eq_true] using hzy
        have hdy :
            { sign := signBit y, mant := 0, exp := 0 } = dy := by
          -- In the `isZero` case, `toDyadic?` returns the canonical dyadic `0`.
          unfold toDyadic? at hy
          have hnaninf : (isNaN y || isInf y) = false := by
            simp [hyNaN, hyInf]
          -- Reduce the nested `if` using the extracted bitfield facts.
          simp (config := { zeta := true }) [hnaninf, hfields.1, hfields.2] at hy
          simpa using hy
        have : dy.mant = 0 := by simp [hdy.symm]
        exact (hy0 this).elim

  -- Reduce to the dyadic branch (finite, nonzero divisor).
  simp (config := { zeta := true }) [hchoose, hxInf, hyInf, hyZero, hx, hy]
  -- The explicit `dx.mant == 0` short-circuit agrees with `roundRatToIEEE32`'s `num == 0` branch.
  cases hx0 : (dx.mant == 0) with
  | true =>
      have hx0' : dx.mant = 0 := (beq_iff_eq).1 hx0
      cases hE : dx.exp - dy.exp <;> simp [hx0', roundRatToIEEE32]
  | false =>
      intro h
      have hx0' : dx.mant ≠ 0 := (beq_eq_false_iff_ne (a := dx.mant) (b := 0)).1 hx0
      exact (hx0' h).elim

/-- IEEE754 square root (ties-to-even). -/
def sqrt (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        if signBit x then canonicalNaN else posInf
      else if isZero x then
        -- sqrt(±0) = ±0
        x
      else if signBit x then
        canonicalNaN
      else
        match toDyadic? x with
        | none => canonicalNaN
        | some d =>
            -- For any finite binary32 input, sqrt is finite and *normal* (no subnormal outputs).
            let expOdd : Bool := (d.exp % 2) != 0
            let mant' : Nat := if expOdd then d.mant * 2 else d.mant
            let expEven : Int := if expOdd then d.exp - 1 else d.exp
            let expHalf : Int := expEven / 2
            let l : Nat := Nat.log2 mant'
            let t : Nat := l / 2
            let p : Nat := 23 - t
            let n : Nat := Nat.shiftLeft mant' (2 * p)
            let q : Nat := Nat.sqrt n
            let r : Nat := n - q * q
            -- Round `sqrt(n)` to the nearest integer.
            --
            -- Write `q = ⌊sqrt(n)⌋` and `r = n - q^2`. The midpoint between `q` and `q+1` is
            -- `q + 1/2`, and for `n : Nat` the comparison reduces to a simple integer test:
            -- we round up iff `r > q`.
            let m0 : Nat :=
              if r > q then q + 1 else q
            let k0 : Int := expHalf + Int.ofNat t
            let k : Int := if m0 == pow2 24 then k0 + 1 else k0
            let m24 : Nat := if m0 == pow2 24 then pow2 23 else m0
            let expNat : Nat := Int.toNat (k + 127)
            let fracNat : Nat := m24 - pow2 23
            ofBits (mkBits false expNat fracNat)

/--
Compare two exact dyadics by exact integer comparison.

This is used internally by executable rounding and special-case logic. The implementation
normalizes the exponents to a common minimum exponent and compares the scaled integer mantissas.
-/
def cmpDyadic (a b : Dyadic) : Ordering :=
  if a.mant == 0 && b.mant == 0 then
    .eq
  else
    let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
    let shA : Nat := Int.toNat (a.exp - e)
    let shB : Nat := Int.toNat (b.exp - e)
    let aNat : Nat := Nat.shiftLeft a.mant shA
    let bNat : Nat := Nat.shiftLeft b.mant shB
    let aInt : Int := if a.sign then -(Int.ofNat aNat) else Int.ofNat aNat
    let bInt : Int := if b.sign then -(Int.ofNat bNat) else Int.ofNat bNat
    compare aInt bInt

/-
Outward-rounded arithmetic (interval-friendly)
=============================================

These ops are meant for *sound enclosures*, not to exactly model hardware rounding-mode flags.
They compute the exact dyadic result and then apply **directed rounding** to float32:

- `roundDyadicDown` rounds toward `-∞` (a lower bound),
- `roundDyadicUp` rounds toward `+∞` (an upper bound).

This matches the way interval arithmetic packages (e.g. INTLAB / IEEE 1788 workflows) implement
outward-rounded endpoints: instead of relying on host rounding modes, we implement the directed
rounding logic explicitly on the binary32 grid.
-/


end IEEE32Exec

end TorchLean.Floats.IEEE754
