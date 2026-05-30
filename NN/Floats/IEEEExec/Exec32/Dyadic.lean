/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Core

/-!
Dyadic helpers for executable IEEE32 arithmetic.

This file contains the integer/rational scaffolding used to describe binary significands,
exponents, and exact dyadic values before rounding to IEEE32.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-- Exact dyadic value `(-1)^sign * mant * 2^exp` used as an intermediate for finite ops. -/
structure Dyadic where
  /-- sign. -/
  sign : Bool
  /-- mant. -/
  mant : Nat
  /-- exp. -/
  exp : Int
  deriving Repr, DecidableEq

/-- `2^k` as a natural number. -/
def pow2 (k : Nat) : Nat :=
  Nat.shiftLeft 1 k

/--
Round `n / 2^shift` to nearest, ties-to-even.

This is the primitive "shift + rounding" operation we use when shrinking a mantissa back down to a
fixed bit width. It is the same tie-breaking policy as IEEE round-to-nearest-even.
-/
def roundShiftRightEven (n : Nat) (shift : Nat) : Nat :=
  if shift == 0 then
    n
  else
    let q := Nat.shiftRight n shift
    let rem := n - Nat.shiftLeft q shift
    let half := pow2 (shift - 1)
    if rem < half then q
    else if rem > half then q + 1
    else if q % 2 == 0 then q else q + 1

/--
Construct a raw binary32 bit-pattern from fields.

`mkBits sign exp frac` places:
- `sign` in bit 31,
- `exp` in bits 30..23 (8 bits),
- `frac` in bits 22..0 (masked to 23 bits).
-/
def mkBits (sign : Bool) (exp : Nat) (frac : Nat) : UInt32 :=
  let s : UInt32 := if sign then (1 : UInt32) <<< 31 else 0
  let e : UInt32 := (UInt32.ofNat exp) <<< 23
  let f : UInt32 := (UInt32.ofNat frac) &&& fracMask
  s ||| e ||| f

/--
Decode a finite binary32 into an exact dyadic value.

Returns `none` for NaN/Inf.
-/
def toDyadic? (x : IEEE32Exec) : Option Dyadic :=
  if isNaN x || isInf x then
    none
  else
    let s := signBit x
    let e := expField x
    let f := fracField x
    if e == 0 then
      if f == 0 then
        some { sign := s, mant := 0, exp := 0 }
      else
        -- subnormal: value = frac * 2^-149
        some { sign := s, mant := f.toNat, exp := -149 }
    else
      -- normal: value = (2^23 + frac) * 2^(e-bias-23) = (2^23+frac) * 2^(e - 150)
      let mant := (pow2 23) + f.toNat
      let exp : Int := (Int.ofNat e.toNat) - 150
      some { sign := s, mant := mant, exp := exp }

/--
If `toDyadic? x = some d` then `x` is not a NaN.

Informal: `toDyadic?` only returns `some _` for finite floats; NaNs map to `none`.
-/
lemma isNaN_eq_false_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isNaN x = false := by
  cases hnan : isNaN x
  · rfl
  · -- `isNaN x = true` forces `toDyadic? x = none`.
    unfold toDyadic? at hx
    have hcond : (isNaN x || isInf x) = true := by
      simp [hnan]
    have : (none : Option Dyadic) = some d := by
      simp [hcond] at hx
    cases this

/--
If `toDyadic? x = some d` then `x` is not an infinity.

Informal: `toDyadic?` only returns `some _` for finite floats; infinities map to `none`.
-/
lemma isInf_eq_false_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isInf x = false := by
  cases hinf : isInf x
  · rfl
  · -- `isInf x = true` forces `toDyadic? x = none`.
    unfold toDyadic? at hx
    have hcond : (isNaN x || isInf x) = true := by
      simp [hinf]
    have : (none : Option Dyadic) = some d := by
      simp [hcond] at hx
    cases this

/-!
## Rounding back to binary32

The general pattern for the finite ops in this file is:

1. decode float32(s) to an exact intermediate representation (`Dyadic` for `+ - * fma sqrt`,
   rationals for `/`),
2. compute the exact result in that intermediate representation,
3. round once to float32 using round-to-nearest, ties-to-even.
-/

/--
Round an exact dyadic value to binary32 (ties-to-even).

This function implements:

- overflow to ±Inf,
- gradual underflow into subnormals (down to exponent `-149`),
- underflow-to-zero below `2^-150` (half the minimum subnormal, where ties-to-even chooses 0),
- mantissa rounding to the 24-bit precision of binary32 normal numbers.
-/
def roundDyadicToIEEE32 (d : Dyadic) : IEEE32Exec :=
  -- Exact 0 becomes signed 0.
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else
    let log2m : Nat := Nat.log2 d.mant
    let k : Int := (Int.ofNat log2m) + d.exp
    -- IEEE754 binary32 exponent range (unbiased): normal [-126,127], subnormal down to -149.
    if k > 127 then
      if d.sign then negInf else posInf
    -- Underflow-to-zero threshold is at half the smallest subnormal: 2^-150 (ties-to-even pick 0).
    -- In terms of `k = ⌊log₂ |x|⌋`, all values with `k < -150` round to 0.
    else if k < -150 then
      if d.sign then negZero else posZero
    else if k < -126 then
      -- subnormal rounding: frac = round_to_even( mant * 2^(exp+149) )
      let fracNat : Nat :=
        match d.exp + 149 with
        | .ofNat sh => Nat.shiftLeft d.mant sh
        | .negSucc sh => roundShiftRightEven d.mant (sh + 1)
      if fracNat == 0 then
        if d.sign then negZero else posZero
      else
        match Nat.decLe (pow2 23) fracNat with
        | isTrue _ =>
            -- Rounds up to the smallest normal: exp=1, frac=0.
            ofBits (mkBits d.sign 1 0)
        | isFalse _ =>
            ofBits (mkBits d.sign 0 fracNat)
    else
      -- normal rounding
      let m24 : Nat :=
        if log2m >= 23 then
          roundShiftRightEven d.mant (log2m - 23)
        else
          Nat.shiftLeft d.mant (23 - log2m)
      let k' : Int := if m24 == pow2 24 then k + 1 else k
      let m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
      if k' > 127 then
        if d.sign then negInf else posInf
      else
        let expNat : Nat := Int.toNat (k' + 127)
        let fracNat : Nat := m24' - pow2 23
        ofBits (mkBits d.sign expNat fracNat)

/-!
## Exact dyadic arithmetic (finite core)

`Dyadic` is closed under `+` and `*` (at the exact level). We use these helpers before rounding
back to float32.
-/

/--
Exact dyadic addition.

We align exponents by shifting the mantissa of the operand with the larger exponent, add signed
integers, and then return an exact dyadic (no rounding yet).
-/
def addDyadic (a b : Dyadic) : Dyadic :=
  if a.exp ≤ b.exp then
    let sh : Nat := Int.toNat (b.exp - a.exp)
    let m1 : Int := if a.sign then -(Int.ofNat a.mant) else (Int.ofNat a.mant)
    let m2s : Nat := Nat.shiftLeft b.mant sh
    let m2 : Int := if b.sign then -(Int.ofNat m2s) else (Int.ofNat m2s)
    let s : Int := m1 + m2
    if s == 0 then
      { sign := a.sign && b.sign, mant := 0, exp := 0 }
    else
      { sign := s < 0, mant := Int.natAbs s, exp := a.exp }
  else
    let sh : Nat := Int.toNat (a.exp - b.exp)
    let m1s : Nat := Nat.shiftLeft a.mant sh
    let m1 : Int := if a.sign then -(Int.ofNat m1s) else (Int.ofNat m1s)
    let m2 : Int := if b.sign then -(Int.ofNat b.mant) else (Int.ofNat b.mant)
    let s : Int := m1 + m2
    if s == 0 then
      { sign := a.sign && b.sign, mant := 0, exp := 0 }
    else
      { sign := s < 0, mant := Int.natAbs s, exp := b.exp }

/-!
## Exact rationals for division

For division we compute an exact rational `num/den` (with `den > 0`) and then round it to binary32
using round-to-nearest, ties-to-even.
-/

/-- Round `num/den` to nearest, ties-to-even (assumes `den > 0`). -/
def roundQuotEven (num den : Nat) : Nat :=
  let q := num / den
  let r := num % den
  let twice := 2 * r
  if twice < den then q
  else if twice > den then q + 1
  else if q % 2 == 0 then q else q + 1

/-- Test whether `num/den < 2^k` without converting to reals. -/
def ratLtPow2 (num den : Nat) (k : Int) : Bool :=
  match k with
  | .ofNat kn => num < Nat.shiftLeft den kn
  | .negSucc kn => Nat.shiftLeft num (kn + 1) < den

/-- Test whether `num/den ≥ 2^k` without converting to reals. -/
def ratGePow2 (num den : Nat) (k : Int) : Bool :=
  match k with
  | .ofNat kn => num ≥ Nat.shiftLeft den kn
  | .negSucc kn => Nat.shiftLeft num (kn + 1) ≥ den

/--
Compute `⌊log₂(num/den)⌋` as an `Int` (assumes `num > 0` and `den > 0`).

We start from the initial estimate `log2(num) - log2(den)` and then adjust by checking against
powers of two.
-/
def floorLog2Rat (num den : Nat) : Int :=
  -- num > 0, den > 0
  let k0 : Int := (Int.ofNat (Nat.log2 num)) - (Int.ofNat (Nat.log2 den))
  let k1 : Int := if ratLtPow2 num den k0 then k0 - 1 else k0
  if ratGePow2 num den (k1 + 1) then k1 + 1 else k1

/--
Round an exact rational `num/den` to binary32 (ties-to-even).

This is the division analogue of `roundDyadicToIEEE32`: it uses the same exponent thresholds and
the same final mantissa rounding policy.
-/
def roundRatToIEEE32 (sign : Bool) (num den : Nat) : IEEE32Exec :=
  if num == 0 then
    if sign then negZero else posZero
  else
    let k : Int := floorLog2Rat num den
    if k > 127 then
      if sign then negInf else posInf
    -- Same underflow threshold as `roundDyadicToIEEE32`.
    else if k < -150 then
      if sign then negZero else posZero
    else if k < -126 then
      -- subnormal: frac = round_to_even( (num/den) * 2^149 )
      let num' := Nat.shiftLeft num 149
      let frac := roundQuotEven num' den
      if frac == 0 then
        if sign then negZero else posZero
      else
        match Nat.decLe (pow2 23) frac with
        | isTrue _ => ofBits (mkBits sign 1 0)
        | isFalse _ => ofBits (mkBits sign 0 frac)
    else
      -- normal: m = round_to_even( (num/den) * 2^(23-k) )
      let shift : Int := 23 - k
      let (num', den') :=
        match shift with
        | .ofNat sh => (Nat.shiftLeft num sh, den)
        | .negSucc sh => (num, Nat.shiftLeft den (sh + 1))
      let m := roundQuotEven num' den'
      let k' : Int := if m == pow2 24 then k + 1 else k
      let m' : Nat := if m == pow2 24 then pow2 23 else m
      if k' > 127 then
        if sign then negInf else posInf
      else
        let expNat : Nat := Int.toNat (k' + 127)
        let fracNat : Nat := m' - pow2 23
        ofBits (mkBits sign expNat fracNat)


end IEEE32Exec

end TorchLean.Floats.IEEE754
