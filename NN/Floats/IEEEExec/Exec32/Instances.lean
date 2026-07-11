/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Transcendentals

/-!
# IEEE32 Executable Instances

This file gives `IEEE32Exec` the standard Lean numeric interfaces needed by tensor specs and
examples. The instances route arithmetic through the executable binary32 operations defined in the
`Exec32` hierarchy.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-- Pretty-print using Lean's `Float` printer (via `toFloat`). -/
instance : ToString IEEE32Exec where
  toString x := toString (toFloat x)

/-- Coerce naturals to binary32 by converting through Lean's `Float` and re-encoding. -/
instance : Coe Nat IEEE32Exec where
  coe n := ofFloat (Float.ofNat n)

/--
Numeral literals for `IEEE32Exec`.

This allows writing:
- `(1 : IEEE32Exec)`
- `(42 : IEEE32Exec)`

The interpretation is `Nat → Float (binary64) → IEEE32Exec` via `ofFloat`, i.e. it rounds the exact
integer to the nearest representable float32 (which is exact for small enough integers).
-/
instance (n : Nat) : OfNat IEEE32Exec n :=
  ⟨(n : IEEE32Exec)⟩

/-- `0.0` as an executable binary32 value (chosen as `+0.0`). -/
instance : Zero IEEE32Exec where
  zero := posZero
/-- `1.0` as an executable binary32 value. -/
instance : One IEEE32Exec where
  one := ofBits 0x3F800000

/-- Unary negation (IEEE-754 sign flip, with NaN payload rules). -/
instance : Neg IEEE32Exec where
  neg := neg
/-- IEEE-754 addition (with NaN/Inf rules). -/
instance : Add IEEE32Exec where
  add := add
/-- IEEE-754 subtraction (with NaN/Inf rules). -/
instance : Sub IEEE32Exec where
  sub := sub
/-- IEEE-754 multiplication (with NaN/Inf rules). -/
instance : Mul IEEE32Exec where
  mul := mul
/-- IEEE-754 division (with NaN/Inf rules). -/
instance : Div IEEE32Exec where
  div := div

/--
Exponentiation instance.

This is a *deterministic* executable choice, not a claim about correctly-rounded `pow`:
we implement a small set of IEEE-like special cases and handle integer exponents exactly enough to
avoid the most common footguns (negative bases, `0^0`, `1^∞`, etc.). For general non-integer
exponents we fall back to `exp (b * log a)` on positive bases.
-/
instance : Pow IEEE32Exec IEEE32Exec where
  pow a b :=
    -- `x^0 = 1` even if `x` is NaN/Inf (common `pow` convention; also avoids `∞*0 = NaN`).
    if isZero b then
      (1 : IEEE32Exec)
    else
      match chooseNaN2 a b with
      | some nan => nan
      | none =>
          -- `1^y = 1` for all non-NaN `y` (including `±Inf`).
          if compare a (1 : IEEE32Exec) = some .eq then
            (1 : IEEE32Exec)
          else if compare b (1 : IEEE32Exec) = some .eq then
            a
          else
            let intOfDyadic? (d : Dyadic) : Option Int :=
              if d.mant == 0 then
                some 0
              else
                match d.exp with
                | .ofNat sh =>
                    let n : Nat := Nat.shiftLeft d.mant sh
                    some (if d.sign then -Int.ofNat n else Int.ofNat n)
                | .negSucc sh =>
                    -- integer iff `mant` is divisible by `2^(sh+1)`.
                    let k := sh + 1
                    let denom := pow2 k
                    if d.mant % denom == 0 then
                      let q := d.mant / denom
                      some (if d.sign then -Int.ofNat q else Int.ofNat q)
                    else
                      none

            let intOfIEEE? (x : IEEE32Exec) : Option Int :=
              match toDyadic? x with
              | some dx => intOfDyadic? dx
              | none => none

            let oddInt (n : Int) : Bool :=
              (Int.natAbs n) % 2 == 1

            let rec powNatLinear (a : IEEE32Exec) : Nat → IEEE32Exec
              | 0 => (1 : IEEE32Exec)
              | n + 1 => mul a (powNatLinear a n)

            let powIntLinear (a : IEEE32Exec) (n : Int) : IEEE32Exec :=
              match n with
              | .ofNat k => powNatLinear a k
              | .negSucc k => div (1 : IEEE32Exec) (powNatLinear a (Nat.succ k))

            let smallPowLimit : Nat := 256

            -- Integer exponent fast path: keep this cheap for common small integers, and use
            -- `exp/log` for huge integer exponents (still deterministic, and avoids O(n) loops).
            let powIntDet (a : IEEE32Exec) (n : Int) : IEEE32Exec :=
              if Int.natAbs n ≤ smallPowLimit then
                powIntLinear a n
              else
                exp (b * log (abs a))

            match compare a posZero with
            | some .eq =>
                -- `0^b` for nonzero `b`.
                match intOfIEEE? b with
                | some n =>
                    let odd := oddInt n
                    match compare b posZero with
                    | some .lt =>
                        -- `±0` raised to a negative integer exponent is an infinity; sign depends on
                        -- oddness (mirrors the real-limit behavior).
                        if signBit a && odd then negInf else posInf
                    | some .gt =>
                        if signBit a && odd then negZero else posZero
                    | _ =>
                        posZero
                | none =>
                    match compare b posZero with
                    | some .lt => posInf
                    | _ => posZero
            | some .lt =>
                -- Negative base: only defined for integer exponents over ℝ.
                match intOfIEEE? b with
                | none => canonicalNaN
                | some n =>
                    let mag := powIntDet a n
                    if oddInt n then neg (abs mag) else abs mag
            | _ =>
                -- Positive base.
                match intOfIEEE? b with
                | some n => powIntDet a n
                | none => exp (b * log a)

/-
Equality and order
==================

These instances are kept explicit and simple:

- `BEq` returns `false` if either side is NaN (matching IEEE comparisons being unordered),
- `BEq` treats `+0` and `-0` as equal (use `bits` if you need to distinguish them),
- `<`/`≤` are defined via `compare`; unordered comparisons are `False`.
-/

/--
Boolean equality with IEEE-754 NaN/zero conventions.

- If either side is NaN, we return `false`.
- If both are zeros (either sign), we return `true`.
- Otherwise we compare raw bits.
-/
instance : BEq IEEE32Exec where
  beq a b :=
    if isNaN a || isNaN b then
      false
    else if isZero a && isZero b then
      true
    else
      a.bits == b.bits

/-- Strict order instance, defined via `IEEE32Exec.lt`. -/
instance : LT IEEE32Exec where
  lt := lt
/-- Non-strict order instance, defined via `IEEE32Exec.le`. -/
instance : LE IEEE32Exec where
  le := le

/-- Decidable `<` inherited from the `compare`-based definition. -/
instance : DecidableRel ((· < ·) : IEEE32Exec → IEEE32Exec → Prop) := by
  intro x y
  -- `x < y` is definitionally `IEEE32Exec.lt x y`.
  change Decidable (IEEE32Exec.lt x y)
  dsimp [IEEE32Exec.lt]
  infer_instance

/-- Decidable `≤` inherited from the `compare`-based definition. -/
instance : DecidableRel ((· ≤ ·) : IEEE32Exec → IEEE32Exec → Prop) := by
  intro x y
  -- `x ≤ y` is definitionally `IEEE32Exec.le x y`.
  change Decidable (IEEE32Exec.le x y)
  dsimp [IEEE32Exec.le]
  cases h : IEEE32Exec.compare x y with
  | none =>
      -- `le` returns `False` on unordered (NaN) comparisons.
      exact isFalse (by intro hFalse; cases hFalse)
  | some o =>
      cases o with
      | lt => exact isTrue trivial
      | eq => exact isTrue trivial
      | gt => exact isFalse (by intro hFalse; cases hFalse)

/-- `min` operator, implemented by IEEE-754 `minimum`. -/
instance : Min IEEE32Exec where
  min := minimum
/-- `max` operator, implemented by IEEE-754 `maximum`. -/
instance : Max IEEE32Exec where
  max := maximum

/-- Provide the `MathFunctions` interface using the deterministic implementations in this file. -/
instance : MathFunctions IEEE32Exec where
  exp x := IEEE32Exec.exp x
  tanh x := IEEE32Exec.tanh x
  cosh x := IEEE32Exec.cosh x
  sqrt x := IEEE32Exec.sqrt x
  abs x := IEEE32Exec.abs x
  log x := IEEE32Exec.log x
  pi := ofBits 0x40490FDB
  cos x := IEEE32Exec.cos x
  sin x := IEEE32Exec.sin x
  sinh x := IEEE32Exec.sinh x

/-- Numeric constants used by the spec library, instantiated at binary32. -/
instance : Numbers IEEE32Exec where
  neg_point_five := ofFloat (-0.5)
  neg_one := ofFloat (-1)
  pointone := ofFloat 0.1
  pointfive := ofFloat 0.5
  zero := posZero
  one := ofBits 0x3F800000
  two := ofFloat 2
  three := ofFloat 3
  four := ofFloat 4
  five := ofFloat 5
  ten := ofFloat 10
  log10 := ofFloat (Float.log 10)
  log10000 := ofFloat (Float.log 10000)
  epsilon := ofFloat (1e-6)

/-- `Context` instance so the spec layer can execute with `IEEE32Exec` scalars. -/
instance : Context IEEE32Exec := {
  decidable_gt := fun x y => inferInstanceAs (Decidable (x > y))
}

end IEEE32Exec

end TorchLean.Floats.IEEE754
