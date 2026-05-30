/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Arithmetic

/-!
Directed executable IEEE32 operations.

This file provides lower and upper rounding variants for arithmetic operations, forming the
runtime side of the interval-enclosure story.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-- `ceil(n / 2^shift)` for naturals, implemented via shifts (used for directed rounding). -/
def shiftRightCeilPow2 (n shift : Nat) : Nat :=
  if shift == 0 then
    n
  else
    let q := Nat.shiftRight n shift
    let rem := n - Nat.shiftLeft q shift
    if rem == 0 then q else q + 1

/-- Directed rounding down (toward `-∞`) for a *positive* dyadic `mant * 2^exp`. -/
def roundDyadicPosDown (mant : Nat) (exp : Int) : IEEE32Exec :=
  -- `mant > 0` by construction at call sites.
  let log2m : Nat := Nat.log2 mant
  let k : Int := (Int.ofNat log2m) + exp
  if k > 127 then
    posMaxFinite
  else if k < -149 then
    posZero
  else if k < -126 then
    -- subnormal: value = frac * 2^-149, so frac = floor(mant * 2^(exp+149))
    let fracNat : Nat :=
      match exp + 149 with
      | .ofNat sh => Nat.shiftLeft mant sh
      | .negSucc sh => Nat.shiftRight mant (sh + 1)
    -- `mkBits` masks the fraction to 23 bits; we reduce explicitly to make proofs easier.
    if fracNat == 0 then posZero else ofBits (mkBits false 0 (fracNat % pow2 23))
  else
    -- normal: m24 = floor(mant * 2^(23 - log2m))
    let m24 : Nat :=
      if log2m >= 23 then
        Nat.shiftRight mant (log2m - 23)
      else
        Nat.shiftLeft mant (23 - log2m)
    let expNat : Nat := Int.toNat (k + 127)
    -- `mkBits` masks the fraction to 23 bits; we reduce explicitly to make proofs easier.
    let fracNat : Nat := (m24 - pow2 23) % pow2 23
    ofBits (mkBits false expNat fracNat)

/-- Directed rounding up (toward `+∞`) for a *positive* dyadic `mant * 2^exp`. -/
def roundDyadicPosUp (mant : Nat) (exp : Int) : IEEE32Exec :=
  -- `mant > 0` by construction at call sites.
  let log2m : Nat := Nat.log2 mant
  let k : Int := (Int.ofNat log2m) + exp
  if k > 127 then
    posInf
  else if k < -149 then
    posMinSubnormal
  else if k < -126 then
    -- subnormal: frac = ceil(mant * 2^(exp+149))
    let fracNat : Nat :=
      match exp + 149 with
      | .ofNat sh => Nat.shiftLeft mant sh
      | .negSucc sh => shiftRightCeilPow2 mant (sh + 1)
    if fracNat == 0 then
      posMinSubnormal
    else
      match Nat.decLe (pow2 23) fracNat with
      | isTrue _ =>
          -- rounds up to the smallest normal: exp=1, frac=0
          ofBits (mkBits false 1 0)
      | isFalse _ =>
          ofBits (mkBits false 0 fracNat)
  else
    -- normal: m24 = ceil(mant * 2^(23 - log2m))
    let m24 : Nat :=
      if log2m >= 23 then
        shiftRightCeilPow2 mant (log2m - 23)
      else
        Nat.shiftLeft mant (23 - log2m)
    let k' : Int := if m24 == pow2 24 then k + 1 else k
    let m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
    if k' > 127 then
      posInf
    else
      let expNat : Nat := Int.toNat (k' + 127)
      let fracNat : Nat := m24' - pow2 23
      ofBits (mkBits false expNat fracNat)

/-- Directed rounding down (toward `-∞`) of an exact dyadic to float32. -/
def roundDyadicDown (d : Dyadic) : IEEE32Exec :=
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else if d.sign then
    -- Negative: rounding down makes it *more negative* → round magnitude up.
    neg (roundDyadicPosUp d.mant d.exp)
  else
    roundDyadicPosDown d.mant d.exp

/-- Directed rounding up (toward `+∞`) of an exact dyadic to float32. -/
def roundDyadicUp (d : Dyadic) : IEEE32Exec :=
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else if d.sign then
    -- Negative: rounding up makes it *less negative* → round magnitude down.
    neg (roundDyadicPosDown d.mant d.exp)
  else
    roundDyadicPosUp d.mant d.exp

/-- `addDown x y` is a float32 lower bound for the exact real sum (when finite). -/
def addDown (x y : IEEE32Exec) : IEEE32Exec :=
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
        | some dx, some dy => roundDyadicDown (addDyadic dx dy)
        | _, _ => canonicalNaN

/-- `addUp x y` is a float32 upper bound for the exact real sum (when finite). -/
def addUp (x y : IEEE32Exec) : IEEE32Exec :=
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
        | some dx, some dy => roundDyadicUp (addDyadic dx dy)
        | _, _ => canonicalNaN

/-- `subDown x y` is a float32 lower bound for the exact real difference (when finite). -/
@[inline] def subDown (x y : IEEE32Exec) : IEEE32Exec :=
  addDown x (neg y)

/-- `subUp x y` is a float32 upper bound for the exact real difference (when finite). -/
@[inline] def subUp (x y : IEEE32Exec) : IEEE32Exec :=
  addUp x (neg y)

/-- `mulDown x y` is a float32 lower bound for the exact real product (when finite). -/
def mulDown (x y : IEEE32Exec) : IEEE32Exec :=
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
              roundDyadicDown { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
        | _, _ => canonicalNaN

/-- `mulUp x y` is a float32 upper bound for the exact real product (when finite). -/
def mulUp (x y : IEEE32Exec) : IEEE32Exec :=
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
              roundDyadicUp { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
        | _, _ => canonicalNaN

/-!
## Directed rounding for exact rationals (division-friendly)

For `divDown`/`divUp` we need outward rounding of an exact rational `num/den` to the float32 grid.
Our dyadic-directed rounders (`roundDyadicDown`/`roundDyadicUp`) already have a clean soundness
proof,
so we reduce rational rounding to dyadic rounding by building a **dyadic enclosure** of `num/den`:

- lower dyadic: `⌊(num/den) * 2^K⌋ * 2^{-K}`,
- upper dyadic: `⌈(num/den) * 2^K⌉ * 2^{-K}`.

We then apply `roundDyadicDown`/`roundDyadicUp` to these dyadics.

This is sound (it produces outward-rounded endpoints), but it is not necessarily optimally tight; a
larger `ratApproxShift` improves tightness at some computational cost.
-/

/-- Number of extra bits used when turning `num/den` into a dyadic enclosure. -/
def ratApproxShift : Nat := 200

/-- `ceil(num/den)` for naturals, totalized (returns `0` when `den = 0`). -/
def quotCeil (num den : Nat) : Nat :=
  if den == 0 then
    0
  else
    let q := num / den
    let r := num % den
    if r == 0 then q else q + 1

/-- Lower dyadic mantissa for `num/den` at scale `2^ratApproxShift`. -/
def ratLowerMant (num den : Nat) : Nat :=
  (Nat.shiftLeft num ratApproxShift) / den

/-- Upper dyadic mantissa for `num/den` at scale `2^ratApproxShift`. -/
def ratUpperMant (num den : Nat) : Nat :=
  quotCeil (Nat.shiftLeft num ratApproxShift) den

/--
Directed rounding down (toward `-∞`) for a rational `±(num/den)` with `den > 0`.

We do not attempt to be "correctly rounded" in the IEEE-754 sense; we only need a sound lower bound.
-/
def roundRatDown (sign : Bool) (num den : Nat) : IEEE32Exec :=
  if num == 0 then
    if sign then negZero else posZero
  else
    let loMant := ratLowerMant num den
    let hiMant := ratUpperMant num den
    let exp : Int := - (Int.ofNat ratApproxShift)
    if sign then
      -- Negative: rounding down makes it more negative → use an upper bound on the magnitude.
      roundDyadicDown { sign := true, mant := hiMant, exp := exp }
    else
      roundDyadicDown { sign := false, mant := loMant, exp := exp }

/--
Directed rounding up (toward `+∞`) for a rational `±(num/den)` with `den > 0`.
-/
def roundRatUp (sign : Bool) (num den : Nat) : IEEE32Exec :=
  if num == 0 then
    if sign then negZero else posZero
  else
    let loMant := ratLowerMant num den
    let hiMant := ratUpperMant num den
    let exp : Int := - (Int.ofNat ratApproxShift)
    if sign then
      -- Negative: rounding up makes it less negative → use a lower bound on the magnitude.
      roundDyadicUp { sign := true, mant := loMant, exp := exp }
    else
      roundDyadicUp { sign := false, mant := hiMant, exp := exp }

/-- `divDown x y` is a float32 lower bound for the exact real quotient (when finite and `y ≠ 0`). -/
def divDown (x y : IEEE32Exec) : IEEE32Exec :=
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
              roundRatDown sign num den
        | _, _ => canonicalNaN

/-- `divUp x y` is a float32 upper bound for the exact real quotient (when finite and `y ≠ 0`). -/
def divUp (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        if signBit x != signBit y then negZero else posZero
      else if isZero y then
        if isZero x then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            let sign := Bool.xor dx.sign dy.sign
            if dx.mant == 0 then
              if sign then negZero else posZero
            else
              let eDiff : Int := dx.exp - dy.exp
              let (num, den) :=
                match eDiff with
                | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
                | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
              roundRatUp sign num den
        | _, _ => canonicalNaN


end IEEE32Exec

end TorchLean.Floats.IEEE754
