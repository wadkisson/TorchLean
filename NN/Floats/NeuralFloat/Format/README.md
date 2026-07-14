# Formats

This folder describes which real numbers belong to a generic radix/exponent grid.  It does not pick
a rounding mode and it does not claim that a runtime backend implements the grid.

- `Magnitude.lean` locates a nonzero real between adjacent radix powers.
- `Digits.lean` relates integer digit counts to those magnitude bounds.
- `Formats.lean` defines fixed-exponent (`FIX`), unbounded (`FLX`), bounded-precision (`FLT`), and
  abrupt-underflow exponent functions.
- `Generic.lean` defines canonical representability and grid inclusion.
- `Theorems.lean` proves structural properties shared by arithmetic and error analysis.

This separation is important for quantization.  A fixed-point or arbitrary-float quantizer first
declares its representable grid here; rounding and saturation are separate policy choices.

The principal reference is Boldo and Melquiond's Flocq library and paper (IEEE ARITH 2011,
doi:10.1109/ARITH.2011.40).  Concrete binary interchange formats are governed by IEEE 754-2019.
