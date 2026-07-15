# Rounding

This folder defines how an exact real value is selected onto a representable grid.

- `Core.lean` contains floor, ceiling, truncation, and nearest-even integer rounding, together with
  the named interface for selecting those four standard choices.
- `Predicates.lean` states validity conditions for rounding functions.
- `Properties.lean` proves the basic directed and nearest-mode laws.
- `Generic.lean` lifts integer rounding to a generic floating-point format.
- `Order.lean` proves monotonicity and sandwich results.
- `Nearest.lean` isolates midpoint-choice semantics.
- `Odd.lean` and `Away.lean` define auxiliary modes used in conversion and double-rounding work.
- `Double.lean` studies when two successive roundings agree with one rounding.

IEEE 754-2019 is the normative reference for the standard rounding directions.  Goldberg (ACM
Computing Surveys, 1991, doi:10.1145/103162.103163) gives the classical numerical background;
Flocq provides the generic-format proof architecture followed here.
