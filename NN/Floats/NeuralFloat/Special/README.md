# Special Execution Policies

This folder is for policies that modify the generic format model.  `FTZ.lean` formalizes
flush-to-zero behavior by replacing sufficiently small nonzero results with zero.  It is kept
separate because FTZ is not equivalent to IEEE gradual underflow and requires different error
bounds.

IEEE 754-2019 specifies gradual underflow through subnormal numbers.  Hardware FTZ modes are
backend behavior and must be selected explicitly in a TorchLean execution contract.
