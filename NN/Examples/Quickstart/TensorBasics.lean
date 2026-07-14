/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Quickstart: Tensor Basics

This is the first stop in the TorchLean examples. It does **not** use sessions, CUDA, or autograd.
It is just about building typed tensors in Lean with a convenient constructor layer.

What it covers:
- 1D and N-D constructors from literal lists (`Tensor.vector`, `Tensor.ofList`, `tensor!`),
- the fact that the element type `α` is the “dtype” (e.g. `Float`, `ℚ`, `Int`),
- Float-literal convenience constructors for executable float32 (`Tensor.float32Vector`),
- why we generally do not try to `print` tensors over `ℝ` (noncomputable / too large).

Run:
  `lake exe torchlean quickstart_tensors`
-/

@[expose] public section


namespace NN.Examples.Quickstart.TensorBasics

open TorchLean

/-- Command-line help for the tensor-basics quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean tensor basics quickstart"
    , ""
    , "Usage:"
    , "  lake exe torchlean quickstart_tensors"
    , ""
    , "This demo has no tutorial-specific flags."
    ]

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "quickstart_tensors" args
  IO.println "== Quickstart: tensor basics =="

  -- The “dtype” here is just the element type `α`.
  let xF := Tensor.vector (α := Float) [0.1, 0.2, 0.3, 0.4]
  let xQ := Tensor.vector (α := ℚ) [0.1, 0.2, 0.3, 0.4]
  let xI := Tensor.vector (α := Int) [1, 2, 3, 4]

  Tensor.print xF
  Tensor.print xQ
  Tensor.print xI

  -- Convenience: build from Float literals then convert to executable float32 (IEEE32Exec).
  let x32 := Tensor.float32Vector [0.1, 0.2, 0.3, 0.4]
  Tensor.print x32

  -- N-D tensor using "nested brackets" (like nested Python lists in PyTorch).
  -- This is often the clearest way to see where each element goes.
  let x3 : Tensor.T Float (Shape.ofDims [2, 2, 2]) :=
    tensor! [
      [ [1, 2], [3, 4] ],
      [ [5, 6], [7, 8] ]
    ]
  Tensor.print x3

  -- The explicit equivalent is `Tensor.ofList`: you provide dims + a flat row-major list.
  -- Row-major means the last dimension changes fastest:
  -- the above `x3` is the same as `Tensor.ofList [2,2,2] [1,2,3,4,5,6,7,8]`.

  -- Showing the intentional “Real tensors refuse to print” behavior.
  let xR := Tensor.vector (α := ℝ) [0.1, 0.2, 0.3, 0.4]
  try
    Tensor.print xR
  catch e =>
    IO.println s!"Expected failure printing Tensor ℝ: {e}"

end NN.Examples.Quickstart.TensorBasics
