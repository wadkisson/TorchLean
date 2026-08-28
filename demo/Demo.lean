/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Verification

/-!
# Demo

Typed model → train → certify. Run `lake exe demo`.
-/

@[expose] public section

open TorchLean

-- A 2→8→1 ReLU MLP. The widths are part of the Lean type, so they cannot
-- silently disagree with `xs` / `ys`.
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

-- Four unit-square corners. Target is y = relu(x1 + x2) + 0.25.
def xs : Tensor Float [4, 2] :=
  tensor! [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

def ys : Tensor Float [4, 1] :=
  tensor! [[0.25], [1.25], [1.25], [2.25]]

def main (_args : List String) : IO Unit := do
  -- Train and verify through one trainer: typed graph + IEEE binary32 reference.
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.adam { lr := 0.03 }
        execution := .typedGraph
        device := .cpu
        scalar := .ieee32Exec }
  -- Held-out point, off the four training corners.
  let x : Tensor Float [2] := tensor! [0.5, -0.25]
  IO.println s!"before {Tensor.pretty (← trainer.predict x)}"
  -- Train, and keep an IBP verifier attached to the updated parameters.
  let trained ← trainer.trainVerified (Data.tensorDataset xs ys)
    { steps := 20, batchSize := 4, logEvery := 10 }
  trained.printSummary
  trained.printPrediction "after" x
  -- Prove an output interval on an ℓ∞ ball of radius 0.05 around x.
  (← trained.verifyRobustLInf x 0.05).printSummary
