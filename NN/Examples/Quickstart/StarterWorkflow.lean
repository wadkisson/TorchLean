/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN

/-!
# Quickstart: Starter Workflow

The smallest useful TorchLean training setup is ordinary model code:

```lean
import NN
open TorchLean

def model :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]
```

No `NN.*` subsystem imports are needed here. The example exercises the first workflow directly:
model construction, data construction, training, evaluation, and verification all come from
`import NN`.
-/

@[expose] public section

namespace NN.Examples.Quickstart.StarterWorkflow

open TorchLean

def model :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

/-- Checks that KAN constructors are available from the public `import NN` surface. -/
def kanModel : nn.M (nn.Sequential (Shape.mat 4 2) (Shape.mat 4 1)) :=
  nn.models.KAN
    { batch := 4
      inDim := 2
      hidden := [8]
      outDim := 1 }

def target (x1 x2 : Float) : Float :=
  let relu (x : Float) := if x < 0.0 then 0.0 else x
  relu (x1 + x2) + 0.25

/--
Tiny in-memory regression dataset.

The important bit is the last line: `Data.tensorDataset xs ys` turns ordinary `Float` tensors into a
runtime-polymorphic dataset, so the trainer can still choose `Float`, executable IEEE32, CPU, CUDA,
eager, or compiled execution later.
-/
def data : Trainer.Dataset (Shape.vec 2) (Shape.vec 1) :=
  let xs : Tensor.T Float (shape![4, 2]) :=
    tensorND! [4, 2] [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
  let ys : Tensor.T Float (shape![4, 1]) :=
    tensorND! [4, 1] [target 0.0 0.0, target 0.0 1.0, target 1.0 0.0, target 1.0 1.0]
  Data.tensorDataset xs ys

def probes : List (Trainer.Probe (Shape.vec 2)) :=
  [ Trainer.Probe.vec2 "origin" 0.0 0.0 (some (toString (target 0.0 0.0)))
  , Trainer.Probe.vec2 "heldout" 0.5 (-0.25) (some (toString (target 0.5 (-0.25)))) ]

/-- Tiny optimizer-choice example using only `import NN`. -/
def optimizerChoiceExample : Except String (String × optim.Optimizer) := do
  let kind ← optim.Kind.parse "adamw"
  pure (kind.name, kind.toOptimizer 0.01)

/-- Tiny adapter type example using only `import NN`. -/
def loraParamsExample : Type :=
  Adapters.LoRA.Params Float 2 1 1

/--
Run the public API example from another command or from `#eval` while developing.

The shape below is the user-facing training path:

- build the trainer from the model,
- attach optimizer/backend choices once,
- call `trainer.eval` for initial inference,
- call `trainer.train`,
- use the returned trained handle for prediction.
- call `trained.verifyRobustLInf` on a small `ℓ∞` box.

The quickstart build only checks that these declarations typecheck; it does not train during
ordinary `lake build`, which keeps CI fast.
-/
def run (_args : List String := []) : IO Unit := do
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.adam { lr := 0.03 }
        backend := .compiled
        dtype := .float32 }
  let heldout : Tensor.T Float (Shape.vec 2) := tensorND! [2] [0.5, -0.25]
  let initial ← trainer.eval heldout
  IO.println s!"initial(heldout) = {Tensor.pretty initial}"
  let trained ← trainer.train data { steps := 25, batchSize := 4, logEvery := 10 } probes
  trained.printSummary
  trained.printPrediction "predict(heldout)" heldout
  let cert ← trained.verifyRobustLInf heldout 0.05
  cert.printSummary

end NN.Examples.Quickstart.StarterWorkflow
