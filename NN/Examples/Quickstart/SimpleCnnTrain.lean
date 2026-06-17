/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Quickstart.Common

/-!
# Simple CNN training example

This is the image-classification companion to the small MLP tutorial.

The code is meant to look like normal model code: define the layers, choose a trainer, batch the
small dataset, and call `trainer.train`. The shape-indexed tensors and checked training task are
still there underneath, but they are not the first thing a new reader has to fight.

Check this tutorial module directly:

- `lake build NN.Examples.Quickstart.SimpleCnnTrain`

For the maintained command-line CNN trainer, use `NN/Examples/Models/Vision/Cnn.lean`:

- `python3 scripts/datasets/download_example_data.py --cifar10`
- `lake build -R -K cuda=true && lake exe torchlean cnn --cuda --n-total 1 --steps 1`

Optional flags:

- `--steps N`
- `--batch N`

See `NN/Examples/Quickstart/README.md` for the shared conventions in this folder.
-/

@[expose] public section


namespace NN.Examples.Quickstart.SimpleCNNTrain

open TorchLean

/-- Default JSON log path used only when the user explicitly passes `--log`. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "quickstart_simple_cnn"

def mkModel {batch : Nat} :
    nn.M (nn.Sequential (Shape.images batch 1 4 4) (Shape.mat batch 2)) :=
  let outC : Nat := 3
  let outH : Nat := (4 - 2) / 1 + 1
  let outW : Nat := (4 - 2) / 1 + 1
  nn.Sequential![
    nn.Conv2d (n := batch) (inC := 1) (inH := 4) (inW := 4)
      { outC := outC, kH := 2, kW := 2, stride := 1, padding := 0 },
    nn.ReLU,
    nn.ClassifierBatch (n := batch) (s := Shape.image outC outH outW) 2
  ]

/-- Command-line help for the simple CNN quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean simple CNN quickstart"
    , ""
    , "Usage:"
    , "  lake exe torchlean quickstart_cnn [options]"
    , ""
    , "Options:"
    , "  --seed N"
    , "  --batch N"
    , "  --steps N"
    , "  --dtype float|ieee32"
    , "  --backend eager|compiled"
    , "  --cpu | --cuda"
    , "  --log PATH"
    ]

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  IO.println "== Quickstart next step: simple CNN training =="
  let (seed, args) ← CLI.seed "SimpleCNNTrain" args 0
  let (batch, args) ← CLI.positiveNatFlag "SimpleCNNTrain" args "batch" 2
  let parsed ←
    _root_.NN.Examples.Quickstart.parseRuntimeTrain
      "SimpleCNNTrain" args defaultLogJson 5 (optim.adam { lr := 0.03 })
  let trainer := Trainer.new (mkModel (batch := batch)) <|
    Trainer.Config.fromRunConfig parsed.run .classification (seed := seed)

  let trainData :=
    Data.batchDataset batch Data.Bands.dataset
      (shuffle := true) (seed := seed) (dropLast := true)
  let trained ← trainer.train trainData parsed.trainOptions
  trained.printSummary
  match Data.Bands.probeSamples with
  | [] => pure ()
  | (name, x, expected) :: _ =>
      let xBatch := Tensor.repeatBatch batch x
      trained.printPrediction s!"{name} expected={expected}" xBatch

end NN.Examples.Quickstart.SimpleCNNTrain
