
module

public import NN.API
public import NN.API.Verification


@[expose] public section

open TorchLean

--mlp model that predicts y = relu(x1 + x2) + 0.25
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def xs : Tensor Float [4, 2] :=
  tensor! [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

def ys : Tensor Float [4, 1] :=
  tensor! [[0.25], [1.25], [1.25], [2.25]]

def main (_args : List String) : IO Unit := do
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.adam { lr := 0.03 }
        execution := .typedGraph
        device := .cpu
        scalar := .ieee32Exec }

  let x : Tensor Float [2] := tensor! [0.5, -0.25]

  --print a models output before training for the above tensor
  IO.println s!"before {Tensor.pretty (← trainer.predict x)}"
  --train
  let trained ← trainer.trainVerified (Data.tensorDataset xs ys)
    { steps := 20, batchSize := 4, logEvery := 10 }
  trained.printSummary
  trained.printPrediction "after" x
  --verify
  (← trained.verifyRobustLInf x 0.05).printSummary
