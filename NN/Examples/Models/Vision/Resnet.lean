/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe torchlean resnet --cpu
  lake build -R -K cuda=true && lake exe torchlean resnet --cuda

This runs a medium ResNet-style model built from `API.nn.resnetBasicBlock` (Conv+BN+residual).
-/

module


public import NN
public import NN.API.Models.Resnet
public import NN.Examples.Models.Common.RealData

/-!
# ResNet Real-Data Example

Runnable `torchlean resnet` example. It trains a compact ResNet-style classifier built from
`API.nn.resnetBasicBlock` on a prepared CIFAR-10 minibatch.

The reusable model wiring lives in `NN.API.Models.Resnet` (`nn.models.resnet`). This file is the
runnable wrapper (CIFAR loader construction + step-limited training loop).

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake build -R -K cuda=true && lake exe torchlean resnet --cuda --n-total 200 --steps 1
```

Tip: the defaults are set for a quick sanity run. For a longer run:

```bash
lake build -R -K cuda=true
lake exe torchlean resnet --cuda --fast-kernels --n-total 5000 --steps 200
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Vision.Resnet

def exeName : String := "torchlean resnet"
def defaultLogJson : System.FilePath := "data/model_zoo/resnet_trainlog.json"

def batch : Nat := 2
def inC : Nat := 3
def inH : Nat := RealData.cifarHeight
def inW : Nat := RealData.cifarWidth

def stemC : Nat := 8
def numClasses : Nat := RealData.cifarClasses

def cfg : nn.models.ResnetConfig :=
  { batch := batch, inC := inC, inH := inH, inW := inW, stemC := stemC, numClasses := numClasses }

abbrev σ : Shape :=
  nn.models.resnetInShape cfg

abbrev τ : Shape :=
  nn.models.resnetOutShape cfg

def mkModel : nn.M (nn.Sequential σ τ) := do
  nn.models.resnet cfg

def loadCifarLoader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (Data.BatchLoader α batch RealData.CifarImage RealData.CifarTarget) := do
  RealData.loadCifarLoader (α := α) exeName batch nRows seed xPath yPath

/--
Train the ResNet-style classifier through the shared Float module-training API.

Only the architecture and loss are local to this file.  The runner-level mechanics -- optimizer
state, exact step counting, CUDA memory watch callbacks, before/after reports, and JSON logging --
stay in `NN.API.train` so future image models do not have to repeat this boilerplate.
-/
def fitCifar (opts : Runtime.Autograd.Torch.Options)
    (xPath yPath : System.FilePath) (nRows seed : Nat) (trainCfg : Common.ModelTrainFlags) :
    IO (train.FitReport Float) := do
  let loader ← loadCifarLoader (α := Float) xPath yPath nRows seed
  nn.withModel mkModel fun model => do
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let module ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let opt := Common.adamOptimizer (α := Float) id (nn.paramShapes model) trainCfg.lr
    let cudaMemWatch :=
      Common.effectiveCudaMemWatch opts trainCfg.train.steps trainCfg.cudaMemWatch
    let (report, _loader') ← train.fitModuleLoaderStepsLoggedFloat module opt opts
      trainCfg.train.steps loader trainCfg.train.log "ResNet CIFAR training"
      #[s!"data=cifar10", s!"x={xPath}", s!"y={yPath}", s!"nRows={nRows}",
        s!"device={if opts.useGpu then "cuda" else "cpu"}", s!"lr={trainCfg.lr}",
        s!"steps={trainCfg.train.steps}", s!"batch={batch}", s!"cuda_mem_watch={cudaMemWatch}"]
      "loss" trainCfg.cudaMemWatch
    pure report

def main (args : List String) : IO UInt32 := do
  Common.runFloat exeName args
    (banner := fun opts =>
      s!"{exeName}: ResNet CIFAR training (device={if opts.useGpu then "cuda" else "cpu"})")
    (k := fun opts rest => do
      let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <|
        RealData.parseCifarFlags rest
      let (trainCfg, rest) ← Common.orThrow exeName <|
        Common.parseModelTrainFlags exeName rest defaultLogJson 1 1e-3
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let report ← fitCifar opts xPath yPath nRows seed trainCfg
      IO.println s!"  steps={trainCfg.train.steps} loss0={report.before} loss1={report.after}")

end NN.Examples.Models.Vision.Resnet
