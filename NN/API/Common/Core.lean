/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Core
public import NN.API.Runtime
public import NN.Runtime.Training.Log
public import NN.Tensor.API

import Mathlib.Algebra.Order.Algebra

/-!
# `NN.API.Common`

Shared helpers used across TorchLean workflows and entrypoints: typed tensor constructors, casting
between scalar backends, and shared `Except`/`IO` utilities.
-/

@[expose] public section

namespace NN
namespace API

namespace Common

open Spec

/-!
## Common Helpers For Workflows And Entry Points

This module contains helper functions that show up repeatedly in executable workflows.

What belongs here:
- casting `Spec.Tensor Float` into an arbitrary scalar backend `α` (given `Float → α`)
- building typed tensors from raw lists (`tensorF`, `vecF`, `matF`)
- bridging `Except String` into `IO` (`orThrow`)
- running model commands under a chosen scalar backend via `--dtype` (see `NN.API.DType`)

What does *not* belong here:
- core semantics/proofs (those live under `NN` / `NN.Proofs`)
- command-specific CLI parsing that belongs with a model or tool entrypoint
-/

/-!
### PyTorch Mapping Notes

If you're coming from PyTorch, this module plays the role of the shared utility layer often written
around:
- `torch.tensor([...], dtype=..., device=...)`
- choosing a dtype/backend based on CLI flags
- dataset loading and runnable workflow code

The main difference is that TorchLean tracks shapes in types, so tensor constructors here return
`Except String ...` rather than silently reshaping.
-/

/-- Cast a tensor elementwise while preserving its type-level shape. -/
def castTensor {α : Type} (cast : Float → α) {s : Spec.Shape} (t : Spec.Tensor Float s) :
    Spec.Tensor α s :=
  Spec.mapTensor cast t

/--
Convert an `Except String α` into an `IO α`, raising a tagged `userError` on failure.

This is handy in `main` functions where we want the process to exit with a readable message.
-/
def orThrow {α : Type} (tag : String := "Example") : Except String α → IO α
  | .ok a => pure a
  | .error msg => throw <| IO.userError s!"{tag}: {msg}"

/-- Standard location for model-example training logs under `data/model_zoo`. -/
def modelZooTrainLog (stem : String) : System.FilePath :=
  System.FilePath.mk s!"data/model_zoo/{stem}_trainlog.json"

/--
Fail with a tagged `userError` if a boolean condition is false.

Executable workflows use this for named precondition checks such as
`Common.check exeName "loss finite" (loss == loss)`.
-/
def check (tag msg : String) (b : Bool) : IO Unit :=
  unless b do throw <| IO.userError s!"{tag}: {msg}"

/-- Validate that a natural-number CLI flag is positive. -/
def requirePositiveNatFlag (exeName flag : String) (value : Nat) : Except String Unit := do
  if value = 0 then
    throw s!"{exeName}: --{flag} must be > 0"

/--
Resolve an optional natural-number CLI flag against a default and require that the result is
strictly positive.

Example parsers use this helper for the common "optional flag + default + positivity check"
case instead of restating the same `getD` / `requirePositiveNatFlag` sequence.
-/
def resolvePositiveNatFlag
    (exeName flag : String)
    (value? : Option Nat)
    (default : Nat) :
    Except String Nat := do
  let value := value?.getD default
  requirePositiveNatFlag exeName flag value
  pure value

/-- Write a prepared `TrainLog` JSON artifact and report the file path. -/
def writeTrainLog (path : System.FilePath) (log : _root_.Runtime.Training.TrainLog) : IO Unit := do
  _root_.Runtime.Training.TrainLog.writeJson path log
  IO.println s!"  wrote TrainLog JSON: {path}"

/-- Write a prepared `TrainLog` to a destination that may be disabled. -/
def writeTrainLogTo (dest : _root_.Runtime.Training.LogDestination)
    (log : _root_.Runtime.Training.TrainLog) : IO Unit := do
  _root_.Runtime.Training.LogDestination.writeTrainLog dest log
  match dest.path? with
  | some path => IO.println s!"  wrote TrainLog JSON: {path}"
  | none => IO.println "  TrainLog JSON disabled"

/--
Write a standard JSON training artifact for routines that record an initial and final loss.

The function uses `Runtime.Training.TrainLog.beforeAfterLoss` and the stable TrainLog JSON format.
The output schema is independent of the model, dataset, and runtime backend.
-/
def writeBeforeAfterLossLog (path : System.FilePath) (title : String) (steps : Nat)
    (beforeLoss afterLoss : Float) (notes : Array String := #[]) : IO Unit := do
  let log := _root_.Runtime.Training.TrainLog.beforeAfterLoss title steps beforeLoss afterLoss notes
  writeTrainLog path log

/--
Write a before/after loss log to an explicit logging destination.

`LogDestination.disabled` is a no-op, mirroring `wandb disabled` for runs where metrics should stay
on stdout only.
-/
def writeBeforeAfterLossLogTo (dest : _root_.Runtime.Training.LogDestination)
    (title : String) (steps : Nat) (beforeLoss afterLoss : Float) (notes : Array String := #[]) :
    IO Unit := do
  let log := _root_.Runtime.Training.TrainLog.beforeAfterLoss title steps beforeLoss afterLoss notes
  writeTrainLogTo dest log

/-- Print the standard before/after loss summary returned by `fit` helpers. -/
def printTrainReport {α : Type} [ToString α] (steps : Nat)
    (report : TorchLean.Trainer.TrainReport α) : IO Unit :=
  IO.println s!"  steps={steps} loss_before={report.before} loss_after={report.after}"

/-- First and last point of a scalar training curve, ready for summaries. -/
structure CurveEndpoints where
  /-- Step used for the final point. -/
  finalStep : Nat
  /-- First recorded metric value. -/
  first : Float
  /-- Last recorded metric value. -/
  last : Float
deriving Repr

/--
Read the first and last values from a scalar curve.

If `curve.steps` is empty, the final step falls back to the last value index. Empty value arrays are
reported as errors instead of printing a fake `0.0` summary.
-/
def curveEndpoints? (curve : _root_.Runtime.Training.Curve) : Option CurveEndpoints := do
  let first ← curve.values[0]?
  let last ← curve.values.back?
  let defaultFinalStep := curve.values.size - 1
  let finalStep := curve.steps.back?.getD defaultFinalStep
  pure { finalStep := finalStep, first := first, last := last }

/-- Require first/last scalar values from a training curve, or raise a user-facing error. -/
def requireCurveEndpoints (context : String) (curve : _root_.Runtime.Training.Curve) :
    IO CurveEndpoints := do
  match curveEndpoints? curve with
  | some endpoints => pure endpoints
  | none => throw <| IO.userError s!"{context}: empty training curve"

/-- Print the standard first/last loss summary for a scalar training curve. -/
def printCurveLossSummary (steps : Nat) (curve : _root_.Runtime.Training.Curve) : IO Unit := do
  let endpoints ← requireCurveEndpoints "printCurveLossSummary" curve
  let finalStep :=
    if curve.steps.isEmpty then steps else endpoints.finalStep
  IO.println s!"  steps={finalStep} loss_before={endpoints.first} loss_after={endpoints.last}"

/-- Write a one-series scalar curve as a standard `TrainLog` JSON artifact. -/
def writeCurveLog (path : System.FilePath) (title : String)
    (curve : _root_.Runtime.Training.Curve) (seriesName : String := "loss")
    (notes : Array String := #[]) : IO Unit := do
  let log := curve.toTrainLog title seriesName (notes := notes)
  writeTrainLog path log

/-- Write a one-series scalar curve to an explicit logging destination. -/
def writeCurveLogTo (dest : _root_.Runtime.Training.LogDestination) (title : String)
    (curve : _root_.Runtime.Training.Curve) (seriesName : String := "loss")
    (notes : Array String := #[]) : IO Unit := do
  let log := curve.toTrainLog title seriesName (notes := notes)
  writeTrainLogTo dest log

/-- Common CLI result for training commands that accept `--steps`, `--batch-size`, and `--log`. -/
structure LoggedTrainFlags where
  /-- Number of optimizer updates. -/
  steps : Nat
  /-- Number of samples consumed by one public in-memory training step. -/
  batchSize : Nat := 1
  /-- Logging destination. Use `--log false` / `off` / `none` to disable. -/
  log : _root_.Runtime.Training.LogDestination
  /-- Path where the JSON `TrainLog` should be written. -/
  logPath : System.FilePath
  /-- CUDA allocator telemetry cadence shared by fixed-sample and custom training loops. -/
  cudaMemWatch : Nat := 0
deriving Repr

/--
Parse common training flags: positive `--steps`, positive optional `--batch-size`, plus optional
`--log`.

`--log <path>` writes the standard local JSON artifact. `--log false`, `--log off`, or
`--log none` disables artifact writing while preserving the parsed step count.
-/
def parseLoggedTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1)
    (allowZeroSteps : Bool := false) :
    Except String (LoggedTrainFlags × List String) := do
  let (logRaw?, args) ← TorchLean.CLI.takeFlagValueOnce args "log"
  let (steps, args) ← TorchLean.CLI.takeStepsFlagDefault args defaultSteps
  let (batchSize?, args) ← TorchLean.CLI.takeNatFlagOnce args "batch-size"
  let (cudaMemWatch?, args) ← TorchLean.CLI.takeNatFlagOnce args "cuda-mem-watch"
  unless allowZeroSteps do
    requirePositiveNatFlag exeName "steps" steps
  let batchSize := batchSize?.getD 1
  requirePositiveNatFlag exeName "batch-size" batchSize
  let log := _root_.Runtime.Training.LogDestination.parse? defaultLogPath logRaw?
  pure ({ steps := steps, batchSize := batchSize, log := log, logPath := log.pathD defaultLogPath,
          cudaMemWatch := cudaMemWatch?.getD 0 }, args)

/--
Training flags shared by runnable model examples.

This covers the knobs almost every example needs: `--steps`, `--log`, CUDA memory watching, and
`--lr`. Model files should reuse this record and add only flags that change that model's actual
behavior, such as text generation settings or evaluation probes.
-/
structure ModelTrainFlags extends LoggedTrainFlags where
  /-- Learning rate for the default Adam optimizer used by examples. -/
  lr : Float
deriving Repr

/-- Parse the standard model-training flags: `--steps`, `--log`, `--lr`, and CUDA telemetry. -/
def parseModelTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (ModelTrainFlags × List String) := do
  let (train, args) ← parseLoggedTrainFlags exeName args defaultLogPath defaultSteps allowZeroSteps
  let (lr, args) ← TorchLean.CLI.takePositiveFloatFlagDefault args exeName "lr" defaultLr
  pure ({ toLoggedTrainFlags := train, lr := lr }, args)

/--
Model-training flags plus an RNG/data-order seed.

Use this when the command needs reproducible initialization, synthetic data, or shuffled row order in
addition to the standard training flags.
-/
structure SeededModelTrainFlags extends ModelTrainFlags where
  /-- Seed used for model initialization, synthetic data, or row order. -/
  seed : Nat
deriving Repr

/--
Parse the standard model-training flags together with `--seed`.

Examples use this when model initialization, synthetic data, or row order needs a reproducible seed.
-/
def parseSeededModelTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (SeededModelTrainFlags × List String) := do
  let (seed, args) ← TorchLean.CLI.takeSeed args 0
  let (train, args) ← parseModelTrainFlags exeName args defaultLogPath defaultSteps defaultLr
    allowZeroSteps
  pure ({ toModelTrainFlags := train, seed := seed }, args)

/-! ### Progress Cadence Helpers -/

/--
Return whether a completed training step should emit a progress report.

The convention is shared across example trainers: `logEvery = 0` disables progress output; otherwise
we log at exact multiples of the completed-step count.
-/
def shouldLogStep (logEvery done : Nat) : Bool :=
  logEvery != 0 && done % logEvery == 0

/-! ### CUDA Memory Watch Helpers -/

/--
Choose a CUDA memory-watch cadence for public examples.

Users can pass `--cuda-mem-watch N` to choose an exact cadence. When no cadence is supplied, long
CUDA runs sample about ten times over the requested training horizon. Short runs and CPU runs stay
quiet by default, so the examples do not print allocator telemetry unless it is likely to be useful.
-/
def effectiveCudaMemWatch (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps requested : Nat) : Nat :=
  API.TorchLean.Trainer.effectiveCudaMemWatch opts steps requested

/-- Standard TrainLog note for the effective CUDA memory-watch cadence. -/
def cudaMemWatchNote (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps requested : Nat) : String :=
  s!"cuda_mem_watch={effectiveCudaMemWatch opts steps requested}"

/--
State for a simple CUDA-memory drift detector.

The first reported sample becomes the baseline.  Later samples compare current CUDA free memory
against that baseline and warn once if the observed per-step drop projects failure before the
requested run length.
-/
abbrev CudaMemWatchState := API.TorchLean.Trainer.CudaMemWatchState

/--
Maybe print a one-line CUDA allocator report.

The report samples the native allocator at a fixed cadence and warns if the observed free-memory
slope would cross zero before the requested training horizon.
-/
def reportCudaMemWatch (opts : _root_.Runtime.Autograd.Torch.Options)
    (watchEvery totalSteps done : Nat) (state? : Option CudaMemWatchState) :
    IO (Option CudaMemWatchState) :=
  API.TorchLean.Trainer.reportCudaMemWatch opts watchEvery totalSteps done state?

/--
Default Adam optimizer constructor used by supervised and vision examples.

The reusable part is the optimizer convention, not the model.  Individual examples still own their
architecture and loss, while this helper keeps the Adam hyperparameter spelling identical across
MLP, CNN, ResNet, ViT, and similar model commands.
-/
def adamOptimizer {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (cast : Float → α) (ps : List Spec.Shape) (lr : Float) :
    TorchLean.Optim.Optimizer α ps :=
  TorchLean.Optim.adam (α := α) (paramShapes := ps)
    (lr := cast lr) (beta1 := cast 0.9) (beta2 := cast 0.999) (epsilon := cast 1e-8)

/--
Run an executable on the concrete `Float` runtime path.

We use this for runnable training commands that produce Float-valued artifacts: CPU/CUDA eager
execution, native kernels, and JSON loss curves.
-/
def runFloat
    (exeName : String) (args : List String)
    (banner : TorchLean.Options → String)
    (k : (opts : TorchLean.Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 :=
  TorchLean.Module.run exeName args (.float k)
    { banner? := some banner, printOk := printOk }

/-- Run a Float-only command after forcing CUDA runtime flags. -/
def runCudaFloat
    (exeName : String)
    (args : List String)
    (banner : TorchLean.Options → String)
    (k : (opts : TorchLean.Options) → (rest : List String) → IO Unit)
    (extraFlags : List String := [])
    (printOk : Bool := true) : IO UInt32 := do
  let requestsCpu := args.any fun arg => arg == "--device=cpu"
    || (args.zip (args.drop 1)).any fun pair => pair == ("--device", "cpu")
  if requestsCpu then
    IO.eprintln s!"{exeName}: this command is GPU-only; remove --device cpu"
    return 1
  let requestsCuda := args.any fun arg => arg == "--device=cuda" || arg == "--device=gpu"
    || (args.zip (args.drop 1)).any fun pair =>
      pair == ("--device", "cuda") || pair == ("--device", "gpu")
  let args := if requestsCuda then args else "--device" :: "cuda" :: args
  let args := extraFlags.foldl
    (fun acc flag => if acc.contains flag then acc else flag :: acc)
    args
  runFloat exeName args banner k printOk

/-- Run a Float-only command after forcing CUDA eager-runtime flags. -/
def runCudaEagerFloat
    (exeName : String)
    (args : List String)
    (banner : TorchLean.Options → String)
    (k : (opts : TorchLean.Options) → (rest : List String) → IO Unit)
    (extraFlags : List String := [])
    (printOk : Bool := true) : IO UInt32 := do
  let requestsCompiled := args.any fun arg => arg == "--backend=compiled"
    || (args.zip (args.drop 1)).any fun pair => pair == ("--backend", "compiled")
  if requestsCompiled then
    IO.eprintln s!"{exeName}: this command requires --backend eager; compiled is proof-compiled host execution, not CUDA graph execution"
    return 1
  runCudaFloat exeName args banner k extraFlags printOk

/-! ### Common model-run parsers -/

/-- Shared corpus-window or training-window count used by finite cyclic examples. -/
structure WindowOptions where
  /-- Number of windows used by the training set, sampler, or cyclic schedule. -/
  windows : Nat
deriving Repr

namespace WindowOptions

/--
Parse the shared `--windows` flag.

The model decides how to use the windows; this helper just keeps the flag spelling and positivity
check consistent.
-/
def parse
    (exeName : String)
    (args : List String)
    (defaultWindows : Nat) :
    Except String (WindowOptions × List String) := do
  let (windows, args) ← TorchLean.CLI.takePositiveNatFlagDefault args exeName "windows" defaultWindows
  pure ({ windows := windows }, args)

end WindowOptions

/-- Optional parameter-checkpoint load/save paths shared by runnable model commands. -/
structure CheckpointOptions where
  /-- Optional checkpoint path loaded before training/generation. -/
  loadParams? : Option System.FilePath
  /-- Optional checkpoint path written after training. -/
  saveParams? : Option System.FilePath
deriving Repr

namespace CheckpointOptions

/-- Parse the shared `--load-params` / `--save-params` flags. -/
def parse
    (args : List String) :
    Except String (CheckpointOptions × List String) := do
  let (loadParams?, args) ← TorchLean.CLI.takePathFlagOnce args "load-params"
  let (saveParams?, args) ← TorchLean.CLI.takePathFlagOnce args "save-params"
  pure ({ loadParams? := loadParams?, saveParams? := saveParams? }, args)

end CheckpointOptions

/-- Diffusion schedule knobs shared by model-zoo diffusion commands. -/
structure DiffusionScheduleFlags where
  /-- Number of diffusion timesteps in the schedule. -/
  T : Nat
  /-- First beta value in the schedule. -/
  betaStart : Float
  /-- Final beta value in the schedule. -/
  betaEnd : Float
deriving Repr

namespace DiffusionScheduleFlags

/-- Parse shared diffusion schedule flags: `--T`, `--beta-start`, and `--beta-end`. -/
def parse
    (args : List String)
    (defaultT : Nat := 100)
    (defaultBetaStart : Float := 1e-4)
    (defaultBetaEnd : Float := 0.12) :
    Except String (DiffusionScheduleFlags × List String) := do
  let (T, args) ← TorchLean.CLI.takeNatFlagDefault args "T" defaultT
  let (betaStart, args) ← TorchLean.CLI.takeFloatFlagDefault args "beta-start" defaultBetaStart
  let (betaEnd, args) ← TorchLean.CLI.takeFloatFlagDefault args "beta-end" defaultBetaEnd
  pure ({ T := T, betaStart := betaStart, betaEnd := betaEnd }, args)

/-- Standard TrainLog metadata for a diffusion schedule. -/
def trainLogNotes (cfg : DiffusionScheduleFlags) : Array String :=
  #[s!"T={cfg.T}", s!"betaStart={cfg.betaStart}", s!"betaEnd={cfg.betaEnd}"]

end DiffusionScheduleFlags

/-- Optional image artifacts emitted by image-generation or reconstruction commands. -/
structure ImageArtifactFlags where
  /-- Optional timestep used for reconstruction-from-noise artifacts. -/
  reconstructStep? : Option Nat
  /-- Optional path for an unconditional sample image. -/
  samplePpm? : Option System.FilePath
  /-- Optional path for the clean reference image. -/
  referencePpm? : Option System.FilePath
  /-- Optional path for the noised/intermediate image. -/
  noisyPpm? : Option System.FilePath
  /-- Optional path for the reconstructed image. -/
  reconstructPpm? : Option System.FilePath
deriving Repr

namespace ImageArtifactFlags

/--
Parse shared image artifact flags for generation/reconstruction commands.

This only parses artifact paths and the optional reconstruction timestep. The model still decides
which images it can write.
-/
def parse (args : List String) : Except String (ImageArtifactFlags × List String) := do
  let (reconstructStep?, args) ← TorchLean.CLI.takeNatFlagOnce args "reconstruct-step"
  let (samplePpm?, args) ← TorchLean.CLI.takePathFlagOnce args "sample-ppm"
  let (referencePpm?, args) ← TorchLean.CLI.takePathFlagOnce args "reference-ppm"
  let (noisyPpm?, args) ← TorchLean.CLI.takePathFlagOnce args "noisy-ppm"
  let (reconstructPpm?, args) ← TorchLean.CLI.takePathFlagOnce args "reconstruct-ppm"
  pure ({ reconstructStep? := reconstructStep?,
          samplePpm? := samplePpm?,
          referencePpm? := referencePpm?,
          noisyPpm? := noisyPpm?,
          reconstructPpm? := reconstructPpm? },
        args)

/-- Standard TrainLog metadata for requested image artifacts. -/
def trainLogNotes (cfg : ImageArtifactFlags) : Array String :=
  (match cfg.reconstructStep? with | none => #[] | some t => #[s!"reconstructStep={t}"]) ++
  (match cfg.samplePpm? with | none => #[] | some p => #[s!"samplePpm={p}"]) ++
  (match cfg.referencePpm? with | none => #[] | some p => #[s!"referencePpm={p}"]) ++
  (match cfg.noisyPpm? with | none => #[] | some p => #[s!"noisyPpm={p}"]) ++
  (match cfg.reconstructPpm? with | none => #[] | some p => #[s!"reconstructPpm={p}"])

end ImageArtifactFlags

/--
Common paired-NPY dataset flags for scientific supervised examples.

This shape is for commands that train on one `(x,y)` tensor pair and evaluate/report on a held-out
pair. The model file supplies the tensor shapes and file defaults.
-/
structure PairedNpyEvalFlags where
  /-- Number of rows loaded from the prepared training tensors. -/
  trainRows : Nat
  /-- Number of rows loaded from the prepared held-out tensors. -/
  testRows : Nat
  /-- Prefix length used for deterministic train/test loss reports. -/
  evalRows : Nat
  /-- Training input `.npy` path. -/
  trainX : System.FilePath
  /-- Training target `.npy` path. -/
  trainY : System.FilePath
  /-- Held-out input `.npy` path. -/
  testX : System.FilePath
  /-- Held-out target `.npy` path. -/
  testY : System.FilePath
deriving Repr

namespace PairedNpyEvalFlags

/--
Parse common train/test paired-NPY flags.

Commands supply their default paths and row counts. This parser keeps the repeated
`--train-rows`, `--test-rows`, `--eval-rows`, `--x`, `--y`, `--test-x`, and `--test-y` flags in one
place.
-/
def parse
    (args : List String)
    (defaultTrainX defaultTrainY defaultTestX defaultTestY : System.FilePath)
    (defaultTrainRows defaultTestRows : Nat)
    (defaultEvalRows : Nat := 16) :
    Except String (PairedNpyEvalFlags × List String) := do
  let (trainRows, args) ← TorchLean.CLI.takeNatFlagDefault args "train-rows" defaultTrainRows
  let (testRows, args) ← TorchLean.CLI.takeNatFlagDefault args "test-rows" defaultTestRows
  let (evalRows, args) ← TorchLean.CLI.takeNatFlagDefault args "eval-rows" defaultEvalRows
  let (trainX, args) ← TorchLean.CLI.takePathFlagDefault args "x" defaultTrainX
  let (trainY, args) ← TorchLean.CLI.takePathFlagDefault args "y" defaultTrainY
  let (testX, args) ← TorchLean.CLI.takePathFlagDefault args "test-x" defaultTestX
  let (testY, args) ← TorchLean.CLI.takePathFlagDefault args "test-y" defaultTestY
  pure ({ trainRows := trainRows,
          testRows := testRows,
          evalRows := evalRows,
          trainX := trainX,
          trainY := trainY,
          testX := testX,
          testY := testY },
        args)

/-- Standard TrainLog metadata for paired train/test NPY tensors. -/
def trainLogNotes (cfg : PairedNpyEvalFlags) : Array String :=
  #[
    s!"train_rows={cfg.trainRows}",
    s!"test_rows={cfg.testRows}",
    s!"eval_rows={cfg.evalRows}",
    s!"train_x={cfg.trainX}",
    s!"train_y={cfg.trainY}",
    s!"test_x={cfg.testX}",
    s!"test_y={cfg.testY}"
  ]

end PairedNpyEvalFlags

/-- Optional CSV artifact path for commands that emit one tabular diagnostic. -/
structure CsvArtifactFlags where
  /-- CSV path for the diagnostic artifact. -/
  plotCsv : System.FilePath
deriving Repr

namespace CsvArtifactFlags

/-- Parse the shared optional `--plot-csv` artifact path. -/
def parse
    (args : List String)
    (defaultPlotCsv : System.FilePath) :
    Except String (CsvArtifactFlags × List String) := do
  let (plotCsv, args) ← TorchLean.CLI.takePathFlagDefault args "plot-csv" defaultPlotCsv
  pure ({ plotCsv := plotCsv }, args)

end CsvArtifactFlags

/--
Generic NPY-backed labeled dataset flags.

Examples provide default paths and data-preparation hints. The repeated flags are `--seed`,
`--n-total`, `--x`, and `--y`.
-/
structure NpyDataFlags where
  /-- Prepared feature/image tensor path. -/
  xPath : System.FilePath
  /-- Prepared label/target tensor path. -/
  yPath : System.FilePath
  /-- Number of rows to read from the prepared arrays. -/
  nRows : Nat
  /-- Data-loader seed. -/
  seed : Nat
deriving Repr

namespace NpyDataFlags

/--
Parse the standard `--seed`, `--n-total`, `--x`, and `--y` flags for NPY-backed datasets.

Examples provide dataset-specific defaults; this parser keeps the repeated NPY dataset flags
consistent.
-/
def parse
    (args : List String)
    (defaultX defaultY : System.FilePath)
    (defaultRows : Nat) :
    Except String (NpyDataFlags × List String) := do
  let (seed, args) ← TorchLean.CLI.takeSeed args (default := 0)
  let (nRows, args) ← TorchLean.CLI.takeNatFlagDefault args "n-total" defaultRows
  let (xPath, args) ← TorchLean.CLI.takePathFlagDefault args "x" defaultX
  let (yPath, args) ← TorchLean.CLI.takePathFlagDefault args "y" defaultY
  pure ({ xPath := xPath, yPath := yPath, nRows := nRows, seed := seed }, args)

/-- Standard TrainLog metadata for an NPY-backed dataset branch. -/
def trainLogNotes (cfg : NpyDataFlags) (datasetName : String) : Array String :=
  #[s!"data={datasetName}", s!"x={cfg.xPath}", s!"y={cfg.yPath}", s!"nRows={cfg.nRows}"]

end NpyDataFlags

/-- Built-in image dataset branches shared by image-model commands. -/
inductive ImageDatasetChoice where
  /-- Prepared 64×64 RGB image tensors. -/
  | imagenet64
  /-- Prepared CIFAR-10 32×32 RGB tensors. -/
  | cifar10
deriving Repr, BEq

namespace ImageDatasetChoice

/-- Parse `--dataset`, `--cifar10`, and `--imagenet64`, rejecting ambiguous selectors. -/
def parse (args : List String) : Except String (ImageDatasetChoice × List String) := do
  let (dataset?, args) ← TorchLean.CLI.takeFlagValueOnce args "dataset"
  let (cifarFlag, args) ← TorchLean.CLI.takeBoolFlagOnce args "cifar10"
  let (imagenetFlag, args) ← TorchLean.CLI.takeBoolFlagOnce args "imagenet64"
  match dataset?, cifarFlag, imagenetFlag with
  | some _, true, _ | some _, _, true | none, true, true =>
      throw "choose only one dataset selector: --dataset, --cifar10, or --imagenet64"
  | some raw, false, false =>
      match raw with
      | "imagenet64" | "imagenet" | "imagenette64" => pure (.imagenet64, args)
      | "cifar10" | "cifar" => pure (.cifar10, args)
      | _ => throw s!"unknown --dataset {raw}; expected imagenet64 or cifar10"
  | none, true, false => pure (.cifar10, args)
  | none, false, true => pure (.imagenet64, args)
  | none, false, false => pure (.imagenet64, args)

end ImageDatasetChoice

/-- Shared forecasting-window data flags: paths, window count, report offset, and seed. -/
structure ForecastWindowDataFlags where
  /-- Prepared input-window tensor path. -/
  xPath : System.FilePath
  /-- Prepared target-window tensor path. -/
  yPath : System.FilePath
  /-- Number of forecasting windows to use. -/
  windows : Nat
  /-- Report window index used for before/after forecast display. -/
  reportOffset : Nat
  /-- Data-loader seed. -/
  seed : Nat
deriving Repr

namespace ForecastWindowDataFlags

/-- Standard TrainLog metadata for forecasting-window datasets. -/
def trainLogNotes (cfg : ForecastWindowDataFlags) : Array String :=
  #[
    s!"windows={cfg.windows}",
    s!"report_index={cfg.reportOffset}",
    s!"x={cfg.xPath}",
    s!"y={cfg.yPath}"
  ]

end ForecastWindowDataFlags

/-- Common arguments for a fixed-sample command backed by supervised `.npy` arrays. -/
structure NpyLoggedTrainFlags extends LoggedTrainFlags, NpyDataFlags where
deriving Repr

/--
Parse a supervised NPY dataset, logged training flags, and require that no command-specific
arguments remain.

Dataset-specific commands provide `parseData`, which owns defaults such as CIFAR or ImageNet paths;
this helper keeps the reusable "NPY data + logged TrainLog" path in one place.
-/
def parseNpyLoggedTrainFlags
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String NpyLoggedTrainFlags := do
  let (data, rest) ← parseData args
  let (train, rest) ← parseLoggedTrainFlags exeName rest defaultLogPath defaultSteps
  TorchLean.CLI.checkNoArgs rest
  pure { toLoggedTrainFlags := train, toNpyDataFlags := data }

/-- Common arguments for a model-training command backed by supervised `.npy` arrays. -/
structure NpyModelTrainFlags extends ModelTrainFlags, NpyDataFlags where
deriving Repr

/--
Parse a supervised NPY dataset and standard model-training flags.

The remaining arguments are returned so callers can still pass runtime/backend flags through a
higher-level runner.
-/
def parseNpyModelTrainFlags
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String (NpyModelTrainFlags × List String) := do
  let (data, rest) ← parseData args
  let (train, rest) ← parseModelTrainFlags exeName rest defaultLogPath defaultSteps defaultLr
  pure ({ toModelTrainFlags := train, toNpyDataFlags := data }, rest)

/-- Common arguments for forecasting-window model-training commands. -/
structure ForecastWindowModelTrainFlags extends ModelTrainFlags, ForecastWindowDataFlags where
deriving Repr

/--
Parse forecasting-window data flags plus standard model-training flags.

The data parser comes from the caller because file defaults often depend on a dataset directory or a
preparation script.
-/
def parseForecastWindowModelTrainFlags
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 100)
    (defaultLr : Float := 0.01)
    (parseData : List String → Except String (ForecastWindowDataFlags × List String)) :
    Except String (ForecastWindowModelTrainFlags × List String) := do
  let (data, rest) ← parseData args
  let (train, rest) ← parseModelTrainFlags exeName rest defaultLogPath defaultSteps defaultLr
  pure ({ toModelTrainFlags := train, toForecastWindowDataFlags := data }, rest)

/-- Common arguments for a model command that reads one supervised CSV. -/
structure CsvModelTrainFlags extends ModelTrainFlags where
  /-- CSV file containing model inputs and targets. -/
  csvPath : System.FilePath
  /-- Seed used for model initialization and data shuffling. -/
  seed : Nat
deriving Repr

/--
Parse common flags for a supervised CSV model runner.

Model files choose their default CSV, default log path, step count, and learning rate.
-/
def parseCsvModelTrainFlags (exeName : String) (args : List String)
    (defaultCsv : System.FilePath) (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (CsvModelTrainFlags × List String) := do
  let (csv?, args) ← TorchLean.CLI.takePathFlagOnce args "csv"
  let csvPath := csv?.getD defaultCsv
  let (seed, args) ← TorchLean.CLI.takeSeed args 0
  let (train, args) ← parseModelTrainFlags exeName args defaultLogPath defaultSteps defaultLr
    allowZeroSteps
  pure ({ toModelTrainFlags := train, csvPath := csvPath, seed := seed }, args)

/-- List generator: `[0, 1, ..., n-1]` mapped through `f`. -/
def listGen {α : Type} (n : Nat) (f : Nat → α) : List α :=
  (List.range n).map f

/--
Build an `N`-D tensor from a raw list of floats and cast it into the chosen scalar backend.

Fails if `xs.length ≠ numel(dims)`.
-/
def tensorF {α : Type} [Context α] (cast : Float → α) (dims : List Nat) (xs : List Float) :
    Except String (Spec.Tensor α (NN.Tensor.shapeOfDims dims)) := do
  let tF ← NN.Tensor.ofList (α := Float) dims xs
  pure (Spec.mapTensor cast tF)

/--
Generate an `N`-D tensor by calling `f` for each element index, then cast into the chosen backend.

The function `f` is indexed by the *flat* element index `0..numel-1`.
-/
def tensorFGen {α : Type} [Context α] (cast : Float → α) (dims : List Nat) (f : Nat → Float) :
    Except String (Spec.Tensor α (NN.Tensor.shapeOfDims dims)) :=
  let n := NN.Tensor.numelDims dims
  tensorF (α := α) cast dims (listGen n f)

/--
Generate an `N`-D tensor by calling `f` for each flat element index, with no failure case.

This is the “total” sibling of `tensorFGen`: since we generate exactly `numel(dims)` values,
the reshape cannot fail, so we avoid an `Except`.
When you want to build a deterministic constant tensor for an example, this is usually the
right tool.
-/
def tensorFGen! {α : Type} [Context α] (cast : Float → α) (dims : List Nat) (f : Nat → Float) :
    Spec.Tensor α (NN.Tensor.shapeOfDims dims) :=
  let xs : List Float := listGen (NN.Tensor.numelDims dims) f
  have hLen : xs.length = NN.Tensor.numelDims dims := by
    simp [xs, listGen]
  let tF : Spec.Tensor Float (NN.Tensor.shapeOfDims dims) :=
    NN.Tensor.ofListOfLength (α := Float) (dims := dims) (xs := xs) hLen
  Spec.mapTensor cast tF

/--
Generate a tensor of a known shape `s` by calling `f` for each flat element index, then cast into
the chosen backend.

This packages the standard shape-cast pattern used when a tensor is generated from flat indices:

```lean
let xDyn : Tensor α (shapeOfDims s.toList) := ...
let x : Tensor α s := by simpa using xDyn
```
-/
def tensorFGenShape! {α : Type} [Context α] (cast : Float → α) (s : Spec.Shape) (f : Nat → Float) :
    Spec.Tensor α s := by
  simpa [NN.Tensor.shapeOfDims_toList] using
    (tensorFGen! (α := α) cast (_root_.Spec.Shape.toList s) f)

/--
1D vector tensor constructor specialized to shape `Vec n`.

Fails if `xs.length ≠ n`.
-/
def vecF {α : Type} [Context α] (cast : Float → α) (n : Nat) (xs : List Float) :
    Except String (Spec.Tensor α (.dim n .scalar)) := do
  let t ← tensorF (α := α) cast [n] xs
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/-- Generator variant of `vecF`. -/
def vecFGen {α : Type} [Context α] (cast : Float → α) (n : Nat) (f : Nat → Float) :
    Except String (Spec.Tensor α (.dim n .scalar)) := do
  let t ← tensorFGen (α := α) cast [n] f
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/--
2D matrix tensor constructor specialized to shape `Mat rows cols`.

Fails if `xs.length ≠ rows * cols`.
-/
def matF {α : Type} [Context α] (cast : Float → α) (rows cols : Nat) (xs : List Float) :
    Except String (Spec.Tensor α (.dim rows (.dim cols .scalar))) := do
  let t ← tensorF (α := α) cast [rows, cols] xs
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/-- Generator variant of `matF`. -/
def matFGen {α : Type} [Context α] (cast : Float → α) (rows cols : Nat) (f : Nat → Float) :
    Except String (Spec.Tensor α (.dim rows (.dim cols .scalar))) := do
  let t ← tensorFGen (α := α) cast [rows, cols] f
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/--
Run a workflow once under a dtype selected from `args` (via `--dtype` / `--float32-mode`).

This logs the chosen dtype and then calls `k` with a cast function `Float → α` for the selected
scalar backend.

In particular:
- `--dtype=float` selects Lean's builtin `Float` (trusted semantics, executable),
- `--dtype=float32` selects TorchLean's verified IEEE32 executable semantics,
- `--dtype=complex` selects TorchLean's parametric complex scalar over Float32,
- `--dtype=real` selects `ℝ` (proof-only; errors at runtime).
-/
def runWithDType (title : String) (args : List String)
    (k :
      ∀ {α : Type},
        [API.Semantics.Scalar α] →
        [DecidableEq Spec.Shape] →
        [ToString α] →
        (Float → α) → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (dtype, _args') ←
    match NN.API.DType.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  NN.API.DType.log dtype
  match (← NN.API.DType.withExec dtype (fun {α} _ _ _ cast => k (α := α) cast)) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/--
Like `runWithDType`, but also provides an `API.Runtime.Scalar α` instance.

Use this when your workflow uses numeric literals (`1.0`, `-3.5`, etc.) at runtime.
-/
def runWithRuntimeDType (title : String) (args : List String)
    (k : ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
      [API.Runtime.Scalar α] → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (dtype, _args') ←
    match NN.API.DType.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  NN.API.DType.log dtype
  match (← NN.API.DType.withRuntime dtype (fun {α} _ _ _ _ => k (α := α))) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

end Common

end API
end NN
