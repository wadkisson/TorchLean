/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Base.Root
public import NN.Examples.ModelZoo.Options

/-!
# TorchLean Model-Zoo Helpers

Shared flags, logs, banners, paths, and runtime helpers for the built-in model-zoo examples.
-/

@[expose] public section

namespace NN.Examples.ModelZoo

open TorchLean

@[inherit_doc NN.API.Common.LoggedTrainFlags]
abbrev LoggedTrainFlags := NN.API.Common.LoggedTrainFlags

@[inherit_doc NN.API.Common.WindowOptions]
abbrev WindowOptions := NN.API.Common.WindowOptions

@[inherit_doc NN.API.Common.CheckpointOptions]
abbrev CheckpointOptions := NN.API.Common.CheckpointOptions

@[inherit_doc NN.API.Common.requirePositiveNatFlag]
abbrev requirePositiveNatFlag := NN.API.Common.requirePositiveNatFlag

@[inherit_doc NN.API.Common.resolvePositiveNatFlag]
abbrev resolvePositiveNatFlag := NN.API.Common.resolvePositiveNatFlag

@[inherit_doc NN.API.Common.effectiveCudaMemWatch]
abbrev effectiveCudaMemWatch := NN.API.Common.effectiveCudaMemWatch

@[inherit_doc NN.API.Common.cudaMemWatchNote]
abbrev cudaMemWatchNote := NN.API.Common.cudaMemWatchNote

@[inherit_doc NN.API.Common.reportCudaMemWatch]
abbrev reportCudaMemWatch := NN.API.Common.reportCudaMemWatch

@[inherit_doc NN.API.Common.shouldLogStep]
abbrev shouldLogStep := NN.API.Common.shouldLogStep

@[inherit_doc NN.API.Common.printCurveLossSummary]
abbrev printCurveLossSummary := NN.API.Common.printCurveLossSummary

/-!
Shared CLI and logging names for the built-in model-zoo examples.

These are public so examples can stay short and readable. They live under `ModelZoo`; ordinary
library code should usually use `Trainer`, `Data`, and `optim` directly.
-/

@[inherit_doc NN.API.Common.ModelTrainFlags]
abbrev TrainFlags := NN.API.Common.ModelTrainFlags

namespace WindowOptions

@[inherit_doc NN.API.Common.WindowOptions.parse]
abbrev parse := NN.API.Common.WindowOptions.parse

end WindowOptions

namespace CheckpointOptions

@[inherit_doc NN.API.Common.CheckpointOptions.parse]
abbrev parse := NN.API.Common.CheckpointOptions.parse

end CheckpointOptions

/-- Standard location for a model-example training log under `data/model_zoo`. -/
def trainLogPath (stem : String) : System.FilePath :=
  System.FilePath.mk s!"data/model_zoo/{stem}_trainlog.json"

@[inherit_doc NN.API.Common.orThrow]
def orThrow {α : Type} (exeName : String) (result : Except String α) : IO α :=
  NN.API.Common.orThrow exeName result

/-- Runtime device label used by example banners and notes. -/
def deviceName (opts : Options) : String :=
  opts.deviceName

/-- `device=...` note string used by example logs. -/
def deviceNote (opts : Options) : String :=
  s!"device={deviceName opts}"

/-- Model-zoo banner with the executable name, a short description, and the selected device. -/
def bannerWithDevice (exeName desc : String) (opts : Options) : String :=
  s!"{exeName}: {desc} (device={deviceName opts})"

/--
Two-line model-zoo banner: a headline with the selected device, then one detail line.
-/
def bannerWithDeviceDetails
    (exeName desc details : String) (opts : Options) : String :=
  bannerWithDevice exeName desc opts ++ "\n" ++ details

@[inherit_doc NN.API.Common.check]
def check (exeName msg : String) (b : Bool) : IO Unit :=
  NN.API.Common.check exeName msg b

@[inherit_doc NN.API.Common.writeBeforeAfterLossLog]
def writeBeforeAfterLossLogPath (path : System.FilePath)
    (title : String) (steps : Nat) (beforeLoss afterLoss : Float) (notes : Array String := #[]) :
    IO Unit :=
  NN.API.Common.writeBeforeAfterLossLog path title steps beforeLoss afterLoss notes

@[inherit_doc NN.API.Common.writeBeforeAfterLossLogTo]
def writeBeforeAfterLossLog
    (dest : Training.LogDestination)
    (title : String) (steps : Nat) (beforeLoss afterLoss : Float) (notes : Array String := #[]) :
    IO Unit :=
  NN.API.Common.writeBeforeAfterLossLogTo dest title steps beforeLoss afterLoss notes

@[inherit_doc NN.API.Common.writeCurveLogTo]
def writeCurveLog
    (dest : Training.LogDestination)
    (title : String) (curve : Training.Curve)
    (seriesName : String := "loss") (notes : Array String := #[]) :
    IO Unit :=
  NN.API.Common.writeCurveLogTo dest title curve seriesName notes

/--
Write a single-curve training log with an explicit series color.

Use this when a command already has a `Training.Curve` and wants a `TrainLog` instead of the default
`"loss"` curve writer.
-/
def writeCurveTrainLog
    (dest : Training.LogDestination)
    (title : String)
    (curve : Training.Curve)
    (seriesName : String)
    (color : String := "#4e79a7")
    (notes : Array String := #[]) :
    IO Unit :=
  NN.API.Common.writeTrainLogTo dest (curve.toTrainLog title seriesName color notes)

/--
Write a multi-series metric history as a TrainLog artifact.

Use this when a command has a `Training.MetricHistory` with named, colored series and wants to write
the usual `TrainLog` artifact without repeating the conversion code in every example.
-/
def writeMetricHistoryLog
    (dest : Training.LogDestination)
    (title : String)
    (history : Training.MetricHistory)
    (notes : Array String := #[]) :
    IO Unit :=
  NN.API.Common.writeTrainLogTo dest (history.toTrainLog (title := title) (notes := notes))

@[inherit_doc NN.API.Common.writeTrainLog]
def writeTrainLogPath (path : System.FilePath) (log : Training.TrainLog) :
    IO Unit :=
  NN.API.Common.writeTrainLog path log

@[inherit_doc NN.API.Common.writeTrainLogTo]
def writeTrainLog
    (dest : Training.LogDestination)
    (log : Training.TrainLog) :
    IO Unit :=
  NN.API.Common.writeTrainLogTo dest log

@[inherit_doc NN.API.Models.TrainFixed.curveFloat]
def trainFixedCurveFloat
    {σ τ : Shape}
    (mkModel : NN.API.nn.M (NN.API.nn.Sequential σ τ))
    (mkModuleDef :
      (model : NN.API.nn.Sequential σ τ) →
        NN.API.TorchLean.Module.ScalarModuleDef (NN.API.nn.paramShapes model) [σ, τ])
    (mkOptim :
      (paramShapes : List Shape) → NN.API.TorchLean.Optim.Optimizer Float paramShapes)
    (opts : Options)
    (sample : SupervisedSample Float σ τ)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO Training.Curve :=
  NN.API.Models.TrainFixed.curveFloat
    (mkModel := mkModel)
    (mkModuleDef := mkModuleDef)
    (mkOptim := mkOptim)
    (opts := opts)
    (sample := sample)
    (steps := steps)
    (cudaMemWatch := cudaMemWatch)

end NN.Examples.ModelZoo
