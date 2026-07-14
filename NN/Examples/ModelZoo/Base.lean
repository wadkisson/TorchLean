/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Common
public import NN.API.Public.Facade.Base.Root

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

@[inherit_doc NN.API.Common.DiffusionScheduleFlags]
abbrev DiffusionScheduleFlags := NN.API.Common.DiffusionScheduleFlags

@[inherit_doc NN.API.Common.ImageArtifactFlags]
abbrev ImageArtifactFlags := NN.API.Common.ImageArtifactFlags

@[inherit_doc NN.API.Common.PairedNpyEvalFlags]
abbrev PairedNpyEvalFlags := NN.API.Common.PairedNpyEvalFlags

@[inherit_doc NN.API.Common.CsvArtifactFlags]
abbrev CsvArtifactFlags := NN.API.Common.CsvArtifactFlags

@[inherit_doc NN.API.Common.NpyDataFlags]
abbrev NpyDataFlags := NN.API.Common.NpyDataFlags

@[inherit_doc NN.API.Common.ImageDatasetChoice]
abbrev ImageDatasetChoice := NN.API.Common.ImageDatasetChoice

@[inherit_doc NN.API.Common.NpyLoggedTrainFlags]
abbrev NpyLoggedTrainFlags := NN.API.Common.NpyLoggedTrainFlags

@[inherit_doc NN.API.Common.NpyModelTrainFlags]
abbrev NpyModelTrainFlags := NN.API.Common.NpyModelTrainFlags

@[inherit_doc NN.API.Common.ForecastWindowDataFlags]
abbrev ForecastWindowDataFlags := NN.API.Common.ForecastWindowDataFlags

@[inherit_doc NN.API.Common.ForecastWindowModelTrainFlags]
abbrev ForecastWindowModelTrainFlags := NN.API.Common.ForecastWindowModelTrainFlags

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

@[inherit_doc NN.API.Common.SeededModelTrainFlags]
abbrev SeededTrainFlags := NN.API.Common.SeededModelTrainFlags

@[inherit_doc NN.API.Common.CsvModelTrainFlags]
abbrev CsvTrainFlags := NN.API.Common.CsvModelTrainFlags

namespace WindowOptions

@[inherit_doc NN.API.Common.WindowOptions.parse]
abbrev parse := NN.API.Common.WindowOptions.parse

end WindowOptions

namespace CheckpointOptions

@[inherit_doc NN.API.Common.CheckpointOptions.parse]
abbrev parse := NN.API.Common.CheckpointOptions.parse

end CheckpointOptions

namespace DiffusionScheduleFlags

@[inherit_doc NN.API.Common.DiffusionScheduleFlags.parse]
abbrev parse := NN.API.Common.DiffusionScheduleFlags.parse

@[inherit_doc NN.API.Common.DiffusionScheduleFlags.trainLogNotes]
abbrev trainLogNotes := NN.API.Common.DiffusionScheduleFlags.trainLogNotes

end DiffusionScheduleFlags

namespace ImageArtifactFlags

@[inherit_doc NN.API.Common.ImageArtifactFlags.parse]
abbrev parse := NN.API.Common.ImageArtifactFlags.parse

@[inherit_doc NN.API.Common.ImageArtifactFlags.trainLogNotes]
abbrev trainLogNotes := NN.API.Common.ImageArtifactFlags.trainLogNotes

end ImageArtifactFlags

namespace PairedNpyEvalFlags

@[inherit_doc NN.API.Common.PairedNpyEvalFlags.parse]
abbrev parse := NN.API.Common.PairedNpyEvalFlags.parse

@[inherit_doc NN.API.Common.PairedNpyEvalFlags.trainLogNotes]
abbrev trainLogNotes := NN.API.Common.PairedNpyEvalFlags.trainLogNotes

end PairedNpyEvalFlags

namespace CsvArtifactFlags

@[inherit_doc NN.API.Common.CsvArtifactFlags.parse]
abbrev parse := NN.API.Common.CsvArtifactFlags.parse

end CsvArtifactFlags

namespace NpyDataFlags

@[inherit_doc NN.API.Common.NpyDataFlags.parse]
abbrev parse := NN.API.Common.NpyDataFlags.parse

@[inherit_doc NN.API.Common.NpyDataFlags.trainLogNotes]
abbrev trainLogNotes := NN.API.Common.NpyDataFlags.trainLogNotes

end NpyDataFlags

namespace ImageDatasetChoice

@[inherit_doc NN.API.Common.ImageDatasetChoice.parse]
abbrev parse := NN.API.Common.ImageDatasetChoice.parse

end ImageDatasetChoice

namespace NpyLoggedTrainFlags

@[inherit_doc NN.API.Common.parseNpyLoggedTrainFlags]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String NpyLoggedTrainFlags :=
  NN.API.Common.parseNpyLoggedTrainFlags exeName args defaultLogPath defaultSteps parseData

end NpyLoggedTrainFlags

namespace NpyModelTrainFlags

@[inherit_doc NN.API.Common.parseNpyModelTrainFlags]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String (NpyModelTrainFlags × List String) :=
  NN.API.Common.parseNpyModelTrainFlags exeName args defaultLogPath defaultSteps defaultLr parseData

end NpyModelTrainFlags

namespace ForecastWindowDataFlags

@[inherit_doc NN.API.Common.ForecastWindowDataFlags.trainLogNotes]
abbrev trainLogNotes := NN.API.Common.ForecastWindowDataFlags.trainLogNotes

end ForecastWindowDataFlags

namespace ForecastWindowModelTrainFlags

@[inherit_doc NN.API.Common.parseForecastWindowModelTrainFlags]
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 100)
    (defaultLr : Float := 0.01)
    (parseData : List String → Except String (ForecastWindowDataFlags × List String)) :
    Except String (ForecastWindowModelTrainFlags × List String) :=
  NN.API.Common.parseForecastWindowModelTrainFlags exeName args defaultLogPath defaultSteps
    defaultLr parseData

end ForecastWindowModelTrainFlags

@[inherit_doc NN.API.Common.modelZooTrainLog]
abbrev trainLogPath := NN.API.Common.modelZooTrainLog

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
