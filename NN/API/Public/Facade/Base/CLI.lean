/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Runtime
public import NN.API.Models
public import NN.API.Public.NN.Transformer
public import NN.API.RL
public import NN.API.Rand
public import NN.API.Samples.Bands
public import NN.API.Text.Bpe
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean CLI Facade

Small command-line parsers used by examples and model-zoo commands.
-/

@[expose] public section

namespace TorchLean

namespace CLI

/-!
Small public command-line parsers for examples.

Use these when a runnable example needs the same compact flag convention as the model zoo.
-/

/-- Drop Lake's `--` separator when present. -/
abbrev dropDashDash := NN.API.CLI.dropDashDash

/-- Return true when the argument list requests command help. -/
abbrev hasHelp := NN.API.CLI.hasHelp

/-- Parse `--seed N`, returning the selected seed and remaining arguments. -/
def seed (exeName : String) (args : List String) (default : Nat := 0) :
    IO (Nat × List String) :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.takeSeed args default

/-- Parse an optional natural-number flag such as `--steps 200`. -/
def natFlag? (exeName : String) (args : List String) (name : String) :
    IO (Option Nat × List String) :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.takeNatFlagOnce args name

/-- Parse an optional natural-number flag and fall back to a default. -/
def natFlagDefault
    (exeName : String)
    (args : List String)
    (name : String)
    (default : Nat) :
    IO (Nat × List String) :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.takeNatFlagDefault args name default

/--
Parse an optional natural-number flag, fall back to a default, and require that the selected value
is strictly positive.
-/
def positiveNatFlag
    (exeName : String)
    (args : List String)
    (name : String)
    (default : Nat) :
    IO (Nat × List String) := do
  NN.API.Common.orThrow exeName <| NN.API.CLI.takePositiveNatFlagDefault args exeName name default

/-- Parse an optional path flag such as `--csv data.csv`. -/
def pathFlag? (exeName : String) (args : List String) (name : String) :
    IO (Option System.FilePath × List String) :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.takePathFlagOnce args name

/-- Parse an optional float flag and fall back to a default. -/
def floatFlagDefault
    (exeName : String)
    (args : List String)
    (name : String)
    (default : Float) :
    IO (Float × List String) :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.takeFloatFlagDefault args name default

/-- Parse an optional path flag and fall back to a default path. -/
def pathFlagDefault
    (exeName : String)
    (args : List String)
    (name : String)
    (default : System.FilePath) :
    IO (System.FilePath × List String) :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.takePathFlagDefault args name default

/--
Parse the `--epochs E --batch N` pair used by the small epoch-oriented tutorial commands.

Step-based model-zoo training commands use the `ModelZoo` / `Trainer.TrainOptions` path instead.
-/
def epochBatch (exeName : String) (args : List String) (defaultEpochs defaultBatch : Nat) :
    IO (NN.API.CLI.EpochBatch × List String) :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.takeEpochBatch args defaultEpochs defaultBatch

/-- Parser used by model-zoo commands that build an `Except` parser first. -/
abbrev takeEpochBatch := NN.API.CLI.takeEpochBatch

/-- Parser used by epoch-oriented tutorial commands that require positive `--epochs` and `--batch`. -/
abbrev takePositiveEpochBatch := NN.API.CLI.takePositiveEpochBatch

/-- Convert an `Except String α` into `IO α` with a shared executable name. -/
def orThrow {α : Type} (exeName : String) (result : Except String α) : IO α :=
  NN.API.Common.orThrow exeName result

/-- Require that no example-specific arguments remain after parsing. -/
def requireNoArgs (exeName : String) (args : List String) : IO Unit :=
  NN.API.Common.orThrow exeName <| NN.API.CLI.requireNoArgs args

/-- Parser used by model-zoo commands that build an `Except` parser first. -/
def takeSeed (args : List String) (default : Nat := 0) : Except String (Nat × List String) :=
  NN.API.CLI.takeSeed args default

/-- Parser used by model-zoo commands that build an `Except` parser first. -/
abbrev takeNatFlagOnce := NN.API.CLI.takeNatFlagOnce

/-- Parser used by model-zoo commands that want a natural-number flag with a default. -/
abbrev takeNatFlagDefault := NN.API.CLI.takeNatFlagDefault

/-- Parser used by model-zoo commands that want a positive natural-number flag. -/
abbrev takePositiveNatFlagDefault := NN.API.CLI.takePositiveNatFlagDefault

/-- Low-level parser used by model-zoo commands that build an `Except` parser first. -/
abbrev takeFloatFlagOnce := NN.API.CLI.takeFloatFlagOnce

/-- Low-level parser used by model-zoo commands that want a float flag with a default. -/
abbrev takeFloatFlagDefault := NN.API.CLI.takeFloatFlagDefault

/-- Low-level parser used by model-zoo commands that require a positive float flag. -/
abbrev takePositiveFloatFlagDefault := NN.API.CLI.takePositiveFloatFlagDefault

/-- Low-level parser used by model-zoo commands that require a nonnegative float flag. -/
abbrev takeNonnegativeFloatFlagDefault := NN.API.CLI.takeNonnegativeFloatFlagDefault

/-- Low-level parser used by model-zoo commands that build an `Except` parser first. -/
abbrev takeFlagValueOnce := NN.API.CLI.takeFlagValueOnce

/-- Low-level parser used by model-zoo commands that want a concrete string-valued flag. -/
abbrev takeFlagValueDefault := NN.API.CLI.takeFlagValueDefault

/-- Low-level parser used by model-zoo commands whose string flag is decoded by a custom parser. -/
def takeParsedFlagDefault {α : Type}
    (args : List String)
    (key : String)
    (default : String)
    (parse : String → Except String α) :
    Except String (α × List String) :=
  NN.API.CLI.takeParsedFlagDefault args key default parse

/-- Low-level parser used by model-zoo commands that build an `Except` parser first. -/
abbrev takePathFlagOnce := NN.API.CLI.takePathFlagOnce

/-- Low-level parser used by model-zoo commands that want a concrete path with a default. -/
abbrev takePathFlagDefault := NN.API.CLI.takePathFlagDefault

/-- Low-level parser used by model-zoo commands that require a path flag to be present. -/
def takeRequiredPathFlag
    (args : List String)
    (key : String)
    (exeName : String := "") :
    Except String (System.FilePath × List String) :=
  NN.API.CLI.takeRequiredPathFlag args key exeName

/-- Low-level parser used by model-zoo commands whose two path flags must appear together. -/
abbrev takePairedPathFlags := NN.API.CLI.takePairedPathFlags

/-- Low-level parser used by model-zoo commands that build an `Except` parser first. -/
abbrev takeBoolFlagOnce := NN.API.CLI.takeBoolFlagOnce

/-- Low-level no-extra-arguments check for parsers that stay in `Except`. -/
abbrev checkNoArgs := NN.API.CLI.requireNoArgs

/-- Check whether a flag appears with an attached value such as `--log out.json`. -/
abbrev hasFlagValue := NN.API.CLI.hasFlagValue

/- Parse an optional `--steps N` flag, falling back to the provided default. -/
abbrev takeStepsFlagDefault := NN.API.CLI.takeStepsFlagDefault

end CLI


end TorchLean
