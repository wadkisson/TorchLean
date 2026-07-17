/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Common
public import NN.API.Public.Facade.Base.Root

/-!
# TorchLean CLI Helpers

Small command-line parsers used by examples and model-zoo commands.
-/

@[expose] public section

namespace TorchLean

namespace CLI

/-!
Small public parsers for example commands.

Use these when a runnable example wants the same compact flag style as the model zoo.
-/

/-- Parse `--seed N`, returning the selected seed and remaining arguments. -/
def seed (exeName : String) (args : List String) (default : Nat := 0) :
    IO (Nat × List String) :=
  NN.API.Common.orThrow exeName <| TorchLean.CLI.takeSeed args default

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
  NN.API.Common.orThrow exeName <| TorchLean.CLI.takePositiveNatFlagDefault args exeName name default

/-- Parse an optional path flag and fall back to a default path. -/
def pathFlagDefault
    (exeName : String)
    (args : List String)
    (name : String)
    (default : System.FilePath) :
    IO (System.FilePath × List String) :=
  NN.API.Common.orThrow exeName <| TorchLean.CLI.takePathFlagDefault args name default

/-- Convert an `Except String α` into `IO α` with a shared executable name. -/
def orThrow {α : Type} (exeName : String) (result : Except String α) : IO α :=
  NN.API.Common.orThrow exeName result

/-- Require that no example-specific arguments remain after parsing. -/
def requireNoArgs (exeName : String) (args : List String) : IO Unit :=
  NN.API.Common.orThrow exeName <| TorchLean.CLI.checkNoArgs args

end CLI


end TorchLean
