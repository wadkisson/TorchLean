/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core

/-!
# Generated Tutorial Dataset Paths

TorchLean tutorials can generate deterministic sample datasets under `NN/Examples/Data/`.
This module centralizes:

- the default data directory,
- the file paths of the generated sample datasets, and
- a CLI helper for overriding the directory via `--data-dir`.

Keeping these paths in one module keeps tutorial code from hardcoding filenames or reimplementing
the same flag parsing in each example.
-/

@[expose] public section

namespace NN
namespace Examples
namespace Data
namespace SamplePaths

/-- Default relative directory containing generated tutorial datasets. -/
def defaultDataDir : System.FilePath :=
  "NN/Examples/Data"

/--
Parse an optional `--data-dir PATH` flag (defaults to `defaultDataDir`).

Tutorials use this when they load generated CSV/NPY files from disk.
-/
def takeDataDir (args : List String) (default : System.FilePath := defaultDataDir) :
    Except String (System.FilePath × List String) := do
  let (dir?, rest) ← NN.API.CLI.takePathFlagOnce args "data-dir"
  pure (dir?.getD default, rest)

/-- `small_regression.csv` (2D regression). -/
def regressionCsv (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_regression.csv"

/-- `small_regression_X.npy` (shape 25x2). -/
def regressionXNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_regression_X.npy"

/-- `small_regression_y.npy` (shape 25x1). -/
def regressionYNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_regression_y.npy"

/-- `small_cifar10like_X.npy` (shape 200x3x32x32). -/
def cifar10likeXNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_cifar10like_X.npy"

/-- `small_cifar10like_y.npy` (shape 200). -/
def cifar10likeYNpy (dataDir : System.FilePath := defaultDataDir) : System.FilePath :=
  dataDir / "small_cifar10like_y.npy"

end SamplePaths
end Data
end Examples
end NN
