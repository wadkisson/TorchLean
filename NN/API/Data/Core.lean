/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Core
public import NN.API.Macros
public import NN.API.Public.TensorPack
public import NN.Runtime.Autograd.Train.IoLoader

import Mathlib.Algebra.Order.Algebra
import Mathlib.Data.List.Basic

/-!
# Datasets, Loaders, and File Sources

This module is TorchLean's public data layer. The intended workflow is:

1. Convert outside-world datasets to canonical `.npy` tensors or small numeric CSV files.
2. Describe those files with `TensorSource`, `SupervisedSource`, or `LabeledSource`.
3. Load them into shape-typed TorchLean tensors and datasets.
4. Train with `batchLoader` / `BatchLoader.epoch`, the public `trainer.train` path, or
   `TorchLean.Trainer.trainDataset` when an manual runner loop is still the right tool.

We keep the implementation small and predictable:
- datasets are in-memory and pure (often backed by `List`)
- loader shuffling is seed-driven and reproducible
- `.npy` is the canonical numeric interchange format
- CSV is supported for small tabular data
- MATLAB `.mat`, PyTorch `.pt/.pth`, NumPy `.npz`, and image folders should be converted to
  `.npy` with `scripts/datasets/torchlean_data_convert.py`
- there are no multiprocessing workers, memory maps, or pinned-memory support

## PyTorch Mapping

This is inspired by `torch.utils.data`:
- Dataset, DataLoader: `https://pytorch.org/docs/stable/data.html`
- TensorDataset: `https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset`
- DataLoader: `https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader`

TorchLean’s key difference is that samples typically carry *type-level shapes* (via `TensorPack`),
so many helpers here are shape-aware by construction.

## Main Entry Points

- `TensorSource`: one file plus expected dimensions.
- `SupervisedSource`: two batched tensors, `X : (N, xDims...)` and `Y : (N, yDims...)`.
- `LabeledSource`: batched inputs plus integer class labels, one-hot encoded on load.
- `TabularSupervisedSource`: one CSV table with input columns followed by target columns.
- `batchLoader`: deterministic, typed minibatching.

For examples and conversion commands, see `NN/Examples/Data/README.md`.
-/

@[expose] public section


namespace NN
namespace API

namespace Data

export _root_.Runtime.Autograd.Train
  (Dataset CsvOptions
   readCsvFloatRows readCsvDatasetPairs readCsvVectorDataset
   readNpy readNpyLeadingAxisPrefix readNpyVector readNpyMatrix
   vectorOfList vectorOfArray matrixOfLists matrixOfArrays
   datasetOfPairs datasetOfListVectors)
/--
Typed analogue of PyTorch's `TensorDataset`.

In TorchLean, a sample is usually a `TensorPack α shapes`, i.e. a shape-tracked tuple of tensors.
-/
abbrev TensorDataset (α : Type) (shapes : List Spec.Shape) :=
  _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α shapes)

/-- Build a dataset from an explicit list of samples. -/
def fromList {a : Type} (xs : List a) : _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.ofList xs

/-- Require that all paths exist, otherwise raise a user-facing error with a shared hint. -/
def requireFiles (exeName : String) (paths : List System.FilePath) (hint : String := "") :
    IO Unit := do
  for p in paths do
    unless (← p.pathExists) do
      let suffix := if hint.isEmpty then "" else "\n" ++ hint
      throw <| IO.userError s!"{exeName}: missing data file: {p}{suffix}"

/-- Require one named data file to exist. -/
def requireFile
    (exeName : String) (label : String) (path : System.FilePath) (hint : String := "") :
    IO Unit := do
  unless (← path.pathExists) do
    let suffix := if hint.isEmpty then "" else "\n" ++ hint
    throw <| IO.userError s!"{exeName}: missing {label}: {path}{suffix}"

/-- Require paired supervised input/target files to exist. -/
def requirePairedFiles
    (exeName : String)
    (xLabel : String) (xPath : System.FilePath)
    (yLabel : String) (yPath : System.FilePath)
    (hint : String := "") : IO Unit := do
  requireFile exeName xLabel xPath hint
  requireFile exeName yLabel yPath hint

/-- Write a small CSV file, creating the parent directory if needed. -/
def writeCsv (path : System.FilePath) (header : List String) (rows : List (List String)) :
    IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let lines := String.intercalate "\n" ((String.intercalate "," header) :: rows.map
    (String.intercalate ",")) ++ "\n"
  IO.FS.writeFile path lines

/--
Write a one-dimensional prediction probe CSV.

Rows are `i,x,input,target,prediction`, where `x = i/(n-1)` for `n > 1`.
This writes the compact prediction table used by plotting examples such as 1D operator learning.
-/
def writeVectorPredictionCsv {n : Nat}
    (path : System.FilePath)
    (input target prediction : Spec.Tensor Float (NN.Tensor.Shape.Vec n)) : IO Unit := do
  let rows := (List.finRange n).map (fun i =>
    let denom := Float.ofNat (Nat.max 1 (n - 1))
    let xpos := Float.ofNat i.val / denom
    [toString i.val, toString xpos, toString (Spec.Tensor.vecGet input i),
      toString (Spec.Tensor.vecGet target i), toString (Spec.Tensor.vecGet prediction i)])
  writeCsv path ["i", "x", "input", "target", "prediction"] rows

/-- Materialize a dataset as a list. -/
def toList {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) : List a :=
  _root_.Runtime.Autograd.Train.Dataset.toList ds

/-- Converting a list to a dataset and back yields the original list. -/
@[simp] theorem toList_fromList {a : Type} (xs : List a) : toList (fromList xs) = xs := by
  simp [toList, fromList,
    _root_.Runtime.Autograd.Train.Dataset.toList,
    _root_.Runtime.Autograd.Train.Dataset.ofList]

/-- Number of elements in the dataset. -/
def size {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) : Nat :=
  _root_.Runtime.Autograd.Train.Dataset.size ds

/-- The size of a dataset built from a list is the list length. -/
@[simp] theorem size_fromList {a : Type} (xs : List a) : size (fromList xs) = xs.length := by
  simp [size, fromList, _root_.Runtime.Autograd.Train.Dataset.size,
    _root_.Runtime.Autograd.Train.Dataset.ofList]

/-- Whether the dataset is empty. -/
def isEmpty {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) : Bool :=
  _root_.Runtime.Autograd.Train.Dataset.isEmpty ds

/--
Build a cycling index function for a *nonempty* list.

`cycleList xs h i` returns `xs[i % xs.length]`.

This is useful for in-memory datasets where a fixed-step “PyTorch-like” loop should avoid repeated
`Option` handling.
-/
def cycleList {a : Type} (xs : List a) (h : xs ≠ []) : Nat → a :=
  fun i =>
    have hlen : 0 < xs.length := by
      simpa using List.length_pos_of_ne_nil h
    xs[i % xs.length]'(Nat.mod_lt _ hlen)

/--
Like `cycleList`, but fail with a message if the list is empty.

Fixed-step dataset code can check emptiness once and then index without `Option`.
-/
def cycleListOrError {a : Type} (xs : List a) (err : String := "empty list") : Except String (Nat →
  a) :=
  match xs with
  | [] => .error err
  | x :: xs => .ok (cycleList (x :: xs) (by simp))

/--
Build a cycling index function for a *nonempty* dataset.

`cycleDataset ds h i` returns `ds[i % ds.size]`.

This is the dataset analogue of `cycleList`. It avoids per-step `Option` handling in fixed-step
training loops.
-/
def cycleDataset {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) (h : ds.data.size ≠ 0) :
  Nat → a :=
  fun i =>
    let n := ds.data.size
    have hn : 0 < n :=
      Nat.pos_of_ne_zero (by simpa [n] using h)
    ds.data[i % n]'(Nat.mod_lt _ hn)

/--
Like `cycleDataset`, but fail with a message if the dataset is empty.

This is the preferred helper for “PyTorch-style” fixed-step loops over in-memory datasets.
-/
def cycleDatasetOrError {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a)
    (err : String := "empty dataset") : Except String (Nat → a) :=
  match h : ds.data.size with
  | 0 => .error err
  | n + 1 =>
      .ok (cycleDataset ds (by
        -- In this branch, `ds.data.size = n+1`, so it is nonzero.
        simp [h]))

/-- Safe indexing into a dataset. -/
def get? {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) (i : Nat) : Option a :=
  _root_.Runtime.Autograd.Train.Dataset.get? ds i

/-- Return the first array element, or a caller-provided error when the array is empty. -/
def firstArrayOrError {a : Type} (xs : Array a) (err : String := "empty array") : Except String a :=
  match xs[0]? with
  | some x => .ok x
  | none => .error err

/-- Map a dataset elementwise (pure, deterministic). -/
def map {a b : Type} (f : a → b) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset b :=
  _root_.Runtime.Autograd.Train.Dataset.map f ds

/-- Append two datasets, preserving order: all samples from `x` followed by all samples from `y`. -/
def append {a : Type} (x y : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.append x y

/-- Split a dataset at position `n` (prefix, suffix). -/
def splitAt {a : Type} (n : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset a × _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.splitAt n ds

/--
Shuffle a dataset deterministically, returning the updated RNG seed and the shuffled dataset.

This is used to implement `DataLoader.shuffle` behavior in a purely functional way.
-/
def shuffle {a : Type} (seed : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    Nat × _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.shuffle seed ds

/-- Deterministically shuffle a dataset when the caller does not need the updated seed. -/
def shuffled {a : Type} (seed : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset a :=
  (shuffle seed ds).snd

/--
Shuffle once and then split at `n`.

This is a small building block for train/val splits.
-/
def randomSplitAt {a : Type} (seed : Nat) (n : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    Nat × (_root_.Runtime.Autograd.Train.Dataset a × _root_.Runtime.Autograd.Train.Dataset a) :=
  let (seed', ds') := shuffle seed ds
  (seed', splitAt n ds')

/--
Split a dataset into equal-sized minibatches (as lists), dropping the final partial batch.

This is a low-level helper; ordinary loader code should use `DataLoader.epoch` or
`Data.batchedSupervised`.
-/
def batches {a : Type} (tag : String) (batchSize : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset
  a) :
    Except String (List (List a)) :=
  _root_.Runtime.Autograd.Train.Dataset.batches tag batchSize ds

/-- Like `batches`, but return each minibatch as an `Array` instead of a `List`. -/
def batchesArray {a : Type} (tag : String) (batchSize : Nat) (ds :
  _root_.Runtime.Autograd.Train.Dataset a) :
    Except String (List (Array a)) :=
  _root_.Runtime.Autograd.Train.Dataset.batchesArray tag batchSize ds

/--
Untyped analogue of PyTorch's `torch.utils.data.DataLoader`.

This is the deterministic, purely-functional loader provided by the TorchLean runtime.
-/
abbrev RawDataLoader (a : Type) :=
  _root_.Runtime.Autograd.Train.DataLoader a

/--
Construct a `RawDataLoader` from a dataset.

If `shuffle := true`, shuffling is deterministic w.r.t. `seed`.
If `dropLast := true`, incomplete final batches are discarded.
-/
def loader {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a)
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) (dropLast : Bool := false) :
    RawDataLoader a :=
  { dataset := ds, batchSize := batchSize, shuffle := shuffle, seed := seed, dropLast := dropLast }

/--
Run one epoch worth of minibatching and return:
- an updated loader (with the new seed), and
- the list of minibatches.
-/
def epoch {a : Type} (name : String) (dl : RawDataLoader a) :
    Except String (RawDataLoader a × List (List a)) :=
  _root_.Runtime.Autograd.Train.DataLoader.epoch name dl

/--
Like `epoch`, but apply a user-provided `collate` function to each minibatch, matching the role of
PyTorch's `collate_fn=` option.
-/
def epochCollate {a b : Type} (name : String) (dl : RawDataLoader a)
    (collate : List a → Except String b) :
    Except String (RawDataLoader a × List b) :=
  _root_.Runtime.Autograd.Train.DataLoader.epochCollate name dl collate

/--
Typed wrapper around `RawDataLoader` for supervised samples.

The batch size `n` is reflected in the type, and `BatchLoader.epoch` returns fully-collated
`dim n` minibatches (so `dropLast=true` is required).
-/
structure BatchLoader (α : Type) (n : Nat) (σ τ : Spec.Shape) where
  /-- Raw underlying data. -/
  raw : RawDataLoader (SupervisedSample α σ τ)

/-- Existential wrapper for loaders when the batch size is chosen at runtime. -/
abbrev AnyBatchLoader (α : Type) (σ τ : Spec.Shape) :=
  Σ n : Nat, BatchLoader α n σ τ

/-!
Note on default arguments:

The underlying CSV loaders take an `opts : CsvOptions := {}` argument.
If we write `abbrev fromCsvRows := readCsvFloatRows`, Lean will *apply the default argument*
and `fromCsvRows` will no longer accept `opts`.

So we eta-expand here to keep the options argument available to callers.
-/

/-- Read a CSV file as a list of rows of floats. -/
abbrev fromCsvRows (path : System.FilePath) (opts : CsvOptions := {}) :=
  readCsvFloatRows path opts

/-- Read a CSV file as `(x, y)` float pairs. -/
abbrev fromCsvPairs (path : System.FilePath) (opts : CsvOptions := {}) :=
  readCsvDatasetPairs path opts

/-- Read a CSV file as length-`n` float vectors. -/
abbrev fromCsvVectors (path : System.FilePath) (n : Nat) (opts : CsvOptions := {}) :=
  readCsvVectorDataset path n opts

/-- Read a `.npy` file into a TorchLean dataset. -/
abbrev fromNpy := readNpy

/-- Read a `.npy` file as a vector dataset. -/
abbrev fromNpyVector := readNpyVector

/-- Read a `.npy` file as a matrix dataset. -/
abbrev fromNpyMatrix := readNpyMatrix

/--
Read the row count from an `.npy` file and check its trailing shape.

For a batched tensor with shape `(N, d₁, ..., dₖ)`, this returns `N` when the trailing dimensions
match `tailShape`.
-/
def availableNpyRows
    (path : System.FilePath) (tailShape : List Nat) (expectedDesc : String) :
    IO (Except String Nat) := do
  let npyMeta ← fromNpy path
  match npyMeta with
  | .error e => pure (.error e)
  | .ok n =>
      match n.shape with
      | rows :: rest =>
          if rest = tailShape then
            pure (.ok rows)
          else
            pure (.error s!"expected {expectedDesc}, got {n.shape}")
      | _ => pure (.error s!"expected {expectedDesc}, got {n.shape}")

/--
Convert a list of `(x, y)` float tensors into a dataset of TorchLean supervised samples.

This casts float data into the selected scalar backend `α` and packs it into a
`TensorPack α [σ, τ]`.
-/
def supervised {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α] {σ τ : Spec.Shape}
    (xs : List (Spec.Tensor Float σ × Spec.Tensor Float τ)) :
    _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ]) :=
  fromList <| xs.map (fun (xF, yF) =>
    let x : Spec.Tensor α σ := Spec.mapTensor (API.Runtime.ofFloat (α := α)) xF
    let y : Spec.Tensor α τ := Spec.mapTensor (API.Runtime.ofFloat (α := α)) yF
    tensorpack! x, y)

/--
Convert a list of `(x, label)` pairs into a dataset of one-hot classification samples.

Labels are given as `Nat` and converted to one-hot targets of shape `Vec classes`.
-/
def labeled {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α] {σ : Spec.Shape}
    (classes : Nat) (xs : List (Spec.Tensor Float σ × Nat)) :
    _root_.Runtime.Autograd.Train.Dataset
      (API.TorchLean.TensorPack α [σ, NN.Tensor.Shape.Vec classes]) :=
  fromList <| xs.map (fun (xF, label) =>
    let x : Spec.Tensor α σ := Spec.mapTensor (API.Runtime.ofFloat (α := α)) xF
    let yF : Spec.Tensor Float (NN.Tensor.Shape.Vec classes) := NN.Tensor.oneHotNat (α := Float)
      classes label
    let y : Spec.Tensor α (NN.Tensor.Shape.Vec classes) :=
      Spec.mapTensor (API.Runtime.ofFloat (α := α)) yF
    tensorpack! x, y)

/-!
## TensorDataset (leading-axis batching)

PyTorch's `TensorDataset` concept is: given one or more tensors that share the same `size(0)`,
build a dataset of samples by slicing each tensor along its leading batch axis.

In TorchLean we do the same thing, but with shapes tracked in the type:

- a batched tensor has shape `.dim n σ`,
- slicing at `i : Fin n` yields a sample of shape `σ`,
- and a batch of multiple tensors is represented as a `TensorPack`.
-/

/--
Slice a batched `TensorPack` along its leading batch axis.

If a sample is represented as a shape-indexed tuple `TensorPack β ss`, then a minibatch of size `n`
is `TensorPack β (ss.map (fun s => .dim n s))`. This function picks a batch index `i : Fin n` and returns
the corresponding single sample.
-/
def unbatchTensorPack {β : Type} {n : Nat} :
    {ss : List Spec.Shape} →
      API.TorchLean.TensorPack β (ss.map (fun s => Spec.Shape.dim n s)) →
      Fin n →
      API.TorchLean.TensorPack β ss
  | [], .nil, _i => .nil
  | _s :: ss, .cons x xs, i =>
      .cons (Spec.getAtSpec x i) (unbatchTensorPack (β := β) (ss := ss) xs i)

/-- Convert a shape-indexed `TensorPack` of `Float` tensors to the runtime scalar type `α`. -/
def castTListOfFloat {α : Type} [API.Runtime.Scalar α] :
    {ss : List Spec.Shape} →
      API.TorchLean.TensorPack Float ss →
      API.TorchLean.TensorPack α ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs =>
      .cons (Spec.mapTensor (API.Runtime.ofFloat (α := α)) x) (castTListOfFloat (ss := ss) xs)

/--
Build a dataset by slicing a *batched* `TensorPack` along the leading batch axis. This gives the
typed counterpart of a tensor dataset built from several aligned arrays.
-/
def tensorDatasetFromLeadingAxis {β : Type} {n : Nat} {ss : List Spec.Shape}
    (xs : API.TorchLean.TensorPack β (ss.map (fun s => Spec.Shape.dim n s))) :
    Dataset (API.TorchLean.TensorPack β ss) :=
  fromList <| (List.finRange n).map (fun i => unbatchTensorPack (β := β) (n := n) (ss := ss) xs i)

/--
Float-to-`α` variant of `tensorDatasetFromLeadingAxis`, for data loaded from disk.
-/
def tensorDatasetFromLeadingAxisFloat {α : Type} [API.Runtime.Scalar α]
    {n : Nat} {ss : List Spec.Shape}
    (xs : API.TorchLean.TensorPack Float (ss.map (fun s => Spec.Shape.dim n s))) :
    Dataset (API.TorchLean.TensorPack α ss) :=
  let samples : List (API.TorchLean.TensorPack α ss) :=
    (List.finRange n).map (fun i =>
      castTListOfFloat (α := α) (unbatchTensorPack (β := Float) (n := n) (ss := ss) xs i))
  fromList samples

/--
Supervised dataset from two batched tensors `X : (n, σ)` and `Y : (n, τ)` by slicing the leading batch axis.

This is the common regression/supervised-learning case: the TorchLean analogue of
`TensorDataset(X, Y)` in PyTorch.
-/
def supervisedFromLeadingAxis {α : Type}
    {n : Nat} {σ τ : Spec.Shape}
    (X : Spec.Tensor α (.dim n σ))
    (Y : Spec.Tensor α (.dim n τ)) :
    Dataset (API.TorchLean.TensorPack α [σ, τ]) :=
  tensorDatasetFromLeadingAxis (β := α) (n := n) (ss := [σ, τ])
    (tensorpack! X, Y)

/-- Float-to-`α` variant of `supervisedFromLeadingAxis`, for data loaded from disk. -/
def supervisedFromLeadingAxisFloat {α : Type} [API.Runtime.Scalar α]
    {n : Nat} {σ τ : Spec.Shape}
    (X : Spec.Tensor Float (.dim n σ))
    (Y : Spec.Tensor Float (.dim n τ)) :
    Dataset (API.TorchLean.TensorPack α [σ, τ]) :=
  tensorDatasetFromLeadingAxisFloat (α := α) (n := n) (ss := [σ, τ])
    (tensorpack! X, Y)

/-!
## Higher-level loaders (PyTorch-style ergonomics)

These are convenience helpers on top of the low-level CSV/NPY readers so example code can stay
"data first" without re-implementing row splitting and casting at every call site.
-/

/-- Load an N-D tensor from a `.npy` file, checking the on-disk shape matches `dims`. -/
def fromNpyTensorND (path : System.FilePath) (dims : List Nat) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  let res ← fromNpy path
  match res with
  | .error e => pure (.error e)
  | .ok data =>
      if data.shape != dims then
        pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")
      else
        pure <| NN.Tensor.tensorND (α := Float) dims data.values.toList

/--
Load an N-D tensor from a `.npy` file, allowing the file to contain more rows on the leading axis.

This is the dataset-loader analogue of taking `tensor[:n]` in PyTorch. The rank and trailing
dimensions must still match exactly; only the leading dimension may be larger than requested.

We use this for dataset sources rather than the stricter `fromNpyTensorND` because an exported
dataset usually has a fixed full size, while local runs often request a bounded prefix. For example,
a CIFAR file may have shape `(50000, 3, 32, 32)` while an example command asks for `n = 80`; the
resulting TorchLean tensor has type-level shape
`(80, 3, 32, 32)`.

This is still a checked loader, not an implicit reshape:

- rank must agree;
- all trailing dimensions must agree;
- the file must contain at least the requested number of rows;
- only C-order NPY files can be prefix-loaded efficiently by the low-level parser.
-/
def fromNpyTensorNDLeadingPrefix (path : System.FilePath) (dims : List Nat) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  let res ← readNpyLeadingAxisPrefix path dims
  match res with
  | .error e => pure (.error e)
  | .ok data =>
      match hDims : dims with
      | expectedN :: expectedTail =>
          match data.shape with
          | actualN :: actualTail =>
              if actualTail != expectedTail then
                pure (.error
                  s!"npy: shape mismatch, expected trailing dims {expectedTail}, got {actualTail}")
              else if actualN < expectedN then
                pure (.error s!"npy: expected at least {expectedN} rows, got {actualN}")
              else
                let dims' := expectedN :: expectedTail
                let count := dims'.foldl (fun acc n => acc * n) 1
                pure <| (NN.Tensor.tensorND (α := Float) dims'
                  (data.values.toList.take count)).map (fun t => by
                    simpa [hDims] using t)
          | [] =>
              pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")
      | [] =>
          match data.shape with
          | [] =>
              pure <| (NN.Tensor.tensorND (α := Float) [] data.values.toList).map (fun t => by
                simpa [hDims] using t)
          | _ =>
              pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")

/-- Load an image tensor from a `.npy` file, checking it has shape `(C, H, W)`. -/
def fromNpyImage (path : System.FilePath) (c h w : Nat) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.Shape.Image c h w))) := do
  let tRes ← fromNpyTensorND path [c, h, w]
  pure <| tRes.map (fun t =>
    (by simpa [NN.Tensor.shapeOfDims, NN.Tensor.Shape.Image] using t))

/-- Load a batch of images from a `.npy` file, checking it has shape `(N, C, H, W)`. -/
def fromNpyImages (path : System.FilePath) (n c h w : Nat) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.Shape.Images n c h w))) := do
  let tRes ← fromNpyTensorND path [n, c, h, w]
  pure <| tRes.map (fun t =>
    (by simpa [NN.Tensor.shapeOfDims, NN.Tensor.Shape.Images] using t))

/-- Parse a float-encoded class label as a `Nat` in `[0, classes)`. -/
def natLabelOfFloat (tag : String) (classes : Nat) (x : Float) : Except String Nat := do
  let n : Nat := x.toUInt64.toNat
  if (n : Float) != x then
    throw s!"{tag}: expected an integer class label, got {x}"
  else if n >= classes then
    throw s!"{tag}: class label {n} out of range (classes={classes})"
  else
    pure n

/--
Labeled dataset from a batched tensor `X : (n, σ)` and a label vector `y : (n,)`.

Labels are stored as floats (common when exporting from NumPy); we validate each label is an
integer in `[0, classes)`, then one-hot encode it.
-/
def labeledFromLeadingAxis {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    (tag : String) (classes : Nat)
    {n : Nat} {σ : Spec.Shape}
    (X : Spec.Tensor Float (.dim n σ))
    (y : Spec.Tensor Float (NN.Tensor.Shape.Vec n)) :
    Except String (Dataset (API.TorchLean.TensorPack α [σ, NN.Tensor.Shape.Vec classes])) := do
  let samples : List (Spec.Tensor Float σ × Nat) ←
    (List.finRange n).mapM (fun i => do
      let x := Spec.getAtSpec X i
      let labelF : Float := Spec.Tensor.toScalar (Spec.getAtSpec y i)
      let label ← natLabelOfFloat tag classes labelF
      pure (x, label))
  pure <| labeled (α := α) (σ := σ) classes samples

/--
Load a supervised dataset from two `.npy` files containing batched arrays:

- `X.npy` has shape `(n, xDims...)`
- `Y.npy` has shape `(n, yDims...)`

and we build a dataset by slicing along the leading batch axis.
-/
def fromNpySupervised {α : Type} [API.Runtime.Scalar α]
    (xPath yPath : System.FilePath) (n : Nat) (xDims yDims : List Nat) :
    IO (Except String (Dataset (API.TorchLean.TensorPack α [NN.Tensor.shapeOfDims xDims,
      NN.Tensor.shapeOfDims yDims]))) := do
  let xRes ← fromNpyTensorNDLeadingPrefix xPath (n :: xDims)
  let yRes ← fromNpyTensorNDLeadingPrefix yPath (n :: yDims)
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok Y => pure (.ok (supervisedFromLeadingAxisFloat (α := α) X Y))

/--
Load a labeled classification dataset from two `.npy` files:

- `X.npy` has shape `(n, xDims...)`
- `y.npy` has shape `(n,)` with float-encoded integer labels in `[0, classes)`

and we build a dataset by slicing along the leading batch axis and one-hot encoding the labels.
-/
def fromNpyLabeled {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    (xPath yPath : System.FilePath) (n : Nat) (xDims : List Nat) (classes : Nat) :
    IO (Except String (Dataset (API.TorchLean.TensorPack α [NN.Tensor.shapeOfDims xDims,
      NN.Tensor.Shape.Vec classes]))) := do
  let xRes ← fromNpyTensorNDLeadingPrefix xPath (n :: xDims)
  let yRes ← fromNpyTensorNDLeadingPrefix yPath [n]
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok y =>
          pure <| labeledFromLeadingAxis (α := α) (σ := NN.Tensor.shapeOfDims xDims) "npy" classes X y

/--
Load a supervised dataset from a CSV with `inDim + outDim` columns per row:

`x1, ..., x_inDim, y1, ..., y_outDim`.
-/
def fromCsvSupervised {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    (path : System.FilePath) (inDim outDim : Nat) (opts : CsvOptions := {}) :
    IO (Except String (Dataset (API.TorchLean.TensorPack α [NN.Tensor.Shape.Vec inDim,
      NN.Tensor.Shape.Vec outDim]))) := do
  let rowsRes ← fromCsvRows path (opts := opts)
  match rowsRes with
  | .error e => pure (.error e)
  | .ok rows =>
      let samplesRes :
          Except String (List (Spec.Tensor Float (NN.Tensor.Shape.Vec inDim) × Spec.Tensor Float
            (NN.Tensor.Shape.Vec outDim))) :=
        rows.mapM (fun row => do
          let xs := row.take inDim
          let ys := row.drop inDim
          let xF ← vectorOfList (tag := "csv") (n := inDim) xs
          let yF ← vectorOfList (tag := "csv") (n := outDim) ys
          pure (xF, yF))
      pure <| samplesRes.map (fun samplesF => supervised (α := α) samplesF)

/--
Load a labeled dataset from a CSV with `inDim + 1` columns per row:

`x1, ..., x_inDim, label` where `label` is in `{0, ..., classes-1}`.
-/
def fromCsvLabeled {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    (path : System.FilePath) (inDim classes : Nat) (opts : CsvOptions := {}) :
    IO (Except String (Dataset (API.TorchLean.TensorPack α [NN.Tensor.Shape.Vec inDim,
      NN.Tensor.Shape.Vec classes]))) := do
  let rowsRes ← fromCsvRows path (opts := opts)
  match rowsRes with
  | .error e => pure (.error e)
  | .ok rows =>
      let samplesRes :
          Except String (List (Spec.Tensor Float (NN.Tensor.Shape.Vec inDim) × Nat)) :=
        rows.mapM (fun row => do
          let xs := row.take inDim
          let labelF := row.getD inDim 0.0
          if row.length != inDim + 1 then
            throw s!"csv: expected {inDim + 1} columns per row (features+label), got {row.length}"
          let xF ← vectorOfList (tag := "csv") (n := inDim) xs
          let label ← natLabelOfFloat (tag := "csv") classes labelF
          pure (xF, label))
      pure <| samplesRes.map (fun samplesF => labeled (α := α) (σ := NN.Tensor.Shape.Vec inDim)
        classes samplesF)

/-!
## Unified file-source layer

The lower-level helpers above stay close to file formats (`fromNpyTensorND`,
`fromCsvRows`, `fromNpySupervised`, ...).  The definitions below give examples and applications a
single scheme:

1. describe each tensor as a `TensorSource`;
2. load it as a typed TorchLean tensor;
3. build supervised/labeled datasets by slicing the leading batch axis, just like PyTorch `TensorDataset`.

Policy for external ecosystems:
- NumPy `.npy` is the canonical interchange format for numeric tensors.
- CSV is supported for small tabular data.
- MATLAB `.mat`, PyTorch checkpoints, HDF5, Parquet, and image archives should be converted by a
  small preparation script into `.npy` tensors plus metadata. The Lean runtime loader intentionally
  handles a small deterministic interchange format rather than every external binary format.
-/

/-- File formats supported directly by the Lean side unified data-source loader. -/
inductive TensorFormat where
  /-- NumPy `.npy`, supporting the subset decoded by `fromNpyTensorND`. -/
  | npy
  /-- Numeric CSV table. CSV sources are interpreted as 2D tensors `[rows, cols]`. -/
  | csv
deriving BEq, Repr

namespace TensorFormat

/-- Human-facing extension used by messages and examples. -/
def extension : TensorFormat → String
  | .npy => ".npy"
  | .csv => ".csv"

end TensorFormat

/--
Description of one tensor stored on disk.

`dims` is the expected tensor shape.  NPY can load any rank supported by `tensorND`; CSV is treated
as a numeric table and therefore expects `dims = [rows, cols]`.
-/
structure TensorSource where
  /-- Path to the file. -/
  path : System.FilePath
  /-- Expected dimensions. -/
  dims : List Nat
  /-- Direct Lean side format. External formats should be preconverted to `.npy`. -/
  format : TensorFormat := .npy
  /-- CSV parsing options, used only when `format = .csv`. -/
  csvOptions : CsvOptions := {}

namespace TensorSource

/--
Load a numeric CSV table as a tensor.

Supported shapes:
- `[rows, cols]`: ordinary numeric table,
- `[n]`: either one column with `n` rows or one row with `n` columns.
-/
def loadCsvTensorND (path : System.FilePath) (dims : List Nat) (opts : CsvOptions := {}) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  match hDims : dims with
  | [rowsExpected, colsExpected] =>
      let rowsRes ← fromCsvRows path (opts := opts)
      match rowsRes with
      | .error e => pure (.error e)
      | .ok rows =>
          if rows.length != rowsExpected then
            pure (.error s!"csv: expected {rowsExpected} rows, got {rows.length}")
          else
            let bad? := rows.zipIdx.find? (fun (row, _i) => row.length != colsExpected)
            match bad? with
            | some (row, i) =>
                pure (.error s!"csv: row {i + 1}: expected {colsExpected} columns, got {row.length}")
            | none =>
                let flat := rows.foldr (fun row acc => row ++ acc) []
                pure <| (NN.Tensor.tensorND (α := Float) [rowsExpected, colsExpected] flat).map
                  (fun t => by
                    simpa [hDims] using t)
  | [n] =>
      let rowsRes ← fromCsvRows path (opts := opts)
      match rowsRes with
      | .error e => pure (.error e)
      | .ok rows =>
          let flat? : Except String (List Float) :=
            if rows.length = n && rows.all (fun row => row.length == 1) then
              .ok (rows.map (fun row => row.getD 0 0.0))
            else
              match rows with
              | [row] =>
                  if row.length = n then .ok row
                  else .error s!"csv: expected one row with {n} columns, got {row.length}"
              | _ =>
                  .error s!"csv: expected a length-{n} vector as one column or one row"
          match flat? with
          | .error e => pure (.error e)
          | .ok flat =>
              pure <| (NN.Tensor.tensorND (α := Float) [n] flat).map
                (fun t => by
                  simpa [hDims] using t)
  | _ =>
      pure (.error s!"csv: TensorSource expects dims=[rows, cols] or dims=[n], got {dims}")

/-- Load a Float tensor from a path/format/dimension tuple. -/
def loadFloatAs (format : TensorFormat) (path : System.FilePath)
    (dims : List Nat) (opts : CsvOptions := {}) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  match format with
  | .npy => fromNpyTensorND path dims
  | .csv => loadCsvTensorND path dims opts

/--
Load a Float tensor, allowing NPY files to contain more rows than requested on the leading axis.

`TensorSource.loadFloatAs` is exact: the file shape must equal `dims`.  This prefix variant is for
dataset-style sources where `dims` starts with the number of rows requested by the current run.  CSV
sources remain exact because CSV has no binary prefix contract; NPY sources use
`fromNpyTensorNDLeadingPrefix`.
-/
def loadFloatLeadingPrefixAs (format : TensorFormat) (path : System.FilePath)
    (dims : List Nat) (opts : CsvOptions := {}) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  match format with
  | .npy => fromNpyTensorNDLeadingPrefix path dims
  | .csv => loadCsvTensorND path dims opts

/-- Load a `TensorSource` as a Float tensor with the statically reflected `shapeOfDims src.dims`. -/
def loadFloat (src : TensorSource) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims src.dims))) := do
  loadFloatAs src.format src.path src.dims src.csvOptions

end TensorSource

/--
Two tensor sources representing supervised data:
- `x` must have shape `(n, xDims...)`,
- `y` must have shape `(n, yDims...)`.
-/
structure SupervisedSource where
  /-- Number of samples along the leading batch axis. -/
  n : Nat
  /-- Per-sample input dimensions. -/
  xDims : List Nat
  /-- Per-sample target dimensions. -/
  yDims : List Nat
  /-- Source for the batched input tensor. -/
  x : TensorSource
  /-- Source for the batched target tensor. -/
  y : TensorSource

namespace SupervisedSource

/-- Construct a supervised source from paths using the same file format for `x` and `y`. -/
def ofPaths (format : TensorFormat) (xPath yPath : System.FilePath)
    (n : Nat) (xDims yDims : List Nat) (csvOptions : CsvOptions := {}) : SupervisedSource :=
  { n, xDims, yDims
    x := { path := xPath, dims := n :: xDims, format, csvOptions }
    y := { path := yPath, dims := n :: yDims, format, csvOptions } }

/--
Load a supervised dataset by slicing the leading batch axis from the two tensors.

This is the preferred public loader for regression/operator-learning examples, regardless of
whether the backing files are `.npy` or small numeric CSV tables.
-/
def load {α : Type} [API.Runtime.Scalar α] (src : SupervisedSource) :
    IO (Except String (Dataset (API.TorchLean.TensorPack α [NN.Tensor.shapeOfDims src.xDims,
      NN.Tensor.shapeOfDims src.yDims]))) := do
  -- Dataset sources interpret `src.n` as "number of rows to use in this run."  For NPY files, the
  -- physical file is allowed to contain more rows; for CSV files, the requested shape remains exact.
  let xRes ← TensorSource.loadFloatLeadingPrefixAs src.x.format src.x.path (src.n :: src.xDims)
    src.x.csvOptions
  let yRes ← TensorSource.loadFloatLeadingPrefixAs src.y.format src.y.path (src.n :: src.yDims)
    src.y.csvOptions
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok Y => pure (.ok (supervisedFromLeadingAxisFloat (α := α) X Y))

end SupervisedSource

/-- Paired `.npy` source for supervised regression or operator-learning datasets. -/
def supervisedNpySource
    (xPath yPath : System.FilePath) (n : Nat)
    (xDims yDims : List Nat) : SupervisedSource :=
  SupervisedSource.ofPaths .npy xPath yPath n xDims yDims

/--
Load paired `.npy` files as concrete `Float` supervised samples.

This is useful for reporting, custom evaluation loops, and native kernels that need concrete
`Float` tensors outside the public trainer facade.
-/
def loadSupervisedNpyFloatSamples
    (xPath yPath : System.FilePath) (n : Nat)
    (xDims yDims : List Nat) :
    IO (Except String (Array (API.SupervisedSample Float (NN.Tensor.shapeOfDims xDims)
      (NN.Tensor.shapeOfDims yDims)))) := do
  let ds ← SupervisedSource.load (α := Float) (supervisedNpySource xPath yPath n xDims yDims)
  pure <| ds.map (fun d => toList d |>.toArray)

/--
Two tensor sources representing labeled classification data:
- `x` must have shape `(n, xDims...)`,
- `y` must have shape `(n,)` and contain integer-valued labels.
-/
structure LabeledSource where
  /-- Number of samples along the leading batch axis. -/
  n : Nat
  /-- Per-sample input dimensions. -/
  xDims : List Nat
  /-- Number of classes for one-hot targets. -/
  classes : Nat
  /-- Source for the batched input tensor. -/
  x : TensorSource
  /-- Source for the label vector. -/
  y : TensorSource

namespace LabeledSource

/-- Construct a labeled source from paths using the same file format for `x` and `y`. -/
def ofPaths (format : TensorFormat) (xPath yPath : System.FilePath)
    (n : Nat) (xDims : List Nat) (classes : Nat) (csvOptions : CsvOptions := {}) : LabeledSource :=
  { n, xDims, classes
    x := { path := xPath, dims := n :: xDims, format, csvOptions }
    y := { path := yPath, dims := [n], format, csvOptions } }

/--
Load a labeled classification dataset by slicing the leading batch axis and one-hot encoding labels.

For CSV label vectors, store labels as a single-column table with `dims = [n, 1]` and use a custom
`TensorSource` if needed; the path constructor above is aimed at `.npy` label vectors.
-/
def load {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α] (src : LabeledSource) :
    IO (Except String (Dataset (API.TorchLean.TensorPack α [NN.Tensor.shapeOfDims src.xDims,
      NN.Tensor.Shape.Vec src.classes]))) := do
  -- Labels use the same prefix-row convention as supervised tensors. This lets one full exported
  -- label vector back different bounded runs without making separate copies on disk.
  let xRes ← TensorSource.loadFloatLeadingPrefixAs src.x.format src.x.path (src.n :: src.xDims)
    src.x.csvOptions
  let yRes ← TensorSource.loadFloatLeadingPrefixAs src.y.format src.y.path [src.n] src.y.csvOptions
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok y =>
          pure <| labeledFromLeadingAxis (α := α) (σ := NN.Tensor.shapeOfDims src.xDims)
            "data-source" src.classes X y

end LabeledSource

/--
Single-table supervised CSV source.

Use this when one CSV row contains both input and target columns:
`x1, ..., x_inDim, y1, ..., y_outDim`.
-/
structure TabularSupervisedSource where
  /-- CSV file path. -/
  path : System.FilePath
  /-- Number of input feature columns. -/
  inDim : Nat
  /-- Number of target columns. -/
  outDim : Nat
  /-- CSV parsing options. -/
  csvOptions : CsvOptions := {}

namespace TabularSupervisedSource

/-- Load a single-table supervised CSV source. -/
def load {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    (src : TabularSupervisedSource) :
    IO (Except String (Dataset (API.TorchLean.TensorPack α [NN.Tensor.Shape.Vec src.inDim,
      NN.Tensor.Shape.Vec src.outDim]))) :=
  fromCsvSupervised (α := α) src.path src.inDim src.outDim (opts := src.csvOptions)

end TabularSupervisedSource

/--
Build a supervised dataset from two matrices `X : n×inDim` and `Y : n×outDim` by pairing rows.
This is the simple regression case of a tensor dataset.
-/
def supervisedRows {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    {n inDim outDim : Nat}
    (X : Spec.Tensor Float (NN.Tensor.Shape.Mat n inDim))
    (Y : Spec.Tensor Float (NN.Tensor.Shape.Mat n outDim)) :
    Dataset (API.TorchLean.TensorPack α [NN.Tensor.Shape.Vec inDim, NN.Tensor.Shape.Vec outDim]) :=
  supervisedFromLeadingAxisFloat (α := α) X Y

/--
Collate a length-`n` supervised batch into a single sample with a leading batch axis.

If your samples are `(x : σ, y : τ)`, the collated sample is:

- `xBatch : (n × σ)` and
- `yBatch : (n × τ)`

In shapes: `TensorPack α [dim n σ, dim n τ]`.
-/
def collateSupervised {α : Type} {σ τ : Spec.Shape} (n : Nat)
    (batch : List (API.TorchLean.TensorPack α [σ, τ])) :
    Except String (API.TorchLean.TensorPack α [Spec.Shape.dim n σ, Spec.Shape.dim n τ]) := do
  if h : batch.length = n then
    let getSample : Fin n → API.TorchLean.TensorPack α [σ, τ] :=
      fun i =>
        let hlt : i.val < batch.length := by simp [h]
        batch.get ⟨i.val, hlt⟩
    let xs : Spec.Tensor α (Spec.Shape.dim n σ) :=
      _root_.Spec.Tensor.dim (fun i =>
        match getSample i with
        | .cons x _ => x)
    let ys : Spec.Tensor α (Spec.Shape.dim n τ) :=
      _root_.Spec.Tensor.dim (fun i =>
        match getSample i with
        | .cons _x (.cons y .nil) => y)
    pure (tensorpack! xs, ys)
  else
    throw s!"collate: expected batch size {n}, got {batch.length}"

/-- Split a list into consecutive length-`n` chunks, dropping any final short chunk. -/
def chunkN {a : Type} (n : Nat) (xs : List a) : List (List a) :=
  if _h : n = 0 then
    []
  else
    -- Use an explicit fuel so Lean sees termination; `take/drop` recursion doesn't.
    let rec go (xs : List a) (fuel : Nat) : List (List a) :=
      match fuel with
      | 0 => []
      | fuel + 1 =>
          if xs.length < n then
            []
          else
            let b := xs.take n
            let rest := xs.drop n
            b :: go rest fuel
    go xs xs.length

/--
Turn a per-sample supervised dataset into a dataset of fixed-size minibatches.

This is useful for metrics (`meanLossDataset`, accuracy, etc.) when your model expects a leading
batch axis.

Notes:
- This drops the final partial batch (PyTorch `drop_last=True` behavior).
- Batches are formed in dataset order (shuffling is the loader's job).
-/
def batchedSupervised {α : Type} {σ τ : Spec.Shape} (n : Nat)
    (ds : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    Except String (_root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [Spec.Shape.dim n σ,
      Spec.Shape.dim n τ])) := do
  if n = 0 then
    throw "batched: batch size must be > 0"
  let groups := chunkN n (_root_.Runtime.Autograd.Train.Dataset.toList ds)
  let full := groups.filter (fun g => g.length = n)
  let batches ← full.mapM (collateSupervised (α := α) (σ := σ) (τ := τ) n)
  pure (fromList batches)

namespace BatchLoader

/-- Extract the underlying per-sample dataset from a typed `BatchLoader`. -/
def dataset {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) :
    Dataset (SupervisedSample α σ τ) :=
  dl.raw.dataset

/-- The batch size `n` carried in the type of a `BatchLoader`. -/
def batchSize {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (_dl : BatchLoader α n σ τ) : Nat :=
  n

/-- Whether the loader is configured to shuffle samples each epoch. -/
def shuffled {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) : Bool :=
  dl.raw.shuffle

/-- RNG seed used for shuffling (if enabled). -/
def seed {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) : Nat :=
  dl.raw.seed

/-- Materialize the dataset as a dataset of full minibatches (dropping any final partial batch). -/
def batchDataset {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) :
    Except String (Dataset (Sample.Batch α n σ τ)) :=
  batchedSupervised (α := α) (σ := σ) (τ := τ) n dl.raw.dataset

/-- Run one epoch: return the updated loader state and a list of typed minibatches. -/
def epoch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ) :
    Except String (BatchLoader α n σ τ × List (Sample.Batch α n σ τ)) := do
  if dl.raw.batchSize != n then
    throw s!"{name}: expected typed batch size {n}, got loader.batchSize={dl.raw.batchSize}"
  if !dl.raw.dropLast then
    throw s!"{name}: BatchLoader requires dropLast=true"
  let (raw', batches) ←
    _root_.Runtime.Autograd.Train.DataLoader.epochCollate name dl.raw
      (fun batch => collateSupervised (α := α) (σ := σ) (τ := τ) n batch)
  pure ({ raw := raw' }, batches)

/-- Like `epoch`, but post-process each minibatch with a user-supplied collate/transform `f`. -/
def epochCollate {α β : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ)
    (f : Sample.Batch α n σ τ → Except String β) :
    Except String (BatchLoader α n σ τ × List β) := do
  let (dl', batches) ← epoch (α := α) (σ := σ) (τ := τ) name dl
  pure (dl', ← batches.mapM f)

/--
Run one epoch and require at least one full typed minibatch.

This is the shared checked boundary for examples that need a nonempty list of full batches. It
keeps the "drop partial batches, but fail if nothing remains" policy with the loader API rather than
repeating it in each dataset-specific helper.
-/
def nonemptyEpoch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ) :
    Except String (BatchLoader α n σ τ × List (Sample.Batch α n σ τ)) := do
  let (dl', batches) ← epoch (α := α) (σ := σ) (τ := τ) name dl
  match batches with
  | _ :: _ => pure (dl', batches)
  | [] =>
      throw s!"{name}: no full minibatch available (batch={n}, rows={Data.size dl.raw.dataset})"

/-- Run one epoch and return its first full typed minibatch. -/
def firstFullBatch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ) :
    Except String (Sample.Batch α n σ τ) := do
  let (_dl', batches) ← nonemptyEpoch (α := α) (σ := σ) (τ := τ) name dl
  match batches with
  | b :: _ => pure b
  | [] => throw s!"{name}: internal error: expected a nonempty minibatch epoch"

end BatchLoader

/--
Public loader API: supervised datasets become fixed-size minibatch loaders by default.

The underlying dataset still stores individual samples; the loader batches them and `epoch`
returns tensors with a leading batch axis. Because the batch size is reflected in the type,
the public batched path requires full batches, so `dropLast` defaults to `true`.
-/
def batchLoader {α : Type} {σ τ : Spec.Shape}
    (ds : Dataset (SupervisedSample α σ τ))
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) (dropLast : Bool := true) :
    BatchLoader α batchSize σ τ :=
  { raw := loader ds batchSize (shuffle := shuffle) (seed := seed) (dropLast := dropLast) }

/--
Load a numeric supervised CSV and immediately wrap it as a typed minibatch loader.

The CSV convention is the same as `TabularSupervisedSource`: each row contains `inDim` feature
columns followed by `outDim` target columns.  This belongs in the data API rather than in an
individual model file because tabular examples, benchmarks, and downstream users all need the same
operation: CSV -> typed dataset -> shuffled minibatch loader.
-/
def tabularCsvLoader {α : Type} [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    (path : System.FilePath) (batchSize inDim outDim : Nat)
    (csvOptions : CsvOptions := {}) (shuffle : Bool := true) (seed : Nat := 0)
    (dropLast : Bool := true) :
    IO (Except String (BatchLoader α batchSize (NN.Tensor.Shape.Vec inDim)
      (NN.Tensor.Shape.Vec outDim))) := do
  let src : TabularSupervisedSource :=
    { path := path, inDim := inDim, outDim := outDim, csvOptions := csvOptions }
  let dsE ← src.load (α := α)
  match dsE with
  | .error e => pure (.error e)
  | .ok ds =>
      pure <| .ok <| batchLoader (α := α) ds batchSize (shuffle := shuffle)
        (seed := seed) (dropLast := dropLast)

/-- Build a batch loader when the batch size is only known at runtime. -/
def loaderAny {α : Type} {σ τ : Spec.Shape}
    (ds : Dataset (SupervisedSample α σ τ))
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) (dropLast : Bool := true) :
    AnyBatchLoader α σ τ :=
  ⟨batchSize, batchLoader (α := α) (σ := σ) (τ := τ) ds batchSize shuffle seed dropLast⟩

end Data

end API
end NN
