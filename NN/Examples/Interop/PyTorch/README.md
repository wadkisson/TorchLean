# PyTorch Interop Examples

This folder contains PyTorch interop examples. The reusable bridge lives in
`NN/Runtime/PyTorch`; the files here are maintained reference examples, runtime checks, and
round-trip artifacts.

Read these examples as boundary exercises. PyTorch may train a small model, emit a `state_dict`, or
expose a graph through `torch.export`. Lean checks the artifact that crosses the boundary: names,
shapes, supported operators, and the TorchLean IR JSON accepted by the importer. The examples stay
small and deterministic so a reader can inspect the artifact rather than treating the Python run as
a black box.

## Two Separate Paths

### 1. Model Agnostic Graph Capture

- Lean entrypoint: `TorchExportCheck.lean`
- Runtime bridge: `NN.Runtime.PyTorch.Export.TorchExport` and `NN.Runtime.PyTorch.Import.TorchExport`
- Command: `lake exe torchlean pytorch_export_check`

This path writes a generated Python adapter, captures small `nn.Module`s through `torch.export`/FX,
emits `torchlean.ir.v1` JSON, and then asks Lean to parse and validate the graph. The importer has a
value graph layer: tensor valued FX nodes lower into `NN.IR.Graph`; tuple valued FX nodes are kept
explicit, and the deterministic `batch_first=True`, `num_heads=1` self attention output of
`nn.MultiheadAttention(...)[0]` is decomposed into ordinary tensor IR primitives. Broader MHA cases
still fail loudly with the unsupported feature named in the error.

Use this path when the architecture is not one of the handwritten family examples. The important
artifact is the TorchLean IR JSON, not the Python module itself. Once Lean accepts the IR, later
workflows can decide whether to run it, compare it against a spec, or feed it to a verifier.

### 2. State Dict Round Trips

- Lean entrypoint: `Roundtrip.lean`
- Example exporters: `MLP/Export.lean`, `CNN/Export.lean`, `Transformer/Export.lean`
- Example importers: `MLP/Import.lean`, `CNN/Import.lean`, `Transformer/Import.lean`

Run:

- `lake exe torchlean pytorch_roundtrip --model mlp --action export`
- `lake exe torchlean pytorch_roundtrip --model mlp --action import`
- `lake exe torchlean pytorch_roundtrip --model cnn --action import`
- `lake exe torchlean pytorch_roundtrip --model transformer --action import`

These examples are model specific: they show how a known MLP/CNN/Transformer JSON
`state_dict` becomes typed Lean tensors and how generated PyTorch code can be produced for the same
architecture. For arbitrary checkpoints, prefer the general state dict adapter in
`NN.Runtime.PyTorch.Export.StateDict`.

Use this path when the architecture is known and the main question is whether the parameter names and
shapes line up. The import side should reject missing tensors, extra assumptions about layout, and
shape mismatches before a verifier or runtime example consumes the weights.

## What These Examples Check

These examples check the boundary format:

- the expected model family name is present;
- tensor names match the family-specific contract;
- JSON tensor shapes match the Lean type expected by the importer;
- supported graph operators lower into `NN.IR.Graph` rather than an unchecked placeholder;
- unsupported graph structure fails with an explicit error.

An imported PyTorch artifact becomes useful in TorchLean once a later checker or theorem consumes
the accepted graph or weight bundle. PyTorch training, PyTorch autograd, and ATen kernels remain
external producers; the Lean side claim starts from the imported object and the explicit boundary
contract above.

## Reference folders

- `MLP/`, `CNN/`, `Transformer/` contain the model-specific round-trip artifacts: JSON weights,
  Python producer scripts, and the Lean import/export helpers for that model.
- Regenerate JSON weights with:
  - `python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py`
  - `python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py`
  - `python3 NN/Examples/Interop/PyTorch/Transformer/train_transformer.py`

## What not to put here

- General PyTorch checkpoint adapters belong in `NN/Runtime/PyTorch`.
- Verification checkpoint bridges belong under `NN/Verification/*`, not under examples.
- Architecture specific verification loaders that are used outside examples should live in their
  owning verification package.
- Per-run generated files from graph capture belong under `.lake/build`, not in this folder.
