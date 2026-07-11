import VersoManual

open Verso.Genre Manual

#doc (Manual) "PyTorch Round Trip" =>
%%%
tag := "pytorch-roundtrip"
%%%

Many TorchLean workflows will still involve Python. A model may be trained in PyTorch because the
dataset, optimizer, or engineering environment belongs there. The question for TorchLean is what
happens after that training run. Can the result come back into Lean as a checked artifact, with
names, shapes, and parameter order made explicit?

The round trip is narrow by design. TorchLean does not import arbitrary `nn.Module` objects. It
supports known model families with known layouts. A bridge this small can be audited, tested, and
connected to later graph and verification work.

For runnable examples around this boundary, see:

- [PyTorch round trip source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- [Torch export check source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/TorchExportCheck.lean)
- [Torch IR and PyTorch example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/TorchIRPyTorch.lean)

The main declarations for this boundary are
[PyTorch export](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Export.lean),
[PyTorch import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Import.lean),
[interop examples](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch.lean), and the spec pages for
[modules](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Module.lean) and [models](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models.lean).

# The Contract

The round trip has five steps:

1. Lean defines the expected model family.
2. Lean exports matching PyTorch code.
3. Python trains or modifies the weights.
4. Python writes a named tensor payload.
5. Lean imports the payload and checks the family tag, parameter names, order, shapes, and scalar
   format.

The boundary between Lean and PyTorch stays small enough that failures are usually local: a
mismatched name, a mismatched shape, or an unexpected model family.

If any of those checks fail, the artifact stops at the boundary. It does not become a TorchLean
model by accident.

The shape of the boundary is intentionally closer to a `state_dict` contract than to Python object
serialization. PyTorch users are used to seeing names such as:

```
layer1.weight
layer1.bias
layer2.weight
layer2.bias
```

TorchLean wants those names to become a checked parameter payload, not an implicit module object.
The import code should be able to say exactly which TorchLean parameter each Python tensor is meant
to fill.

# What Lean Checks On Import

The importer is strict by design. It checks the payload before constructing the typed parameter
bundle used by the rest of TorchLean.

The checks include:

- the model family tag,
- the parameter names Lean expects for that family,
- the parameter order used by the typed bundle,
- the shape attached to each tensor,
- the number of values implied by that shape,
- the scalar payload format.

For example, if Lean expects:

```
linear1.weight : shape![8, 2]
```

but the JSON contains:

```
linear1.weight : shape![2, 8]
```

the import fails before the value becomes a model parameter. A round trip should either reconstruct
the same typed layout or stop at the boundary with a concrete error.

# What Gets Serialized

The round trip does not serialize an arbitrary Python object graph. Instead it serializes a small
amount of model family metadata plus a named tensor payload. The exact schema depends on the family,
but the shape of the contract is always the same:

- choose a known architecture family,
- agree on parameter order and tensor layout,
- serialize tensors by name into JSON,
- check that layout again on the Lean side before constructing typed parameters.

Lean then reconstructs the typed parameter bundle that the rest of TorchLean expects. The JSON file
is transport, not semantics.

For a small MLP, the payload can be read informally as:

```
{
  "family": "mlp",
  "parameters": [
    { "name": "linear1.weight", "shape": [8, 2], "values": [...] },
    { "name": "linear1.bias",   "shape": [8],    "values": [...] },
    { "name": "linear2.weight", "shape": [1, 8], "values": [...] },
    { "name": "linear2.bias",   "shape": [1],    "values": [...] }
  ]
}
```

This is not a replacement for PyTorch serialization. It is a small exchange format for known
TorchLean families. If the training pipeline needs the full PyTorch object graph, keep that object
in Python and export only the checked payload that TorchLean understands.

# Model Families

The repository uses three family examples:

- MLP: the smallest parameter and shape contract.
- CNN: convolution weights and image layouts.
- Transformer encoder: attention projections and many named parameters.

## MLP

Open:

- Lean entrypoint: [PyTorch round trip source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- Python: [MLP training script](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/MLP/train_mlp.py)
- Output JSON: `NN/Examples/Interop/PyTorch/MLP/mlp.json`

The MLP example is the smallest place to read the round trip as a contract rather than as infrastructure.
The architecture is small enough to inspect exported names, compare them against PyTorch's
`state_dict`, and see exactly what Lean checks on re-import.

The same path also shows the common failure modes. If the exported JSON is wrong, Lean usually
rejects it because the family name, tensor shape, or parameter ordering does not match the expected
typed layout.

Example command sequence:

```
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model mlp --action export
python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model mlp --action import
```

## CNN

Open:

- Lean entrypoint: [PyTorch round trip source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- Python: [CNN training script](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/CNN/train_cnn.py)
- Output JSON: `NN/Examples/Interop/PyTorch/CNN/cnn.json`

The CNN example exercises nontrivial tensor shapes and weight layouts, while remaining small enough
to debug by hand when the importer reports a mismatch.

Example command sequence:

```
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model cnn --action export
python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model cnn --action import
```

## Transformer (Encoder)

Open:

- Lean entrypoint: [PyTorch round trip source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- Python: [Transformer training script](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Transformer/train_transformer.py)
- Output JSON: `NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json`

The transformer example shows why the boundary must stay explicit. Once attention layers and
multiple projections appear, Lean needs a known layout to check, not a vague promise that every
PyTorch module can be imported.

The same pattern scales to larger model families: Lean checks the declared shapes and parameter
layout, Python runs the training loop, and the JSON payload transports only named parameters back
across the boundary.

Example command sequence:

```
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model transformer --action export
python3 NN/Examples/Interop/PyTorch/Transformer/train_transformer.py
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model transformer --action import
```

# TorchLean, IR, and Generated PyTorch Code

A second interop path complements the JSON round trip workflow:

- file: [Torch IR and PyTorch example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/TorchIRPyTorch.lean)

This example compiles a TorchLean model to the shared IR and then emits runnable PyTorch code for a
curated set of architectures:

- `linear`, `mlp`, `sum`, `autoencoder`
- `mha`, `mha-mask`
- `transformer`

Typical usage:

```
lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py
python3 exported_model.py
```

PyTorch users get a translation example for architectures richer than a single linear layer, and the
example shows where the compiled IR acts as the interchange format.

# A Small Python View

The Python half stays ordinary. TorchLean does not replace the PyTorch workflow; it makes the
boundary between PyTorch and Lean explicit enough to audit.

```
import torch

model = build_exported_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(steps):
    pred = model(x_train)
    loss = torch.nn.functional.mse_loss(pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

save_torchlean_json(model, "mlp.json")
```

The round trip has exactly that shape: Lean defines the expected structure, Python performs the
training, and Lean checks the returned payload against the same structure.

# Tensor Exchange Versus Model Import

It is useful to distinguish three interop layers:

- *tensor exchange* moves arrays between frameworks;
- *parameter import* fills a known TorchLean model family with checked weights;
- *semantic import* claims that a foreign program has the same meaning as a TorchLean graph.

DLPack belongs mostly to the first layer: it is a standard in-memory tensor exchange format used by
array and tensor libraries. It can help avoid extra copies, but it does not by itself tell Lean that
a Python model has the same architecture, parameter names, or proof semantics.

TorchLean's current round trip implements the second layer: it checks family metadata and named
parameter tensors. Semantic import is stronger and requires an IR denotation and a proof relating the
foreign program to it.

# Choosing The Interop Boundary

Use the round trip when the training workflow belongs in Python but the returned artifact should
enter TorchLean with typed parameters and auditable metadata. Run the training loop in Lean when its
state transitions must remain explicit. Lower to IR when the model must be inspected as a graph or
connected to a theorem.

# Guarantees And Limits

When the round trip succeeds, the result is a controlled bridge between Lean and Python:

- Lean can emit a model skeleton and companion files.
- Python can train or export weights using a matching layout.
- Lean can read the exported payload back in and continue on the proof or verification side.

The scope is narrow:

- arbitrary PyTorch training needs its own semantic or artifact bridge,
- the importer covers the supported artifact formats rather than the full PyTorch ecosystem,
- binary32 behavior is handled by the floating point bridge rather than by the round trip format.

Binary32 claims still require the relevant TorchLean float backend and the theorems in the
floating point chapters.

For a direct comparison with PyTorch, read *TorchLean vs PyTorch*.
Verification after import: *Graphs and IR*, then *Verification*.

The round trip is a checked artifact workflow, not a universal conversion tool. Python can remain
the right place to train. Lean becomes the place where the returned object is named, shaped,
inspected, and prepared for graph analysis or verification work.

# References

- TorchLean paper (George et al., 2026): project overview, shared IR architecture, and verification-driven
  architecture. https://arxiv.org/abs/2602.22631
- PyTorch serialization notes: the official reference for `state_dict`-style save/load workflows.
  https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch `nn.Module` documentation: helpful background for parameter naming and module-family
  conventions. https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- DLPack documentation: useful background for tensor exchange, distinct from model-family import.
  https://dmlc.github.io/dlpack/latest/
