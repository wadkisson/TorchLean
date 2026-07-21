import VersoManual

open Verso.Genre Manual

#doc (Manual) "PyTorch Roundtrip" =>
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

The checked-in
[round-trip program](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
uses the same importer for an MLP, a convolutional network, and a transformer encoder. The examples
below differ in parameter layout, but not in the rule at the boundary: Python supplies a named
payload, and Lean either reconstructs the expected typed parameters or rejects it.

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
