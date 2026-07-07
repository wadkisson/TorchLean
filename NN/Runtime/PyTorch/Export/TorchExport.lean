/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core

/-!
# PyTorch `nn.Module` Graph Capture Adapter

This module emits the Python side of the model-import bridge:

```text
PyTorch nn.Module
  --torch.export / FX capture-->
TorchLean graph JSON (`torchlean.ir.v1`)
  --NN.Runtime.PyTorch.Import.TorchExport.parseGraph-->
NN.IR.Graph
```

Why generate a Python adapter instead of parsing PyTorch objects in Lean?

- PyTorch checkpoints are Python/Pickle/zip artifacts; PyTorch is the external loader that knows
  how to read them.
- `torch.export` and FX already know how to inspect PyTorch programs.
- Lean should receive a small, explicit artifact and then validate it with TorchLean's own IR
  checkers.

The generated adapter is conservative by design. It lowers only ops that already exist in
`NN.IR.OpKind`; unsupported operators fail with a clear message. This is exactly the behavior we
want for verification: better to reject a model than silently erase semantics.

References:
- `torch.export`: `https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html`
- `torch.fx`: `https://docs.pytorch.org/docs/stable/fx.html`
-/

@[expose] public section

namespace Export
namespace PyTorch
namespace TorchExport

open Export.PyTorch

/-- Options for the generated PyTorch graph-capture script. -/
structure GraphBridgeOptions where
  /-- Name of the Python helper function emitted into the script. -/
  functionName : String := "export_torchlean_graph_json"
  /-- If true, use `torch.export.export` first and fall back to FX symbolic tracing. -/
  preferTorchExport : Bool := true
  /-- If true, include raw PyTorch target strings in each node for debugging. -/
  includeDebugTargets : Bool := true
deriving Repr

/--
Emit a Python script that captures a PyTorch module and writes TorchLean graph JSON.

The generated script expects a Python file containing a zero-argument model constructor or class:

```bash
python export_torchlean_graph.py my_model.py MyModel out_graph.json --example-shape 1,4
```

The first implementation target is the shared IR subset: elementwise ops, matmul, reductions,
reshape/permute/flatten/concat, softmax, layernorm, and simple pooling/conv metadata when PyTorch
exposes enough static arguments. Linear layers can appear either as `aten.linear` or as lower-level
`matmul/add` depending on PyTorch's graph capture.
-/
def generateGraphBridgeScript (opts : GraphBridgeOptions := {}) : String :=
  joinLines
    [ "import argparse"
    , "import importlib.util"
    , "import json"
    , "import operator"
    , "from pathlib import Path"
    , "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    , "FORMAT = \"torchlean.ir.v1\""
    , ""
    , "def _shape_of(value):"
    , indentFour "if hasattr(value, \"shape\"):"
    , indentEight "return [int(x) for x in tuple(value.shape)]"
    , indentFour "if isinstance(value, (tuple, list)) and value and hasattr(value[0], \"shape\"):"
    , indentEight "return [int(x) for x in tuple(value[0].shape)]"
    , indentFour "return []"
    , ""
    , "def _node_shape(node):"
    , indentFour "return _shape_of(node.meta.get(\"val\", node.meta.get(\"tensor_meta\")))"
    , ""
    , "def _shape_from_arg(arg):"
    , indentFour "return _node_shape(arg) if hasattr(arg, \"meta\") else []"
    , ""
    , "def _tuple_shapes_of_node(node):"
    , indentFour "val = node.meta.get(\"val\", None)"
    , indentFour "if isinstance(val, (tuple, list)) and not hasattr(val, \"shape\"):"
    , indentEight "return [_shape_of(x) for x in val]"
    , indentFour "tm = node.meta.get(\"tensor_meta\", None)"
    , indentFour "if hasattr(tm, \"shape\"):"
    , indentEight "return []"
    , indentFour "if isinstance(tm, (tuple, list)):"
    , indentEight "return [_shape_of(x) for x in tm]"
    , indentFour "return []"
    , ""
    , "def _is_getitem(node):"
    , indentFour "return node.target is operator.getitem or _target_name(node.target) == \"<built-in function getitem>\""
    , ""
    , "def _is_getattr(node):"
    , indentFour "return node.target is getattr or _target_name(node.target) == \"<built-in function getattr>\""
    , ""
    , "def _tuple_kind(node, model=None):"
    , indentFour "if node.op == \"call_module\" and model is not None:"
    , indentEight "mod = model.get_submodule(str(node.target))"
    , indentEight "if isinstance(mod, nn.MultiheadAttention):"
    , indentEight "    return {"
    , indentEight "        \"kind\": \"multihead_attention\","
    , indentEight "        \"embed_dim\": int(mod.embed_dim),"
    , indentEight "        \"num_heads\": int(mod.num_heads),"
    , indentEight "        \"batch_first\": bool(mod.batch_first),"
    , indentEight "        \"dropout_zero\": bool(float(mod.dropout) == 0.0),"
    , indentEight "        \"bias\": bool(mod.in_proj_bias is not None),"
    , indentEight "    }"
    , indentFour "return {\"kind\": \"py_tuple\"}"
    , ""
    , "def _load_model(module_path: str, ctor_name: str):"
    , indentFour "spec = importlib.util.spec_from_file_location(\"torchlean_user_model\", module_path)"
    , indentFour "if spec is None or spec.loader is None:"
    , indentEight "raise RuntimeError(f\"could not load Python module from {module_path}\")"
    , indentFour "mod = importlib.util.module_from_spec(spec)"
    , indentFour "spec.loader.exec_module(mod)"
    , indentFour "ctor = getattr(mod, ctor_name)"
    , indentFour "model = ctor()"
    , indentFour "model.eval()"
    , indentFour "return model"
    , ""
    , "def _target_name(target):"
    , indentFour "return str(target).replace(\"torch.ops.\", \"\")"
    , ""
    , "def _node_refs(obj, node_to_id):"
    , indentFour "refs = []"
    , indentFour "def visit(x):"
    , indentEight "if x in node_to_id:"
    , indentEight "    if node_to_id[x] is not None:"
    , indentEight "        refs.append(node_to_id[x])"
    , indentEight "elif isinstance(x, (tuple, list)):"
    , indentEight "    for y in x: visit(y)"
    , indentEight "elif isinstance(x, dict):"
    , indentEight "    for y in x.values(): visit(y)"
    , indentFour "visit(obj)"
    , indentFour "return refs"
    , ""
    , "def _first_int(x, default=0):"
    , indentFour "if isinstance(x, int): return int(x)"
    , indentFour "if isinstance(x, (tuple, list)) and x: return int(x[0])"
    , indentFour "return default"
    , ""
    , "def _pair(x, default):"
    , indentFour "if isinstance(x, int): return int(x), int(x)"
    , indentFour "if isinstance(x, (tuple, list)) and len(x) >= 2: return int(x[0]), int(x[1])"
    , indentFour "return default, default"
    , ""
    , "def _normalize_axis(axis, rank):"
    , indentFour "axis = int(axis)"
    , indentFour "return axis + rank if axis < 0 else axis"
    , ""
    , "def _lower_kind(node, model=None):"
    , indentFour "name = _target_name(node.target)"
    , indentFour "args = node.args"
    , indentFour "kwargs = dict(node.kwargs)"
    , indentFour "axis = kwargs.get(\"dim\", kwargs.get(\"axis\", None))"
    , indentFour "if axis is None and len(args) > 1 and isinstance(args[1], int): axis = args[1]"
    , indentFour "if node.op == \"call_module\" and model is not None:"
    , indentEight "mod = model.get_submodule(str(node.target))"
    , indentEight "if isinstance(mod, nn.Linear): return {\"kind\": \"linear\"}"
    , indentEight "if isinstance(mod, nn.ReLU): return {\"kind\": \"relu\"}"
    , indentEight "if isinstance(mod, nn.Tanh): return {\"kind\": \"tanh\"}"
    , indentEight "if isinstance(mod, nn.Sigmoid): return {\"kind\": \"sigmoid\"}"
    , indentEight "if isinstance(mod, nn.Softmax): return {\"kind\": \"softmax\", \"axis\": int(mod.dim)}"
    , indentEight "if isinstance(mod, nn.Flatten): return {\"kind\": \"flatten\", \"value_shape\": _shape_from_arg(args[0]) if args else []}"
    , indentEight "if isinstance(mod, nn.LayerNorm):"
    , indentEight "    rank = len(_node_shape(node))"
    , indentEight "    norm_rank = len(tuple(mod.normalized_shape))"
    , indentEight "    return {\"kind\": \"layernorm\", \"axis\": max(0, rank - norm_rank)}"
    , indentEight "if isinstance(mod, nn.Conv2d):"
    , indentEight "    kH, kW = _pair(mod.kernel_size, 1)"
    , indentEight "    stride = _first_int(mod.stride, 1)"
    , indentEight "    padding = _first_int(mod.padding, 0)"
    , indentEight "    return {\"kind\": \"conv2d\", \"inC\": int(mod.in_channels), \"outC\": int(mod.out_channels), \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding}"
    , indentEight "if isinstance(mod, nn.BatchNorm2d):"
    , indentEight "    return {\"kind\": \"batch_norm2d_nchw_eval\", \"channels\": int(mod.num_features), \"eps\": float(mod.eps)}"
    , indentEight "if isinstance(mod, nn.MaxPool2d):"
    , indentEight "    kH, kW = _pair(mod.kernel_size, 1)"
    , indentEight "    stride = _first_int(mod.stride if mod.stride is not None else mod.kernel_size, kH)"
    , indentEight "    padding = _first_int(mod.padding, 0)"
    , indentEight "    return {\"kind\": \"max_pool2d_pad\" if padding else \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indentEight "if isinstance(mod, nn.AvgPool2d):"
    , indentEight "    kH, kW = _pair(mod.kernel_size, 1)"
    , indentEight "    stride = _first_int(mod.stride if mod.stride is not None else mod.kernel_size, kH)"
    , indentEight "    padding = _first_int(mod.padding, 0)"
    , indentEight "    return {\"kind\": \"avg_pool2d_pad\" if padding else \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indentFour "if name in (\"aten.add.Tensor\", \"aten.add.Scalar\") or node.target is operator.add: return {\"kind\": \"add\"}"
    , indentFour "if name in (\"aten.sub.Tensor\", \"aten.sub.Scalar\") or node.target is operator.sub: return {\"kind\": \"sub\"}"
    , indentFour "if name in (\"aten.mul.Tensor\", \"aten.mul.Scalar\") or node.target is operator.mul: return {\"kind\": \"mul_elem\"}"
    , indentFour "if name.startswith(\"aten.relu\") or node.target is torch.relu or node.target is F.relu: return {\"kind\": \"relu\"}"
    , indentFour "if name.startswith(\"aten.tanh\") or node.target is torch.tanh: return {\"kind\": \"tanh\"}"
    , indentFour "if name.startswith(\"aten.sigmoid\") or node.target is torch.sigmoid: return {\"kind\": \"sigmoid\"}"
    , indentFour "if name.startswith(\"aten.exp\") or node.target is torch.exp: return {\"kind\": \"exp\"}"
    , indentFour "if name.startswith(\"aten.log\") or node.target is torch.log: return {\"kind\": \"log\"}"
    , indentFour "if name.startswith(\"aten.sin\") or node.target is torch.sin: return {\"kind\": \"sin\"}"
    , indentFour "if name.startswith(\"aten.cos\") or node.target is torch.cos: return {\"kind\": \"cos\"}"
    , indentFour "if name.startswith(\"aten.abs\") or node.target is torch.abs: return {\"kind\": \"abs\"}"
    , indentFour "if name.startswith(\"aten.sqrt\") or node.target is torch.sqrt: return {\"kind\": \"sqrt\"}"
    , indentFour "if name.startswith(\"aten.reciprocal\"): return {\"kind\": \"inv\"}"
    , indentFour "if name.startswith(\"aten.maximum\") or node.target is torch.maximum: return {\"kind\": \"max_elem\"}"
    , indentFour "if name.startswith(\"aten.minimum\") or node.target is torch.minimum: return {\"kind\": \"min_elem\"}"
    , indentFour "if name.startswith(\"aten.matmul\") or name.startswith(\"aten.mm\") or node.target is torch.matmul: return {\"kind\": \"matmul\"}"
    , indentFour "if name.startswith(\"aten.sum\") and axis is None: return {\"kind\": \"sum\"}"
    , indentFour "if name.startswith(\"aten.sum\"): return {\"kind\": \"reduce_sum\", \"axis\": int(axis)}"
    , indentFour "if name.startswith(\"aten.mean\"): return {\"kind\": \"reduce_mean\", \"axis\": int(axis)}"
    , indentFour "if name.startswith(\"aten.softmax\") or \"softmax\" in name or node.target is F.softmax:"
    , indentEight "rank = len(_node_shape(node))"
    , indentEight "return {\"kind\": \"softmax\", \"axis\": _normalize_axis(axis if axis is not None else -1, rank)}"
    , indentFour "if name.startswith(\"aten.reshape\") or name.startswith(\"aten.view\") or name in (\"reshape\", \"view\"):"
    , indentEight "return {\"kind\": \"reshape\", \"in_shape\": _shape_from_arg(args[0]) if args else [], \"out_shape\": _node_shape(node)}"
    , indentFour "if name == \"contiguous\":"
    , indentEight "s = _shape_from_arg(args[0]) if args else _node_shape(node)"
    , indentEight "return {\"kind\": \"reshape\", \"in_shape\": s, \"out_shape\": _node_shape(node)}"
    , indentFour "if name.startswith(\"aten.flatten\") or name == \"flatten\" or node.target is torch.flatten: return {\"kind\": \"flatten\", \"value_shape\": _shape_from_arg(args[0]) if args else []}"
    , indentFour "if name.startswith(\"aten.permute\") or name == \"permute\":"
    , indentEight "perm = list(args[1]) if len(args) > 1 and isinstance(args[1], (tuple, list)) else [int(x) for x in args[1:]] if name == \"permute\" else list(kwargs.get(\"dims\", []))"
    , indentEight "return {\"kind\": \"permute\", \"perm\": [int(x) for x in perm]}"
    , indentFour "if name.startswith(\"aten.cat\") or node.target is torch.cat:"
    , indentEight "dim = int(kwargs.get(\"dim\", args[1] if len(args) > 1 and isinstance(args[1], int) else 0))"
    , indentEight "return {\"kind\": \"concat\", \"axis\": dim}"
    , indentFour "if name.startswith(\"aten.transpose\") or name == \"transpose\":"
    , indentEight "d0 = int(args[1]); d1 = int(args[2])"
    , indentEight "if d0 == 0 and d1 == 1: return {\"kind\": \"swap_first_two\"}"
    , indentEight "if d0 == -1 and d1 == -2: return {\"kind\": \"transpose3d_last_two\"}"
    , indentEight "if d0 == -2 and d1 == -1: return {\"kind\": \"transpose3d_last_two\"}"
    , indentFour "if name.startswith(\"aten.layer_norm\") or node.target is F.layer_norm:"
    , indentEight "return {\"kind\": \"layernorm\", \"axis\": 1}"
    , indentFour "if name.startswith(\"aten.linear\") or node.target is F.linear: return {\"kind\": \"linear\"}"
    , indentFour "if \"batch_norm\" in name or node.target is F.batch_norm:"
    , indentEight "shape = _node_shape(node)"
    , indentEight "channels = int(shape[1]) if len(shape) >= 2 else 0"
    , indentEight "eps = float(kwargs.get(\"eps\", args[7] if len(args) > 7 else 1e-5))"
    , indentEight "return {\"kind\": \"batch_norm2d_nchw_eval\", \"channels\": channels, \"eps\": eps}"
    , indentFour "if name.startswith(\"aten.conv2d\") or node.target is F.conv2d:"
    , indentEight "stride = _first_int(kwargs.get(\"stride\", args[3] if len(args) > 3 else 1), 1)"
    , indentEight "padding = _first_int(kwargs.get(\"padding\", args[4] if len(args) > 4 else 0), 0)"
    , indentEight "weight = args[1] if len(args) > 1 else None"
    , indentEight "wshape = _shape_of(weight.meta.get(\"val\") if hasattr(weight, \"meta\") else weight)"
    , indentEight "outC, inC, kH, kW = (wshape + [0, 0, 0, 0])[:4]"
    , indentEight "return {\"kind\": \"conv2d\", \"inC\": inC, \"outC\": outC, \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding}"
    , indentFour "if name.startswith(\"aten.max_pool2d\") or node.target is F.max_pool2d:"
    , indentEight "kH, kW = _pair(args[1] if len(args) > 1 else kwargs.get(\"kernel_size\", 1), 1)"
    , indentEight "stride = _first_int(kwargs.get(\"stride\", args[2] if len(args) > 2 else kH), kH)"
    , indentEight "padding = _first_int(kwargs.get(\"padding\", args[3] if len(args) > 3 else 0), 0)"
    , indentEight "return {\"kind\": \"max_pool2d_pad\" if padding else \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indentFour "if name.startswith(\"aten.avg_pool2d\") or node.target is F.avg_pool2d:"
    , indentEight "kH, kW = _pair(args[1] if len(args) > 1 else kwargs.get(\"kernel_size\", 1), 1)"
    , indentEight "stride = _first_int(kwargs.get(\"stride\", args[2] if len(args) > 2 else kH), kH)"
    , indentEight "padding = _first_int(kwargs.get(\"padding\", args[3] if len(args) > 3 else 0), 0)"
    , indentEight "return {\"kind\": \"avg_pool2d_pad\" if padding else \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indentFour "if _is_getitem(node):"
    , indentEight "idx = int(args[1]) if len(args) > 1 and isinstance(args[1], int) else 0"
    , indentEight "return {\"kind\": \"tuple_getitem\", \"index\": idx}"
    , indentFour "if _is_getattr(node):"
    , indentEight "attr = args[1] if len(args) > 1 else \"<unknown>\""
    , indentEight "raise NotImplementedError(f\"unsupported PyTorch attribute projection: {attr}. If this came from a tuple-returning op such as torch.sort(...).values, add a value-graph/lowering rule for that producer instead of treating the attribute as an ordinary tensor op.\")"
    , indentFour "raise NotImplementedError(f\"unsupported PyTorch op: {name}\")"
    , ""
    , "def _capture(model, example):"
    , indentFour "if " ++ pyBool opts.preferTorchExport ++ ":"
    , indentEight "try:"
    , indentEight "    ep = torch.export.export(model, (example,))"
    , indentEight "    return ep.graph"
    , indentEight "except Exception:"
    , indentEight "    pass"
    , indentFour "from torch.fx import symbolic_trace"
    , indentFour "from torch.fx.passes.shape_prop import ShapeProp"
    , indentFour "gm = symbolic_trace(model)"
    , indentFour "ShapeProp(gm).propagate(example)"
    , indentFour "return gm.graph"
    , ""
    , s!"def {opts.functionName}(model, example, json_path: str):"
    , indentFour "graph = _capture(model, example)"
    , indentFour "nodes = []"
    , indentFour "node_to_id = {}"
    , indentFour "output_id = None"
    , indentFour "input_id = None"
    , indentFour "for node in graph.nodes:"
    , indentEight "if node.op == \"output\":"
    , indentEight "    refs = _node_refs(node.args, node_to_id)"
    , indentEight "    output_id = refs[0] if refs else None"
    , indentEight "    continue"
    , indentEight "if len(getattr(node, \"users\", {})) == 0:"
    , indentEight "    # FX often leaves dead tuple projections behind, e.g. attention weights from"
    , indentEight "    # `y, _ = mha(...)`. They are not part of the exported value, so we omit them"
    , indentEight "    # instead of forcing every unused container projection to have a tensor lowering."
    , indentEight "    node_to_id[node] = None"
    , indentEight "    continue"
    , indentEight "if node.op in (\"get_attr\",):"
    , indentEight "    continue"
    , indentEight "node_id = len(nodes)"
    , indentEight "node_to_id[node] = node_id"
    , indentEight "shape = _node_shape(node)"
    , indentEight "tuple_shapes = _tuple_shapes_of_node(node)"
    , indentEight "value_meta = {\"value_kind\": \"tuple\", \"tuple_shapes\": tuple_shapes} if tuple_shapes else {\"value_kind\": \"tensor\", \"shape\": shape}"
    , indentEight "if node.op == \"placeholder\":"
    , indentEight "    if input_id is not None:"
    , indentEight "        # torch.export lifts parameters/buffers into placeholders. TorchLean IR keeps those"
    , indentEight "        # tensors in an external parameter store, so they are not dataflow parents."
    , indentEight "        node_to_id[node] = None"
    , indentEight "        continue"
    , indentEight "    kind = {\"kind\": \"input\"}"
    , indentEight "    parents = []"
    , indentEight "    if input_id is None: input_id = node_id"
    , indentEight "elif tuple_shapes:"
    , indentEight "    # Preserve tuple/list-valued FX nodes in the import artifact. They are lowered to the"
    , indentEight "    # tensor-only TorchLean IR only when a later semantic lowering rule exists."
    , indentEight "    kind = _tuple_kind(node, model)"
    , indentEight "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
    , indentEight "elif node.op in (\"call_function\", \"call_method\", \"call_module\"):"
    , indentEight "    kind = _lower_kind(node, model)"
    , indentEight "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
    , indentEight "else:"
    , indentEight "    raise NotImplementedError(f\"unsupported FX node op: {node.op}\")"
    , indentEight "entry = {\"id\": node_id, \"parents\": parents, **value_meta, **kind}"
    , indentEight "if " ++ pyBool opts.includeDebugTargets ++ ":"
    , indentEight "    entry[\"debug_target\"] = _target_name(node.target)"
    , indentEight "nodes.append(entry)"
    , indentFour "if input_id is None or output_id is None:"
    , indentEight "raise RuntimeError(\"could not identify graph input/output\")"
    , indentFour "payload = {\"format\": FORMAT, \"input_id\": input_id, \"output_id\": output_id, \"nodes\": nodes}"
    , indentFour "with open(json_path, \"w\", encoding=\"utf-8\") as f:"
    , indentEight "json.dump(payload, f, indent=2, sort_keys=True)"
    , indentFour "return payload"
    , ""
    , "def main():"
    , indentFour "parser = argparse.ArgumentParser(description=\"Export a PyTorch nn.Module to TorchLean IR JSON\")"
    , indentFour "parser.add_argument(\"module\", help=\"Python file containing the model class/constructor\")"
    , indentFour "parser.add_argument(\"ctor\", help=\"Zero-argument model class or constructor name\")"
    , indentFour "parser.add_argument(\"json\", help=\"Output graph JSON path\")"
    , indentFour "parser.add_argument(\"--example-shape\", required=True, help=\"Comma-separated example input shape, e.g. 1,4\")"
    , indentFour "args = parser.parse_args()"
    , indentFour "shape = tuple(int(x) for x in args.example_shape.split(',') if x)"
    , indentFour "model = _load_model(args.module, args.ctor)"
    , indentFour "example = torch.randn(*shape)"
    , indentFour s!"payload = {opts.functionName}(model, example, args.json)"
    , indentFour "print(f\"wrote {len(payload['nodes'])} TorchLean IR nodes to {args.json}\")"
    , ""
    , "if __name__ == \"__main__\":"
    , indentFour "main()"
    ]

end TorchExport
end PyTorch
end Export
