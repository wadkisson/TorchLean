import VersoManual

open Verso.Genre Manual

#doc (Manual) "Graphs And Specs" =>
%%%
tag := "graphs-ir"
%%%


GraphSpec preserves an architecture as a typed Lean term. A backend pass needs a more ordinary
piece of data: an array of operation nodes that can be imported, traversed, serialized, checked, and
assigned to kernels. That representation is `NN.IR.Graph`.

The easiest way to understand it is to build one. Consider

$$`
G(x)=\tanh\!\left(
  \sum_k
  \left[W_2\operatorname{ReLU}(W_1x+b_1)+b_2\right]_k
\right).
`

TorchLean stores this as six nodes:

```
0  input   []    [4]
1  linear  [0]   [5]
2  relu    [1]   [5]
3  linear  [2]   [3]
4  sum     [3]   scalar
5  tanh    [4]   scalar
```

The parent list gives data dependencies. Because every parent ID is smaller than the node ID, the
array order is already a topological execution order.

# Build The Six Nodes

The executable deep dive constructs the graph directly:

```
import NN.IR.Graph

open NN.IR
open Spec

def xShape : Shape := shape![4]
def hShape : Shape := shape![5]
def yShape : Shape := shape![3]

def graph : Graph :=
  let n0 : Node :=
    { id := 0, parents := [],  kind := .input,  outShape := xShape }
  let n1 : Node :=
    { id := 1, parents := [0], kind := .linear, outShape := hShape }
  let n2 : Node :=
    { id := 2, parents := [1], kind := .relu,   outShape := hShape }
  let n3 : Node :=
    { id := 3, parents := [2], kind := .linear, outShape := yShape }
  let n4 : Node :=
    { id := 4, parents := [3], kind := .sum,    outShape := .scalar }
  let n5 : Node :=
    { id := 5, parents := [4], kind := .tanh,   outShape := .scalar }
  { nodes := #[n0, n1, n2, n3, n4, n5] }
```

Unlike GraphSpec, this datatype does not make every edge shape-correct by construction. Node IDs,
raw axis numbers, and declared output shapes are ordinary data. That is deliberate: an importer
must be able to construct a candidate graph from an external document before Lean knows it is
valid.

The price of an import-friendly representation is an explicit validation phase.

# Structure And Shape Are Separate Checks

`Graph.checkWellFormed` checks graph structure:

- node ID equals its array position;
- parents occur earlier in topological order;
- the operation has an admissible number of parents;
- designated input and constant nodes have the required arity.

`Graph.checkShapes` follows with operation-specific shape rules. For the graph above it checks, among
other things, that:

- the payload for node 1 accepts `[4]` and produces `[5]`;
- ReLU preserves `[5]`;
- the payload for node 3 accepts `[5]` and produces `[3]`;
- `sum` produces a scalar;
- `tanh` preserves that scalar.

Try changing only node 2's declared output from `[5]` to `[4]`. The graph remains topologically
well formed, but the shape checker rejects the edge into the second linear layer. This is why a
claim that an imported graph is “validated” should name both checks.

The distinction also matters in the backend adapter:
`NN.Backend.IR.checkedPlanGraphNodesWithRegistry` currently calls `checkWellFormed` before planning,
but it does not replace an importer's shape check. A caller accepting untrusted graph data must not
infer shape validity from a successful plan alone.

# Parameters Live In A Payload

The two `.linear` nodes mention only their activation parent. Their weights and biases are stored in
a payload keyed by node ID:

# The Mathematical Specification

The model we trained earlier can be written on one line:

$$`F_\theta(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2.`

That formula is the mathematical center of the example. It says nothing about mutable buffers,
CUDA launches, PyTorch modules, reverse-mode tapes, or JSON files. It does say exactly how the four
parameter tensors and the input determine the output. TorchLean's `NN.Spec` library is where such
formulas live as Lean definitions.

The distinction is important because executable ML code changes form constantly. A linear layer
may be evaluated by a nested Lean function, a CPU loop, cuBLAS, or an ATen kernel. The specification
does not try to imitate those implementations. It gives them a common statement to implement.

# The Tensor Behind The Formula

A specification tensor is defined recursively from its shape:

- a scalar shape stores one value;
- a dimension of length `n` stores a function from `Fin n` to a smaller tensor.

Thus a vector of length two is, mathematically, two scalar values indexed by `Fin 2`; a matrix of
shape `[3, 2]` is three such vectors. The index type prevents an out-of-range lookup. It also makes
shape induction natural: a proof about an arbitrary tensor can follow the same scalar-or-dimension
recursion as the datatype.

This is not a claim that CUDA stores a matrix as nested Lean functions. Native runtimes use flat,
contiguous buffers. The specification chooses the representation that makes mathematical reasoning
clear; a layout contract is needed when an implementation flattens that value into memory.

The public aliases hide most of the recursive spelling:

```
import NN.Spec.Layers.Linear
import NN.Spec.Layers.Activation

open Spec
open Spec.Tensor

#check Tensor ℝ (shape![2])
#check Tensor ℝ (shape![3, 2])
#check LinearSpec ℝ 2 3
```

For a linear map from two inputs to three outputs,
`LinearSpec ℝ 2 3` contains

$$`W\in\mathbb{R}^{3\times 2},
\qquad b\in\mathbb{R}^{3}.`

The order is the same convention used by PyTorch's `nn.Linear`: output features index the rows of
the weight matrix.

# One Layer, Forward And Backward

The forward definition is deliberately short:

```
def linearSpec {α : Type} [Add α] [Mul α] [Zero α]
    {inDim outDim : Nat}
    (layer : LinearSpec α inDim outDim)
    (x : Tensor α (shape![inDim])) :
    Tensor α (shape![outDim]) :=
  addSpec (matVecMulSpec layer.weights x) layer.bias
```

At coordinate `i`, this means

# GraphSpec: One Architecture, Several Meanings

The pure MLP formula from the previous chapter is excellent for a theorem:

$$`x\mapsto W_2\operatorname{ReLU}(W_1x+b_1)+b_2.`

It is less convenient for a tool that wants to enumerate layers, export parameter shapes, replace
one operation, or lower the same architecture to several targets. For those tasks TorchLean uses
GraphSpec, a small typed language in which the architecture itself is data.

GraphSpec sits between two other layers:

```
public model builders
        ↓
GraphSpec architecture and parameter ABI
        ↓
pure interpretation / TorchLean program / sequential model / DAG tools
        ↓
canonical NN.IR.Graph and runtime-specific lowering
```

It is not a second tensor runtime, and it is not the low-level backend IR. Its job is to preserve
model structure while making every input, output, and parameter shape explicit.

# Write The Running MLP As A Graph

The complete architecture is:

```
import NN.GraphSpec.Models.Mlp

open NN
open NN.GraphSpec
open NN.GraphSpec.Models
open Spec

def mlpGraph (input hidden output : Nat) :
    Graph
      [ shape![hidden, input], shape![hidden],
        shape![output, hidden], shape![output] ]
      (shape![input])
      (shape![output]) :=
  Graph.linear input hidden >>>
  Graph.relu (shape![hidden]) >>>
  Graph.linear hidden output
```

Read the type from right to left:

- the graph consumes one tensor of shape `[input]`;
- it produces one tensor of shape `[output]`;
- its parameter environment contains exactly four tensors;
- the order is `W₁`, `b₁`, `W₂`, `b₂`.

For the `2 → 3 → 1` model used by the GraphSpec tutorial, the parameter shapes are

```
W₁ : [3, 2]   six values
b₁ : [3]      three values
W₂ : [1, 3]   three values
b₂ : [1]      one value
```

There are thirteen trainable scalars. That count is not recovered from strings such as
`"layer1.weight"`. It follows from the graph's type.
