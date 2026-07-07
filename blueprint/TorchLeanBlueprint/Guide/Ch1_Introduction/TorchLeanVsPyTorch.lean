import VersoManual

open Verso.Genre Manual

#doc (Manual) "TorchLean vs. PyTorch" =>
%%%
tag := "torchlean_vs_pytorch"
%%%

PyTorch and TorchLean should not be compared as if they were trying to solve the same problem.
PyTorch is the right default for broad, high-performance ML engineering. TorchLean asks a narrower
question: what extra structure do we need when a model must be run, inspected, lowered to a graph,
imported or exported, and connected to a formal claim?

The useful comparison is therefore not "which library is better?" but "which library should own
which part of the workflow?" PyTorch is excellent when the primary task is to train, debug, scale,
profile, and deploy models in the existing Python ecosystem. TorchLean becomes useful when the model
or artifact must also be a Lean object: something with typed shapes, named parameters, graph
denotation, numerical semantics, and checkable Lean claims.

# A Fair Comparison

The short version is:

- PyTorch optimizes for breadth, speed, ecosystem integration, and deployment practice.
- TorchLean optimizes for semantic access: typed shapes, explicit parameters, graph denotations,
  numerical meanings, and checker-friendly artifacts.
- A realistic workflow can use both systems.

For the main design dimensions:

- Main goal: PyTorch is a production ML engineering system. TorchLean is proof-aware ML
  infrastructure.
- Tensor shapes: PyTorch treats shapes dynamically. TorchLean often carries shapes in Lean types.
- Parameters: PyTorch modules own mutable state. TorchLean keeps architecture and payload as
  explicit values.
- Autograd: PyTorch provides a mature, broad engine. TorchLean exposes supported gradients through
  definitions that can be related to Lean semantics.
- Graphs: PyTorch graph tools support capture, compilation, and deployment. TorchLean uses a shared
  IR with a Lean denotation.
- Floating point: PyTorch follows backend behavior. TorchLean names real, float32, executable
  IEEE-754, and native-runtime layers separately.

That is why TorchLean is not a drop-in replacement for PyTorch. It does not try to match PyTorch's
operator coverage, distributed training stack, profiler ecosystem, or deployment maturity. The
tradeoff goes the other direction: for the supported fragments, TorchLean asks for more explicit
structure so later code can state what was checked and what was proved.

# Modules And Parameters

In PyTorch, a model is usually an `nn.Module`. The module owns its parameters and buffers, and
calling `model(x)` runs `forward` using that internal state. That ergonomic choice is exactly right
for many training scripts.

```
# PyTorch
model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 5),
)
out = model(x)
```

TorchLean keeps the authoring style familiar, but separates the model description from the parameter
payload that will be executed, saved, lowered, or verified.

```
-- TorchLean
import NN

open TorchLean

def model : nn.M (nn.Sequential (Shape.vec 10) (Shape.vec 5)) :=
  nn.Sequential![
    nn.Linear 10 32,
    nn.GELU,
    nn.Linear 32 5
  ]

def trainer :=
  Trainer.new model { task := .classification, seed := 2026 }
```

Informally, the forward pass is a function of both the architecture and the parameters:

$$`\operatorname{forward}(\operatorname{architecture},\theta,x)=y`

The first major difference is how visible the parameters are. In PyTorch, `state_dict()` exposes the
parameters when needed. In TorchLean, the parameter bundle is already part of the ordinary data
path, so graph lowering, checkpoint exchange, and theorem statements can refer to the same weights.

This difference affects audits. In PyTorch, an audit often asks whether the `state_dict`, module
definition, preprocessing code, and export script agreed. In TorchLean, the aim is to make that
agreement a typed path: architecture plus payload plus input convention becomes the object consumed
by graph lowering and verification. The agreement may still depend on an external import boundary,
but the boundary is named.

# Shapes And Types

PyTorch tensors are dynamically shaped. That flexibility is one of PyTorch's strengths: a script can
build tensors from files, batch them in many ways, and dispatch to highly optimized kernels at
runtime. The cost is that many shape mistakes are discovered only when a particular execution path
hits a mismatched operation, or worse, when broadcasting performs a valid but unintended operation.

TorchLean uses Lean's dependent types for the core tensor APIs. A tensor carries both a scalar type
and a shape:

$$`\operatorname{Tensor}(\alpha,s)`

This does not mean every possible data problem is solved statically. Files can still be malformed,
an imported payload can still be rejected, and dynamic loaders can still fail. The difference is
that once a tensor has entered the typed core, many dimensional contracts are checked before the
runtime kernel is reached.

That tradeoff is intentional. TorchLean gives up some of Python's dynamism in exchange for a
stronger statement about the computations that do elaborate.

The shape type is not a claim about performance, accuracy, or robustness. It is a static contract.
It can rule out a large family of dimensional mistakes, but it does not prove that the labels are
correct, that the dataset is representative, or that the model generalizes. TorchLean uses types for
structural facts and separate checks or theorems for semantic facts.

# Training And Autograd

A standard PyTorch loop mutates optimizer state and accumulates gradients in tensor fields:

```
for step in range(steps):
    opt.zero_grad()
    pred = model(x)
    loss = mse(pred, y)
    loss.backward()
    opt.step()
```

TorchLean keeps the same conceptual rhythm, but the training state is explicit. Gradients are
returned by the differentiation machinery, optimizer state is passed through the step, and logs are
values that can be rendered or persisted.

$$`(\theta,\mathrm{optState},\mathrm{rng},x,y)
\longmapsto
(\theta',\mathrm{optState}',\mathrm{rng}',\mathrm{report})`

PyTorch's autograd is mature, broad, and deeply optimized. TorchLean's advantage is semantic access:
for supported fragments, the backward pass can be related to a Lean specification and the resulting
artifacts can be inspected inside the same formal environment.

This is a narrow claim. A successful TorchLean training run is still a runtime event, not a proof
that the optimizer found a good model. A Lean theorem about a supported backward rule is a theorem
about that rule's denotation, not a benchmark result. The value of putting training machinery near
specifications is that a later proof can name the same parameter bundle, loss, graph, or derivative
definition without translating through an informal diagram.

For the public training API, see the [public API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean) and the
[training runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Train.lean).

# Graphs And Verification

PyTorch has several graph-oriented tools, including tracing, FX, `torch.export`, and compiler
pipelines. They are engineering tools for capture, transformation, optimization, and deployment.
They are not, by themselves, Lean theorem statements about a model.

This is not a criticism of those tools. PyTorch's
[`torch.export`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html) is
captures a full graph representation of a PyTorch program for portable, Python-less execution
contexts. That is a different contract from a Lean denotation. TorchLean can interoperate
with external graph artifacts, but the formal claim eventually has to name a Lean object and the
boundary through which the external artifact entered.

TorchLean's graph path has a different target. A supported model can be lowered to
[NN.IR.Graph](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean), a graph whose nodes name their operations and have a Lean denotation.
Verifier passes and certificate checkers can then talk about that graph directly:

$$`\operatorname{NN.IR.Graph.denoteAll}(g,\theta,x)`

A robustness checker, for example, should not have to trust a training script. It should receive a
graph, a parameter payload, an input region, and a certificate or bound propagation result whose
meaning is defined in Lean.

TorchLean differs most clearly from an ordinary tensor runtime here: the runnable workflow stays
connected to a semantic object that proofs and checkers can cite.

# Import And Export

PyTorch remains valuable at the boundary. It has the ecosystem for large datasets, pretrained
checkpoints, distributed training, debugging tools, and deployment practices. TorchLean therefore
supports interop, but it keeps the contract focused.

The supported round trip is family based:

1. choose a known architecture family,
2. agree on parameter names, order, and tensor shapes,
3. train or edit the payload in Python,
4. serialize the payload,
5. re-import it into Lean and check the names and shapes again.

The relevant APIs are [PyTorch export](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Export.lean),
[PyTorch import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Import.lean), and the
[round trip examples](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch.lean). A narrow contract is easier to audit
and easier to connect to verification.

Interop claims should be written at the right strength:

- "The file was parsed" is an input/output claim.
- "The names and shapes match this architecture family" is a structural check.
- "The imported payload denotes the same function as the source module" requires a semantic bridge
  for the supported operators and preprocessing convention.
- "The deployed CUDA path implements the same function" is an additional runtime or proof boundary.

TorchLean tries to make the first two straightforward and to expose the hypotheses needed for the
last two.

# Floating Point And Kernels

PyTorch users normally rely on the behavior of the selected backend: CPU kernels, CUDA kernels,
vendor libraries, compiler fusions, and device specific floating point choices. That default is
sound engineering, but it leaves a verification question: which numeric semantics did the claim use?

TorchLean names the relevant layers separately:

- real valued specifications for clean mathematical statements,
- float32 proof models such as `FP32`,
- executable IEEE binary32 models such as `IEEE32Exec`,
- native runtime kernels behind explicit assumptions.

The separation does not make native kernels disappear from the trusted base. It makes the boundary
precise: a theorem can say whether it is about real semantics, executable binary32 semantics, or a
native backend related to those semantics by a stated bridge.

# When To Use Which

Use PyTorch when you need the full production ecosystem: broad model coverage, mature GPU kernels,
distributed training, pretrained checkpoints, and standard deployment integrations.

Use TorchLean when the supported model family needs to be a Lean object: when shapes should be part
of the type, when the graph and payload should be inspectable, when imported weights need a checked
contract, or when a verifier result should connect to a formal statement.

A realistic workflow can use both. Train in PyTorch when that is the right engineering choice.
Export a known architecture and payload. Import it into TorchLean. Inspect the graph. Check the
certificate. State the remaining assumptions. TorchLean's contribution is the semantic layer around
that workflow: the model, graph, payload, certificate, and theorem statement can be kept in one
checked vocabulary.

# References

- Paszke et al., ["PyTorch: An Imperative Style, High-Performance Deep Learning Library"](https://arxiv.org/abs/1912.01703),
  NeurIPS 2019.
- PyTorch `torch.export` documentation: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html
- ONNX project documentation: https://onnx.ai/
- de Moura et al., ["The Lean Theorem Prover"](https://lean-lang.org/papers/system.pdf), CADE 2015.
