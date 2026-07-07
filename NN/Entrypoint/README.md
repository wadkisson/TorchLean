# `NN/Entrypoint`

This directory contains curated umbrella imports for the major TorchLean subsystems.

Lean projects age better when users do not have to import implementation leaves directly. The
entrypoint modules are the stable doors into TorchLean: one for the public API, one for tensors, one
for specs, one for runtime code, one for verification, and so on. Internal files can move as the
library grows; the entrypoints should remain the names downstream code can rely on.

Most model and training files should still start with the full public import:

```lean
import NN
open TorchLean
```

The focused entrypoints are for subsystem files, documentation examples, and downstream projects
that deliberately want a smaller import path.

## Entrypoint Map

| Entrypoint | Use it when |
| --- | --- |
| `NN.Entrypoint.API` | You want the public `TorchLean.*` namespaces without the whole proof/model zoo API. |
| `NN.Entrypoint.Tensor` | You are working with shaped tensors and tensor constructors. |
| `NN.Entrypoint.Spec` | You need mathematical tensor, layer, model, loss, or dynamical-system definitions. |
| `NN.Entrypoint.Runtime` | You need executable tensors, autograd, trainers, optimizers, PyTorch interop, or CUDA runtime code. |
| `NN.Entrypoint.IR` | You need the op-tagged graph representation, shape checks, payloads, and graph semantics. |
| `NN.Entrypoint.GraphSpec` | You need typed architecture descriptions that lower into TorchLean runtime or IR. |
| `NN.Entrypoint.Verification` | You need certificate checkers, robustness workflows, PINN/ODE/geometry/VNN-COMP support, or proof-backed verification handles. |
| `NN.Entrypoint.MLTheory` | You need CROWN/LiRPA theory, optimizer laws, learning theory, generative objectives, SSL algebra, or approximation theorems. |
| `NN.Entrypoint.Proofs` | You need the proof library umbrella for tensor, autograd, runtime approximation, or RL proofs. |
| `NN.Entrypoint.Floats` | You need FP32/IEEE-style executable semantics, interval floats, or finite-path floating-point bridges. |
| `NN.Entrypoint.TorchLeanModels` | You need model-family entrypoints used by the model zoo and examples. |
| `NN.Entrypoint.Widgets` | You need editor widgets for tensors, graphs, runtime traces, verification views, or logs. |

## Choosing An Import

For normal model code, `import NN` is the cleanest choice. It gives the public API, model helpers,
runtime conveniences, and examples the shape users expect.

For implementation and proof files, a narrower entrypoint often gives a clearer dependency story. A
file proving graph semantics should not need CUDA runtime imports. A certificate checker should not
pull in the model zoo unless it is actually checking a model-zoo artifact. Keeping those imports
small makes build failures easier to understand and helps the dependency graph stay auditable.

## Rule Of Thumb

Do not import a subsystem entrypoint to shorten an example. If the code is ordinary user code, use
`import NN`. If the file is proving something about one layer, importing the focused entrypoint
makes the dependency boundary easier to see.

For example, a training tutorial should use:

```lean
import NN
open TorchLean
```

but a checker that only needs graph semantics and certificate structures should prefer
`NN.Entrypoint.IR` plus `NN.Entrypoint.Verification`.

## Adding A New Entrypoint

Add a new entrypoint only when a real subsystem boundary has emerged. A good entrypoint name should
describe a stable concept, not a temporary folder layout. After adding one, update this table and make
sure at least one small downstream file imports it directly; otherwise it is probably not a public
boundary yet.
