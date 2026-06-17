/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Library

/-!
# NN

Canonical TorchLean umbrella import.

This re-exports `NN.Library`, the curated umbrella for TorchLean's reusable API, runtime, proof,
verification, and widget surface. Ordinary downstream files can start here:

```lean
import NN
open TorchLean
```

The public names remain under the `TorchLean` namespace because that is the project API, but the
module import is `NN`. For smaller imports, use `NN.Entrypoint.API`, `NN.Entrypoint.Tensor`,
`NN.Entrypoint.IR`, or another subsystem entrypoint.
-/

@[expose] public section
