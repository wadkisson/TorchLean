/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd

/-!
# API Public

Clean public API for TorchLean model code.

User code should use the umbrella import:

```lean
import NN
open TorchLean
```

This module remains the lower-level subsystem surface used by the facade. The stable user names are
exported by `NN.Entrypoint.API` and the `NN` umbrella through `NN.API.Public.Facade`:

- `TorchLean.nn` (model/layer builders)
- `TorchLean.optim` (optimizer configs)
- `TorchLean.Trainer` (train/evaluate APIs)
- `TorchLean.Data` (datasets/loaders + CSV/NPY readers)
- `TorchLean.Loss` and `TorchLean.Metrics`

Advanced users can still import `NN.API.Public` directly when they are extending TorchLean itself
or deliberately working below the `NN` umbrella. Import `NN.Entrypoint.API` when the focused
`TorchLean.*` facade names should be available without the broader `NN` umbrella.

The callback-heavy training namespace lives in `NN.API.Public.Training`. It is deliberately not
re-exported from this umbrella module: ordinary code should get training through `TorchLean.Trainer`,
while files that truly need callback runners should import the advanced training
module explicitly.
-/
