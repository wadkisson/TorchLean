/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Public.NN
public import NN.API.Public.Training
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd

/-!
# API Public

PyTorch-style public API for TorchLean.

Most user code should be able to `import NN` and then work with the public namespaces exported here:

- `API.nn`     (model/layer builders)
- `API.optim`  (optimizer configs for training)
- `API.Adapters` (LoRA and other model adapters)
- `API.train`  (fit/predict helpers)
- `API.Data`   (datasets/loaders + CSV/NPY readers)
- `API.autograd` (grad/vjp/jacobian helpers)
- `API.rand` (deterministic RNG helpers)
- `API.text` (tokenizers and text-model helpers)
- `API.ssl` (self-supervised sample/objective helpers)
-/
