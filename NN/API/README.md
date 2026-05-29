# NN/API

This directory collects the public API modules for TorchLean. The goal is to give users a small set
of imports with names that stay stable as the runtime and specification layers evolve.

Recommended imports:

* `import NN` for most users.
* `import NN.API.Public` for the PyTorch-style surface (`NN.API.nn`, `NN.API.optim`, `NN.API.train`, ...).
* `import NN.API.Runtime` when you need the lower level runtime surface (`NN.API.TorchLean.*`).

`NN.API.Public` is the import users should remember when they want the PyTorch-style surface:
model builders, training, seeded builders, autograd helpers, datasets, tokenizers, and adapters.

`NN.API.Runtime` is for code that works directly with the executable runtime: tensor operations,
layer helpers, autograd, module execution, supervised training, and session-level tools.

Design goal:

Keep the public surface stable and discoverable while the runtime implementation evolves behind it.
Users should not have to chase imports across `NN/Runtime/*`.
