/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Base
public import NN.API.Public.Facade.Core
public import NN.API.Public.Facade.NN
public import NN.API.Public.Facade.Runtime
public import NN.API.Public.Facade.Trainer
public import NN.API.Public.Facade.Data
public import NN.API.Public.Facade.Classical

/-!
# TorchLean Public API

The canonical user-facing `TorchLean.*` namespaces exported by `NN.API` and `NN`. Neural architectures are
available under `TorchLean.nn.models`; classical and statistical models are available under
`TorchLean.classical`.

Repository model-zoo commands are intentionally separate from the application API. Import
`NN.Examples.ModelZoo` in command implementations that need those adapters.
-/
