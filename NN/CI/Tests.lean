/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module


public import NN.Tests.Runtime.Floats.AllAutogradTests
public import NN.Tests.Runtime.Floats.PINNDerivResidual
public import NN.Tests.Runtime.Floats.Suite
public import NN.Tests.Runtime.Floats.TorchLeanIRExecEquivCheck
public import NN.Tests.Runtime.Floats.TorchLeanIndexShapeCheck
public import NN.Tests.Runtime.Floats.TorchLeanOpsCheck
public import NN.Tests.Runtime.Floats.TorchLeanSpecMlpEquivCheck
public import NN.Tests.Runtime.Floats.Utils
public import NN.Tests.Runtime.Rationals.AutogradEngineTest
public import NN.Tests.Runtime.Rationals.Suite
public import NN.Widgets

/-!
# Tests CI Suite

Focused CI import suite. `NN.CI.All` combines every suite for exhaustive repository validation.

Local usage:

```bash
lake build NN.CI.All
```
-/

@[expose] public section
