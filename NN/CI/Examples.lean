/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Data.SamplePaths
public import NN.Examples.Zoo
public import NN.Examples.DeepDives


/-!
# Examples CI Suite

Focused CI import suite. `NN.CI.All` combines every suite for exhaustive repository validation.

Local usage:

```bash
lake build NN.CI.All
```
-/

@[expose] public section
