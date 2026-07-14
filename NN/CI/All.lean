/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.CI.Examples
public import NN.CI.Floats
public import NN.CI.Foundation
public import NN.CI.Runtime
public import NN.CI.Tests
public import NN.CI.Theory
public import NN.CI.Verification

/-!
# Complete CI Import Surface

This CI-only umbrella keeps every maintained source module buildable in one environment. Application
code should import `NN` or a focused subsystem instead.

```bash
lake build NN.CI.All
```
-/
