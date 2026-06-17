# NN/Entrypoint

This directory contains curated umbrella imports for the major TorchLean subsystems.

Use these modules when you want one import for a subsystem without depending on the internal file
layout.

Examples:

* `NN/Entrypoint/Spec.lean` (pure spec layer)
* `NN/Entrypoint/Runtime.lean` (runtime execution layer)
* `NN/Entrypoint/IR.lean` (op-tagged graph IR)
* `NN/Entrypoint/Verification.lean` (verification infrastructure)
* `NN/Entrypoint/Proofs.lean` (proof library umbrella)

Most model/training code should prefer:

```lean
import NN
open TorchLean
```

Use these entrypoints when you specifically want a subsystem umbrella without importing the whole
library surface.
