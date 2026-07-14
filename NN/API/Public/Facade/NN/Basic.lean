/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Base

/-!
# TorchLean NN Basics

Foundational model and seed operations exposed under `TorchLean.nn`.
-/

@[expose] public section

namespace TorchLean

namespace nn

export NN.API.nn
  (Sequential LayerDef M
   manualSeed run runGlobal nextSeed nextSeeds freshSeed
   paramShapes paramRequiresGrad initParams updateBuffers programWithMode forwardProgram)

end nn

end TorchLean
