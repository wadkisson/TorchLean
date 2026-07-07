/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Base

/-!
# TorchLean NN Facade Basics

Foundational public names and seed operations for the `TorchLean.nn` namespace.
-/

@[expose] public section

namespace TorchLean

namespace nn

abbrev Sequential := NN.API.nn.Sequential

@[inherit_doc NN.API.nn.LayerDef]
abbrev LayerDef := NN.API.nn.LayerDef

@[inherit_doc NN.API.nn.M]
abbrev M := NN.API.nn.M

@[inherit_doc NN.API.nn.manualSeed]
abbrev manualSeed := NN.API.nn.manualSeed

@[inherit_doc NN.API.nn.run]
abbrev run {α : Type 2} (seed : Nat) (x : M α) : α :=
  NN.API.nn.run seed x

@[inherit_doc NN.API.nn.runGlobal]
abbrev runGlobal {α : Type} (x : M α) : IO α :=
  NN.API.nn.runGlobal x

@[inherit_doc NN.API.nn.nextSeed]
abbrev nextSeed := NN.API.nn.nextSeed

@[inherit_doc NN.API.nn.nextSeeds]
abbrev nextSeeds := NN.API.nn.nextSeeds

@[inherit_doc NN.API.nn.freshSeed]
abbrev freshSeed := NN.API.nn.freshSeed

@[inherit_doc NN.API.nn.paramShapes]
abbrev paramShapes {σ τ : Shape} (model : Sequential σ τ) : List Shape :=
  NN.API.nn.paramShapes model

export NN.API.nn
  (paramRequiresGrad initParams updateBuffers programWithMode forwardProgram)

end nn

end TorchLean
