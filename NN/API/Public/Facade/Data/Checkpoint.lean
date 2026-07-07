/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Data.Datasets

/-!
# TorchLean Public Checkpoints

Checkpoint loading and saving operations for exact Float bit payloads.
-/

@[expose] public section

namespace TorchLean

namespace Checkpoint

/--
Load a JSON checkpoint containing exact `Float.toBits` parameter values.

The loader checks the saved tensor pack against the model's parameter shapes. Examples such as
`gpt2_saved` use this boundary for inference from saved weights.
-/
def loadParamBits {paramShapes : List Shape} (path : System.FilePath) :
    IO (nn.ParamTensors Float paramShapes) :=
  try
    NN.API.TorchLean.ParamIO.loadParamBits (paramShapes := paramShapes) path
  catch e =>
    throw <| IO.userError s!"Checkpoint: failed to load {path}: {e}"

/--
Load a JSON checkpoint containing exact `Float.toBits` parameter values for one checked model.

Model-specialized companion to `loadParamBits`: callers hand over the model they are about to
evaluate instead of repeating `paramShapes := nn.paramShapes model`.
-/
def loadModelParamBits {σ τ : Shape}
    (model : nn.Sequential σ τ) (path : System.FilePath) :
    IO (nn.ParamTensors Float (nn.paramShapes model)) :=
  loadParamBits (paramShapes := nn.paramShapes model) path

/--
Turn loaded parameter tensors into runtime parameter handles.

Use this when an example wants inference-only execution without constructing a trainer first.
-/
def toRuntimeParams {paramShapes : List Shape}
    (ps : nn.ParamTensors Float paramShapes) :
    IO (NN.API.TorchLean.ParamList Float paramShapes) :=
  _root_.Runtime.Autograd.Torch.ParamList.ofTList (α := Float) (ss := paramShapes) ps

/-- Load exact `Float.toBits` parameters into an existing runtime module. -/
def loadModuleParamBits {paramShapes inputShapes : List Shape}
    (m : Module.ScalarModule Float paramShapes inputShapes)
    (path : System.FilePath) : IO Unit :=
  NN.API.TorchLean.ParamIO.loadModuleParamsBits (paramShapes := paramShapes)
    (inputShapes := inputShapes) m path

/--
Load exact `Float.toBits` parameters into an existing runtime module attached to one checked model.

The model determines the parameter and input shape indices at the call site.
-/
def loadModelIntoModule {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : Module.ScalarModule Float (nn.paramShapes model) [σ, τ])
    (path : System.FilePath) : IO Unit :=
  loadModuleParamBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ]) m path

/--
Load exact `Float.toBits` parameters into a model-attached runtime module when a checkpoint path is
present.
-/
def loadModelIntoModuleIfSome {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : Module.ScalarModule Float (nn.paramShapes model) [σ, τ])
    (path? : Option System.FilePath) : IO Unit :=
  match path? with
  | none => pure ()
  | some path => loadModelIntoModule model m path

/-- Save a runtime module's parameters as exact `Float.toBits` JSON. -/
def saveModuleParamBits {paramShapes inputShapes : List Shape}
    (m : Module.ScalarModule Float paramShapes inputShapes)
    (path : System.FilePath) : IO Unit :=
  NN.API.TorchLean.ParamIO.saveModuleParamsBits (paramShapes := paramShapes)
    (inputShapes := inputShapes) m path

/--
Save a model-attached runtime module's exact `Float.toBits` parameters when an output path is
present, and print the standard confirmation line.
-/
def saveModelIntoPathIfSome {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : Module.ScalarModule Float (nn.paramShapes model) [σ, τ])
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      saveModuleParamBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ]) m path
      IO.println s!"  wrote params: {path}"

end Checkpoint

end TorchLean
