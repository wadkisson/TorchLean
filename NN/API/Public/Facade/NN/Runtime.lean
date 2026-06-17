/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN.Params

/-!
# TorchLean NN Runtime

Compiled/eager prediction and scalar-module operations for checked sequential models.
-/

@[expose] public section

namespace TorchLean

namespace nn

abbrev TensorConv (α : Type) :=
  _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α

/-- Evaluate one model input without building an autograd tape. -/
abbrev eval1NoGrad {σ τ : Shape}
    (opts : Options)
    (model : Sequential σ τ)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape]
    [TensorConv α]
    (params : NN.API.TorchLean.ParamList α (paramShapes model))
    (x : Tensor.T α σ) : IO (Tensor.T α τ) :=
  NN.API.TorchLean.NN.Seq.eval1NoGrad (α := α) opts model params x

/-- Evaluate one model input with the selected runtime options. -/
abbrev eval1 {σ τ : Shape}
    (opts : Options)
    (model : Sequential σ τ)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape]
    [TensorConv α]
    (params : NN.API.TorchLean.ParamList α (paramShapes model))
    (x : Tensor.T α σ) : IO (Tensor.T α τ) :=
  NN.API.TorchLean.NN.Seq.eval1 (α := α) opts model params x

/-- Compile a sequential model for repeated prediction. -/
abbrev compileOut {σ τ : Shape} (model : Sequential σ τ)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape] :
    IO (_root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes model ++ [σ]) τ) :=
  NN.API.TorchLean.NN.Seq.compileOut (α := α) model

/-- Run one compiled prediction with explicit parameter tensors. -/
abbrev predict1 {σ τ : Shape} (model : Sequential σ τ)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape]
    (compiled : _root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes model ++ [σ]) τ)
    (params : ParamTensors α (paramShapes model))
    (x : Tensor.T α σ) : Tensor.T α τ :=
  NN.API.TorchLean.NN.Seq.predict1 (α := α) model compiled params x

@[inherit_doc NN.API.nn.withModel]
abbrev withModel {σ τ : Shape} {β : Type}
    (mk : M (Sequential σ τ)) (k : (model : Sequential σ τ) → IO β) : IO β :=
  NN.API.nn.withModel mk k

@[inherit_doc NN.API.nn.scalarModuleDef]
abbrev scalarModuleDef {σ τ : Shape} (model : Sequential σ τ)
    (loss : ∀ {α : Type}, [Runtime.TensorScalar α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar) :
    ScalarModuleDef (paramShapes model) [σ, τ] :=
  NN.API.nn.scalarModuleDef model loss

@[inherit_doc NN.API.nn.mseScalarModuleDef]
abbrev mseScalarModuleDef {σ τ : Shape} (model : Sequential σ τ)
    (reduction : LossReduction := .mean) :
    ScalarModuleDef (paramShapes model) [σ, τ] :=
  NN.API.nn.mseScalarModuleDef model reduction

@[inherit_doc NN.API.nn.crossEntropyOneHotScalarModuleDef]
abbrev crossEntropyOneHotScalarModuleDef {σ τ : Shape} (model : Sequential σ τ)
    (reduction : LossReduction := .mean) :
    ScalarModuleDef (paramShapes model) [σ, τ] :=
  NN.API.nn.crossEntropyOneHotScalarModuleDef model reduction

end nn

end TorchLean
