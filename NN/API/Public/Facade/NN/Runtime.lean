/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN.Params

/-!
# TorchLean NN Runtime

Forward, prediction, compiled inference, and scalar-module operations for checked sequential models.
-/

@[expose] public section

namespace TorchLean

namespace nn

abbrev TensorConv (α : Type) :=
  _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α

/-- Evaluation-mode layer behavior, matching PyTorch's `model.eval()` concept. -/
abbrev eval : NN.API.TorchLean.NN.Mode :=
  .eval

/-- Training-mode layer behavior, matching PyTorch's `model.train()` concept. -/
abbrev train : NN.API.TorchLean.NN.Mode :=
  .train

/-- Run one eager forward pass under an explicit mode. -/
abbrev forward {σ τ : Shape}
    (model : Sequential σ τ)
    (opts : Options)
    (mode : NN.API.TorchLean.NN.Mode)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape]
    [TensorConv α]
    (params : NN.API.TorchLean.ParamList α (paramShapes model))
    (x : Tensor.T α σ) : IO (Tensor.T α τ) :=
  NN.API.TorchLean.NN.Seq.forward (α := α) opts mode model params x

/-- Run eval-mode eager inference from live parameters. -/
abbrev predict {σ τ : Shape}
    (model : Sequential σ τ)
    (opts : Options)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape]
    [TensorConv α]
    (params : NN.API.TorchLean.ParamList α (paramShapes model))
    (x : Tensor.T α σ) : IO (Tensor.T α τ) :=
  NN.API.TorchLean.NN.Seq.predict (α := α) opts model params x

/--
A compiled sequential model.

Public wrapper returned by `nn.compile`. It stores the compiled artifact and carries the
parameter-shape ABI in its type, so callers can run `compiled.forward params x` without passing the
source model again.
-/
structure Compiled (paramShapes : List Shape) (σ τ : Shape) (α : Type) where
  artifact : _root_.Runtime.Autograd.Torch.CompiledGraph α (paramShapes ++ [σ]) τ

namespace Compiled

/-- Run a compiled model forward with explicit parameter tensors. -/
def forward {σ τ : Shape} {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    {paramShapes : List Shape}
    (compiled : Compiled paramShapes σ τ α)
    (params : ParamTensors α paramShapes)
    (x : Tensor.T α σ) : Tensor.T α τ :=
  let args : NN.API.TorchLean.TList α (paramShapes ++ [σ]) :=
    _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := paramShapes) (ss₂ := [σ]) params (.cons x .nil)
  _root_.Runtime.Autograd.Torch.CompiledGraph.forward compiled.artifact args

end Compiled

/-- Compile a sequential model into a reusable callable wrapper. -/
def compile {σ τ : Shape} (model : Sequential σ τ)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape] :
    IO (Compiled (paramShapes model) σ τ α) := do
  let artifact ← NN.API.TorchLean.NN.Seq.compileForward (α := α) model
  pure { artifact := artifact }

/-- Compile a sequential model under an explicit mode. -/
def compileWithMode {σ τ : Shape}
    (model : Sequential σ τ)
    (mode : NN.API.TorchLean.NN.Mode)
    {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape] :
    IO (Compiled (paramShapes model) σ τ α) := do
  let artifact ← NN.API.TorchLean.NN.Seq.compileForwardWithMode (α := α) mode model
  pure { artifact := artifact }

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

namespace Runtime
namespace Autograd
namespace TorchLean
namespace NN
namespace Seq

/-- Dot-notation wrapper for the public compiled-model API: `let c ← model.compile`. -/
def compile {σ τ : _root_.Spec.Shape}
    (model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ)
    {α : Type} [_root_.Context α] [DecidableEq _root_.Spec.Shape] :
    IO (_root_.TorchLean.nn.Compiled (_root_.TorchLean.nn.paramShapes model) σ τ α) :=
  _root_.TorchLean.nn.compile (α := α) model

/-- Dot-notation wrapper for explicit-mode compilation: `let c ← model.compileWithMode nn.train`. -/
def compileWithMode {σ τ : _root_.Spec.Shape}
    (model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ)
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    {α : Type} [_root_.Context α] [DecidableEq _root_.Spec.Shape] :
    IO (_root_.TorchLean.nn.Compiled (_root_.TorchLean.nn.paramShapes model) σ τ α) :=
  _root_.TorchLean.nn.compileWithMode (α := α) model mode

end Seq
end NN
end TorchLean
end Autograd
end Runtime
