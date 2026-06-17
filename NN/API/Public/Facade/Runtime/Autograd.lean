/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Runtime Autograd

Public autograd APIs for model- and function-level differentiation.
-/

@[expose] public section

namespace TorchLean

namespace autograd

namespace model

export NN.API.autograd.model
  (Params OutputLoss gradParams gradInputs gradX gradTarget
   ValueAndGrads valueAndGrads valueAndGradParams valueAndGradParamsScalar
   valueAndGradX valueAndGradTarget vjpParams vjpInputs vjpInput jacrevParams
   jvpParams hvpParams)

/--
Pack explicit weight and bias tensors for the public `nn.linear` constructor.

This is the parameter-side companion to `nn.linear inDim outDim seedW seedB`.
-/
def linearParams {α : Type} {inDim outDim : Nat} {seedW seedB : Nat}
    (w : Tensor.T α (Shape.mat outDim inDim))
    (b : Tensor.T α (Shape.vec outDim)) :
    Params (nn.linear inDim outDim seedW seedB) α :=
  nn.ParamTensors.of2 w b

/--
Pack one tensor into the typed parameter/result container used by the public autograd APIs.
-/
def pack1 {α : Type} {s₁ : Shape} :
    Tensor.T α s₁ → nn.ParamTensors α [s₁] :=
  fun x => nn.ParamTensors.of1 x

/--
Pack two tensors into the typed parameter/result container used by the public autograd APIs.
-/
def pack2 {α : Type} {s₁ s₂ : Shape}
    (x₁ : Tensor.T α s₁) (x₂ : Tensor.T α s₂) :
    nn.ParamTensors α [s₁, s₂] :=
  nn.ParamTensors.of2 x₁ x₂

/--
Pack three tensors into the typed parameter/result container used by the public autograd APIs.
-/
def pack3 {α : Type} {s₁ s₂ s₃ : Shape}
    (x₁ : Tensor.T α s₁) (x₂ : Tensor.T α s₂) (x₃ : Tensor.T α s₃) :
    nn.ParamTensors α [s₁, s₂, s₃] :=
  nn.ParamTensors.of3 x₁ x₂ x₃

/--
Pack four tensors into the typed parameter/result container used by the public autograd APIs.
-/
def pack4 {α : Type} {s₁ s₂ s₃ s₄ : Shape}
    (x₁ : Tensor.T α s₁) (x₂ : Tensor.T α s₂)
    (x₃ : Tensor.T α s₃) (x₄ : Tensor.T α s₄) :
    nn.ParamTensors α [s₁, s₂, s₃, s₄] :=
  nn.ParamTensors.of4 x₁ x₂ x₃ x₄

/-- Unpack a one-entry typed autograd parameter/result container. -/
def unpack1 {α : Type} {s₁ : Shape}
    (xs : nn.ParamTensors α [s₁]) :
    Tensor.T α s₁ :=
  nn.ParamTensors.unpack1 xs

/-- Unpack a two-entry typed autograd parameter/result container. -/
def unpack2 {α : Type} {s₁ s₂ : Shape}
    (xs : nn.ParamTensors α [s₁, s₂]) :
    Tensor.T α s₁ × Tensor.T α s₂ :=
  nn.ParamTensors.unpack2 xs

/-- Unpack a three-entry typed autograd parameter/result container. -/
def unpack3 {α : Type} {s₁ s₂ s₃ : Shape}
    (xs : nn.ParamTensors α [s₁, s₂, s₃]) :
    Tensor.T α s₁ × Tensor.T α s₂ × Tensor.T α s₃ :=
  nn.ParamTensors.unpack3 xs

/-- Unpack a four-entry typed autograd parameter/result container. -/
def unpack4 {α : Type} {s₁ s₂ s₃ s₄ : Shape}
    (xs : nn.ParamTensors α [s₁, s₂, s₃, s₄]) :
    Tensor.T α s₁ × Tensor.T α s₂ × Tensor.T α s₃ × Tensor.T α s₄ :=
  nn.ParamTensors.unpack4 xs

namespace OutputLoss

export NN.API.autograd.model.OutputLoss
  (mse crossEntropyOneHot detach)

end OutputLoss

end model

namespace fn1

export NN.API.autograd.fn1
  (Fn jacfwd hessian vjp jacrev grad valueAndGradScalar)

end fn1

end autograd


end TorchLean
