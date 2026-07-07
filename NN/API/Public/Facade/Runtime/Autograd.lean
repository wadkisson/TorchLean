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

Parameter-side companion to `nn.linear inDim outDim seedW seedB`.
-/
def linearParams {α : Type} {inDim outDim : Nat} {seedW seedB : Nat}
    (w : Tensor.T α (Shape.mat outDim inDim))
    (b : Tensor.T α (Shape.vec outDim)) :
    Params (nn.linear inDim outDim seedW seedB) α :=
  nn.ParamTensors.pair w b

/--
Pack one tensor into the typed parameter/result container used by the public autograd APIs.
-/
def packSingleton {α : Type} {s₁ : Shape} :
    Tensor.T α s₁ → nn.ParamTensors α [s₁] :=
  fun x => nn.ParamTensors.singleton x

/--
Pack two tensors into the typed parameter/result container used by the public autograd APIs.
-/
def packPair {α : Type} {s₁ s₂ : Shape}
    (x₁ : Tensor.T α s₁) (x₂ : Tensor.T α s₂) :
    nn.ParamTensors α [s₁, s₂] :=
  nn.ParamTensors.pair x₁ x₂

/--
Pack three tensors into the typed parameter/result container used by the public autograd APIs.
-/
def packTriple {α : Type} {s₁ s₂ s₃ : Shape}
    (x₁ : Tensor.T α s₁) (x₂ : Tensor.T α s₂) (x₃ : Tensor.T α s₃) :
    nn.ParamTensors α [s₁, s₂, s₃] :=
  nn.ParamTensors.triple x₁ x₂ x₃

/--
Pack four tensors into the typed parameter/result container used by the public autograd APIs.
-/
def packQuad {α : Type} {s₁ s₂ s₃ s₄ : Shape}
    (x₁ : Tensor.T α s₁) (x₂ : Tensor.T α s₂)
    (x₃ : Tensor.T α s₃) (x₄ : Tensor.T α s₄) :
    nn.ParamTensors α [s₁, s₂, s₃, s₄] :=
  nn.ParamTensors.quad x₁ x₂ x₃ x₄

/-- Unpack a one-entry typed autograd parameter/result container. -/
def unpackSingleton {α : Type} {s₁ : Shape}
    (xs : nn.ParamTensors α [s₁]) :
    Tensor.T α s₁ :=
  nn.ParamTensors.unpackSingleton xs

/-- Unpack a two-entry typed autograd parameter/result container. -/
def unpackPair {α : Type} {s₁ s₂ : Shape}
    (xs : nn.ParamTensors α [s₁, s₂]) :
    Tensor.T α s₁ × Tensor.T α s₂ :=
  nn.ParamTensors.unpackPair xs

/-- Unpack a three-entry typed autograd parameter/result container. -/
def unpackTriple {α : Type} {s₁ s₂ s₃ : Shape}
    (xs : nn.ParamTensors α [s₁, s₂, s₃]) :
    Tensor.T α s₁ × Tensor.T α s₂ × Tensor.T α s₃ :=
  nn.ParamTensors.unpackTriple xs

/-- Unpack a four-entry typed autograd parameter/result container. -/
def unpackQuad {α : Type} {s₁ s₂ s₃ s₄ : Shape}
    (xs : nn.ParamTensors α [s₁, s₂, s₃, s₄]) :
    Tensor.T α s₁ × Tensor.T α s₂ × Tensor.T α s₃ × Tensor.T α s₄ :=
  nn.ParamTensors.unpackQuad xs

namespace OutputLoss

export NN.API.autograd.model.OutputLoss
  (mse crossEntropyOneHot detach)

end OutputLoss

end model

namespace func

export NN.API.autograd.func
  (Fn jacfwd hessian vjp jacrev grad valueAndGradScalar)

end func

end autograd


end TorchLean
