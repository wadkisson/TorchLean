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
Construct the model-indexed parameter pack for a deterministic linear layer.

Unlike the generic `nn.ParamTensors.pair`, the result is stated directly as `Params` for the
corresponding model, so callers do not need to unfold the model's dependent parameter shape.
-/
def linearParams {α : Type} {inDim outDim : Nat} {seedW seedB : Nat}
    (w : Tensor.T α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor.T α (.dim outDim .scalar)) :
    Params (TorchLean.nn.deterministic.linear inDim outDim seedW seedB) α :=
  nn.ParamTensors.pair w b

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
