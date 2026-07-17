/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer.Results

/-!
# TorchLean Trainer Verification Helpers

Trained-result convenience methods for uniform `ℓ∞` IBP checks.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace Regression

namespace TrainResult

/--
Verify the trained model on a uniform `ℓ∞` input ball.

The method lives on `TrainResult`, not on the untrained trainer, because the verifier needs the
actual trained parameter values.
-/
def verifyRobustLInf {σ τ : Shape}
    (result : TrainResult σ τ) (center : Tensor.T Float σ) (eps : Float) :
    IO VerificationReport :=
  result.verify { center := center, eps := eps }

end TrainResult

end Regression

end Implementation

end Trainer

end TorchLean
