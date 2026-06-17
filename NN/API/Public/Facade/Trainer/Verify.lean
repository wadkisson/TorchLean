/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer.Results

/-!
# TorchLean Trainer Verification Helpers

Public verifier request builders and trained-result convenience methods.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace Regression

namespace TrainResult

/-- Run `verify` and print the resulting certified output interval. -/
def printVerification {σ τ : Shape}
    (result : TrainResult σ τ) (request : LInfIBPRequest σ) : IO Unit := do
  let report ← result.verify request
  report.printSummary

/--
Verify the trained model on a uniform `ℓ∞` input ball.

The method lives on `TrainResult`, not on the untrained trainer, because the verifier needs the
actual trained parameter values.
-/
def verifyRobustLInf {σ τ : Shape}
    (result : TrainResult σ τ) (center : Tensor.T Float σ) (eps : Float) :
    IO VerificationReport :=
  result.verify { center := center, eps := eps }

/-- Verify a uniform `ℓ∞` input ball and print the resulting certified output interval. -/
def printRobustLInf {σ τ : Shape}
    (result : TrainResult σ τ) (center : Tensor.T Float σ) (eps : Float) : IO Unit := do
  let report ← result.verifyRobustLInf center eps
  report.printSummary

end TrainResult

end Regression

end Implementation

namespace Verify

/-- Build a uniform `ℓ∞` IBP request for trained-model verification. -/
def lInfIBP {σ : Shape} (center : Tensor.T Float σ) (eps : Float) :
    Implementation.Regression.LInfIBPRequest σ :=
  { center := center, eps := eps }

end Verify

end Trainer

end TorchLean
