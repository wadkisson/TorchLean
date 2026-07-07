/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer.Core

/-!
# TorchLean Trainer Results

Public trained-result handles and verification reports for the unified trainer facade.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace Regression

/--
Uniform `ℓ∞` verification request for a trained regression model.

The request is deliberately small: one center input and one radius. It maps cleanly to TorchLean's
checked verifier path:

1. compile the trained forward model to verifier IR,
2. seed the distinguished input node with an `ℓ∞` box,
3. run IBP,
4. report the certified output interval.

The report is not a replacement for richer robustness specifications; it is the compact public door
into the checked verifier path.
-/
structure LInfIBPRequest (σ : Shape) where
  /-- Center of the input perturbation box, written as a normal `Float` tensor. -/
  center : Tensor.T Float σ
  /-- Uniform `ℓ∞` radius around `center`. -/
  eps : Float

/--
Public result of an `ℓ∞` IBP verification run.

Bounds are rendered as strings because the trained handle hides the runtime-selected scalar type
inside a closure. That keeps the API usable from ordinary scripts while still running the verifier
over the same scalar backend used for training.
-/
structure VerificationReport where
  /-- Number of IR nodes in the compiled verifier graph. -/
  nodes : Nat
  /-- Flattened output dimension reported by the verifier. -/
  outputDim : Nat
  /-- Lower bound on the flattened output box. -/
  lo : String
  /-- Upper bound on the flattened output box. -/
  hi : String

namespace VerificationReport

/-- One-line verification summary. -/
def summary (report : VerificationReport) : String :=
  s!"IBP nodes={report.nodes} output_dim={report.outputDim} lo={report.lo} hi={report.hi}"

/-- Print the verification summary to stdout. -/
def printSummary (report : VerificationReport) : IO Unit :=
  IO.println (summary report)

instance : ToString VerificationReport where
  toString := summary

end VerificationReport

/--
Result of training a regression trainer.

The trained runner stays alive behind the returned closures, so callers can immediately reuse the
trained model for inference without reopening the runtime API directly.
-/
structure TrainResult (σ τ : Shape) where
  /-- Before/after loss summary for the completed training run. -/
  report : TrainSummary
  /-- Run one Float input through the trained model and return the output tensor. -/
  predict : Tensor.T Float σ → IO (Tensor.T Float τ)
  /-- Run several Float inputs through the trained model and return their output tensors. -/
  predictBatch : List (Tensor.T Float σ) → IO (List (Tensor.T Float τ))
  /-- Run a public robustness check against the trained model. -/
  verify : LInfIBPRequest σ → IO VerificationReport

/--
Result of training from a step-indexed regression stream.

Generated or resampled workloads may not have one static dataset to summarize:
diffusion noising schedules, PDE collocation batches, RL replay-style batches, and similar loops.
The trained model handle is the same one returned by ordinary training; the extra field is the
explicit loss curve collected from a caller-provided evaluation sample.
-/
structure StreamTrainResult (σ τ : Shape) where
  /-- Trained model handle with prediction and verification closures. -/
  result : TrainResult σ τ
  /-- Evaluation loss curve recorded during stream training. -/
  curve : Training.Curve

/--
Result of training two regression trainers in one alternating stream.

GAN-style examples need this shape: one checked model is stepped on one stream, another checked
model is stepped on a related stream, and the report is a task-specific scalar such as total
generator/discriminator loss. The public result still returns ordinary trained handles for both
models, so post-training code uses `predict` instead of touching module state directly.
-/
structure PairStreamTrainResult (σ₁ τ₁ σ₂ τ₂ : Shape) where
  /-- Trained handle for the first trainer. -/
  first : TrainResult σ₁ τ₁
  /-- Trained handle for the second trainer. -/
  second : TrainResult σ₂ τ₂
  /-- Task-specific curve recorded by the caller-provided evaluation function. -/
  curve : Training.Curve

namespace TrainResult

/-- One-line summary for the completed training run. -/
def summary {σ τ : Shape} (result : TrainResult σ τ) : String :=
  result.report.summary

/-- Print the before/after training summary to stdout. -/
def printSummary {σ τ : Shape} (result : TrainResult σ τ) : IO Unit :=
  IO.println result.summary

/--
Run one regression prediction and print it with a user-provided label.

Small "train, then inspect one heldout example" helper used by tutorials.
-/
def printPrediction {σ τ : Shape}
    (result : TrainResult σ τ) (label : String) (x : Tensor.T Float σ) : IO Unit := do
  let yhat ← result.predict x
  IO.println s!"{label} = {Tensor.pretty yhat}"

instance {σ τ : Shape} : ToString (TrainResult σ τ) where
  toString := summary

end TrainResult

namespace StreamTrainResult

/-- One-line summary for the trained stream run. -/
def summary {σ τ : Shape} (result : StreamTrainResult σ τ) : String :=
  result.result.summary

/-- Print the stream training summary to stdout. -/
def printSummary {σ τ : Shape} (result : StreamTrainResult σ τ) : IO Unit :=
  IO.println result.summary

/--
Run one prediction through the trained model produced by stream training.

Stream training carries an extra loss curve. This forwarding method keeps stream examples concise:
call `stream.predict x` on the stream result instead of reaching through the trained handle field.
-/
def predict {σ τ : Shape}
    (result : StreamTrainResult σ τ) (x : Tensor.T Float σ) : IO (Tensor.T Float τ) :=
  result.result.predict x

/-- Run several predictions through the trained model produced by stream training. -/
def predictBatch {σ τ : Shape}
    (result : StreamTrainResult σ τ) (xs : List (Tensor.T Float σ)) :
    IO (List (Tensor.T Float τ)) :=
  result.result.predictBatch xs

instance {σ τ : Shape} : ToString (StreamTrainResult σ τ) where
  toString := summary

end StreamTrainResult

namespace PairStreamTrainResult

/-- One-line summary for the two trained models. -/
def summary {σ₁ τ₁ σ₂ τ₂ : Shape} (result : PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) :
    String :=
  s!"first: {result.first.summary}; second: {result.second.summary}"

/-- Print the trained-handle summary for both models. -/
def printSummary {σ₁ τ₁ σ₂ τ₂ : Shape} (result : PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) :
    IO Unit :=
  IO.println result.summary

/--
Print a before/after summary for the paired task curve.

Paired-model examples usually care about a coupled metric such as total GAN loss. That metric lives
in `result.curve`, not in either trained handle alone, so this operation gives examples one standard
before/after line without peeking at `curve.values`.
-/
def printCurveSummary {σ₁ τ₁ σ₂ τ₂ : Shape}
    (result : PairStreamTrainResult σ₁ τ₁ σ₂ τ₂)
    (metric : String := "loss") : IO Unit := do
  let endpoints ← NN.API.Common.requireCurveEndpoints "PairStreamTrainResult.printCurveSummary"
    result.curve
  IO.println
    s!"  steps={endpoints.finalStep} {metric}0={endpoints.first} {metric}{endpoints.finalStep}={endpoints.last}"

instance {σ₁ τ₁ σ₂ τ₂ : Shape} : ToString (PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) where
  toString := summary

end PairStreamTrainResult

end Regression

namespace CrossEntropy

/--
Result of training a general one-hot cross-entropy trainer.

Sequence-model friendly trained handle: it owns the trained runner and gives callers ordinary Float
prediction tensors. Text examples can then decode logits however they like without threading runtime
runner/module state through the example.
-/
structure TrainResult (σ τ : Shape) where
  /-- Before/after loss summary for the completed training run. -/
  report : TrainSummary
  /-- Run one Float input through the trained model and return logits/output tensor. -/
  predict : Tensor.T Float σ → IO (Tensor.T Float τ)
  /-- Run several Float inputs through the trained model. -/
  predictBatch : List (Tensor.T Float σ) → IO (List (Tensor.T Float τ))

namespace TrainResult

/-- One-line summary for the completed training run. -/
def summary {σ τ : Shape} (result : TrainResult σ τ) : String :=
  result.report.summary

/-- Print the before/after training summary to stdout. -/
def printSummary {σ τ : Shape} (result : TrainResult σ τ) : IO Unit :=
  IO.println result.summary

/-- Run one prediction and print it with a user-provided label. -/
def printPrediction {σ τ : Shape}
    (result : TrainResult σ τ) (label : String) (x : Tensor.T Float σ) : IO Unit := do
  let yhat ← result.predict x
  IO.println s!"{label} = {Tensor.pretty yhat}"

instance {σ τ : Shape} : ToString (TrainResult σ τ) where
  toString := summary

end TrainResult

end CrossEntropy

namespace Custom

/--
Result of training a custom supervised trainer.

The trained handle mirrors `CrossEntropy.TrainResult`: custom objectives affect training, but
inference is still just "run the checked model on a Float tensor". Keeping that API identical
means examples can switch from a canned loss to a task-specific loss without rewriting their
prediction/reporting code.
-/
structure TrainResult (σ τ : Shape) where
  /-- Before/after loss summary for the completed training run. -/
  report : TrainSummary
  /-- Run one Float input through the trained model and return its output tensor. -/
  predict : Tensor.T Float σ → IO (Tensor.T Float τ)
  /-- Run several Float inputs through the trained model. -/
  predictBatch : List (Tensor.T Float σ) → IO (List (Tensor.T Float τ))

namespace TrainResult

/-- One-line summary for the completed training run. -/
def summary {σ τ : Shape} (result : TrainResult σ τ) : String :=
  result.report.summary

/-- Print the before/after training summary to stdout. -/
def printSummary {σ τ : Shape} (result : TrainResult σ τ) : IO Unit :=
  IO.println result.summary

/-- Run one prediction and print it with a user-provided label. -/
def printPrediction {σ τ : Shape}
    (result : TrainResult σ τ) (label : String) (x : Tensor.T Float σ) : IO Unit := do
  let yhat ← result.predict x
  IO.println s!"{label} = {Tensor.pretty yhat}"

instance {σ τ : Shape} : ToString (TrainResult σ τ) where
  toString := summary

end TrainResult

end Custom

end Implementation

end Trainer

end TorchLean
