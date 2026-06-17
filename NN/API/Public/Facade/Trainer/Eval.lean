/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer.Train
public import NN.API.Public.Facade.Trainer.Verify

/-!
# TorchLean Public Trainer Methods

Unified trained result and public methods on `Trainer.Handle`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

/-- Evaluate one Float input through a runtime runner and return a Float output. -/
def evalWithRunner {σ τ : Shape} {task : NN.API.TorchLean.Trainer.Task σ τ}
    {α : Type} [Runtime.SemanticScalar α] [DecidableEq Shape] [ToString α] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (runner : NN.API.train.Advanced.Runner α task) (x : Tensor.T Float σ) :
    IO (Tensor.T Float τ) := do
  Advanced.evalMode (task := task) runner
  let x' := Tensor.castFloat (Runtime.ofFloat (α := α)) x
  let y ← Advanced.predict (task := task) runner x'
  Tensor.toFloatIO y

/-- Evaluate one input through a custom-loss trainer without first running training. -/
def evalCustomWithRunConfig {σ τ : Shape}
    (trainer : Custom σ τ) (run : RunConfig) (x : Tensor.T Float σ) :
    IO (Tensor.T Float τ) := do
  let opts := run.toOptions
  let runFor
      {α : Type} [Runtime.SemanticScalar α] [DecidableEq Shape] [ToString α] [Runtime.Scalar α]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
      IO (Tensor.T Float τ) := do
    Module.withScalarLossModel
      (α := α) (mkModel := pure trainer.model) (opts := opts) (loss := trainer.loss)
      (k := fun model m => do
        let x' := Tensor.castFloat (Runtime.ofFloat (α := α)) x
        let y ← Module.predict1 (α := α) opts model m x'
        Tensor.toFloatIO y)
  match run.dtype with
  | .float => runFor (α := Float)
  | _ =>
      if opts.useGpu then
        throw <| IO.userError
          "TorchLean.Trainer.eval: CUDA execution currently requires dtype Float"
      let outRef : IO.Ref (Option (Tensor.T Float τ)) ← IO.mkRef none
      match (← NN.API.DType.withRuntime run.dtype (fun {α} _ _ _ _ => do
          let out ← runFor (α := α)
          outRef.set (some out))) with
      | .ok () =>
          let some out ← outRef.get
            | throw <| IO.userError
                "TorchLean.Trainer.eval: internal error: eval result was not initialized"
          pure out
      | .error msg => throw <| IO.userError msg

/-- Build the regression dispatch record used by `Handle.train`. -/
def regressionHandle {σ τ : Shape} (trainer : Handle σ τ)
    (reduction : Loss.Reduction := .mean) : Regression σ τ :=
  { model := trainer.model
    reduction := reduction
    runtime := trainer.runtime }

/-- Build the cross-entropy dispatch record used by `Handle.train`. -/
def crossEntropyHandle {σ τ : Shape} (trainer : Handle σ τ)
    (reduction : Loss.Reduction := .mean) : CrossEntropy σ τ :=
  { model := trainer.model
    reduction := reduction
    runtime := trainer.runtime }

/-- Build the custom-loss dispatch record used by `Handle.train`. -/
def customHandle {σ τ : Shape} (trainer : Handle σ τ)
    (loss : ∀ {α : Type}, [Runtime.TensorScalar α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar) :
    Custom σ τ :=
  { model := trainer.model
    loss := loss
    runtime := trainer.runtime }

end Implementation

/--
Evaluate one input using the trainer's current model and runtime settings.

Inference before any training call. After training, use the returned trained handle's
`trained.eval` / `trained.evalBatch` methods to evaluate the trained parameters.
-/
def Handle.eval {σ τ : Shape} (trainer : Handle σ τ) (x : Tensor.T Float σ) :
    IO (Tensor.T Float τ) := do
  match trainer.task with
  | .regression reduction =>
      let impl := Implementation.regressionHandle trainer reduction
      Implementation.Regression.Internal.withRunnerFromRunConfig impl impl.runConfig
        (fun {_} _ _ _ _ _ runner => Implementation.evalWithRunner runner x)
  | .classification reduction =>
      let impl := Implementation.crossEntropyHandle trainer reduction
      Implementation.CrossEntropy.Internal.withRunnerFromRunConfig impl impl.runConfig
        (fun {_} _ _ _ _ _ runner => Implementation.evalWithRunner runner x)
  | .crossEntropy reduction =>
      let impl := Implementation.crossEntropyHandle trainer reduction
      Implementation.CrossEntropy.Internal.withRunnerFromRunConfig impl impl.runConfig
        (fun {_} _ _ _ _ _ runner => Implementation.evalWithRunner runner x)
  | .custom loss =>
      let impl := Implementation.customHandle trainer loss
      Implementation.evalCustomWithRunConfig impl impl.runConfig x

/-- Evaluate a list of inputs using the trainer's current model and runtime settings. -/
def Handle.evalBatch {σ τ : Shape} (trainer : Handle σ τ) (xs : List (Tensor.T Float σ)) :
    IO (List (Tensor.T Float τ)) :=
  xs.mapM trainer.eval

/-- Result returned by the unified public `Trainer.train` method. -/
inductive TrainResult (σ τ : Shape) where
  /-- Trained regression model. -/
  | regression (result : Implementation.Regression.TrainResult σ τ)
  /-- Trained one-hot cross-entropy model. -/
  | crossEntropy (result : Implementation.CrossEntropy.TrainResult σ τ)
  /-- Trained custom-loss model. -/
  | custom (result : Implementation.Custom.TrainResult σ τ)

namespace TrainResult

/-- The before/after scalar summary for this training run. -/
def report {σ τ : Shape} : TrainResult σ τ → TrainSummary
  | .regression result => result.report
  | .crossEntropy result => result.report
  | .custom result => result.report

/-- One-line summary suitable for quickstarts and scripts. -/
def summary {σ τ : Shape} (result : TrainResult σ τ) : String :=
  result.report.summary

/-- Print the standard before/after training summary. -/
def printSummary {σ τ : Shape} (result : TrainResult σ τ) : IO Unit :=
  IO.println (summary result)

/-- Evaluate one input using the trained model. -/
def eval {σ τ : Shape} : TrainResult σ τ → Tensor.T Float σ → IO (Tensor.T Float τ)
  | .regression result, x => result.predict x
  | .crossEntropy result, x => result.predict x
  | .custom result, x => result.predict x

/-- Evaluate a list of inputs using the trained model. -/
def evalBatch {σ τ : Shape} : TrainResult σ τ → List (Tensor.T Float σ) →
    IO (List (Tensor.T Float τ))
  | .regression result, xs => result.predictBatch xs
  | .crossEntropy result, xs => result.predictBatch xs
  | .custom result, xs => result.predictBatch xs

/-- Print one prediction from a unified trained handle. -/
def printPrediction {σ τ : Shape} (result : TrainResult σ τ)
    (name : String) (x : Tensor.T Float σ) : IO Unit := do
  let y ← result.eval x
  IO.println s!"{name} = {Tensor.pretty y}"

/-- Verify an `ℓ∞` input ball for a trained regression handle. -/
def verifyRobustLInf {σ τ : Shape}
    (result : TrainResult σ τ) (center : Tensor.T Float σ) (eps : Float) :
    IO Implementation.Regression.VerificationReport := do
  match result with
  | .regression result => result.verifyRobustLInf center eps
  | .crossEntropy _ =>
      throw <| IO.userError
        "Trainer.TrainResult.verifyRobustLInf: verification is currently implemented for trained regression handles"
  | .custom _ =>
      throw <| IO.userError
        "Trainer.TrainResult.verifyRobustLInf: verification is currently implemented for trained regression handles"

instance {σ τ : Shape} : ToString (TrainResult σ τ) where
  toString := summary

end TrainResult

/--
Train a unified public trainer.

Main user-facing training method: one trainer value, one task field, and one trained result.
-/
def Handle.train {σ τ : Shape} (trainer : Handle σ τ)
    (data : Dataset σ τ) (trainOptions : TrainOptions := {}) (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  match trainer.task with
  | .regression reduction =>
      let out ← (Implementation.regressionHandle trainer reduction).train data trainOptions probes
      pure (.regression out)
  | .classification reduction =>
      let out ← (Implementation.crossEntropyHandle trainer reduction).train data trainOptions probes
      pure (.crossEntropy out)
  | .crossEntropy reduction =>
      let out ← (Implementation.crossEntropyHandle trainer reduction).train data trainOptions probes
      pure (.crossEntropy out)
  | .custom loss =>
      let out ← (Implementation.customHandle trainer loss).train data trainOptions
      pure (.custom out)

/--
Train a unified regression trainer from a Float sample stream.

Generated-data examples use this when there is no fixed `Dataset` to hand to `trainer.train`.
-/
def Handle.trainStreamFloat {σ τ : Shape}
    (trainer : Handle σ τ)
    (opts : Options)
    (sampleAt : Nat → SupervisedSample Float σ τ)
    (evalSample : SupervisedSample Float σ τ)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String → (Tensor.T Float σ → IO (Tensor.T Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (Implementation.Regression.StreamTrainResult σ τ) := do
  match trainer.task with
  | .regression reduction =>
      (Implementation.regressionHandle trainer reduction).trainStreamFloat opts sampleAt evalSample
        trainOptions
        (curveEvery := curveEvery) (cudaMemWatch := cudaMemWatch) (onEval := onEval)
  | _ =>
      throw <| IO.userError
        "Trainer.trainStreamFloat: stream training currently expects task := .regression"

/--
Train two unified regression trainers from coupled Float streams.

GAN-style examples use this path when two regression trainers have to step together, without opening
the lower-level runtime modules directly.
-/
def Handle.trainPairStreamFloat {σ₁ τ₁ σ₂ τ₂ : Shape}
    (first : Handle σ₁ τ₁)
    (second : Handle σ₂ τ₂)
    (opts : Options)
    (firstSampleAt : Nat → SupervisedSample Float σ₁ τ₁)
    (secondSamplesAt : Nat → List (SupervisedSample Float σ₂ τ₂))
    (evalTotal :
      (Tensor.T Float σ₁ → IO (Tensor.T Float τ₁)) →
      (Tensor.T Float σ₂ → IO (Tensor.T Float τ₂)) →
      IO Float)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 1)
    (cudaMemWatch : Nat := 0) :
    IO (Implementation.Regression.PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) := do
  match first.task, second.task with
  | .regression r1, .regression r2 =>
      Implementation.Regression.trainPairStreamFloat
        (Implementation.regressionHandle first r1) (Implementation.regressionHandle second r2) opts
        firstSampleAt secondSamplesAt evalTotal trainOptions
        (curveEvery := curveEvery) (cudaMemWatch := cudaMemWatch)
  | _, _ =>
      throw <| IO.userError
        "Trainer.trainPairStreamFloat: both trainers must use task := .regression"

/--
Train a unified cross-entropy trainer after the scalar type has already been selected.

Advanced scalar-selected cross-entropy training. Use this path from dispatchers such as
`ModelZoo.runAnyOrFloatNoCast`, where the callback already has a concrete scalar `α`.
-/
def Handle.trainSelectedCrossEntropy {σ τ : Shape} {α : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [ToString α] [Runtime.Scalar α]
    (trainer : Handle σ τ)
    (opts : Options) (data : Dataset σ τ) (trainOptions : TrainOptions := {})
    (probes : List (Probe σ) := []) :
    IO (Implementation.CrossEntropy.TrainResult σ τ) := do
  match trainer.task with
  | .crossEntropy reduction =>
      (Implementation.crossEntropyHandle trainer reduction).trainSelected (α := α) opts data
        trainOptions probes
  | .classification reduction =>
      (Implementation.crossEntropyHandle trainer reduction).trainSelected (α := α) opts data
        trainOptions probes
  | _ =>
      throw <| IO.userError
        "Trainer.trainSelectedCrossEntropy: expected task := .crossEntropy or task := .classification"


end Trainer

end TorchLean
