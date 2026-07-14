/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime.Training.Core

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Supervised Training

Supervised tasks, runners, steppers, optimizer configs, trainer aliases, and the low-level session
exports that back executable examples.
-/

namespace Supervised

/-! ## Dataset training loops -/

/--
Train on a small in-memory list of supervised samples for a fixed number of steps.

This is the simplest training-loop helper: it is intended for examples and small synthetic datasets.
For loader-based training, see `trainLoader`.
-/
def trainSamples {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig) (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) := do
  trainMode runner
  let before ← meanLoss runner samples
  unless samples.isEmpty do
    let batchSize := effectiveTrainBatchSize cfg.batchSize
    let restRef ← IO.mkRef samples
    let nextBatch : IO (List (API.TorchLean.TensorPack α [σ, τ])) :=
      nextCyclicBatch "Supervised.train" samples restRef batchSize
    let logBatchLoss (stepIdx : Nat) (batch : List (API.TorchLean.TensorPack α [σ, τ])) :
        IO Unit := do
      if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
        let loss ← meanLoss runner batch
        IO.println s!"step {stepIdx}: loss={loss}"
    match cfg.optimizer with
    | .sgd lr momentum =>
        if momentum == 0.0 then
          for stepIdx in [0:cfg.steps] do
            let batch ← nextBatch
            let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
            for sample in batch do
              updateRunnerBuffers runner sample
              API.TorchLean.Module.step runner.module lrα sample
            logBatchLoss stepIdx batch
        else
          let opt := API.TorchLean.Optim.momentumSGD
            (α := α) (paramShapes := paramShapes task)
            (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
          let st0 : API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α
            (paramShapes task) ←
            API.TorchLean.Module.initOptim runner.module opt
          let mut st := st0
          for stepIdx in [0:cfg.steps] do
            let batch ← nextBatch
            let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
            st := momentumSGDStateWithLR (paramShapes := paramShapes task) lrα st
            for sample in batch do
              updateRunnerBuffers runner sample
              st ← API.TorchLean.Module.stepWith runner.module opt st sample
            logBatchLoss stepIdx batch
    | .adagrad lr epsilon =>
        let opt := API.TorchLean.Optim.adagrad
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr)
          (epsilon := API.Runtime.ofFloat epsilon)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdaGrad.State α (paramShapes task) ←
          API.TorchLean.Module.initOptim runner.module opt
        let mut st := st0
        for stepIdx in [0:cfg.steps] do
          let batch ← nextBatch
          let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
          st := adagradStateWithLR (paramShapes := paramShapes task) lrα st
          for sample in batch do
            updateRunnerBuffers runner sample
            st ← API.TorchLean.Module.stepWith runner.module opt st sample
          logBatchLoss stepIdx batch
    | .rmsprop lr decay epsilon =>
        let opt := API.TorchLean.Optim.rmsprop
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr)
          (decay := API.Runtime.ofFloat decay)
          (epsilon := API.Runtime.ofFloat epsilon)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.RMSProp.State α (paramShapes task) ←
          API.TorchLean.Module.initOptim runner.module opt
        let mut st := st0
        for stepIdx in [0:cfg.steps] do
          let batch ← nextBatch
          let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
          st := rmspropStateWithLR (paramShapes := paramShapes task) lrα st
          for sample in batch do
            updateRunnerBuffers runner sample
            st ← API.TorchLean.Module.stepWith runner.module opt st sample
          logBatchLoss stepIdx batch
    | .adam lr beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adam
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr)
          (beta1 := API.Runtime.ofFloat beta1)
          (beta2 := API.Runtime.ofFloat beta2)
          (epsilon := API.Runtime.ofFloat epsilon)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adam.State α (paramShapes task) ←
          API.TorchLean.Module.initOptim runner.module opt
        let mut st := st0
        for stepIdx in [0:cfg.steps] do
          let batch ← nextBatch
          let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
          st := adamStateWithLR (paramShapes := paramShapes task) lrα st
          for sample in batch do
            updateRunnerBuffers runner sample
            st ← API.TorchLean.Module.stepWith runner.module opt st sample
          logBatchLoss stepIdx batch
    | .adamw lr weightDecay beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adamw
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
          (beta1 := API.Runtime.ofFloat beta1)
          (beta2 := API.Runtime.ofFloat beta2)
          (epsilon := API.Runtime.ofFloat epsilon)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α (paramShapes task) ←
          API.TorchLean.Module.initOptim runner.module opt
        let mut st := st0
        for stepIdx in [0:cfg.steps] do
          let batch ← nextBatch
          let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
          st := adamwStateWithLR (paramShapes := paramShapes task) lrα st
          for sample in batch do
            updateRunnerBuffers runner sample
            st ← API.TorchLean.Module.stepWith runner.module opt st sample
          logBatchLoss stepIdx batch
    | .adadelta lr rho epsilon =>
        let opt := API.TorchLean.Optim.adadelta
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr)
          (rho := API.Runtime.ofFloat rho)
          (epsilon := API.Runtime.ofFloat epsilon)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adadelta.State α (paramShapes task) ←
          API.TorchLean.Module.initOptim runner.module opt
        let mut st := st0
        for stepIdx in [0:cfg.steps] do
          let batch ← nextBatch
          let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
          st := adadeltaStateWithLR (paramShapes := paramShapes task) lrα st
          for sample in batch do
            updateRunnerBuffers runner sample
            st ← API.TorchLean.Module.stepWith runner.module opt st sample
          logBatchLoss stepIdx batch
  let after ← meanLoss runner samples
  pure { before := before, after := after }

/-- Train over a dataset by materializing it as a list. -/
def trainDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  trainSamples runner cfg dataset.toList

/--
Train over a `DataLoader` for `cfg.epochs` epochs, returning the final report and updated loader.

This corresponds to the common PyTorch pattern:
`for epoch in ...: for batch in loader: step(batch)`.
-/
def trainLoader {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderTrainConfig)
    (dl : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α × _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) := do
  trainMode runner
  let nextEpoch
      (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
      IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) ×
        List (List (API.TorchLean.TensorPack α [σ, τ]))) :=
    match _root_.Runtime.Autograd.Train.DataLoader.epoch "Supervised.trainLoader" loader with
    | .ok out => pure out
    | .error msg => throw <| IO.userError s!"Supervised.trainLoader: {msg}"

  let before ← meanLossDataset runner dl.dataset

  match cfg.optimizer with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let rec trainSgdBatches
            (epoch : Nat)
            (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
            IO Unit := do
          match batches with
          | [] => pure ()
          | batch :: rest =>
              let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch)
              for sample in batch do
                updateRunnerBuffers runner sample
                API.TorchLean.Module.step runner.module lrα sample
              trainSgdBatches epoch rest

        let rec runSgdEpochs (remaining : Nat)
            (epoch : Nat)
            (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
            IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) := do
          match remaining with
          | 0 => pure loader
          | n + 1 =>
              let (loader', batches) ← nextEpoch loader
              trainSgdBatches epoch batches
              runSgdEpochs n (epoch + 1) loader'

        let dl' ← runSgdEpochs cfg.epochs 0 dl
        let after ← meanLossDataset runner dl'.dataset
        pure ({ before := before, after := after }, dl')
      else
        let opt := API.TorchLean.Optim.momentumSGD
          (α := α) (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
        let st0 ← API.TorchLean.Module.initOptim runner.module opt

        let rec trainMomSamples (epoch stepIdx : Nat) (state : opt.State)
            (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
            IO (opt.State × Nat) := do
          match samples with
          | [] => pure (state, stepIdx)
          | sample :: rest =>
              updateRunnerBuffers runner sample
              let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
              if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
                let loss ← API.TorchLean.Module.forward runner.module sample
                IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
              trainMomSamples epoch (stepIdx + 1) state' rest

        let rec trainMomBatches (epoch stepIdx : Nat) (state : opt.State)
            (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
            IO (opt.State × Nat) := do
          match batches with
          | [] => pure (state, stepIdx)
          | batch :: rest =>
              let (state', stepIdx') ← trainMomSamples epoch stepIdx state batch
              trainMomBatches epoch stepIdx' state' rest

        let rec runMomEpochs (epoch remaining : Nat)
            (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
            (st : opt.State) :
            IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
              := do
          match remaining with
          | 0 => pure (loader, st)
          | n + 1 =>
              let (loader', batches) ← nextEpoch loader
              let stSched : opt.State :=
                match cfg.scheduler with
                | none => st
                | some _ =>
                    momentumSGDStateWithLR
                      (paramShapes := paramShapes task)
                      (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                      st
              let (st', _) ← trainMomBatches epoch 0 stSched batches
              runMomEpochs (epoch + 1) n loader' st'

        let (dl', _) ← runMomEpochs 0 cfg.epochs dl st0
        let after ← meanLossDataset runner dl'.dataset
        pure ({ before := before, after := after }, dl')

  | .adagrad lr epsilon =>
      let opt := API.TorchLean.Optim.adagrad
        (α := α) (lr := API.Runtime.ofFloat lr)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdaGradSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdaGradSamples epoch (stepIdx + 1) state' rest

      let rec trainAdaGradBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdaGradSamples epoch stepIdx state batch
            trainAdaGradBatches epoch stepIdx' state' rest

      let rec runAdaGradEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adagradStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdaGradBatches epoch 0 stSched batches
            runAdaGradEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdaGradEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

  | .rmsprop lr decay epsilon =>
      let opt := API.TorchLean.Optim.rmsprop
        (α := α) (lr := API.Runtime.ofFloat lr)
        (decay := API.Runtime.ofFloat decay)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainRMSPropSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainRMSPropSamples epoch (stepIdx + 1) state' rest

      let rec trainRMSPropBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainRMSPropSamples epoch stepIdx state batch
            trainRMSPropBatches epoch stepIdx' state' rest

      let rec runRMSPropEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  rmspropStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainRMSPropBatches epoch 0 stSched batches
            runRMSPropEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runRMSPropEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

  | .adam lr beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adam
        (α := α) (lr := API.Runtime.ofFloat lr)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdamSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdamSamples epoch (stepIdx + 1) state' rest

      let rec trainAdamBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdamSamples epoch stepIdx state batch
            trainAdamBatches epoch stepIdx' state' rest

      let rec runAdamEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adamStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdamBatches epoch 0 stSched batches
            runAdamEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdamEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adamw
        (α := α) (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdamWSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdamWSamples epoch (stepIdx + 1) state' rest

      let rec trainAdamWBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdamWSamples epoch stepIdx state batch
            trainAdamWBatches epoch stepIdx' state' rest

      let rec runAdamWEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adamwStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdamWBatches epoch 0 stSched batches
            runAdamWEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdamWEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

  | .adadelta lr rho epsilon =>
      let opt := API.TorchLean.Optim.adadelta
        (α := α) (lr := API.Runtime.ofFloat lr)
        (rho := API.Runtime.ofFloat rho)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdadeltaSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdadeltaSamples epoch (stepIdx + 1) state' rest

      let rec trainAdadeltaBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdadeltaSamples epoch stepIdx state batch
            trainAdadeltaBatches epoch stepIdx' state' rest

      let rec runAdadeltaEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adadeltaStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdadeltaBatches epoch 0 stSched batches
            runAdadeltaEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdadeltaEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

end Supervised

end TorchLean
end API
end NN
