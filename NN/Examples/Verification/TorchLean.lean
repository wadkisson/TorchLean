/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.CrownOpsWorkflow
public import NN.Verification.TorchLean.IBPWorkflow
public import NN.Verification.TorchLean.MlpTrainVerifyWorkflow
public import NN.Verification.TorchLean.TransformerIBPWorkflow

/-!
# TorchLean Verification Workflows

End-to-end examples that build TorchLean models, lower them into verification artifacts, and run
IBP/CROWN-style checks.
-/

@[expose] public section
