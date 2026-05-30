/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine

/-!
# CROWN Graph Backward Objective API

Public entrypoint for objective-dependent backward CROWN bounds. The implementation lives with the
forward affine engine in `Graph/Engine`, because the backward pass reuses the same affine-transfer
machinery. This module gives users a focused import for the backward chapter without exposing those
implementation helpers as public API.
-/
