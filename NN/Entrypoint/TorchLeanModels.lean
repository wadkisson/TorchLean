/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.TorchLean

/-!
# TorchLean executable model zoo

This module re-exports TorchLean’s small runnable model constructors (MLP/CNN/Transformer/etc.).
It is the executable model-zoo counterpart to the pure specs in `NN.Spec.Models.*`.

The implementations live under `NN.GraphSpec.Models.TorchLean.*`, because they are architecture
constructors. The runtime namespace stays focused on execution machinery: ops, backends, sessions,
losses, optimizers, and training loops.

`NN.TorchLeanModels` is the short import path for those constructors, so example code can refer
to `TorchLeanModels.mlp` without exposing the internal GraphSpec module layout.
-/

@[expose] public section


namespace NN
namespace TorchLeanModels

export _root_.NN.GraphSpec.Models.TorchLean
  (mlp autoencoder twoConvCnn softmaxRegression mlpClassifier transformerBlock
   fno1d fno1dParamShapes
   resnet18Model resnet18Program resnet18InitParams
  )

end TorchLeanModels
end NN
