/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN.Runtime

/-!
# Neural-Network Layers

This module exposes the canonical layer API under `TorchLean.nn`. It introduces no forwarding
definitions: every exported name is the declaration from `NN.API.nn`.
-/

@[expose] public section

namespace TorchLean
namespace nn

export NN.API.nn
  (M Sequential LayerDef
   Conv Pool LayerNorm RMSNorm ChannelNorm MultiheadAttention
   run lift mapLeading withSeed withSeedPair
   linear linearV
   relu silu gelu sigmoid tanh softmax
   sum flatten flattenBatch
   conv maxPool avgPool globalAvgPool
   rnn gru mamba lstm
   embedding learnedPositionalEmbedding sinusoidalPositionalEncoding rope
   layerNorm batchNorm instanceNorm groupNorm
   multiheadAttention transformerEncoderBlock transformerEncoderStack
   dropout)

namespace blocks

export NN.API.nn.blocks
  (Activation MLP ConvAct ConvActPool
   activation mlp convAct convActPool
   residual residualLayer residualBlock
   addBranches addBranchesLayer
   TransformerEncoderBlock TransformerEncoderStack
   transformerEncoderBlock transformerEncoderBlockWithMask
   transformerEncoderStack transformerEncoderStackWithMask
   transformerEncoderClassifier)

end blocks

namespace heads

export NN.API.nn.heads (classifier regressor classifierBatch regressorBatch)

end heads

namespace functional

export NN.API.nn.functional
  (square checkpoint exp log scale shift affine detach addB mulB embedding mean
   dropoutSeeded)

end functional

namespace deterministic

export NN.API.nn.deterministic (linear)

end deterministic
end nn
end TorchLean
