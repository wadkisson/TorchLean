/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN.Runtime

/-!
# TorchLean NN Layers

User-facing layer constructors, layer configs, blocks, and functional operations.
-/

@[expose] public section

namespace TorchLean

namespace nn

abbrev Conv2dConfig := NN.API.nn.Conv2d

@[inherit_doc NN.API.nn.Conv]
abbrev ConvConfig := NN.API.nn.Conv

@[inherit_doc NN.API.nn.pure.Conv]
abbrev Conv := NN.API.nn.pure.Conv

@[inherit_doc NN.API.nn.BatchNorm2d]
abbrev BatchNorm2dConfig := NN.API.nn.BatchNorm2d

@[inherit_doc NN.API.nn.pure.MaxPool]
abbrev MaxPool := NN.API.nn.pure.MaxPool

@[inherit_doc NN.API.nn.pure.MultiheadAttention]
abbrev MultiheadAttention := NN.API.nn.pure.MultiheadAttention

@[inherit_doc NN.API.nn.pure.blocks.TransformerEncoderBlock]
abbrev TransformerEncoderBlockConfig := NN.API.nn.pure.blocks.TransformerEncoderBlock

export NN.API.nn.pure
  (linear rnn gru mamba lstm
   embedding learnedPositionalEmbedding sinusoidalPositionalEncoding rope
   relu silu gelu sigmoid tanh softmax sum flatten flattenBatch
   flattenStart1 dropout flattenLinear
   conv2dCHWWith conv2dCHW conv2d convCHWWith convCHW conv
   maxPool2dWith maxPool2dCHW maxPool2d maxPoolWith maxPoolCHW maxPool
   avgPool2dWith avgPool2dCHW avgPool2d avgPoolWith avgPoolCHW avgPool
   globalAvgPoolCHW globalAvgPoolNCHW
   layerNormWith layerNorm rmsNormWith rmsNorm
   batchNorm2dNCHWWith batchNorm2d instanceNorm2dWith instanceNorm2d
   groupNorm2dNCHW multiheadAttentionWith multiheadAttention)

namespace blocks

export NN.API.nn.blocks
  (Activation MLP mlp down2)

@[inherit_doc NN.API.nn.blocks.ResNetBasicBlock]
abbrev ResNetBasicBlockConfig := NN.API.nn.blocks.ResNetBasicBlock

end blocks

namespace functional

export NN.API.nn.functional
  (square mean detach stopGrad)

end functional

abbrev Conv2dLayer {n inC inH inW : Nat} (cfg : Conv2dConfig)
    [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (Shape.images n inC inH inW)
      (Shape.images n cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1))) :=
  NN.API.nn.conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc NN.API.nn.conv]
abbrev Conv2d {n inC inH inW : Nat} (cfg : ConvConfig)
    [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (Shape.images n inC inH inW)
      (Shape.images n cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1))) :=
  NN.API.nn.conv (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc NN.API.nn.pure.maxPool2d]
abbrev MaxPool2d {n inC inH inW : Nat} (cfg : MaxPool)
    [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (Shape.images n inC inH inW)
      (Shape.images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1))) :=
  pure <| NN.API.nn.pure.maxPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc NN.API.nn.batchNorm2d]
abbrev BatchNorm2d {n c h w : Nat} (cfg : BatchNorm2dConfig := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    M (Sequential (Shape.images n c h w) (Shape.images n c h w)) :=
  NN.API.nn.batchNorm2d (n := n) (c := c) (h := h) (w := w) cfg

@[inherit_doc NN.API.nn.batchNorm]
abbrev BatchNorm {n c h w : Nat} (cfg : BatchNorm2dConfig := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    M (Sequential (Shape.images n c h w) (Shape.images n c h w)) :=
  NN.API.nn.batchNorm (n := n) (c := c) (h := h) (w := w) cfg

@[inherit_doc NN.API.nn.resnetBasicBlock]
abbrev ResNetBasicBlock {n inC h w : Nat} (cfg : blocks.ResNetBasicBlockConfig)
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w] [NeZero cfg.outC] :
    M (Sequential
      (Shape.images n inC h w)
      (Shape.images n cfg.outC
        (if cfg.downsample then blocks.down2 h else h)
        (if cfg.downsample then blocks.down2 w else w))) :=
  NN.API.nn.resnetBasicBlock (n := n) (inC := inC) (h := h) (w := w) cfg

@[inherit_doc NN.API.nn.Linear]
abbrev Linear (inDim outDim : Nat) (pfx : Shape := Shape.scalar) :
    M (Sequential (pfx.appendDim inDim) (pfx.appendDim outDim)) :=
  NN.API.nn.Linear inDim outDim (pfx := pfx)

@[inherit_doc NN.API.nn.ReLU]
abbrev ReLU {s : Shape} : M (Sequential s s) := NN.API.nn.ReLU (s := s)

@[inherit_doc NN.API.nn.GELU]
abbrev GELU {s : Shape} : M (Sequential s s) := NN.API.nn.GELU (s := s)

@[inherit_doc NN.API.nn.SiLU]
abbrev SiLU {s : Shape} : M (Sequential s s) := NN.API.nn.SiLU (s := s)

@[inherit_doc NN.API.nn.Sigmoid]
abbrev Sigmoid {s : Shape} : M (Sequential s s) := NN.API.nn.Sigmoid (s := s)

@[inherit_doc NN.API.nn.Tanh]
abbrev Tanh {s : Shape} : M (Sequential s s) := NN.API.nn.Tanh (s := s)

@[inherit_doc NN.API.nn.Softmax]
abbrev Softmax {s : Shape} : M (Sequential s s) := NN.API.nn.Softmax (s := s)

@[inherit_doc NN.API.nn.Flatten]
abbrev Flatten {s : Shape} : M (Sequential s (.dim (Shape.size s) .scalar)) :=
  NN.API.nn.Flatten (s := s)

@[inherit_doc NN.API.nn.FlattenBatch]
abbrev FlattenBatch {n : Nat} {s : Shape} :
    M (Sequential (.dim n s) (Shape.mat n (Shape.size s))) :=
  NN.API.nn.FlattenBatch (n := n) (s := s)

@[inherit_doc NN.API.nn.pure.heads.classifierBatch]
abbrev ClassifierBatch {n : Nat} {s : Shape} (classes : Nat) :
    M (Sequential (.dim n s) (Shape.mat n classes)) :=
  pure <| NN.API.nn.pure.heads.classifierBatch (n := n) (s := s) classes

@[inherit_doc NN.API.nn.pure.heads.regressorBatch]
abbrev RegressorBatch {n : Nat} {s : Shape} (outDim : Nat := 1) :
    M (Sequential (.dim n s) (Shape.mat n outDim)) :=
  pure <| NN.API.nn.pure.heads.regressorBatch (n := n) (s := s) outDim

@[inherit_doc NN.API.nn.pure.sum]
abbrev Sum {s : Shape} : M (Sequential s Shape.scalar) :=
  pure <| NN.API.nn.pure.sum (s := s)

@[inherit_doc NN.API.nn.pure.multiheadAttention]
abbrev MultiheadAttentionLayer {batch n dModel : Nat} (cfg : MultiheadAttention)
    [NeZero n]
    (mask : Option (Tensor.T Bool (Shape.mat n n)) := none) :
    M (Sequential (shape![batch, n, dModel]) (shape![batch, n, dModel])) :=
  pure <| NN.API.nn.pure.multiheadAttention
    (batch := batch) (n := n) (dModel := dModel) cfg (mask := mask)

@[inherit_doc NN.API.nn.pure.blocks.transformerEncoderBlock]
abbrev TransformerEncoderBlock {batch n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderBlockConfig)
    (mask : Option (Tensor.T Bool (Shape.mat n n)) := none) :
    M (Sequential (shape![batch, n, dModel]) (shape![batch, n, dModel])) :=
  pure <| NN.API.nn.pure.blocks.transformerEncoderBlock
    (batch := batch) (n := n) (dModel := dModel) cfg (mask := mask)

end nn

end TorchLean
