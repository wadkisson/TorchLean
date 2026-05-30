/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Runtime.Core

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Layer Helpers

Sequential layer constructors over the lower-level TorchLean runtime surface. These keep the direct
runtime API available while the higher-level `NN.API.Public.nn` namespace provides named-field configs.
-/

namespace Layers

/-!
### Sequential Layer Helpers

`Runtime.Autograd.TorchLean.NN` exposes *layers* (`LayerDef σ τ`) and *sequential models*
(`Seq σ τ`). This namespace provides direct `Seq` constructors and common derived shapes such as
`flattenLinear`.

For the more fully-documented public surface (named-field configs, blocks, etc.), see
`NN.API.Public` under `API.nn`.
-/

/-- Lift a single layer into a sequential model. -/
def of {σ τ : Spec.Shape} (layer : API.TorchLean.NN.LayerDef σ τ) :
    API.TorchLean.NN.Seq σ τ :=
  API.TorchLean.NN.seq1 layer

/-- Linear layer over vectors (returns a 1-layer `Seq`). -/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  of <| API.TorchLean.NN.linear inDim outDim seedW seedB

/-- Pointwise ReLU activation, preserving the input shape. -/
def relu {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.relu (s := s)

/-- Elementwise SiLU/Swish. -/
def silu {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.silu (s := s)

/-- Pointwise GELU activation, preserving the input shape. -/
def gelu {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.gelu (s := s)

/-- Pointwise logistic sigmoid activation, preserving the input shape. -/
def sigmoid {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.sigmoid (s := s)

/-- Pointwise hyperbolic tangent activation, preserving the input shape. -/
def tanh {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.tanh (s := s)

/-- Softmax over the flattened tensor entries for the current runtime layer convention. -/
def softmax {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.softmax (s := s)

/-- Pointwise square map, preserving the input shape. -/
def square {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.square (s := s)

/-- Reduce-sum to a scalar. -/
def sum {s : Spec.Shape} : API.TorchLean.NN.Seq s Spec.Shape.scalar :=
  of <| API.TorchLean.NN.sum (s := s)

/-- Flatten any input shape into a 1D vector of length `Spec.Shape.size s`. -/
def flatten {s : Spec.Shape} : API.TorchLean.NN.Seq s (.dim (Spec.Shape.size s) .scalar) :=
  of <| API.TorchLean.NN.flatten (s := s)

/-- Dropout layer that is active in training mode and identity in eval mode. -/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.dropout (s := s) p seed

/-- `Flatten -> Linear` head, with the input dimension computed from the input shape. -/
def flattenLinear {s : Spec.Shape} (outDim : Nat) (seedW seedB : Nat := 0) :
    API.TorchLean.NN.Seq s (NN.Tensor.Shape.Vec outDim) :=
  (flatten (s := s)) >>> (linear (Spec.Shape.size s) outDim seedW seedB)

/-- Sequential 2D convolution layer for CHW inputs. -/
def conv2d (inC outC kH kW stride padding inH inW : Nat)
    {hInC : inC ≠ 0} {hKH : kH ≠ 0} {hKW : kW ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  of <| API.TorchLean.NN.conv2d
    (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) (h1 := hInC) (h2 := hKH) (h3 := hKW)
    (seedK := seedK) (seedB := seedB) (kInit := kInit)

/-- Sequential max-pooling layer for CHW inputs. -/
def maxPool2d (kH kW inH inW inC stride : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  of <| API.TorchLean.NN.maxPool2d
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := hKH) (h2 := hKW)

/-- Sequential padded max-pooling layer for CHW inputs. -/
def maxPool2dPad (kH kW inH inW inC stride padding : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  of <| API.TorchLean.NN.maxPool2dPad
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
      padding)
    (h1 := hKH) (h2 := hKW)

/-- Sequential average-pooling layer for CHW inputs. -/
def avgPool2d (kH kW inH inW inC stride : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  of <| API.TorchLean.NN.avgPool2d
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    hKH hKW

/-- Sequential padded average-pooling layer for CHW inputs. -/
def avgPool2dPad (kH kW inH inW inC stride padding : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  of <| API.TorchLean.NN.avgPool2dPad
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
      padding)
    hKH hKW

/--
Global average-pooling over `C×H×W` inputs.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` followed by
flattening the spatial axes.
-/
def globalAvgPoolCHW (c h w : Nat)
    {hC : c > 0} {hH : h > 0} {hW : w > 0} :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.CHW c h w) (NN.Tensor.Shape.Vec c) :=
  of <| API.TorchLean.NN.globalAvgPool2dChw (c := c) (h := h) (w := w)
    (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)

/--
Global average-pooling over `N×C×H×W` inputs.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` and then reshaping
to `(N, C)`.
-/
def globalAvgPoolNCHW (n c h w : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0} :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (.dim n (.dim c .scalar)) :=
  of <| API.TorchLean.NN.globalAvgPool2dNchw (n := n) (c := c) (h := h) (w := w)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)

/--
Sequence-wise layer normalization.

PyTorch analogy: `torch.nn.LayerNorm(embedDim)` applied to each position in a sequence.
-/
def layerNorm (batch seqLen embedDim : Nat)
    {hSeq : seqLen > 0} {hEmbed : embedDim > 0}
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.NN.Seq (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| API.TorchLean.NN.layerNorm (batch := batch)
    (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Sequence-wise RMS normalization.

PyTorch analogy: an `RMSNorm`-style layer over `(seqLen × embedDim)` tensors.
-/
def rmsNorm (batch seqLen embedDim : Nat)
    {hSeq : seqLen > 0} {hEmbed : embedDim > 0}
    (seedGamma : Nat := 0) :
    API.TorchLean.NN.Seq (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| API.TorchLean.NN.rmsNorm (batch := batch)
    (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := seedGamma)

/--
Mode-aware batch norm on a single `C×H×W` image tensor.

PyTorch analogy: `torch.nn.BatchNorm2d(channels)` on a single sample, with the layer's mode
controlling whether running statistics are updated or reused.
-/
def batchNormCHW (channels height width : Nat)
    {hC : channels > 0} {hH : height > 0} {hW : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW channels height width)
      (NN.Tensor.Shape.CHW channels height width) :=
  of <| API.TorchLean.NN.batchnormChannelFirstMode
    (channels := channels) (height := height) (width := width)
    (h_c := hC) (h_h := hH) (h_w := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)
    (seedMean := seedMean) (seedVar := seedVar)

/--
Eval-mode batch norm on a single `C×H×W` image tensor with explicit running statistics.

PyTorch analogy: `torch.nn.BatchNorm2d(...).eval()` with `running_mean` and `running_var`.
-/
def batchNormEvalCHW (channels height width : Nat)
    {hC : channels > 0} {hH : height > 0} {hW : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW channels height width)
      (NN.Tensor.Shape.CHW channels height width) :=
  of <| API.TorchLean.NN.batchnormChannelFirstEval
    (channels := channels) (height := height) (width := width)
    (h_c := hC) (h_h := hH) (h_w := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)
    (seedMean := seedMean) (seedVar := seedVar)

/--
Instance normalization over `N×C×H×W` tensors.

PyTorch analogy: `torch.nn.InstanceNorm2d(c, affine=True)` with `NCHW` layout.
-/
def instanceNorm2dNCHW (n c h w : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0}
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  of <| API.TorchLean.NN.instanceNorm2dNchw
    (n := n) (c := c) (h := h) (w := w)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Group normalization over `N×C×H×W` tensors.

PyTorch analogy: `torch.nn.GroupNorm(groups, c)` with `NCHW` layout.
-/
def groupNorm2dNCHW (n c h w groups : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0} {hG : groups > 0}
    (hGE : c ≥ groups) (hDiv : c % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  of <| API.TorchLean.NN.groupNorm2dNchw
    (n := n) (c := c) (h := h) (w := w) (groups := groups)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW) (h_g_pos := hG)
    hGE hDiv
    (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Batch norm over `N×C×H×W` tensors in training mode.

PyTorch analogy: `torch.nn.BatchNorm2d(c)` during training, where batch statistics are used.
-/
def batchNorm2dNCHW (n c h w : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  of <| API.TorchLean.NN.batchNorm2dNchwMode
    (n := n) (c := c) (h := h) (w := w)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)
    (seedMean := seedMean) (seedVar := seedVar)

/--
Multi-head self-attention over sequence embeddings.

PyTorch analogy: `torch.nn.MultiheadAttention(embed_dim=dModel, num_heads=numHeads)` in self-
attention mode, with explicit `n × dModel` shapes.
-/
def attention (batch n dModel numHeads headDim : Nat)
    {hN : n ≠ 0}
    (seedW : Nat := 0)
    (mask : Option (_root_.Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    API.TorchLean.NN.Seq (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  of <| API.TorchLean.NN.multiHeadAttention (batch := batch)
    (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
    (h1 := hN) (seedW := seedW) (mask := mask)

end Layers

end TorchLean
end API
end NN
