/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Indexing

/-!
# CUDA Tape Operations: Convolution and Pooling
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Tape

/-!
## Conv2D + pooling (ConvPool FFI)
-/

/-- Conv2D forward/backward via ConvPool FFI (single image, channels-first). -/
def conv2d
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape) (kernelId biasId inputId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  have _ := h3
  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kernel ← requireValue (t := t) kernelId (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))
  let bias ← requireValue (t := t) biasId (.dim outC .scalar)
  let input ← requireValue (t := t) inputId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let y := torchleanConv2dFwdCuda input kernel bias inC32 inH32 inW32 outC32 kH32 kW32 stride32
    pad32
  let outShape : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : Node :=
    { name := some "conv2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConv2dBwdCuda input kernel dLdy.buf
            inC32 inH32 inW32 outC32 kH32 kW32 stride32 pad32
        pure [
          (kernelId, { s := .dim outC (.dim inC (.dim kH (.dim kW .scalar))), buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dInput })
        ] }
  pure (t.addNode node)

/-!
### ConvTranspose2D (ConvPool FFI)
-/

/-- ConvTranspose2D forward/backward via ConvPool FFI (single image, channels-first). -/
def convTranspose2d
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape) (kernelId biasId inputId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  have _ := h3
  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kernel ← requireValue (t := t) kernelId (.dim inC (.dim outC (.dim kH (.dim kW .scalar))))
  let bias ← requireValue (t := t) biasId (.dim outC .scalar)
  let input ← requireValue (t := t) inputId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - 1) * stride - 2 * padding + kH
  let outW : Nat := (inW - 1) * stride - 2 * padding + kW
  let y :=
    torchleanConvTranspose2dFwdCuda input kernel bias inC32 inH32 inW32 outC32 kH32 kW32 stride32
      pad32
  let outShape : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : Node :=
    { name := some "conv_transpose2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvTranspose2dBwdCuda input kernel dLdy.buf
            inC32 inH32 inW32 outC32 kH32 kW32 stride32 pad32
        pure [
          (kernelId, { s := .dim inC (.dim outC (.dim kH (.dim kW .scalar))), buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dInput })
        ] }
  pure (t.addNode node)

/-!
### Generic naming wrappers

The CUDA tape exposes `conv`/`max_pool`/`avg_pool`/`smooth_max_pool` using the same names as the
CPU tape. These dispatch to the ConvPool CUDA FFI entrypoints that take per-axis parameters as
`Array Nat` (rank ≤ 8).

The `*2d*` wrappers remain as concise convenience names for the common rank-2 case.
-/

/-- N-D convolution (CUDA) via ConvPool FFI (rank ≤ 8). -/
def conv
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape) (kernelId biasId inputId : Nat)
  (hInC : inC ≠ 0)
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
  Result (Tape × Nat) := do
  have _ := hInC
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: conv: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: conv: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: conv: stride must be > 0"

  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let kernelShape : Shape :=
    Shape.ofList (outC :: inC :: kernel.toList)
  let inputShape : Shape :=
    Shape.ofList (inC :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (outC :: outSpatial.toList)

  let kernelBuf ← requireValue (t := t) kernelId kernelShape
  let biasBuf ← requireValue (t := t) biasId (.dim outC .scalar)
  let inputBuf ← requireValue (t := t) inputId inputShape

  let y :=
    torchleanConvFwdCuda inputBuf kernelBuf biasBuf
      inSpatialArr kernelSpatialArr strideArr paddingArr
      inC32 outC32

  let node : Node :=
    { name := some "conv"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvBwdCuda inputBuf kernelBuf dLdy.buf
            inSpatialArr kernelSpatialArr strideArr paddingArr
            inC32 outC32
        pure [
          (kernelId, { s := kernelShape, buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := inputShape, buf := dInput })
        ] }
  pure (t.addNode node)

/-- N-D transposed convolution (CUDA) via ConvPool FFI (rank ≤ 8). -/
def convTranspose
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape) (kernelId biasId inputId : Nat)
  (hInC : inC ≠ 0)
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
  Result (Tape × Nat) := do
  have _ := hInC
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: conv_transpose: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: conv_transpose: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: conv_transpose: stride must be > 0"

  let inC32 ← u32 inC
  let outC32 ← u32 outC
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  -- NOTE: for transposed conv, kernel layout is `(inC, outC, kernelSpatial...)`.
  let kernelShape : Shape :=
    Shape.ofList (inC :: outC :: kernel.toList)
  let inputShape : Shape :=
    Shape.ofList (inC :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a - 1) * stride.get a - 2 * padding.get a + kernel.get a)
  let outShape : Shape :=
    Shape.ofList (outC :: outSpatial.toList)

  let kernelBuf ← requireValue (t := t) kernelId kernelShape
  let biasBuf ← requireValue (t := t) biasId (.dim outC .scalar)
  let inputBuf ← requireValue (t := t) inputId inputShape

  let y :=
    torchleanConvTransposeFwdCuda inputBuf kernelBuf biasBuf
      inSpatialArr kernelSpatialArr strideArr paddingArr
      inC32 outC32

  let node : Node :=
    { name := some "conv_transpose"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvTransposeBwdCuda inputBuf kernelBuf dLdy.buf
            inSpatialArr kernelSpatialArr strideArr paddingArr
            inC32 outC32
        pure [
          (kernelId, { s := kernelShape, buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := inputShape, buf := dInput })
        ] }
  pure (t.addNode node)

/-- MaxPool2D via ConvPool FFI (no padding). -/
def maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 : UInt32 := 0
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - kH) / stride + 1
  let outW : Nat := (inW - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanMaxPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "max_pool2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanMaxPool2dBwdCuda x dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- MaxPool2D via ConvPool FFI (with symmetric padding). -/
def maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanMaxPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "max_pool2d_pad"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanMaxPool2dBwdCuda x dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- N-D max pooling (CUDA) via ConvPool FFI (rank ≤ 8). -/
def maxPool
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: max_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: max_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: max_pool: stride must be > 0"

  let inC32 ← u32 C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.toList)

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanMaxPoolFwdCuda xBuf inSpatialArr kernelArr strideArr paddingArr inC32

  let node : Node :=
    { name := some "max_pool"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanMaxPoolBwdCuda xBuf dLdy.buf
            inSpatialArr kernelArr strideArr paddingArr inC32
        pure [(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)

/-- Smooth max-pool2d (log-sum-exp surrogate) via ConvPool FFI (no padding). -/
def smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  if beta == 0.0 then
    throw "autograd: cuda: smooth_max_pool2d: beta must be nonzero"
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 : UInt32 := 0
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - kH) / stride + 1
  let outW : Nat := (inW - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanSmoothMaxPool2dFwdCuda x beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "smooth_max_pool2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanSmoothMaxPool2dBwdCuda x dLdy.buf beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- Smooth max-pool2d (log-sum-exp surrogate) via ConvPool FFI (with symmetric padding). -/
def smoothMaxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  if beta == 0.0 then
    throw "autograd: cuda: smooth_max_pool2d_pad: beta must be nonzero"
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanSmoothMaxPool2dFwdCuda x beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "smooth_max_pool2d_pad"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanSmoothMaxPool2dBwdCuda x dLdy.buf beta inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- N-D smooth max pooling (CUDA) via ConvPool FFI (rank ≤ 8). -/
def smoothMaxPool
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  if beta == 0.0 then
    throw "autograd: cuda: smooth_max_pool: beta must be nonzero"
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: smooth_max_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: smooth_max_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: smooth_max_pool: stride must be > 0"

  let inC32 ← u32 C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.toList)

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanSmoothMaxPoolFwdCuda xBuf beta
      inSpatialArr kernelArr strideArr paddingArr
      inC32

  let node : Node :=
    { name := some "smooth_max_pool"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanSmoothMaxPoolBwdCuda xBuf dLdy.buf beta
            inSpatialArr kernelArr strideArr paddingArr
            inC32
        pure [(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)

/-- AvgPool2D via ConvPool FFI (no padding). -/
def avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 : UInt32 := 0
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH - kH) / stride + 1
  let outW : Nat := (inW - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanAvgPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "avg_pool2d"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanAvgPool2dBwdCuda dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- AvgPool2D via ConvPool FFI (with symmetric padding). -/
def avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := h1
  have _ := h2
  let inC32 ← u32 inC
  let inH32 ← u32 inH
  let inW32 ← u32 inW
  let kH32 ← u32 kH
  let kW32 ← u32 kW
  let stride32 ← u32 stride
  let pad32 ← u32 padding
  let x ← requireValue (t := t) xId (.dim inC (.dim inH (.dim inW .scalar)))
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
  let y := torchleanAvgPool2dFwdCuda x inC32 inH32 inW32 kH32 kW32 stride32 pad32
  let node : Node :=
    { name := some "avg_pool2d_pad"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx := torchleanAvgPool2dBwdCuda dLdy.buf inC32 inH32 inW32 kH32 kW32 stride32 pad32
        pure [(xId, { s := .dim inC (.dim inH (.dim inW .scalar)), buf := dx })] }
  pure (t.addNode node)

/-- N-D average pooling (CUDA) via ConvPool FFI (rank ≤ 8). -/
def avgPool
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  have _ := hKernel
  if d = 0 then
    throw "autograd: cuda: avg_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: avg_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, stride.get i ≠ 0) then
    throw "autograd: cuda: avg_pool: stride must be > 0"

  let inC32 ← u32 C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.get i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.get i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.get i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.get i)

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.toList)
  let outSpatial : Vector Nat d :=
    Vector.ofFn (fun a =>
      (inSpatial.get a + 2 * padding.get a - kernel.get a) / stride.get a + 1)
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.toList)

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanAvgPoolFwdCuda xBuf
      inSpatialArr kernelArr strideArr paddingArr
      inC32

  let node : Node :=
    { name := some "avg_pool"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanAvgPoolBwdCuda dLdy.buf
            inSpatialArr kernelArr strideArr paddingArr
            inC32
        pure [(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)
end Tape

end Cuda
end Autograd
end Runtime

