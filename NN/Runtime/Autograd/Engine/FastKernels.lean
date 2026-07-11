/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.DGemm
public import NN.Runtime.Autograd.Engine.Cuda.Kernels

/-!
# Matmul Reference and cuBLAS Routines

Low-level matrix-multiplication routines used to compare the CPU reference implementation with
the explicit FP32 and FP64 cuBLAS paths. User-facing execution selects kernels through the runtime
device and backend profile; this module does not define a separate execution mode.
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace FastKernels

/--
Precision selector for GPU-backed fast matmul over Lean `Float` tensors.

- `.fp32` routes through `Cuda.Buffer` and cuBLAS SGEMM, matching the precision used by the eager
  CUDA tensor-buffer path.
- `.fp64` routes through the host `FloatArray` DGEMM bridge and cuBLAS DGEMM, preserving Lean
  `Float` precision for matmul-only research paths.
-/
inductive GpuMatmulPrecision where
  | fp32
  | fp64
deriving Repr, DecidableEq

/--
 Convert an `(m×n)` matrix tensor into an array-of-rows representation.

 This is purely a representation change to make runtime loops faster/easier to write.
 -/
def matToRows {α : Type} {m n : Nat} :
    Tensor α (.dim m (.dim n .scalar)) → Array (Array α)
  | .dim rows =>
      Array.ofFn (fun i : Fin m =>
        match rows i with
        | .dim cols =>
            Array.ofFn (fun j : Fin n =>
              match cols j with
              | .scalar a => a))

/--
Fast (runtime-only) 2D matmul kernel.

This is a tight-loop kernel (array-of-rows representation) intended to avoid the overhead of the
spec-layer definitions when running eager autograd.
-/
def matmulForward {α : Type} [Context α]
    {m n p : Nat}
    (a : Tensor α (.dim m (.dim n .scalar)))
    (b : Tensor α (.dim n (.dim p .scalar))) :
    Tensor α (.dim m (.dim p .scalar)) :=
  let matmulLean (a : Tensor α (.dim m (.dim n .scalar))) (b : Tensor α (.dim n (.dim p .scalar))) :
      Tensor α (.dim m (.dim p .scalar)) :=
    let aArr := matToRows (α := α) (m := m) (n := n) a
    let bArr := matToRows (α := α) (m := n) (n := p) b
    let cArr : Array (Array α) :=
      Array.ofFn (fun i : Fin m =>
        Array.ofFn (fun k : Fin p =>
          Id.run do
            let mut acc : α := 0
            for j in [0:n] do
              acc := acc + (aArr[i]!)[j]! * (bArr[j]!)[k]!
            return acc))
    Tensor.dim (fun i : Fin m =>
      Tensor.dim (fun k : Fin p =>
        Tensor.scalar (cArr[i]!)[k]!))
  matmulLean a b

namespace Cuda

/-- Convert an FFI dimension to `UInt32`, failing before the native call on overflow. -/
def natToU32! (n : Nat) : UInt32 :=
  let u := UInt32.ofNat n
  if u.toNat = n then u else panic! "fast matmul dimension does not fit in UInt32"

/-- 2D matmul forward via cuBLAS DGEMM (`torchlean_dgemm_cuda` / `Cuda.torchleanDgemmCuda`). -/
def matmulForwardcuBLAS64 {m n p : Nat}
    (a : Tensor Float (.dim m (.dim n .scalar)))
    (b : Tensor Float (.dim n (.dim p .scalar))) :
    Tensor Float (.dim m (.dim p .scalar)) :=
  let aRows := matToRows a
  let bRows := matToRows b
  let flatA : FloatArray :=
    Id.run do
      let mut out : Array Float := Array.mkEmpty (m * n)
      for row in aRows do
        for x in row do
          out := out.push x
      return FloatArray.mk out
  let flatB : FloatArray :=
    Id.run do
      let mut out : Array Float := Array.mkEmpty (n * p)
      for row in bRows do
        for x in row do
          out := out.push x
      return FloatArray.mk out
  let flatC := Runtime.Autograd.Cuda.torchleanDgemmCuda flatA flatB
    (natToU32! m) (natToU32! n) (natToU32! p)
  Tensor.dim (fun i : Fin m =>
    Tensor.dim (fun j : Fin p =>
      Tensor.scalar (flatC.get! (i.val * p + j.val))))

/--
2D matmul forward via the float32 CUDA buffer path.

This path uploads Lean `Float` values to `Cuda.Buffer` (rounding to float32), calls the existing
`Buffer.bmm` SGEMM implementation with `batch = 1`, then downloads the float32 result back to Lean
`Float`.
-/
def matmulForwardcuBLAS32 {m n p : Nat}
    (a : Tensor Float (.dim m (.dim n .scalar)))
    (b : Tensor Float (.dim n (.dim p .scalar))) :
    Tensor Float (.dim m (.dim p .scalar)) :=
  let aBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a)
  let bBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b)
  let cBuf := Runtime.Autograd.Cuda.Buffer.bmm aBuf bBuf
    (natToU32! 1) (natToU32! m) (natToU32! n) (natToU32! p)
  Runtime.Autograd.Cuda.Convert.unflattenFloatUnsafe
    (s := .dim m (.dim p .scalar))
    (Runtime.Autograd.Cuda.Buffer.toFloatArray cBuf)

/-- Dispatch to the requested GPU matmul precision. -/
def matmulForwardcuBLASWith (precision : GpuMatmulPrecision) {m n p : Nat}
    (a : Tensor Float (.dim m (.dim n .scalar)))
    (b : Tensor Float (.dim n (.dim p .scalar))) :
    Tensor Float (.dim m (.dim p .scalar)) :=
  match precision with
  | .fp32 => matmulForwardcuBLAS32 (m := m) (n := n) (p := p) a b
  | .fp64 => matmulForwardcuBLAS64 (m := m) (n := n) (p := p) a b

end Cuda

end FastKernels

end Autograd
end Runtime
