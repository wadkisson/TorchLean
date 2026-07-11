/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Entrypoint.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: LibTorch SDPA

Explicit smoke test for the optional LibTorch scaled-dot-product-attention provider.

This module is intentionally not part of the default CUDA suite: the externs here require building
with `-K cuda=true -K libtorch=true`. The test compares the LibTorch SDPA bridge against the native
TorchLean fused attention bridge without a mask and with a hard boolean mask. The masked case
includes a fully blocked row, which must produce zero output rather than a finite-sentinel
approximation or a NaN.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace LibTorchSDPA

open Spec

abbrev batch : Nat := 2
abbrev n : Nat := 2
abbrev d : Nat := 2

abbrev s : Shape := shape![batch, n, d]
abbrev maskShape : Shape := shape![batch, n, n]

def q : Tensor Float s :=
  tensorND! [batch, n, d] [
    0.10, -0.20,
    0.30,  0.05,
   -0.15,  0.25,
    0.40, -0.10
  ]

def k : Tensor Float s :=
  tensorND! [batch, n, d] [
    0.05,  0.20,
   -0.10,  0.30,
    0.15, -0.25,
    0.35,  0.10
  ]

def v : Tensor Float s :=
  tensorND! [batch, n, d] [
    0.20, -0.05,
    0.10,  0.30,
   -0.20,  0.15,
    0.05, -0.10
  ]

def dOut : Tensor Float s :=
  tensorND! [batch, n, d] [
    1.00, 0.50,
   -0.25, 0.75,
    0.30, 1.20,
   -0.60, 0.40
  ]

/-- `1` marks an allowed key. The last query row is fully blocked. -/
def hardMask : Tensor Float maskShape :=
  tensorND! [batch, n, n] [
    1.0, 0.0,
    1.0, 1.0,
    1.0, 0.0,
    0.0, 0.0
  ]

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: LibTorch SDPA ==="

  let qBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) q)
  let kBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) k)
  let vBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) v)
  let dOutBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) dOut)
  let emptyMask := Runtime.Autograd.Cuda.Buffer.zeros 0

  let batch32 := UInt32.ofNat batch
  let n32 := UInt32.ofNat n
  let d32 := UInt32.ofNat d
  let scale : Float := 1.0 / Float.sqrt (Float.ofNat d)

  let nativeY := Runtime.Autograd.Cuda.Buffer.flashAttentionFwd
    qBuf kBuf vBuf emptyMask 0 batch32 n32 d32 scale
  let libTorchY ← Runtime.Autograd.Cuda.Buffer.libTorchSDPAFwd
    qBuf kBuf vBuf emptyMask 0 batch32 n32 d32 scale

  let nativeYT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeY
  let libTorchYT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchY
  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch sdpa forward" libTorchYT nativeYT (tol := 2e-2)

  let (nativeDQ, nativeDK, nativeDV) := Runtime.Autograd.Cuda.Buffer.flashAttentionBwd
    qBuf kBuf vBuf emptyMask dOutBuf 0 batch32 n32 d32 scale
  let (libTorchDQ, libTorchDK, libTorchDV) ← Runtime.Autograd.Cuda.Buffer.libTorchSDPABwd
    qBuf kBuf vBuf emptyMask dOutBuf 0 batch32 n32 d32 scale

  let nativeDQT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeDQ
  let nativeDKT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeDK
  let nativeDVT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeDV
  let libTorchDQT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchDQ
  let libTorchDKT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchDK
  let libTorchDVT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchDV

  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch sdpa dQ" libTorchDQT nativeDQT (tol := 2e-2)
  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch sdpa dK" libTorchDKT nativeDKT (tol := 2e-2)
  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch sdpa dV" libTorchDVT nativeDVT (tol := 2e-2)

  let maskBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := maskShape) hardMask)
  let nativeMaskedY := Runtime.Autograd.Cuda.Buffer.flashAttentionFwd
    qBuf kBuf vBuf maskBuf 1 batch32 n32 d32 scale
  let libTorchMaskedY ← Runtime.Autograd.Cuda.Buffer.libTorchSDPAFwd
    qBuf kBuf vBuf maskBuf 1 batch32 n32 d32 scale
  let nativeMaskedYT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeMaskedY
  let libTorchMaskedYT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchMaskedY
  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch hard-masked sdpa forward" libTorchMaskedYT nativeMaskedYT (tol := 2e-2)

  let (nativeMaskedDQ, nativeMaskedDK, nativeMaskedDV) :=
    Runtime.Autograd.Cuda.Buffer.flashAttentionBwd
      qBuf kBuf vBuf maskBuf dOutBuf 1 batch32 n32 d32 scale
  let (libTorchMaskedDQ, libTorchMaskedDK, libTorchMaskedDV) ←
    Runtime.Autograd.Cuda.Buffer.libTorchSDPABwd
      qBuf kBuf vBuf maskBuf dOutBuf 1 batch32 n32 d32 scale
  let nativeMaskedDQT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeMaskedDQ
  let nativeMaskedDKT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeMaskedDK
  let nativeMaskedDVT ← Tests.Cuda.Utils.bufferToTensor (s := s) nativeMaskedDV
  let libTorchMaskedDQT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchMaskedDQ
  let libTorchMaskedDKT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchMaskedDK
  let libTorchMaskedDVT ← Tests.Cuda.Utils.bufferToTensor (s := s) libTorchMaskedDV
  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch hard-masked sdpa dQ" libTorchMaskedDQT nativeMaskedDQT (tol := 2e-2)
  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch hard-masked sdpa dK" libTorchMaskedDKT nativeMaskedDKT (tol := 2e-2)
  Tests.Cuda.Utils.assertTensorApprox (s := s)
    "libtorch hard-masked sdpa dV" libTorchMaskedDVT nativeMaskedDVT (tol := 2e-2)

  let shortQ := Runtime.Autograd.Cuda.Buffer.zeros 1
  let rejected ← try
    let _ ← Runtime.Autograd.Cuda.Buffer.libTorchSDPAFwd
      shortQ kBuf vBuf emptyMask 0 batch32 n32 d32 scale
    pure false
  catch _ =>
    pure true
  unless rejected do
    throw <| IO.userError "libtorch sdpa accepted a Q buffer with the wrong size"
  IO.println "== LibTorch SDPA bridge: OK =="

def main : IO Unit :=
  run

end LibTorchSDPA
end Cuda
end Tests

def main : IO Unit :=
  Tests.Cuda.LibTorchSDPA.main
