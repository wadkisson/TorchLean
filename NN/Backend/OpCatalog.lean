/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Types

/-!
# Shared Backend Op Catalog

Operation lists shared by Native CUDA and Reference capsule registries.

Keeping one catalog means adding a pointwise/view/reduction/conv-pool op updates both backends.
Backend-specific specials (attention, FFT, selective scan, sin/cos, custom summaries) stay in the
provider modules.
-/

@[expose] public section

namespace NN
namespace Backend
namespace OpCatalog

/-- Pointwise ops registered for both native CUDA and reference providers. -/
def pointwiseOps : List BackendOp :=
  [ .add, .sub, .mul, .scale, .abs, .sqrt, .clamp, .max, .min
  , .sigmoid, .tanh, .softplus, .exp, .log, .inv, .safeLog, .logSoftmax ]

/-- Reduction ops registered for both providers. -/
def reductionOps : List BackendOp :=
  [ .sum, .reduceSum, .reduceMean ]

/-- View / indexing ops registered for both providers. -/
def viewOps : List BackendOp :=
  [ .flatten, .reshape, .permute, .transpose2d, .swapAdjacentAtDepth
  , .transpose3dFirstToLast, .transpose3dLastToFirst, .transpose3dLastTwo
  , .broadcastTo, .concatVectors, .concatLeadingAxis, .sliceLeadingAxisRange
  , .gatherScalar, .gatherScalarNat, .gatherRow, .gatherVecNat, .gatherRowsNat
  , .scatterAddVec, .scatterAddRow ]

/-- Conv/pool ops that use the generic conv-pool capsule builders. -/
def convPoolOps : List BackendOp :=
  [ .conv, .convTranspose, .convTranspose2d
  , .maxPool2d, .maxPool2dPad, .smoothMaxPool, .smoothMaxPool2d
  , .avgPool2d, .avgPool2dPad ]

/-- Reference-only pointwise trig ops (not currently wired as native CUDA capsules). -/
def referenceExtraPointwiseOps : List BackendOp :=
  [ .sin, .cos ]

end OpCatalog
end Backend
end NN
