/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops

/-!
# Backward Passes and Optimizers

Gradient extraction and optimizer updates for eager sessions, including the CUDA paths that keep
parameter mirrors and moment buffers on device.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Internal

namespace EagerSession

/--
Run reverse-mode backprop on the CUDA tape, returning device gradients for all tape entries.

This is the CUDA analogue of `backwardDenseAll`, but it does *not* download gradients back to the
host. This is primarily useful for implementing GPU-native optimizer steps.
-/
def backwardDenseAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α] [Zero α]
  [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array Runtime.Autograd.Cuda.AnyBuffer) := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: backwardDenseAllCuda called on non-CUDA eager session"
  let t ← s.cudaTape.get
  let seedAny ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := sh) seed
  okOrThrow <|
    Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t) (outId := out.id) (seed := seedAny)

/-- Run CUDA backward from a scalar loss with seed `1`, returning device gradient buffers. -/
def backwardScalarDenseAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α]
  [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array Runtime.Autograd.Cuda.AnyBuffer) := do
  backwardDenseAllCuda (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Accumulate one CUDA gradient contribution into a sparse map.

Ownership rule: the contribution buffer `g` is consumed by this function. When a contribution is
first inserted into the map, we store an owned copy and release the incoming buffer. That extra copy
is intentional: CUDA backward rules are allowed to return a fresh buffer, but view-like rules may
also pass through an upstream buffer. Copy-on-insert keeps this sparse accumulator correct for every
op without requiring every local VJP to expose aliasing metadata. When a second contribution arrives,
we sum into a fresh buffer and release both inputs.

This rule is what lets sparse CUDA backprop avoid the dense "one zero buffer per tape node"
representation without leaking transient gradients across long training loops.
-/
def addCudaGradToMap (t : Runtime.Autograd.Cuda.Tape)
    (gradsRef : IO.Ref CudaGradMap) (id : Nat)
    (g : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw <| IO.userError "torch: invalid parent id during CUDA backward"
  if node.requires_grad = false then
    checkCudaAnyBufferSize s!"discarded gradient for node {id}" g
    releaseCudaAnyBuffer g
  else if _h : g.s = node.value.s then
    let g' : Runtime.Autograd.Cuda.AnyBuffer := { s := node.value.s, buf := g.buf }
    checkCudaAnyBufferSize s!"gradient contribution for node {id} ({node.name})" g'
    let grads ← gradsRef.get
    match grads.get? id with
    | none =>
        let owned ← ownedCudaAnyBuffer s!"owned gradient for node {id} ({node.name})" g'
        releaseCudaAnyBuffer g'
        gradsRef.set (grads.insert id owned)
    | some old =>
        if _hold : old.s = node.value.s then
          let old' : Runtime.Autograd.Cuda.AnyBuffer := { s := node.value.s, buf := old.buf }
          checkCudaAnyBufferSize s!"accumulated gradient for node {id} ({node.name})" old'
          let summed ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.add old' g'
          releaseCudaAnyBuffer old'
          releaseCudaAnyBuffer g'
          gradsRef.set (grads.insert id summed)
        else
          releaseCudaAnyBuffer g'
          throw <| IO.userError "torch: CUDA gradient map has wrong shape for node"
  else
    releaseCudaAnyBuffer g
    throw <| IO.userError "torch: CUDA gradient contribution has wrong shape for parent"

/--
Run scalar-loss CUDA backprop and return gradients only for trainable parameter leaves.

The returned map stays on device so CUDA optimizers can update parameters without downloading dense
gradient arrays to the host.
-/
def backwardScalarParamGradsCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α)
    [One α] [DecidableEq Shape]
    (loss : TensorRef α Shape.scalar) : IO CudaGradMap := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: backwardScalarParamGradsCuda called on non-CUDA eager session"
  let t ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let seedAny ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := Shape.scalar)
    (Tensor.scalar (1 : α))
  checkCudaAnyBufferSize "scalar CUDA backward seed" seedAny
  let gradsRef ← IO.mkRef ((Std.HashMap.emptyWithCapacity).insert loss.id seedAny : CudaGradMap)
  for off in [0:t.nodes.size] do
    let id := t.nodes.size - 1 - off
    let grads ← gradsRef.get
    match grads.get? id with
    | none => pure ()
    | some dLdy =>
        let node ← match t.getNode? id with
          | some n => pure n
          | none => throw <| IO.userError "torch: internal CUDA tape node missing"
        if node.requires_grad then
          checkCudaAnyBufferSize s!"upstream gradient for node {id} ({node.name})" dLdy
          let contribs ← okOrThrow <| node.backward dLdy
          for (pid, pg) in contribs do
            addCudaGradToMap t gradsRef pid pg
        if params.contains id then
          pure ()
        else
          let gradsNow ← gradsRef.get
          match gradsNow.get? id with
          | none => pure ()
          | some stale =>
              releaseCudaAnyBuffer stale
              gradsRef.set (gradsNow.erase id)
  gradsRef.get

/--
Run reverse-mode backprop and return a dense gradient array for all tape entries.

`seed` is the upstream gradient for `out` (like PyTorch's `backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α] [Zero α]
  [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Runtime.AnyTensor α)) := do
  if Options.device s.opts == .cuda then
    let gradsDev ← backwardDenseAllCuda (α := α) s (sh := sh) out seed
    gradsDev.mapM (fun g => CudaBridge.TensorConv.ofAnyBuffer (α := α) g)
  else
    let t ← s.tape.get
    okOrThrow (Runtime.Autograd.Tape.backwardDenseAll (t := t) (outId := out.id)
      (seed := Runtime.Autograd.AnyTensor.mk seed))

/--
Run backward from a scalar loss with seed `1`.

PyTorch comparison: `loss.backward()` for a scalar loss.
-/
def backwardScalarDenseAll {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α]
  [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array (Runtime.AnyTensor α)) := do
  backwardDenseAll (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Extract the gradient for a particular `TensorRef` from a dense gradient array.
-/
def grad {α : Type} {sh : Shape} [DecidableEq Shape]
  (grads : Array (Runtime.AnyTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch: gradient array out of bounds"
  if h : gAny.s = sh then
    pure (Tensor.castShape gAny.t h)
  else
    throw <| IO.userError
      s!"torch: grad shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty gAny.s})"

/--
Apply an SGD update to all parameters recorded via `use`.

PyTorch comparison: `for p in params: p.data -= lr * p.grad`.
-/
def sgdStepAll {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  (lr : α) (grads : Array (Runtime.AnyTensor α)) : IO Unit := do
  if Options.device s.opts == .cuda then
    let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
    let t0 ← s.cudaTape.get
    let m ← s.paramsByLeaf.get
    for (id, p) in m.toList do
      let gAny ← match grads[id]? with
        | some g => pure g
        | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
      if hs : gAny.s = p.s then
        let gT : Tensor α p.s := Tensor.castShape gAny.t hs
        let gDev ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := p.s) gT
        let pBuf ← okOrThrow <|
          Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
        let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
          { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gDev.buf (-lrF) }
        p.setCuda updatedDev
        -- The uploaded host gradient is only a transient bridge buffer for this update.
        let released := Runtime.Autograd.Cuda.Buffer.release gDev.buf
        AnyParam.observeCudaCleanupFlag released
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"
  else
    let m ← s.paramsByLeaf.get
    for (id, p) in m.toList do
      let gAny ← match grads[id]? with
        | some g => pure g
        | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
      if hs : gAny.s = p.s then
        let pv ← p.get
        if hp : pv.s = p.s then
          let pvT : Tensor α p.s := Tensor.castShape pv.t hp
          let gT : Tensor α p.s := Tensor.castShape gAny.t hs
          let updated : Tensor α p.s :=
            Tensor.materialize <| subSpec pvT (scaleSpec (α := α) (s := p.s) gT lr)
          p.set (Runtime.Autograd.AnyTensor.mk updated)
        else
          throw <| IO.userError "torch: internal param shape mismatch"
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"

/--
Apply an SGD update to all parameters recorded via `use`, using CUDA device gradients.

This avoids downloading the full dense gradient array and keeps updated parameters in each
`Param`'s CUDA mirror. Host tensors are synchronized later by explicit parameter readback.
-/
def sgdStepAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
  (lr : α) (grads : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: sgdStepAllCuda called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let t0 ← s.cudaTape.get
  let m ← s.paramsByLeaf.get
  for (id, p) in m.toList do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gAny.buf (-lrF) }
      p.setCuda updatedDev
    else
      throw <| IO.userError "torch: internal grad shape mismatch during SGD"

/--
Apply SGD from a sparse CUDA gradient map.

This is the path used by the CUDA trainer.  It updates only parameter leaves and avoids allocating
zero gradients for every forward activation in the tape.
-/
def sgdStepAllCudaMap {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
    (lr : α) (grads : CudaGradMap) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: sgdStepAllCudaMap called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  for (id, p) in params.toList do
    let gAny ← match grads.get? id with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient map missing parameter during CUDA SGD"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gAny.buf (-lrF) }
      p.setCuda updatedDev
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA SGD"

/-- Device-side Adam moment buffers for one parameter leaf. -/
structure CudaAdamParamState where
  /-- First moment buffer. -/
  m : Runtime.Autograd.Cuda.Buffer
  /-- Second moment buffer. -/
  v : Runtime.Autograd.Cuda.Buffer
  /-- Adam step counter for this parameter. -/
  t : Nat

/-- Adam moment state keyed by parameter leaf id. -/
abbrev CudaAdamState := Std.HashMap Nat CudaAdamParamState

/--
Apply an Adam update to all parameters recorded via `use`, using CUDA device gradients.

This is the CUDA analogue of the generic `TorchLean.Optim.adam` path.  It keeps Adam moments as
device buffers and keeps updated parameters in each `Param`'s CUDA mirror, so the next CUDA forward
can reuse them without a host upload. Host tensors are synchronized later by explicit readback.
-/
def adamStepAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
    (stateRef : IO.Ref CudaAdamState)
    (lr beta1 beta2 epsilon : α)
    (grads : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamStepAllCuda called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let beta1F ← CudaBridge.TensorConv.toFloat (α := α) beta1
  let beta2F ← CudaBridge.TensorConv.toFloat (α := α) beta2
  let epsF ← CudaBridge.TensorConv.toFloat (α := α) epsilon
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  for (id, p) in params.toList do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient array out of bounds during CUDA Adam"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mScaled := Runtime.Autograd.Cuda.Buffer.scale st.m beta1F
      let m' := Runtime.Autograd.Cuda.Buffer.axpy mScaled gAny.buf oneMinusBeta1
      let g2 := Runtime.Autograd.Cuda.Buffer.mul gAny.buf gAny.buf
      let vScaled := Runtime.Autograd.Cuda.Buffer.scale st.v beta2F
      let v' := Runtime.Autograd.Cuda.Buffer.axpy vScaled g2 oneMinusBeta2
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let mHat := Runtime.Autograd.Cuda.Buffer.scale m' mHatScale
      let vHat := Runtime.Autograd.Cuda.Buffer.scale v' vHatScale
      let sqrtVHat := Runtime.Autograd.Cuda.Buffer.sqrt vHat
      let epsBuf := Runtime.Autograd.Cuda.Buffer.full n epsF
      let denom :=
        Runtime.Autograd.Cuda.Buffer.add sqrtVHat epsBuf
      let update := Runtime.Autograd.Cuda.Buffer.div mHat denom
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf update (-lrF) }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      releaseCudaBuffer mScaled
      releaseCudaBuffer g2
      releaseCudaBuffer vScaled
      releaseCudaBuffer mHat
      releaseCudaBuffer vHat
      releaseCudaBuffer sqrtVHat
      releaseCudaBuffer epsBuf
      releaseCudaBuffer denom
      releaseCudaBuffer update
      state := state.insert id { m := m', v := v', t := t' }
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA Adam"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

/-- Apply Adam using an already-computed sparse CUDA gradient map. -/
def adamStepAllCudaMap {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
    (stateRef : IO.Ref CudaAdamState)
    (lr beta1 beta2 epsilon : α)
    (grads : CudaGradMap) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamStepAllCudaMap called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let beta1F ← CudaBridge.TensorConv.toFloat (α := α) beta1
  let beta2F ← CudaBridge.TensorConv.toFloat (α := α) beta2
  let epsF ← CudaBridge.TensorConv.toFloat (α := α) epsilon
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  for (id, p) in params.toList do
    let gAny ← match grads.get? id with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient map missing parameter during CUDA Adam"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mScaled := Runtime.Autograd.Cuda.Buffer.scale st.m beta1F
      let m' := Runtime.Autograd.Cuda.Buffer.axpy mScaled gAny.buf oneMinusBeta1
      let g2 := Runtime.Autograd.Cuda.Buffer.mul gAny.buf gAny.buf
      let vScaled := Runtime.Autograd.Cuda.Buffer.scale st.v beta2F
      let v' := Runtime.Autograd.Cuda.Buffer.axpy vScaled g2 oneMinusBeta2
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let mHat := Runtime.Autograd.Cuda.Buffer.scale m' mHatScale
      let vHat := Runtime.Autograd.Cuda.Buffer.scale v' vHatScale
      let sqrtVHat := Runtime.Autograd.Cuda.Buffer.sqrt vHat
      let epsBuf := Runtime.Autograd.Cuda.Buffer.full n epsF
      let denom := Runtime.Autograd.Cuda.Buffer.add sqrtVHat epsBuf
      let update := Runtime.Autograd.Cuda.Buffer.div mHat denom
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf update (-lrF) }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      releaseCudaBuffer mScaled
      releaseCudaBuffer g2
      releaseCudaBuffer vScaled
      releaseCudaBuffer mHat
      releaseCudaBuffer vHat
      releaseCudaBuffer sqrtVHat
      releaseCudaBuffer epsBuf
      releaseCudaBuffer denom
      releaseCudaBuffer update
      state := state.insert id { m := m', v := v', t := t' }
    else
      releaseCudaAnyBuffer gAny
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA Adam"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

/--
Apply an AdamW update to all parameters recorded via `use`, using CUDA device gradients.

This mirrors `Optim.AdamW.update`: moments are formed from the raw gradient, weight decay is applied
directly to parameters, then the Adam update is applied. Like `adamStepAllCuda`, it keeps updated
parameter buffers resident on device and only synchronizes the host copy when readback is requested.
-/
def adamWStepAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
    (stateRef : IO.Ref CudaAdamState)
    (lr weightDecay beta1 beta2 epsilon : α)
    (grads : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamWStepAllCuda called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let wdF ← CudaBridge.TensorConv.toFloat (α := α) weightDecay
  let beta1F ← CudaBridge.TensorConv.toFloat (α := α) beta1
  let beta2F ← CudaBridge.TensorConv.toFloat (α := α) beta2
  let epsF ← CudaBridge.TensorConv.toFloat (α := α) epsilon
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  for (id, p) in params.toList do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient array out of bounds during CUDA AdamW"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mScaled := Runtime.Autograd.Cuda.Buffer.scale st.m beta1F
      let m' := Runtime.Autograd.Cuda.Buffer.axpy mScaled gAny.buf oneMinusBeta1
      let g2 := Runtime.Autograd.Cuda.Buffer.mul gAny.buf gAny.buf
      let vScaled := Runtime.Autograd.Cuda.Buffer.scale st.v beta2F
      let v' := Runtime.Autograd.Cuda.Buffer.axpy vScaled g2 oneMinusBeta2
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let mHat := Runtime.Autograd.Cuda.Buffer.scale m' mHatScale
      let vHat := Runtime.Autograd.Cuda.Buffer.scale v' vHatScale
      let sqrtVHat := Runtime.Autograd.Cuda.Buffer.sqrt vHat
      let epsBuf := Runtime.Autograd.Cuda.Buffer.full n epsF
      let denom :=
        Runtime.Autograd.Cuda.Buffer.add sqrtVHat epsBuf
      let update := Runtime.Autograd.Cuda.Buffer.div mHat denom
      let decayedParam := Runtime.Autograd.Cuda.Buffer.axpy pBuf pBuf (-(lrF * wdF))
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy decayedParam update (-lrF) }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      releaseCudaBuffer mScaled
      releaseCudaBuffer g2
      releaseCudaBuffer vScaled
      releaseCudaBuffer mHat
      releaseCudaBuffer vHat
      releaseCudaBuffer sqrtVHat
      releaseCudaBuffer epsBuf
      releaseCudaBuffer denom
      releaseCudaBuffer update
      releaseCudaBuffer decayedParam
      state := state.insert id { m := m', v := v', t := t' }
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA AdamW"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

/--
Apply AdamW from a sparse CUDA gradient map.

The dense-array AdamW function remains available for callers that explicitly ask for all tape
gradients, but normal training should use this sparse map so activation gradients can be released as
soon as their contributions have been propagated.
-/
def adamWStepAllCudaMap {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α)
    [DecidableEq Shape]
    (stateRef : IO.Ref CudaAdamState)
    (lr weightDecay beta1 beta2 epsilon : α)
    (grads : CudaGradMap) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamWStepAllCudaMap called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let wdF ← CudaBridge.TensorConv.toFloat (α := α) weightDecay
  let beta1F ← CudaBridge.TensorConv.toFloat (α := α) beta1
  let beta2F ← CudaBridge.TensorConv.toFloat (α := α) beta2
  let epsF ← CudaBridge.TensorConv.toFloat (α := α) epsilon
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  for (id, p) in params.toList do
    let gAny ← match grads.get? id with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient map missing parameter during CUDA AdamW"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mScaled := Runtime.Autograd.Cuda.Buffer.scale st.m beta1F
      let m' := Runtime.Autograd.Cuda.Buffer.axpy mScaled gAny.buf oneMinusBeta1
      let g2 := Runtime.Autograd.Cuda.Buffer.mul gAny.buf gAny.buf
      let vScaled := Runtime.Autograd.Cuda.Buffer.scale st.v beta2F
      let v' := Runtime.Autograd.Cuda.Buffer.axpy vScaled g2 oneMinusBeta2
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let mHat := Runtime.Autograd.Cuda.Buffer.scale m' mHatScale
      let vHat := Runtime.Autograd.Cuda.Buffer.scale v' vHatScale
      let sqrtVHat := Runtime.Autograd.Cuda.Buffer.sqrt vHat
      let epsBuf := Runtime.Autograd.Cuda.Buffer.full n epsF
      let denom := Runtime.Autograd.Cuda.Buffer.add sqrtVHat epsBuf
      let update := Runtime.Autograd.Cuda.Buffer.div mHat denom
      let decayedParam := Runtime.Autograd.Cuda.Buffer.axpy pBuf pBuf (-(lrF * wdF))
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy decayedParam update (-lrF) }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      releaseCudaBuffer mScaled
      releaseCudaBuffer g2
      releaseCudaBuffer vScaled
      releaseCudaBuffer mHat
      releaseCudaBuffer vHat
      releaseCudaBuffer sqrtVHat
      releaseCudaBuffer epsBuf
      releaseCudaBuffer denom
      releaseCudaBuffer update
      releaseCudaBuffer decayedParam
      state := state.insert id { m := m', v := v', t := t' }
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA AdamW"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
