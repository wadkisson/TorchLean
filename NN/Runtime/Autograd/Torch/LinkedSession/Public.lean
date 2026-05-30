/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.LinkedSession.Autograd

/-!
# Proof-Linked Session: Public Names
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

/-! ## Public re-exports (stable names for docs) -/

-- The proof-linked session lives under `Internal` to keep the surface area small, but we expose
-- a stable public name layer for the blueprint and for downstream users who want the proved hook.

/-- Public alias for the proof-linked session state (internal definition re-export). -/
abbrev SessionIRState (α : Type) : Type := Internal.SessionIRState α

namespace SessionIRState

/-- Empty `SessionIRState` (no parameters/graph recorded yet). -/
abbrev empty {α : Type} : SessionIRState α := Internal.SessionIRState.empty (α := α)

end SessionIRState

/-- Public alias for the proof-linked session object (internal definition re-export). -/
abbrev SessionIR (α : Type) : Type := Internal.SessionIR α

namespace SessionIR

/-- Create a new proof-linked session (records a graph + supports proved backprop hook). -/
abbrev new {α : Type} (opts : Options := {}) : IO (SessionIR α) :=
  Internal.SessionIR.new (α := α) opts

/--
Compute dense gradients for all tracked refs w.r.t. an output tensor and a seed.

This mirrors the "backward with custom seed" pattern in tensor AD systems.
-/
abbrev backwardDenseAll {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
    {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
    IO (Array (Runtime.AnyTensor α)) :=
  Internal.SessionIR.backwardDenseAll (α := α) (s := s) out seed

/-- Dense gradients for all tracked refs w.r.t. a scalar loss (seed is implicitly `1`). -/
abbrev backwardScalarDenseAll {α : Type} (s : SessionIR α) [Add α] [Zero α] [One α] [DecidableEq
  Shape]
    (loss : TensorRef α Shape.scalar) : IO (Array (Runtime.AnyTensor α)) :=
  Internal.SessionIR.backwardScalarDenseAll (α := α) (s := s) loss

/-- Extract the gradient tensor for a specific ref from a dense gradient array. -/
abbrev grad {α : Type} {sh : Shape} [DecidableEq Shape]
    (grads : Array (Runtime.AnyTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) :=
  Internal.SessionIR.grad (α := α) (grads := grads) x

end SessionIR

/--
Public proof hook: the runtime reverse-mode loop on the compiled tape equals proved IR backprop.

This is a re-export of the internal theorem so downstream users can cite a stable name.
-/
theorem backwardDenseFrom_compileAuxData_eq_backpropAllCtx
    {α : Type} [DecidableEq Shape] [CommSemiring α]
    (st : SessionIRState α) (seed : _root_.Proofs.Autograd.Algebra.TList α (st.Γ ++ st.ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (Proofs.Autograd.Algebra.Graph.compileAuxData (α := α) (Δ := Array Nat) (Γ := st.Γ)
          (ss := st.ss) st.g st.x st.nat).1)
        (grads0 := _root_.Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := st.Γ ++ st.ss)
          seed)
      =
      .ok
        (_root_.Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := st.Γ ++ st.ss)
          (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Array Nat) (Γ :=
            st.Γ) (ss := st.ss) st.g st.x st.nat seed)) := by
  simpa [SessionIRState] using
    (Internal.SessionIR.backwardDenseFrom_compileAuxData_eq_backpropAllCtx (α := α) (st := st) seed)

end Torch
end Autograd
end Runtime
