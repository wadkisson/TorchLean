/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved

/-!
# Verified forward compiler bridge

This module is the public naming layer for the verified forward compiler bridge.

The implementation proof lives in `NN.Verification.TorchLean.Proved`; this file exposes the names we
want users and downstream modules to reach for: `compileForward`, `compileForward_wellFormed`, and
`compileForward_correct`.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Verified

open _root_.Spec
open _root_.Spec.Tensor
open NN.Verification.TorchLean

open NN.Verification.TorchLean.Proved

/-- Compile a verified single-input forward program into the verifier IR. -/
abbrev compileForward
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Proved.Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    NN.Verification.TorchLean.CompiledIR α :=
  NN.Verification.TorchLean.Proved.compileForward (α := α) (paramShapes := paramShapes)
    (inShape := inShape) (outShape := outShape) p params

/-- Compile-time structural safety of the compiled verifier graph. -/
theorem compileForward_wellFormed
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Proved.Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    (compileForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
      params).graph.wellFormed = true :=
  NN.Verification.TorchLean.Proved.compileForward_wellFormed (α := α)
    (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params

/-- Short, explicit alias for the main end-to-end compiler correctness theorem. -/
theorem compileForward_correct
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Proved.Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) :
    NN.Verification.TorchLean.runForwardIR (α := α) (inShape := inShape) (outShape := outShape)
        (c := NN.Verification.TorchLean.Proved.compileForward (α := α) (paramShapes := paramShapes)
          (inShape := inShape) (outShape := outShape) p params)
        x
      =
    NN.Verification.TorchLean.Proved.evalForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
      (outShape := outShape) p params x :=
  NN.Verification.TorchLean.Proved.runForwardIR_eq_evalForward (α := α)
    (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params x

end NN.Verification.TorchLean.Verified
