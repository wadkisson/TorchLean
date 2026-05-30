/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness

/-!
# Verified Forward Fragment: Public Names

Short public names for the compiler and its two main correctness theorems.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

/-- Compile a verified single-input forward program into the verifier IR. -/
abbrev compileForward1
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    NN.Verification.TorchLean.CompiledIR α :=
  compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params

/-- Graph structural safety for the concise compiler name. -/
theorem compileForward1_wellFormed
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    (compileForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params).graph.wellFormed = true :=
  Correctness.compileVerifiedForward1_wellFormed (α := α) (paramShapes := paramShapes) (inShape := inShape)
    (outShape := outShape) p params

/-- Main end-to-end compiler correctness using the short name. -/
theorem evalCompiledForward1_eq_evalForward1
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) :
    NN.Verification.TorchLean.evalCompiledForward1 (α := α) (inShape := inShape) (outShape := outShape)
        (c := compileForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
          params)
        x
      =
    NN.Verification.TorchLean.Proved.evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
      (outShape := outShape) p params x :=
  Correctness.evalCompiledForward1_compileVerifiedForward1_eq_evalForward1 (α := α)
    (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params x

/-- Convenience spelling for the same short correctness lemma. -/
theorem evalCompiledForward1_forward1_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) :
    NN.Verification.TorchLean.evalCompiledForward1 (α := α) (inShape := inShape) (outShape := outShape)
        (c := compileForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
          params)
        x
      =
    NN.Verification.TorchLean.Proved.evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
      (outShape := outShape) p params x :=
  evalCompiledForward1_eq_evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
    (outShape := outShape) p params x


end NN.Verification.TorchLean.Proved
