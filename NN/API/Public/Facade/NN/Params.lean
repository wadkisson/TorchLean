/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN.Basic

/-!
# TorchLean NN Parameter Packs

Public parameter-pack operations for checked model weights.
-/

@[expose] public section

namespace TorchLean

namespace nn

abbrev ParamTensors := TorchLean.ParamTensors

namespace ParamTensors

/-- Build a one-tensor parameter pack. -/
def of1 {α : Type} {s1 : Shape}
    (x1 : Tensor.T α s1) : ParamTensors α [s1] :=
  tensorpack! x1

/-- Unpack a one-tensor parameter pack. -/
def unpack1 {α : Type} {s1 : Shape}
    (xs : ParamTensors α [s1]) : Tensor.T α s1 :=
  NN.API.tensorpack.unpack1 xs

/-- Build a two-tensor parameter pack. -/
def of2 {α : Type} {s1 s2 : Shape}
    (x1 : Tensor.T α s1) (x2 : Tensor.T α s2) : ParamTensors α [s1, s2] :=
  tensorpack! x1, x2

/-- Unpack a two-tensor parameter pack. -/
def unpack2 {α : Type} {s1 s2 : Shape}
    (xs : ParamTensors α [s1, s2]) : Tensor.T α s1 × Tensor.T α s2 :=
  NN.API.tensorpack.unpack2 xs

/-- Build a three-tensor parameter pack. -/
def of3 {α : Type} {s1 s2 s3 : Shape}
    (x1 : Tensor.T α s1) (x2 : Tensor.T α s2) (x3 : Tensor.T α s3) :
    ParamTensors α [s1, s2, s3] :=
  tensorpack! x1, x2, x3

/-- Unpack a three-tensor parameter pack. -/
def unpack3 {α : Type} {s1 s2 s3 : Shape}
    (xs : ParamTensors α [s1, s2, s3]) :
    Tensor.T α s1 × Tensor.T α s2 × Tensor.T α s3 :=
  NN.API.tensorpack.unpack3 xs

/-- Build a four-tensor parameter pack. -/
def of4 {α : Type} {s1 s2 s3 s4 : Shape}
    (x1 : Tensor.T α s1) (x2 : Tensor.T α s2) (x3 : Tensor.T α s3) (x4 : Tensor.T α s4) :
    ParamTensors α [s1, s2, s3, s4] :=
  tensorpack! x1, x2, x3, x4

/-- Unpack a four-tensor parameter pack. -/
def unpack4 {α : Type} {s1 s2 s3 s4 : Shape}
    (xs : ParamTensors α [s1, s2, s3, s4]) :
    Tensor.T α s1 × Tensor.T α s2 × Tensor.T α s3 × Tensor.T α s4 :=
  NN.API.tensorpack.unpack4 xs

/-- Build a seven-tensor parameter pack. -/
def of7 {α : Type} {s1 s2 s3 s4 s5 s6 s7 : Shape}
    (x1 : Tensor.T α s1) (x2 : Tensor.T α s2) (x3 : Tensor.T α s3) (x4 : Tensor.T α s4)
    (x5 : Tensor.T α s5) (x6 : Tensor.T α s6) (x7 : Tensor.T α s7) :
    ParamTensors α [s1, s2, s3, s4, s5, s6, s7] :=
  tensorpack! x1, x2, x3, x4, x5, x6, x7

/-- Unpack a seven-tensor parameter pack. -/
def unpack7 {α : Type} {s1 s2 s3 s4 s5 s6 s7 : Shape}
    (xs : ParamTensors α [s1, s2, s3, s4, s5, s6, s7]) :
    Tensor.T α s1 × Tensor.T α s2 × Tensor.T α s3 × Tensor.T α s4 ×
      Tensor.T α s5 × Tensor.T α s6 × Tensor.T α s7 :=
  match xs with
  | .cons x1 (.cons x2 (.cons x3 (.cons x4 (.cons x5 (.cons x6 (.cons x7 .nil)))))) =>
      (x1, x2, x3, x4, x5, x6, x7)

end ParamTensors

/--
View a shape-indexed parameter list as the concrete parameter tensors for `model`.

Use this when two model fragments share the same parameter layout by definition.
-/
def paramTensorsOf {α : Type} {σ τ : Shape}
    (model : Sequential σ τ)
    (params : ParamTensors α (paramShapes model)) :
    ParamTensors α (paramShapes model) := by
  simpa using params

end nn

end TorchLean
