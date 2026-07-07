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
def singleton {α : Type} {shape : Shape}
    (value : Tensor.T α shape) : ParamTensors α [shape] :=
  tensorpack! value

/-- Unpack a one-tensor parameter pack. -/
def unpackSingleton {α : Type} {shape : Shape}
    (params : ParamTensors α [shape]) : Tensor.T α shape :=
  NN.API.tensorpack.unpackSingleton params

/-- Build a two-tensor parameter pack. -/
def pair {α : Type} {leftShape rightShape : Shape}
    (left : Tensor.T α leftShape) (right : Tensor.T α rightShape) :
    ParamTensors α [leftShape, rightShape] :=
  tensorpack! left, right

/-- Unpack a two-tensor parameter pack. -/
def unpackPair {α : Type} {leftShape rightShape : Shape}
    (params : ParamTensors α [leftShape, rightShape]) :
    Tensor.T α leftShape × Tensor.T α rightShape :=
  NN.API.tensorpack.unpackPair params

/-- Build a three-tensor parameter pack. -/
def triple {α : Type} {firstShape secondShape thirdShape : Shape}
    (first : Tensor.T α firstShape) (second : Tensor.T α secondShape)
    (third : Tensor.T α thirdShape) :
    ParamTensors α [firstShape, secondShape, thirdShape] :=
  tensorpack! first, second, third

/-- Unpack a three-tensor parameter pack. -/
def unpackTriple {α : Type} {firstShape secondShape thirdShape : Shape}
    (params : ParamTensors α [firstShape, secondShape, thirdShape]) :
    Tensor.T α firstShape × Tensor.T α secondShape × Tensor.T α thirdShape :=
  NN.API.tensorpack.unpackTriple params

/-- Build a four-tensor parameter pack. -/
def quad {α : Type} {firstShape secondShape thirdShape fourthShape : Shape}
    (first : Tensor.T α firstShape) (second : Tensor.T α secondShape)
    (third : Tensor.T α thirdShape) (fourth : Tensor.T α fourthShape) :
    ParamTensors α [firstShape, secondShape, thirdShape, fourthShape] :=
  tensorpack! first, second, third, fourth

/-- Unpack a four-tensor parameter pack. -/
def unpackQuad {α : Type} {firstShape secondShape thirdShape fourthShape : Shape}
    (params : ParamTensors α [firstShape, secondShape, thirdShape, fourthShape]) :
    Tensor.T α firstShape × Tensor.T α secondShape × Tensor.T α thirdShape × Tensor.T α fourthShape :=
  NN.API.tensorpack.unpackQuad params

/-- Build a seven-tensor parameter pack. -/
def septuple {α : Type}
    {firstShape secondShape thirdShape fourthShape fifthShape sixthShape seventhShape : Shape}
    (first : Tensor.T α firstShape) (second : Tensor.T α secondShape)
    (third : Tensor.T α thirdShape) (fourth : Tensor.T α fourthShape)
    (fifth : Tensor.T α fifthShape) (sixth : Tensor.T α sixthShape)
    (seventh : Tensor.T α seventhShape) :
    ParamTensors α
      [firstShape, secondShape, thirdShape, fourthShape, fifthShape, sixthShape, seventhShape] :=
  tensorpack! first, second, third, fourth, fifth, sixth, seventh

/-- Unpack a seven-tensor parameter pack. -/
def unpackSeptuple {α : Type}
    {firstShape secondShape thirdShape fourthShape fifthShape sixthShape seventhShape : Shape}
    (params : ParamTensors α
      [firstShape, secondShape, thirdShape, fourthShape, fifthShape, sixthShape, seventhShape]) :
    Tensor.T α firstShape × Tensor.T α secondShape × Tensor.T α thirdShape ×
      Tensor.T α fourthShape × Tensor.T α fifthShape × Tensor.T α sixthShape ×
      Tensor.T α seventhShape :=
  match params with
  | .cons first (.cons second (.cons third (.cons fourth (.cons fifth (.cons sixth (.cons seventh .nil)))))) =>
      (first, second, third, fourth, fifth, sixth, seventh)

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
