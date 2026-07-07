/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.Spec
public import NN.API.Macros
public import NN.API.Public.TensorPack
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils
public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Runtime.Autograd.TorchLean.Backend
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.Verification.TorchLean.Compile

/-!
# Pipeline (iii): All-in-Lean TwoStage refinement + IBP/CROWN check

This file corresponds to **Figure 7 (iii)** in the TorchLean paper (`arXiv:2602.22631`).

Everything runs *inside Lean*:
- Stage 1: sample training points in a box and train parameters (SGD) under exact `IEEE32Exec`.
- Stage 2: for each round, run a small PGD loop on the input `x` to find “counterexample-ish”
  points, then train on them (CEGIS flavor).
- Final: compile the same TorchLean loss program to the shared verifier IR and run in-repo IBP/CROWN
  bound propagation to check the loss on a small box around the origin.

Notes:
- This workflow uses the in-repo IBP/CROWN engine, so its bounds are meant to exercise TorchLean's
  own verifier path rather than reproduce every optimization used by external α/β-CROWN systems.
- The point is the *trust boundary*: the whole compute path, including float32 semantics, is inside
  Lean, and external tooling is not required to run the end-to-end pipeline.

Run:
`lake exe verify -- twostage-torchlean-cegis-van`
-/

@[expose] public section


open Spec
open Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIII.AllInLean

open _root_.TorchLean.Floats.IEEE754
open Runtime
open Runtime.Autograd
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

open NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
open NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils

abbrev α : Type := IEEE32Exec

abbrev xDim : Nat := Core.xDim
abbrev uDim : Nat := Core.uDim

abbrev xShape : Shape := Core.xShape
abbrev uShape : Shape := Core.uShape

abbrev paramShapes (width : Nat) : List Shape :=
  Core.paramShapes width

/-- Local alias for `ExecUtils.nat` (coercion `Nat → IEEE32Exec`). -/
abbrev nat (k : Nat) : α := ExecUtils.nat k

/-- Learning rate for the stage-1 and stage-2 SGD loops. -/
def lr : α := ExecUtils.defaultLr

/-- PGD step size when searching for counterexample-ish inputs. -/
def pgdStepSize : α := ExecUtils.defaultPgdStepSize

/-- Radius of the training box `[-rad, rad]^2` (also used for clamping PGD iterates). -/
def rad : α := ExecUtils.defaultRad

/-- Half-width of the small box around the origin used for the final IBP/CROWN post-check. -/
def epsCheck : α := ExecUtils.defaultEpsCheck

def lossProg (width : Nat) :
    ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
      TorchLean.Program β (paramShapes width ++ [xShape]) Shape.scalar :=
  Core.lossProgram width

def initParamsF (width : Nat) : NN.API.TensorPack Float (paramShapes width) :=
  let wC : Tensor Float (.dim uDim (.dim xDim .scalar)) :=
    _root_.Runtime.Autograd.Torch.Init.xavierW uDim xDim (seed := 0)
  let bC : Tensor Float (.dim uDim .scalar) :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := .dim uDim .scalar) (sch := .zeros) (seed := 1)
  let w1 : Tensor Float (.dim width (.dim xDim .scalar)) :=
    _root_.Runtime.Autograd.Torch.Init.xavierW width xDim (seed := 2)
  let b1 : Tensor Float (.dim width .scalar) :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := .dim width .scalar) (sch := .zeros) (seed := 3)
  let w2 : Tensor Float (.dim 1 (.dim width .scalar)) :=
    _root_.Runtime.Autograd.Torch.Init.xavierW 1 width (seed := 4)
  let b2 : Tensor Float (.dim 1 .scalar) :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := .dim 1 .scalar) (sch := .zeros) (seed := 5)
  tensorpack! wC, bC, w1, b1, w2, b2

def moduleDef (width : Nat) : TorchLean.Module.ScalarModuleDef (paramShapes width) [xShape]
  :=
  { initParams := initParamsF width
    loss := Core.lossProgram width }

abbrev lossΓ (width : Nat) : List Shape := paramShapes width ++ [xShape]

/--
One PGD step on the input `x` using a *compiled* loss graph.

We differentiate only w.r.t. `x` (parameters are treated as constants during the PGD inner loop),
then clamp to the training box `[-rad, rad]^2`.
-/
def pgdStepCompiled
    (width : Nat)
    (cLoss : _root_.Runtime.Autograd.Torch.CompiledScalar α (lossΓ width))
    (params : NN.API.TensorPack α (paramShapes width))
    (x : Tensor α xShape) : Tensor α xShape :=
  let args : NN.API.TensorPack α (lossΓ width) :=
    _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := paramShapes width) (ss₂ := [xShape]) params (.cons x .nil)
  let gAll : NN.API.TensorPack α (lossΓ width) :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.backward (α := α) (Γ := lossΓ width) cLoss args
  let gx : NN.API.TensorPack α [xShape] :=
    (_root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.splitAppend (α := α)
      (ss₁ := paramShapes width) (ss₂ := [xShape]) gAll).2
  let .cons g .nil := gx
  let x' := Tensor.addSpec x (Tensor.scaleSpec g pgdStepSize)
  clampStateVector (-rad) rad x'

/--
Final post-check: compile the TorchLean loss to the shared verifier IR, then run IBP and CROWN
over a small box around the origin.
-/
def checkBox (width : Nat) (params : NN.API.TensorPack α (paramShapes width)) (eps : α :=
  epsCheck) : IO Unit := do
  IO.println "Stage 2 check: IBP + CROWN on the scalar loss over a small box"
  let compiled ←
    match NN.Verification.TorchLean.compileForward
          (α := α) (paramShapes := paramShapes width) (inShape := xShape) (outShape := Shape.scalar)
          (lossProg width (β := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

  let x0 : Tensor α xShape := Spec.zeros (α := α) xShape
  let xB : FlatBox α := NN.MLTheory.CROWN.FlatBox.lInfBall (α := α) x0 eps
  let ps : ParamStore α := compiled.seedInputBox xB

  let ibp := runIBP (α := α) compiled.graph ps
  let outB ← compiled.outputBoxOrThrow ibp
  IO.println s!"[IBP] scalar loss box dim={outB.dim}"

  match NN.MLTheory.CROWN.Graph.outputBoxCROWN? compiled.graph ps xB
      compiled.inputId compiled.outputId xDim with
  | .ok outB =>
      IO.println s!"[CROWN] loss lo = {pretty outB.lo}"
      IO.println s!"[CROWN] loss hi = {pretty outB.hi}"
  | .error msg =>
      IO.println s!"[CROWN] {msg}"

/-- Main entrypoint for the all-in-Lean pipeline (width is a parameter; CLI default is
  `defaultWidth`). -/
def run (width : Nat) (args : List String) : IO Unit := do
  let longRun : Bool := args.any (· = "--long")
  let twoRun : Bool := args.any (· = "--two")
  let paperRun : Bool := args.any (· = "--paper")
  let stage1Steps : Nat := (if longRun then 20 else if paperRun then 10 else if twoRun then 2 else
    1)
  let stage2Rounds : Nat := (if longRun then 10 else if paperRun then 10 else 1)
  let pgdSteps : Nat := (if longRun then 20 else if paperRun then 10 else 1)

  IO.println "== TwoStage TorchLean CEGIS workflow (IEEE32Exec) =="
  IO.println
    s!"width={width} stage1Steps={stage1Steps} stage2Rounds={stage2Rounds} pgdSteps={pgdSteps}"

  let mod ← TorchLean.Module.ScalarModuleDef.instantiate (α := α) (moduleDef width)
    IEEE32Exec.ofFloat .compiled
  let tr := mod.trainer
  let cLoss ← TorchLean.Autodiff.compileLoss
    (α := α) (paramShapes := paramShapes width) (inputShapes := [xShape]) (lossProg width)

  -- Stage 1: initialization pass on random x in [-rad, rad]^2
  let mut seed : UInt64 := 1
  for i in [0:stage1Steps] do
    let (seed', x) := sampleStateVector seed rad
    seed := seed'
    let xs : NN.API.TensorPack α [xShape] := tensorpack! x
    let currentLoss := _root_.Runtime.Autograd.Torch.scalarOf (←
      _root_.Runtime.Autograd.Torch.ScalarTrainer.forwardT tr xs)
    _root_.Runtime.Autograd.Torch.ScalarTrainer.stepT tr lr xs
    if i % 5 = 0 then
      IO.println s!"[stage1] step {i}: loss={currentLoss}"

  -- Stage 2: PGD on x to find violations, then train on them
  for round in [0:stage2Rounds] do
    let (seed', x0) := sampleStateVector seed rad
    seed := seed'
    let params ← tr.getParams
    let mut x := x0
    for _k in [0:pgdSteps] do
      x := pgdStepCompiled width cLoss params x
    let xs : NN.API.TensorPack α [xShape] := tensorpack! x
    let lossFound := _root_.Runtime.Autograd.Torch.scalarOf (←
      _root_.Runtime.Autograd.Torch.ScalarTrainer.forwardT tr xs)
    _root_.Runtime.Autograd.Torch.ScalarTrainer.stepT tr lr xs
    IO.println s!"[stage2] round {round}: loss={lossFound}"

  let params ← tr.getParams
  checkBox width params (eps := epsCheck)

/-- Default hidden width used by the Pipeline III all-in-Lean workflow. -/
def defaultWidth : Nat := 100

/-- CLI entrypoint (all-in-Lean pipeline, default width). -/
def main (args : List String) : IO Unit :=
  run (width := defaultWidth) args

end NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIII.AllInLean
