/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN
public import NN.Runtime.External.Process
public import NN.Spec.Core.TensorOps
public import NN.Runtime.Autograd.TorchLean.Norm
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# TorchLeanOpsCheck

Runtime checks for TorchLean operator wrappers over the float runtime.

The file is intentionally fixture-driven: each helper builds one small tensor example, runs it
through the relevant backend boundary, and compares the result against either another TorchLean
backend, a closed form, or PyTorch when it is available.
-/

@[expose] public section

open Lean
open Spec
open Tensor
open NN.API
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanOpsCheck

/-! ## Shared fixtures -/

/-- BatchNorm parity fixture: batch size. -/
abbrev bnN : Nat := 2
/-- BatchNorm parity fixture: channel count. -/
abbrev bnC : Nat := 2
/-- BatchNorm parity fixture: image height. -/
abbrev bnH : Nat := 2
/-- BatchNorm parity fixture: image width. -/
abbrev bnW : Nat := 2
/-- NCHW shape used by the BatchNorm runtime and PyTorch parity checks. -/
abbrev bnShape : Shape := .dim bnN (.dim bnC (.dim bnH (.dim bnW .scalar)))

/-- Scratch directory for small Python parity scripts emitted by this test module. -/
def workDir : System.FilePath :=
  Runtime.External.Process.artifactWorkDir "ops_check"

/-- Path for the generated BatchNorm parity script. -/
def batchNormParityScriptPath : System.FilePath :=
  workDir / "batchnorm_parity.py"

/-- Assert that an IO action fails. Used for runtime boundary checks where rejection is success. -/
def expectRuntimeRejection {α : Type} (msg : String) (act : IO α) : IO Unit := do
  let mut rejected := false
  try
    let _ ← act
  catch _ =>
    rejected := true
  unless rejected do
    throw <| IO.userError msg

/-- Flatten a typed Nat vector into an array for simple test comparison. -/
def natTensorToArray {n : Nat} (t : Tensor Nat (.dim n .scalar)) : Array Nat := Id.run do
  let mut out := #[]
  match t with
  | Tensor.dim f =>
      for i in List.finRange n do
        match f i with
        | Tensor.scalar x => out := out.push x
  out

/--
Run the eager token-id conversion on a fixed three-element float vector.

The helper is intentionally small: the test below wants to isolate the adapter boundary, not the
full GPT loss.
-/
def readTokenIdsFromFloatVector (xs : Tensor Float (.dim 3 .scalar)) :
    IO (Tensor Nat (.dim 3 .scalar)) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let xRef ← Runtime.Autograd.Torch.Internal.EagerSession.input
    (α := Float) sess (sh := .dim 3 .scalar) xs
  Runtime.Autograd.Torch.Internal.EagerSession.tokenIdsFromFloatVec
    (α := Float) sess (k := 3) xRef

/--
Regression check for float-encoded token ids.

Good integer-valued floats must round-trip to Nat ids, while fractional and negative values must be
rejected. This protects the causal-LM path from silently turning bad labels into different labels.
-/
def checkTokenIdFloatVectorConversion : IO Unit := do
  let good ← readTokenIdsFromFloatVector (tensor! [0.0, 2.0, 5.0])
  let got := natTensorToArray good
  if got != #[0, 2, 5] then
    throw <| IO.userError s!"token id conversion: got {got}, expected #[0, 2, 5]"
  expectRuntimeRejection "token id conversion accepted a fractional id" <|
    readTokenIdsFromFloatVector (tensor! [0.0, 1.5, 2.0])
  expectRuntimeRejection "token id conversion accepted a negative id" <|
    readTokenIdsFromFloatVector (tensor! [0.0, -1.0, 2.0])

/-! ## Eager/compiled operator parity -/

/-- Evaluate a fixed 2x3 by 3x2 matrix product through the selected public backend. -/
def evalMatmulFixture (backend : TorchLean.Backend) :
    IO (Tensor Float (.dim 2 (.dim 2 .scalar))) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let a : Tensor Float (.dim 2 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (i.val + 2 * j.val + 1))))
  let b : Tensor Float (.dim 3 (.dim 2 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (3 * i.val + j.val + 1))))
  let aR ← TorchLean.Session.const sess (sh := .dim 2 (.dim 3 .scalar)) a
  let bR ← TorchLean.Session.const sess (sh := .dim 3 (.dim 2 .scalar)) b
  let cR ← TorchLean.Session.matmul sess (m := 2) (n := 3) (p := 2) aR bR
  TorchLean.Session.getValue sess (sh := .dim 2 (.dim 2 .scalar)) cR

/-- Evaluate a fixed length-2 plus length-3 vector concatenation through the selected backend. -/
def evalConcatFixture (backend : TorchLean.Backend) : IO (Tensor Float (.dim 5 .scalar)) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let a : Tensor Float (.dim 2 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val +
    1)))
  let b : Tensor Float (.dim 3 .scalar) := Tensor.dim (fun i => Tensor.scalar (10.0 + Float.ofNat
    i.val))
  let aR ← TorchLean.Session.const sess (sh := .dim 2 .scalar) a
  let bR ← TorchLean.Session.const sess (sh := .dim 3 .scalar) b
  let cR ← TorchLean.Session.concatVectors sess (n := 2) (m := 3) aR bR
  TorchLean.Session.getValue sess (sh := .dim 5 .scalar) cR

/-- Evaluate a fixed 2D max-pooling example through the selected public backend. -/
def evalMaxPool2dFixture (backend : TorchLean.Backend) :
    IO (Tensor Float (.dim 1 (.dim 2 (.dim 2 .scalar)))) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let x : Tensor Float (.dim 1 (.dim 4 (.dim 4 .scalar))) :=
    Tensor.dim (fun _c =>
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          Tensor.scalar (Float.ofNat (i.val * 10 + j.val)))))
  let xR ← TorchLean.Session.const sess (sh := .dim 1 (.dim 4 (.dim 4 .scalar))) x
  let yR ← TorchLean.Session.maxPool2d sess (kH := 2) (kW := 2) (inH := 4) (inW := 4) (inC := 1)
    (stride := 2)
    (h1 := by decide) (h2 := by decide) xR
  TorchLean.Session.getValue sess (sh := .dim 1 (.dim 2 (.dim 2 .scalar))) yR

/-- Evaluate a fixed 2D average-pooling example through the selected public backend. -/
def evalAvgPool2dFixture (backend : TorchLean.Backend) :
    IO (Tensor Float (.dim 1 (.dim 2 (.dim 2 .scalar)))) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let x : Tensor Float (.dim 1 (.dim 4 (.dim 4 .scalar))) :=
    Tensor.dim (fun _c =>
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          Tensor.scalar (Float.ofNat (i.val * 10 + j.val)))))
  let xR ← TorchLean.Session.const sess (sh := .dim 1 (.dim 4 (.dim 4 .scalar))) x
  let yR ← TorchLean.Session.avgPool2d sess (kH := 2) (kW := 2) (inH := 4) (inW := 4) (inC := 1)
    (stride := 2)
    (by decide) (by decide) xR
  TorchLean.Session.getValue sess (sh := .dim 1 (.dim 2 (.dim 2 .scalar))) yR

/-! ## BatchNorm fixture and PyTorch parity -/

/-- NCHW input with different signs per channel, chosen so mean/variance are easy to inspect. -/
def bnInput : Tensor Float bnShape :=
  Tensor.dim (fun n =>
    Tensor.dim (fun c =>
      Tensor.dim (fun h =>
        Tensor.dim (fun w =>
          let base := Float.ofNat (n.val * 8 + c.val * 4 + h.val * 2 + w.val + 1)
          Tensor.scalar (if c.val = 0 then base else -base)))))

/-- BatchNorm scale parameter for the two channels. -/
def bnGamma : Tensor Float (.dim bnC .scalar) :=
  tensor! [1.0, 0.5]

/-- BatchNorm shift parameter for the two channels. -/
def bnBeta : Tensor Float (.dim bnC .scalar) :=
  tensor! [0.0, 0.1]

/-- Running mean used by the eval-mode BatchNorm fixture. -/
def bnMean : Tensor Float (.dim bnC .scalar) :=
  tensor! [2.0, -3.0]

/-- Running variance used by the eval-mode BatchNorm fixture. -/
def bnVar : Tensor Float (.dim bnC .scalar) :=
  tensor! [4.0, 9.0]

/--
Run training-mode BatchNorm and return the output together with the computed channel statistics.

This checks the TorchLean runtime wrapper directly, not only the exported graph path.
-/
def evalBatchNormNchwTrain :
    IO (Tensor Float bnShape × Tensor Float (.dim bnC .scalar) × Tensor Float (.dim bnC .scalar)) :=
    do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float
      (Tensor Float bnShape × Tensor Float (.dim bnC .scalar) × Tensor Float (.dim bnC .scalar)) :=
    do
      let xR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := bnShape) bnInput
      let gR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := .dim bnC .scalar)
        bnGamma
      let bR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := .dim bnC .scalar)
        bnBeta
      let (yR, meanR, varR) ← Runtime.Autograd.TorchLean.Norm.batchNorm2dNchwTrainStats
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
        (n := bnN) (c := bnC) (h := bnH) (w := bnW)
        (by decide) (by decide) (by decide) (by decide) xR gR bR
      let sess ← read
      let y ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := bnShape) sess yR
      let mean ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := .dim bnC .scalar) sess meanR
      let var ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := .dim bnC .scalar) sess varR
      pure (y, mean, var)
  action sess

/-- Run eval-mode BatchNorm using fixed running statistics. -/
def evalBatchNormNchwEval :
    IO (Tensor Float bnShape) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float bnShape) := do
    let xR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := bnShape) bnInput
    let gR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := .dim bnC .scalar)
      bnGamma
    let bR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := .dim bnC .scalar)
      bnBeta
    let mR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := .dim bnC .scalar)
      bnMean
    let vR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := .dim bnC .scalar)
      bnVar
    let yR ← Runtime.Autograd.TorchLean.Norm.batchNorm2dNchwEval
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (n := bnN) (c := bnC) (h := bnH) (w := bnW)
      (by decide) (by decide) (by decide) (by decide) xR gR bR mR vR
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := bnShape) sess yR
  action sess

/-- Closed-form BatchNorm expression when the mean and variance come from the current batch. -/
def expectedBatchNormFromBatchStats (x gamma beta mean var : Float) : Float :=
  ((x - mean) / Float.sqrt (var + Numbers.epsilon)) * gamma + beta

/-- Closed-form BatchNorm expression when the mean and variance are fixed running statistics. -/
def expectedBatchNormFromRunningStats (x gamma beta mean var : Float) : Float :=
  ((x - mean) / Float.sqrt (var + Numbers.epsilon)) * gamma + beta

/-- Flatten an NCHW tensor in PyTorch's row-major order for JSON parity comparisons. -/
def flattenNchwRowMajor {n c h w : Nat}
    (t : Tensor Float (.dim n (.dim c (.dim h (.dim w .scalar))))) : Array Float := Id.run do
  let mut out := #[]
  for ni in List.finRange n do
    for ci in List.finRange c do
      for hi in List.finRange h do
        for wi in List.finRange w do
          out := out.push (nchwVal t ni ci hi wi)
  out

/-- Small Python program used to compare TorchLean BatchNorm against PyTorch. -/
def batchNormParityScript : String :=
  String.intercalate "\n"
    [ "import json"
    , "import torch"
    , "import torch.nn.functional as F"
    , ""
    , "x = torch.tensor(["
    , "    [[[1., 2.], [3., 4.]], [[-5., -6.], [-7., -8.]]],"
    , "    [[[9., 10.], [11., 12.]], [[-13., -14.], [-15., -16.]]],"
    , "], dtype=torch.float32)"
    , "gamma = torch.tensor([1.0, 0.5], dtype=torch.float32)"
    , "beta = torch.tensor([0.0, 0.1], dtype=torch.float32)"
    , "running_mean = torch.tensor([2.0, -3.0], dtype=torch.float32)"
    , "running_var = torch.tensor([4.0, 9.0], dtype=torch.float32)"
    , "eps = 1e-5"
    , "mean = x.mean(dim=(0, 2, 3))"
    , "var = x.var(dim=(0, 2, 3), unbiased=False)"
    , "train = F.batch_norm(x, None, None, gamma, beta, training=True, eps=eps)"
    , "eval = F.batch_norm(x, running_mean, running_var, gamma, beta, training=False, eps=eps)"
    , "print(json.dumps({"
    , "    'mean': mean.flatten().tolist(),"
    , "    'var': var.flatten().tolist(),"
    , "    'train': train.flatten().tolist(),"
    , "    'eval': eval.flatten().tolist(),"
    , "}))"
    ]

/--
Compare TorchLean BatchNorm output and statistics against PyTorch, when PyTorch is installed.

The fallback skip is deliberate: the Lean-side closed-form checks still run on machines without a
Python/PyTorch environment.
-/
def checkBatchNormNchwAgainstPyTorch
    (trainY evalY : Tensor Float bnShape)
    (mean var : Tensor Float (.dim bnC .scalar)) : IO Unit := do
  if !(← pythonHasTorch) then
    IO.println "torchlean_ops_check: PyTorch BatchNorm parity skipped (`torch` not installed)"
    return ()
  IO.FS.createDirAll workDir
  IO.FS.writeFile batchNormParityScriptPath batchNormParityScript
  let out ← Runtime.External.Process.runStdoutChecked
    (ctx := "torchlean_ops_check: batchnorm pytorch parity")
    (cmd := "python3")
    (args := #[batchNormParityScriptPath.toString])
    (cwd := some ".")
  let pyJson ←
    match Json.parse out with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"torchlean_ops_check: bad BatchNorm parity JSON: {e}\n{out}")
  let readField key := do
    match jsonFloatArrayField pyJson key with
    | .ok xs => pure xs
    | .error e => throw (IO.userError s!"torchlean_ops_check: {e}")
  assertArrayApprox "batchnorm_nchw pytorch mean"
    #[vecVal mean ⟨0, by decide⟩, vecVal mean ⟨1, by decide⟩] (← readField "mean")
  assertArrayApprox "batchnorm_nchw pytorch var"
    #[vecVal var ⟨0, by decide⟩, vecVal var ⟨1, by decide⟩] (← readField "var")
  assertArrayApprox "batchnorm_nchw pytorch train" (flattenNchwRowMajor trainY) (← readField "train")
  assertArrayApprox "batchnorm_nchw pytorch eval" (flattenNchwRowMajor evalY) (← readField "eval")

/--
Run the full BatchNorm check: closed-form expectations first, then optional PyTorch parity.
-/
def checkBatchNormNchw : IO Unit := do
  let (trainY, mean, var) ← evalBatchNormNchwTrain
  let evalY ← evalBatchNormNchwEval

  assertApprox "batchnorm_nchw mean[0] expected" (vecVal mean ⟨0, by decide⟩) 6.5
  assertApprox "batchnorm_nchw mean[1] expected" (vecVal mean ⟨1, by decide⟩) (-10.5)
  assertApprox "batchnorm_nchw var[0] expected" (vecVal var ⟨0, by decide⟩) 17.25
  assertApprox "batchnorm_nchw var[1] expected" (vecVal var ⟨1, by decide⟩) 17.25

  for n in List.finRange bnN do
    for c in List.finRange bnC do
      for h in List.finRange bnH do
        for w in List.finRange bnW do
          let x := nchwVal bnInput n c h w
          let gamma := vecVal bnGamma c
          let beta := vecVal bnBeta c
          let trainExpected :=
            expectedBatchNormFromBatchStats x gamma beta (vecVal mean c) (vecVal var c)
          let evalExpected :=
            expectedBatchNormFromRunningStats x gamma beta (vecVal bnMean c) (vecVal bnVar c)
          assertApprox s!"batchnorm_nchw train[{n.val},{c.val},{h.val},{w.val}] expected"
            (nchwVal trainY n c h w) trainExpected 1e-5
          assertApprox s!"batchnorm_nchw eval[{n.val},{c.val},{h.val},{w.val}] expected"
            (nchwVal evalY n c h w) evalExpected 1e-5

  checkBatchNormNchwAgainstPyTorch trainY evalY mean var

/-- Entrypoint called by the curated float runtime suite. -/
def run : IO Unit := do
  IO.println "torchlean_ops_check: begin"
  checkTokenIdFloatVectorConversion

  let mmE ← evalMatmulFixture .eager
  let mmC ← evalMatmulFixture .compiled
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      assertApprox s!"matmul[{i.val},{j.val}] eager/compiled" (matVal mmE i j) (matVal mmC i j) 1e-5

  let cvE ← evalConcatFixture .eager
  let cvC ← evalConcatFixture .compiled
  for i in List.finRange 5 do
    assertApprox s!"concat[{i.val}] eager/compiled" (vecVal cvE i) (vecVal cvC i) 1e-5

  let mpE ← evalMaxPool2dFixture .eager
  let mpC ← evalMaxPool2dFixture .compiled
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"max_pool2d[{hi.val},{wi.val}] eager/compiled"
        (chwVal mpE ⟨0, by decide⟩ hi wi)
        (chwVal mpC ⟨0, by decide⟩ hi wi)
        1e-5

  let apE ← evalAvgPool2dFixture .eager
  let apC ← evalAvgPool2dFixture .compiled
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"avg_pool2d[{hi.val},{wi.val}] eager/compiled"
        (chwVal apE ⟨0, by decide⟩ hi wi)
        (chwVal apC ⟨0, by decide⟩ hi wi)
        1e-5

  checkBatchNormNchw

  IO.println "torchlean_ops_check: ok"

end TorchLeanOpsCheck
end Floats
end Tests
