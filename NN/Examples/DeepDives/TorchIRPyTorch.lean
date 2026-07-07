/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean IR to PyTorch

Tutorial: TorchLean → IR (`NN.IR.Graph`) → emitted PyTorch code.

Run:
  `lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py`
Then:
  `python3 exported_model.py`
-/

@[expose] public section


namespace NN.Examples.DeepDives.TorchIRPyTorch

open TorchLean

/-! ## Architectures -/

def archLinear : nn.M (nn.Sequential (Shape.vec 2) (Shape.vec 1)) :=
  nn.Linear 2 1

def archMLP : nn.M (nn.Sequential (Shape.vec 2) (Shape.vec 1)) :=
  nn.Sequential![
    nn.Linear 2 3,
    nn.ReLU,
    nn.Linear 3 1
  ]

def archSumReduce : nn.M (nn.Sequential (Shape.vec 4) Shape.scalar) :=
  nn.Sum (s := Shape.vec 4)

def archAutoencoder : nn.M (nn.Sequential (Shape.vec 3) (Shape.vec 3)) :=
  nn.Sequential![
    nn.Linear 3 2,
    nn.Tanh,
    nn.Linear 2 3
  ]

def archCNN : nn.M (nn.Sequential (Shape.images 1 1 4 4) (shape![1, 3])) :=
  let featDim : Nat := Shape.size (Shape.CHW 2 2 2)
  nn.Sequential![
    nn.Conv2d (n := 1) (inC := 1) (inH := 4) (inW := 4)
      { outC := 2, kH := 3, kW := 3, stride := 1, padding := 0 },
    nn.ReLU,
    nn.FlattenBatch (n := 1) (s := Shape.CHW 2 2 2),
    nn.Linear featDim 3 (Shape.vec 1)
  ]

def archConvMLP :
    nn.M (nn.Sequential (Shape.images 1 1 3 3) (shape![1, 1])) :=
  -- Conv output is `Images 1 1 2 2`; flattening gives `Mat 1 4`.
  let featDim : Nat := Shape.size (Shape.CHW 1 2 2)
  nn.Sequential![
    nn.Conv2d (n := 1) (inC := 1) (inH := 3) (inW := 3)
      { outC := 1, kH := 2, kW := 2, stride := 1, padding := 0 },
    nn.ReLU,
    nn.FlattenBatch (n := 1) (s := Shape.CHW 1 2 2),
    nn.Linear featDim 3 (Shape.vec 1),
    nn.ReLU,
    nn.Linear 3 1 (Shape.vec 1)
  ]

def archMHA :
    nn.M (nn.Sequential (shape![1, 4, 8]) (shape![1, 4, 8])) :=
  nn.MultiheadAttentionLayer (batch := 1) (n := 4) (dModel := 8)
    { numHeads := 2, headDim := 4 }

def archMHAMask : Tensor.T Bool (Shape.mat 4 4) :=
  text.causalMask 4

def archMHAMasked :
    nn.M (nn.Sequential (shape![1, 4, 8]) (shape![1, 4, 8])) :=
  nn.MultiheadAttentionLayer (batch := 1) (n := 4) (dModel := 8)
    { numHeads := 2, headDim := 4 } (mask := some archMHAMask)

def archTransformer :
    nn.M (nn.Sequential (shape![1, 2, 2]) (shape![1, 2, 2])) :=
  nn.TransformerEncoderBlock (batch := 1) (n := 2) (dModel := 2)
    { numHeads := 1
    , headDim := 2
    , ffnHidden := 2 }

/-! ## CLI parsing -/

def usage : String :=
  String.intercalate "\n"
    [ "TorchLean → IR → PyTorch exporter"
    , ""
    , "Usage:"
    , "  lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp --seed 123 > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py"
    , ""
    , "Then: python3 exported_model.py"
    ]

/-! ## Export driver -/

def emitSeq {σ τ : Shape} (className : String) (model : nn.Sequential σ τ) : IO Unit := do
  let ps := nn.paramShapes model
  let prog : _root_.Runtime.Autograd.TorchLean.Program Float (ps ++ [σ]) τ :=
    nn.forwardProgram (model := model) (α := Float)
  let params := nn.initParams (m := model)
  let compiled ←
    match NN.Verification.TorchLean.compileForward
        (α := Float) (paramShapes := ps) (inShape := σ) (outShape := τ)
        (model := prog) (params := params) with
    | .error e => throw <| IO.userError e
    | .ok c => pure c

  let code ←
    match Export.IRPyTorch.emit
        (g := compiled.graph) (ps := compiled.ps) (inputId := compiled.inputId) (outputId :=
          compiled.outputId)
        (opts := { className := className }) with
    | .error e => throw <| IO.userError e
    | .ok s => pure s

  IO.println code

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  let help := args.contains "--help" || args.contains "-h"
  if help then
    IO.println usage
  else
    let (seed, args) ← CLI.orThrow "TorchIRPyTorch" <| CLI.takeSeed args 0
    let (arch, rest) ← CLI.orThrow "TorchIRPyTorch" <|
      CLI.takeFlagValueDefault args "arch" "mlp"
    CLI.requireNoArgs "TorchIRPyTorch" rest
    if arch == "linear" then
      emitSeq (className := "TorchLeanLinear") (nn.run seed archLinear)
    else if arch == "mlp" then
      emitSeq (className := "TorchLeanMLP") (nn.run seed archMLP)
    else if arch == "sum" then
      emitSeq (className := "TorchLeanSumReduce") (nn.run seed archSumReduce)
    else if arch == "autoencoder" then
      emitSeq (className := "TorchLeanAutoencoder") (nn.run seed archAutoencoder)
    else if arch == "cnn" then
      throw <| IO.userError
        "torch_ir_pytorch: --arch cnn is not in the supported exporter fragment yet (conv lowering uses scatter)"
    else if arch == "conv-mlp" then
      throw <| IO.userError
        "torch_ir_pytorch: --arch conv-mlp is not in the supported exporter fragment yet (conv lowering uses scatter)"
    else if arch == "mha" then
      emitSeq (className := "TorchLeanMHA") (nn.run seed archMHA)
    else if arch == "mha-mask" then
      emitSeq (className := "TorchLeanMHAMasked") (nn.run seed archMHAMasked)
    else if arch == "transformer" then
      emitSeq (className := "TorchLeanTransformerBlock") (nn.run seed archTransformer)
    else
      throw <| IO.userError
        (s!"unknown --arch {arch} (supported: linear | mlp | sum | autoencoder | " ++
          s!"mha | mha-mask | transformer)")

end NN.Examples.DeepDives.TorchIRPyTorch
