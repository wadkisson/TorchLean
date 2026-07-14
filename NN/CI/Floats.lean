/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats
public import NN.Floats.Arb
public import NN.Floats.Arb.Oracle
public import NN.Floats.Calc
public import NN.Floats.Calc.Bracket
public import NN.Floats.Calc.Operations
public import NN.Floats.Calc.Plus
public import NN.Floats.Calc.Round
public import NN.Floats.FP32
public import NN.Floats.FP32.Core
public import NN.Floats.FP32.Error
public import NN.Floats.FP32.Notation
public import NN.Floats.FP32.RuntimeApprox
public import NN.Floats.Float32
public import NN.Floats.IEEEExec
public import NN.Floats.IEEEExec.BridgeERealTotal
public import NN.Floats.IEEEExec.BridgeFP32
public import NN.Floats.IEEEExec.BridgeFP32Expr
public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.IEEEExec.BridgeInitFloat32
public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.DivDirectedRoundingSoundness
public import NN.Floats.IEEEExec.ERealSemantics
public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.IEEEExec.MinMaxERealSoundness
public import NN.Floats.IEEEExec.MkBitsToReal
public import NN.Floats.IEEEExec.NatLemmas
public import NN.Floats.IEEEExec.Notation
public import NN.Floats.IEEEExec.Reductions
public import NN.Floats.IEEEExec.RoundDyadicToIEEE32Bounds
public import NN.Floats.IEEEExec.RoundQuotEvenBounds
public import NN.Floats.IEEEExec.RoundShiftRightEven
public import NN.Floats.IEEEExec.SpecialRules
public import NN.Floats.IEEEExec.TranscendentalRules
public import NN.Floats.Interval
public import NN.Floats.Interval.FP32
public import NN.Floats.Interval.IEEEExec32
public import NN.Floats.Interval.IEEEExec32AddSoundness
public import NN.Floats.Interval.IEEEExec32ArbTrans
public import NN.Floats.Interval.IEEEExec32DivSoundness
public import NN.Floats.Interval.IEEEExec32MinMaxSoundness
public import NN.Floats.Interval.IEEEExec32MulSoundness
public import NN.Floats.Interval.IEEEExec32NoNaN
public import NN.Floats.Interval.Quantized
public import NN.Floats.Interval.RealBounds
public import NN.Floats.Interval.Rounders
public import NN.Floats.NeuralFloat
public import NN.Floats.NeuralFloat.Scalar.Conversion
public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.Error.Bounds
public import NN.Floats.NeuralFloat.Format.Formats
public import NN.Floats.NeuralFloat.Metadata
public import NN.Floats.NeuralFloat.Scalar.NF
public import NN.Floats.NeuralFloat.Scalar.NNOps
public import NN.Floats.NeuralFloat.Rounding.Core


/-!
# Floats CI Suite

Focused CI import suite. `NN.CI.All` combines every suite for exhaustive repository validation.

Local usage:

```bash
lake build NN.CI.All
```
-/

@[expose] public section
