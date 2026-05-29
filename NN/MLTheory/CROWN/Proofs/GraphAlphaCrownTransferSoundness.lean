/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.AlphaBeta

/-!
# α-CROWN Graph Transfer Soundness

Soundness theorems for the concrete α-CROWN and α/β-CROWN graph transfer rules over `ℝ`.

The main results are:
- `AlphaCrownTransferSoundness.alphaCrown_transfer_sound`
- `AlphaCrownTransferSoundness.alphaBetaCrown_transfer_sound`

These theorems show that the executable checker steps satisfy the abstract `CrownTransferSound`
interface used by the generic graph certificate soundness theorem.
-/
