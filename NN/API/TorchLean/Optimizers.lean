/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime

/-!
# Optimizer Convenience Constructors

This module provides a compact PyTorch-shaped optimizer surface for the TorchLean trainer API.
The ordinary trainer configuration exposes self-contained optimizers such as SGD, AdamW, and
Adadelta. Runtime-level extension points such as Muon and GaLore-style projected updates live in
the public runtime facade, where their backend/projection arguments are explicit.

## PyTorch Mapping

The names and default hyperparameters mirror common PyTorch optimizers:
- SGD: `https://pytorch.org/docs/stable/generated/torch.optim.SGD.html`
- AdaGrad: `https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html`
- RMSProp: `https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html`
- Adam: `https://pytorch.org/docs/stable/generated/torch.optim.Adam.html`
- AdamW: `https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html`
- Adadelta: `https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html`

General optimizer docs:
`https://pytorch.org/docs/stable/optim.html`

TorchLean runtime extension points:
- `TorchLean.optim.runtimeMuon` carries an orthogonalization backend.
- `TorchLean.optim.galore.projectedSGD` carries a projector/lift pair.
-/

@[expose] public section


namespace NN
namespace API
namespace TorchLean
namespace Optimizers

/-- Public optimizer config alias for the high-level trainer API. -/
abbrev Config := API.TorchLean.Trainer.Optimizer

-- Re-export constructors from `API.TorchLean.Trainer` (canonical).
/-- Construct an SGD optimizer configuration. -/
abbrev sgd := API.TorchLean.Trainer.sgd

/-- Construct a momentum-SGD optimizer configuration. -/
abbrev momentumSGD := API.TorchLean.Trainer.momentumSGD

/-- Construct an AdaGrad optimizer configuration. -/
abbrev adagrad := API.TorchLean.Trainer.adagrad

/-- Construct an RMSProp optimizer configuration. -/
abbrev rmsprop := API.TorchLean.Trainer.rmsprop

/-- Construct an Adam optimizer configuration. -/
abbrev adam := API.TorchLean.Trainer.adam

/-- Construct an AdamW optimizer configuration. -/
abbrev adamw := API.TorchLean.Trainer.adamw

/-- Construct an Adadelta optimizer configuration. -/
abbrev adadelta := API.TorchLean.Trainer.adadelta

end Optimizers
end TorchLean
end API
end NN
