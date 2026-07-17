/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Schedulers.Core

/-!
# PyTorch-Compatible Learning-Rate Schedulers

Learning-rate schedulers for TorchLean runtime training.

Schedulers are small deterministic state machines that answer:

- “what learning rate should we use *at this step*?”
- “how do we advance to the next step?”

TorchLean keeps schedulers explicit and pure so:
- runtime code can store scheduler state in a record (or serialize it),
- proofs and specs can refer to the exact schedule that was used.

Step counter convention:
- `currentStep` is **0-indexed**. The first call to `getLr` uses `currentStep = 0`.
- `step` increments the counter by 1.

PyTorch analogy: these mirror common `torch.optim.lr_scheduler.*` schedules, but expressed as
simple Lean structures with `getLr` and `step`.

Organization:
- the first section defines TorchLean-native schedules with total, easy-to-reason-about behavior;
- the later `*LR` section defines PyTorch-compatible variants when PyTorch's exact phase and
  step-count conventions matter;

References (common schedules we implement):
- Cosine annealing / SGDR (Loshchilov–Hutter, 2017): https://arxiv.org/abs/1608.03983
- Cyclical learning rates (Smith, 2017): https://arxiv.org/abs/1506.01186
- 1cycle policy (Smith, 2018): https://arxiv.org/abs/1803.09820

PyTorch references:
- `torch.optim.lr_scheduler` overview:
  https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
-/

@[expose] public section


namespace Optim

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

open MathFunctions

/-! ## PyTorch-compatible scheduler variants -/

/-!
The schedulers below use formulas and step-count conventions chosen to match PyTorch's
`torch.optim.lr_scheduler.*` semantics more directly.

Important convention note (PyTorch `last_epoch`):
- In modern PyTorch, schedulers effectively start at `last_epoch = 0` right after construction
  (fresh run with `last_epoch = -1` in the constructor triggers an initial internal step).
- We model that behavior by using a `current_step : Nat := 0` counter.
  Think: `current_step` corresponds to PyTorch's `last_epoch` after construction.

These schedulers are *LR-only* (they do not mutate optimizer momentum/betas). If you need the full
PyTorch OneCycle momentum behavior, consider adding a separate momentum schedule and stepping both
in lockstep.
-/

/-! ### StepLR -/

/--
PyTorch-compatible `StepLR`.

Semantics:
- `current_step = 0` yields `base_lr`.
- Every `step_size` steps, multiply LR by `gamma`.
- When `step_size = 0`, this degenerates to a constant schedule (total, no exceptions).

PyTorch reference: `torch.optim.lr_scheduler.StepLR`.
-/
structure StepLR (α : Type) where
  /-- Base learning rate (what PyTorch calls `base_lrs[i]`). -/
  baseLr : α
  /-- Step interval (`step_size`). -/
  stepSize : Nat
  /-- Multiplicative decay factor (`gamma`). -/
  gamma : α
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

/-- Current learning rate for `StepLR` at `current_step`. -/
def StepLR.getLr (scheduler : StepLR α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLr
  else
    let decayCount := scheduler.currentStep / scheduler.stepSize
    scheduler.baseLr * (scheduler.gamma ^ (decayCount : α))

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/-- The PyTorch-compatible `StepLR` is also totalized to a constant when `step_size = 0`. -/
theorem StepLR.getLr_zero_stepSize
    (baseLr gamma : α) (currentStep : Nat) :
    StepLR.getLr
      { baseLr := baseLr
        stepSize := 0
        gamma := gamma
        currentStep := currentStep } = baseLr := by
  simp [StepLR.getLr]

/-- Advance `StepLR` by one step. -/
def StepLR.step (scheduler : StepLR α) : StepLR α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/-- Constructor for `StepLR` starting at `current_step = 0`. -/
def stepLR (baseLr : α) (stepSize : Nat) (gamma : α) : StepLR α :=
  { baseLr := baseLr, stepSize := stepSize, gamma := gamma }

/-! ### CosineAnnealingLR -/

/--
PyTorch-compatible `CosineAnnealingLR`.

Key behavior difference from TorchLean's `CosineAnnealingScheduler` above:
- PyTorch's `CosineAnnealingLR` continues the cosine curve past `T_max` (it is periodic with period
  `2*T_max`), rather than clamping to `eta_min`.

PyTorch reference: `torch.optim.lr_scheduler.CosineAnnealingLR`.
-/
structure CosineAnnealingLR (α : Type) where
  /-- Base learning rate (`base_lrs[i]`). -/
  baseLr : α
  /-- Maximum number of steps in a half-cycle (`T_max`). -/
  tMax : Nat
  /-- Minimum learning rate (`eta_min`). -/
  etaMin : α
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

/-- Current learning rate for `CosineAnnealingLR` at `current_step`. -/
def CosineAnnealingLR.getLr (scheduler : CosineAnnealingLR α) : α :=
  if scheduler.tMax = 0 then
    scheduler.baseLr
  else
    scheduler.etaMin
      + (scheduler.baseLr - scheduler.etaMin)
          * (1 + cos ((pi : α) * (scheduler.currentStep : α) / (scheduler.tMax : α)))
          / (1 + 1)

/-- Advance `CosineAnnealingLR` by one step. -/
def CosineAnnealingLR.step (scheduler : CosineAnnealingLR α) : CosineAnnealingLR α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/-! ### OneCycleLR (LR-only) -/

/-- Anneal strategy used by `OneCycleLR` (matches PyTorch `"cos"` or `"linear"`). -/
inductive OneCycleAnnealStrategy
  | cos
  | linear
  deriving Repr, DecidableEq

/--
PyTorch-compatible `OneCycleLR` (LR-only).

Notes:
- This mirrors PyTorch's `OneCycleLR` *learning-rate* schedule only. PyTorch can also cycle momentum
  (or Adam's `beta1`); TorchLean keeps this scheduler pure and LR-only.
- PyTorch defines:
  - `initial_lr = max_lr / div_factor`
  - `min_lr = initial_lr / final_div_factor`
  (note: `min_lr` is derived from `initial_lr`, not directly from `max_lr`).
- PyTorch uses "phase end steps" that are floats:
  - phase 1 ends at `pct_start * total_steps - 1`
  - phase 2 ends at `total_steps - 1` (and `three_phase` inserts a middle phase).
  This means the boundary can be fractional; the schedule uses interpolation ratios (`pct`)
  computed from these float endpoints. We match that behavior using `α` arithmetic.

PyTorch reference: `torch.optim.lr_scheduler.OneCycleLR`.
-/
structure OneCycleLR (α : Type) where
  /-- Peak learning rate (`max_lr`). -/
  maxLr : α
  /-- Total number of steps (`total_steps`). -/
  totalSteps : Nat
  /-- Fraction of steps spent increasing LR (`pct_start`). -/
  pctStart : α
  /-- `div_factor` used to derive `initial_lr = max_lr / div_factor`. -/
  divFactor : α
  /-- `final_div_factor` used to derive `min_lr = initial_lr / final_div_factor`. -/
  finalDivFactor : α
  /-- Anneal strategy (`cos` or `linear`). -/
  annealStrategy : OneCycleAnnealStrategy := .cos
  /-- Use PyTorch's `three_phase` variant when `true`. -/
  threePhase : Bool := false
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

namespace OneCycleLR

/-- Derived initial LR (`max_lr / div_factor`). -/
def initialLr (s : OneCycleLR α) : α :=
  s.maxLr / s.divFactor

/-- Derived minimum LR (`initial_lr / final_div_factor`). -/
def minLr (s : OneCycleLR α) : α :=
  initialLr s / s.finalDivFactor

/-- PyTorch-compatible anneal helper (no clamping). -/
def anneal (s : OneCycleLR α) (startLR endLR pct : α) : α :=
  match s.annealStrategy with
  | .cos => SchedulerUtils.cosineAnnealRaw startLR endLR pct
  | .linear => SchedulerUtils.linearInterpolationRaw startLR endLR pct

end OneCycleLR

/-- Current learning rate for `OneCycleLR` at `current_step` (LR-only). -/
def OneCycleLR.getLr (s : OneCycleLR α) : α :=
  let initLR := OneCycleLR.initialLr (α := α) s
  let minLR := OneCycleLR.minLr (α := α) s
  if s.totalSteps = 0 then
    initLR
  else
    -- PyTorch raises when `step_num > total_steps`. We clamp to keep the function total.
    let stepNat := if s.currentStep ≤ s.totalSteps then s.currentStep else s.totalSteps
    let stepNum : α := stepNat
    let total : α := s.totalSteps
    if s.threePhase then
      let end1 : α := s.pctStart * total - 1
      let end2 : α := (Numbers.two : α) * s.pctStart * total - (Numbers.two : α)
      let end3 : α := (s.totalSteps - 1 : Nat)
      if stepNum > end1 then
        if stepNum > end2 then
          let pct := SchedulerUtils.safeDiv (stepNum - end2) (end3 - end2)
          OneCycleLR.anneal (α := α) s initLR minLR pct
        else
          let pct := SchedulerUtils.safeDiv (stepNum - end1) (end2 - end1)
          OneCycleLR.anneal (α := α) s s.maxLr initLR pct
      else
        let pct := SchedulerUtils.safeDiv stepNum (end1 - 0)
        OneCycleLR.anneal (α := α) s initLR s.maxLr pct
    else
      let end1 : α := s.pctStart * total - 1
      let end2 : α := (s.totalSteps - 1 : Nat)
      if stepNum > end1 then
        let pct := SchedulerUtils.safeDiv (stepNum - end1) (end2 - end1)
        OneCycleLR.anneal (α := α) s s.maxLr minLR pct
      else
        let pct := SchedulerUtils.safeDiv stepNum (end1 - 0)
        OneCycleLR.anneal (α := α) s initLR s.maxLr pct

/-- Advance `OneCycleLR` by one step. -/
def OneCycleLR.step (s : OneCycleLR α) : OneCycleLR α :=
  { s with currentStep := s.currentStep + 1 }

end Optim
