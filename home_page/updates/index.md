---
title: Updates
usemathjax: true
---

# TorchLean Updates

This is the release log for public releases, recent fixes, validation notes,
and user-facing issues.

For correctness assumptions and trusted boundaries, use
[Trust Boundaries](https://github.com/lean-dojo/TorchLean/blob/main/TRUST_BOUNDARIES.md).
For source-level provenance and third-party notes, use
[Third-Party Notices](https://github.com/lean-dojo/TorchLean/blob/main/THIRD_PARTY_NOTICES.md).

## Index

- [June 2026: CUDA, CROWN, and PINN Reliability Update](#june-2026-cuda-crown-and-pinn-reliability-update)
- [June 2026: Lean 4.31 Migration](#june-2026-lean-431-migration)
- [May 2026: Repository Modularization and Comment Cleanup](#may-2026-repository-modularization-and-comment-cleanup)
- [May 2026: Runtime API Update](#may-2026-runtime-api-update)
- [May 2026: CUDA Training Stability Update](#may-2026-cuda-training-stability-update)
- [May 2026: Introductory Data Note](#may-2026-introductory-data-note)
- [May 2026: TorchLean Released](#may-2026-torchlean-released)

## June 2026: CUDA, CROWN, and PINN Reliability Update

This update tightens several runtime and proof-facing edges that matter for
longer local validation runs and downstream users of the public API surface.

On the native side, CUDA and CPU-stub shape arithmetic now share checked size
helpers for products, byte counts, and additions that are computed at the Lean
FFI boundary. Rank-polymorphic broadcast, reduce, swap, gather/scatter, flash
attention, convolution/pooling, tensor-copy, and spectral-convolution paths now
reject impossible sizes before allocation or kernel launch. cuFFT and metadata
upload failure paths also clean up scratch buffers and plans before reporting
the boundary failure.

The CUDA test suite now includes value-checking coverage for higher-rank
`broadcastTo`, `reduceFromBroadcastTo`, `reduceSumAxis`, and
`swapAdjacentAtDepth` cases. These tests exercise the rank-polymorphic native
paths instead of only checking that output buffers have the expected length.

On the proof side, the graph CROWN certificate theorem now exposes a concrete
non-vacuous result: callers provide the actual certificate and semantic value
entries for the node being enclosed. The IEEE32 statement remains
evaluator-parametric, but it now requires a no-self-dependency side condition
for the evaluator trace, so the old identity-evaluator hook cannot be used as a
semantic bridge. The 2-layer MLP CROWN implementation also exposes the affine
forms used by `boundAffineCrown`, making the structural relation between the
forms and the returned box explicit.

The Python PINN trainers were refactored to share dataset loading, MLP
construction, expression evaluation, gradient, constant parsing, and export
helpers through `scripts/verification/pinn/pinn_common.py`. The restricted
expression evaluator no longer requires NumPy merely to import; NumPy aliases
are enabled when NumPy is installed.

The focused public API entrypoint now imports the facade namespaces directly, so
`import NN.Entrypoint.API` exposes the documented `TorchLean.nn`,
`TorchLean.optim`, `TorchLean.Trainer`, `TorchLean.Data`, `TorchLean.Loss`, and
`TorchLean.Metrics` names without requiring the broader `NN` umbrella import.

Validation for this update included:

- `lake test`
- `lake build NN.CI.All`
- `lake lint -R -K cuda=true -K cuda_home=/usr/local/cuda-13.0`
- `scripts/checks/check.sh --cuda --cuda-home /usr/local/cuda-13.0`
- `scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda-13.0 --all-tools`
- focused Lean checks for the CROWN MLP and graph CROWN certificate modules
- PyTorch CUDA smoke tests for the PINN trainers on an A100 GPU

CUDA sanitizer reported zero memcheck/initcheck/synccheck errors and zero
racecheck hazards on the exercised runtime suite.

## June 2026: Lean 4.31 Migration

TorchLean now builds with `leanprover/lean4:v4.31.0`. The root Lake manifest,
Mathlib pin, documentation generator pin, Verso blueprint toolchain, website
metadata, README, and formalization metadata were moved together so the public
repository state agrees about the Lean version.

The migration fixed proof-term breakages in differentiability and autograd
composition files where Lean 4.31 became stricter about composed functions and
eventual equality. The full repository build was rerun on 4.31.

## May 2026: Repository Modularization and Comment Cleanup

TorchLean has had a large repository cleanup pass. The goal was simple: make
the public source tree easier to read, easier to review, and easier to extend
without changing the intended behavior of the library.

This pass is a source-structure and documentation-quality update, not a change
to TorchLean's mathematical or runtime semantics:

- large proof and runtime files were split along existing conceptual boundaries;
- umbrella modules were kept only where they clarify the public import surface;
- obsolete import shells and example names were removed;
- comments were rewritten in a more mathlib-style voice, explaining definitions
  and trust boundaries as part of the maintained source;
- examples, API docs, the Verso guide, and website pages were rebuilt against the
  new module layout.

This pass does not change model semantics, verification claims, CUDA behavior,
or trusted boundaries.

## May 2026: Runtime API Update

This earlier update added runtime pieces that were needed for longer training
runs and a cleaner public API. The current Lean toolchain is recorded in the
repository root, the blueprint package, and the formalization metadata.

The runtime pieces added in that pass were:

- runtime-side Float initializers for large CPU/CUDA modules;
- shape-indexed initializer plans, so each parameter receives exactly one
  initializer;
- step-indexed training streams for batches produced by a rule, simulator,
  replay buffer, or file-backed window source;
- integer-token embedding and row-wise cross-entropy helpers for GPT-style
  language-model training;
- CUDA allocator diagnostics and explicit workspace-buffer ownership in the eager
  runtime.

The guide now has dedicated sections for step streams, runtime initialization,
CUDA memory ownership, and integer-token GPT training.

## May 2026: CUDA Training Stability Update

Longer CUDA training runs exposed a practical bug in the eager runtime. A
GPT-style example could train normally for thousands of updates and then stop
with a CUDA allocation failure. The surprising part was that the same kind of
problem could show up even on compact examples if enough per-step CUDA objects
were kept alive.

The issue was not that the model suddenly needed more parameters. It was that
some intermediate values created during training -- tape entries, gradient buffers,
and kernel workspace buffers -- could stay attached to the run longer than they
needed to. Over many optimizer updates, that turns into allocator pressure. The
fix was to make the training loop and CUDA eager backend more explicit about
which values are returned to the caller and which intermediate buffers can be
released after the step finishes.

The CUDA training path used by the public model examples now does three things:

- loader-based model commands now treat `--steps` as optimizer updates;
- CUDA eager/autograd releases ephemeral tape, gradient, and workspace buffers
  after the values needed by the caller have been extracted;
- longer CUDA example runs report allocator telemetry by default, with
  `--cuda-mem-watch N` available when a user wants an exact sampling cadence.

The allocator report is a runtime diagnostic. It prints live and peak allocator
state while a run is still in progress, and it warns if the observed free-memory
trend would run out before the requested training horizon. If a long run is
going wrong, the terminal should show that early instead of leaving the user to
discover it at the end. It does not change the Lean-side trust-boundary story;
it makes the native runtime behavior easier to inspect.

Validation for this update included:

- `lake build`
- `lake build -K cuda=true`
- `lake exe torchlean mlp --cpu --steps 10 --log false`
- `lake exe torchlean mlp --steps 10 --dtype float --backend eager --log false`
- `lake exe -K cuda=true torchlean mlp --cuda --fast-kernels --steps 1000 --log false`
- `lake exe -K cuda=true torchlean cnn --cuda --steps 1000 --log false`
- `lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --steps 1200 --generate 0 --log false`
- `lake exe -K cuda=true torchlean fno1d_burgers --cuda --fast-kernels --steps 50 --log false`

The validation commands above checked two things: representative losses or MSE
values went down, and the CUDA allocator stayed bounded on the exercised runs.
CUDA execution remains an implementation path; the mathematical trust boundary
is still documented separately.

## May 2026: Introductory Data Note

Some examples use real public datasets and do not download them during
`lake build`. If a model reports a missing dataset, run the downloader it
prints. For the README MLP example:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
```

For the text and vision examples:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --cifar10
```

Keeping data downloads explicit makes ordinary builds deterministic and avoids
silently committing large external data into the repository.

## May 2026: TorchLean Released

TorchLean is public as a Lean 4 framework for writing, running, inspecting, and
verifying neural-network programs. The initial release brings together the parts
of the project that are meant to be used as one system:

- typed tensors, layers, model APIs, loaders, training loops, and examples;
- a shared graph IR for execution, inspection, verification, and import/export;
- finite-precision semantics, including executable IEEE-style Float32 models and
  explicit runtime agreement boundaries;
- autograd and optimizer support for runnable examples;
- verification examples including IBP/CROWN-style bounds, certificate checkers,
  VNN-COMP-style bundles, Bug Zoo contracts, and 3D geometry certificates;
- optional CUDA/native runtime paths with documented trust boundaries;
- generated API docs, a guide, examples pages, third-party notices, and an AI
  assistance disclosure.

Start with the README example, then use the Guide and Examples pages for the
longer walkthroughs.
