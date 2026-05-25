---
title: Updates
usemathjax: true
---

# TorchLean Updates

This is the release log where we record the public releases, recent fixes,
validation notes, and any user-facing issues worth knowing about.

For correctness assumptions and trusted boundaries, use
[Trust Boundaries](https://github.com/lean-dojo/TorchLean/blob/main/TRUST_BOUNDARIES.md).
For source-level provenance and release hygiene, use
[Third-Party Notices](https://github.com/lean-dojo/TorchLean/blob/main/THIRD_PARTY_NOTICES.md).

## Index

- [May 2026: CUDA Training Stability Update](#may-2026-cuda-training-stability-update)
- [May 2026: Quickstart Data Note](#may-2026-quickstart-data-note)
- [May 2026: TorchLean Released](#may-2026-torchlean-released)

## May 2026: CUDA Training Stability Update

Recent CUDA training runs exposed two practical runtime issues in the model
examples:

- Long training logs could retain too much per-step structure and make ordinary
  runs harder to scale.
- CUDA eager/autograd paths needed more aggressive release of temporary gradient,
  tape, and downloaded device buffers after each optimization step.

The current runtime and examples now treat `--steps` as optimizer updates,
stream or sample long training logs, and release CUDA/autograd temporary buffers
after the values needed by the caller have been extracted.

Fresh-clone validation included:

- `lake build`
- `lake build -K cuda=true`
- `lake test -R -K cuda=true`
- `lake exe -K cuda=true torchlean mlp --cuda --fast-kernels --steps 1000`
- `lake exe -K cuda=true torchlean mlp --cuda --fast-kernels --steps 50000`
- `lake exe -K cuda=true torchlean cnn --cuda --steps 2000`
- `lake exe -K cuda=true torchlean gpt2 --cuda --steps 100 --generate 0`

These checks cover the public build, CUDA build, curated CUDA runtime stress
tests, and representative model-level CUDA training runs. CUDA execution remains
an implementation path; the mathematical trust boundary is still documented
separately.

## May 2026: Quickstart Data Note

Some examples intentionally use real public datasets and do not download them
during `lake build`. If a model reports a missing dataset, run the downloader it
prints. For the README MLP quickstart:

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

Start with the README quickstart, then use the Guide and Examples pages for the
longer walkthroughs.
