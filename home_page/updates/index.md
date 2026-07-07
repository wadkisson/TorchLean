---
title: Updates
usemathjax: true
---

<nav class="timeline-nav" aria-label="TorchLean update timeline">
  <a href="#june-2026-reliability">June 2026 reliability</a>
  <a href="#june-2026-lean-431">Lean 4.31</a>
  <a href="#may-2026-cleanup">Comment cleanup</a>
  <a href="#may-2026-runtime-api">Runtime API</a>
  <a href="#may-2026-cuda-stability">CUDA stability</a>
  <a href="#may-2026-data-note">Data note</a>
  <a href="#may-2026-release">Initial release</a>
</nav>

<div class="updates-timeline">

<article class="update-card" id="june-2026-reliability" markdown="1">
  <div class="update-date">June 2026</div>
  <div class="update-body" markdown="1">

## CUDA, CROWN, and PINN Reliability

<p class="update-kicker">Native runtime, certificates, scientific ML</p>
<p class="update-summary">
The June reliability pass tightened CUDA allocation checks, made CROWN certificate statements more
direct, and moved duplicated PINN training code into shared helpers.
</p>

<div class="update-grid">
  <section>
    <h3>Runtime checks</h3>
    <p>
      CUDA and CPU stubs now use shared checked size arithmetic for products,
      byte counts, and additions at the Lean FFI boundary. Broadcast, reduction,
      swap, gather/scatter, attention, convolution/pooling, tensor-copy, and
      spectral-convolution paths reject impossible sizes before allocation or
      kernel launch.
    </p>
  </section>
  <section>
    <h3>Proof API</h3>
    <p>
      The graph CROWN certificate theorem now returns the enclosure for the
      node being checked. The IEEE32 version records the no-self-dependency
      condition on the evaluator trace, and the two-layer MLP CROWN code exposes
      the affine forms used by <code>boundAffineCrown</code>.
    </p>
  </section>
  <section>
    <h3>PINNs</h3>
    <p>
      Python PINN trainers now share dataset loading, MLP construction,
      expression evaluation, gradients, constant parsing, and export helpers
      through <code>scripts/verification/pinn/pinn_common.py</code>.
    </p>
  </section>
</div>

The focused API import now exposes the short `TorchLean.*` namespaces directly.
With `import NN.Entrypoint.API`, users get `TorchLean.nn`, `TorchLean.optim`,
`TorchLean.Trainer`, `TorchLean.Data`, `TorchLean.Loss`, and
`TorchLean.Metrics` without importing the broader `NN` umbrella.

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake test`
- `lake build NN.CI.All`
- `lake lint -R -K cuda=true -K cuda_home=/usr/local/cuda-13.0`
- `scripts/checks/check.sh --cuda --cuda-home /usr/local/cuda-13.0`
- `scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda-13.0 --all-tools`
- focused Lean checks for the CROWN MLP and graph CROWN certificate modules
- PyTorch CUDA regression runs for the PINN trainers on an A100 GPU

CUDA sanitizer reported zero memcheck/initcheck/synccheck errors and no
racecheck hazards on the exercised runtime suite.
</div>

  </div>
</article>

<article class="update-card" id="june-2026-lean-431" markdown="1">
  <div class="update-date">June 2026</div>
  <div class="update-body" markdown="1">

## Lean 4.31 Migration

<p class="update-kicker">Toolchain alignment</p>
<p class="update-summary">
TorchLean now builds with <code>leanprover/lean4:v4.31.0</code>. The root Lake
manifest, Mathlib pin, documentation generator pin, Verso blueprint toolchain,
website metadata, README, and formalization metadata were moved together.
</p>

The migration fixed proof-term breakages in differentiability and autograd
composition files where Lean 4.31 became stricter about composed functions and
eventual equality. The full repository build was rerun on the new toolchain.

  </div>
</article>

<article class="update-card" id="may-2026-cleanup" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Repository Modularization and Comment Cleanup

<p class="update-kicker">Source organization</p>
<p class="update-summary">
The public source tree is easier to read, easier to review, and easier to extend without changing
TorchLean's intended behavior.
</p>

<div class="update-grid">
  <section>
    <h3>Structure</h3>
    <ul>
      <li>Large proof and runtime files were split along conceptual boundaries.</li>
      <li>Umbrella modules were kept only where they make imports clearer.</li>
      <li>Obsolete import shells and example names were removed.</li>
    </ul>
  </section>
  <section>
    <h3>Documentation</h3>
    <p>
      Comments were rewritten in a more mathlib-style voice: definitions state
      mathematical intent, runtime boundaries name assumptions, and examples
      avoid stale narration.
    </p>
  </section>
  <section>
    <h3>Scope</h3>
    <p>
      Model semantics, verification claims, CUDA behavior, and trusted boundaries are unchanged.
    </p>
  </section>
</div>

The examples and website pages were rebuilt against the new module layout.

  </div>
</article>

<article class="update-card" id="may-2026-runtime-api" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Runtime API Update

<p class="update-kicker">Training loops, streams, initialization</p>
<p class="update-summary">
Longer examples now use the same public runtime API for initialization, minibatches, optimizer
steps, logging, and checkpoint-style parameter files.
</p>

<div class="update-grid">
  <section>
    <h3>Initialization</h3>
    <p>
      Runtime-side Float initializers and shape-indexed initializer plans give parameter bundles a
      clear construction story: the model declares the parameter shape, the initializer plan selects
      the distribution or constant, and the runtime builds each tensor once.
    </p>
  </section>
  <section>
    <h3>Streams</h3>
    <p>
      Step-indexed training streams make minibatches explicit. A batch may come from a rule, a
      simulator, a replay buffer, or a file-backed window source, but the training loop sees a typed
      stream of inputs, targets, and metadata.
    </p>
  </section>
  <section>
    <h3>Language models</h3>
    <p>
      Integer-token embedding and row-wise cross-entropy helpers let GPT-style examples train on
      token ids directly, instead of expanding every target into a one-hot vector.
    </p>
  </section>
</div>

The runtime documentation now follows the same path as the examples: initialize parameters, produce
batches, run forward/autograd, update parameters, save reports, and state the native or external
boundary when a backend is selected.

  </div>
</article>

<article class="update-card" id="may-2026-cuda-stability" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## CUDA Training Stability

<p class="update-kicker">Memory lifetime, allocator diagnostics</p>
<p class="update-summary">
Longer CUDA training runs exposed allocator pressure from intermediate values
that could stay attached to a run longer than needed.
</p>

The issue was not model size. Some intermediate values created during training
-- tape entries, gradient buffers, and kernel workspace buffers -- could remain
alive across many optimizer steps. The fix made the training loop and CUDA eager
backend explicit about which values are returned to the caller and which buffers
can be released after the step finishes.

<div class="update-grid">
  <section>
    <h3>Step counts</h3>
    <p>Loader-based model commands now treat <code>--steps</code> as optimizer updates.</p>
  </section>
  <section>
    <h3>Buffer lifetime</h3>
    <p>CUDA eager/autograd releases ephemeral tape, gradient, and workspace buffers after each step.</p>
  </section>
  <section>
    <h3>Diagnostics</h3>
    <p>Longer CUDA examples report allocator telemetry by default, with <code>--cuda-mem-watch N</code> for exact sampling.</p>
  </section>
</div>

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake build`
- `lake build -K cuda=true`
- `lake exe torchlean mlp --cpu --steps 10 --log false`
- `lake exe torchlean mlp --steps 10 --dtype float --backend eager --log false`
- `lake exe -K cuda=true torchlean mlp --cuda --fast-kernels --steps 1000 --log false`
- `lake exe -K cuda=true torchlean cnn --cuda --steps 1000 --log false`
- `lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --steps 1200 --generate 0 --log false`
- `lake exe -K cuda=true torchlean fno1d_burgers --cuda --fast-kernels --steps 50 --log false`

The validation checked representative losses or MSE values going down and the
CUDA allocator staying bounded on the exercised runs.
</div>

  </div>
</article>

<article class="update-card" id="may-2026-data-note" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Introductory Data Note

<p class="update-kicker">Reproducible builds</p>
<p class="update-summary">
Some examples use public datasets and do not download them during
<code>lake build</code>. Keeping data downloads explicit makes ordinary builds
deterministic and avoids silently committing large external data.
</p>

For the model-zoo MLP example:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
```

For the text and vision examples:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --cifar10
```

  </div>
</article>

<article class="update-card" id="may-2026-release" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## TorchLean Released

<p class="update-kicker">Initial public release</p>
<p class="update-summary">
TorchLean became public as a Lean 4 framework for writing, running, inspecting,
and verifying neural-network programs.
</p>

<div class="update-grid">
  <section>
    <h3>Core system</h3>
    <p>
      Typed tensors, layers, model APIs, loaders, training loops, examples, and
      a shared graph IR for execution, inspection, verification, and
      import/export.
    </p>
  </section>
  <section>
    <h3>Semantics</h3>
    <p>
      Finite-precision semantics, executable IEEE-style Float32 models,
      autograd, optimizer support, and explicit runtime agreement boundaries.
    </p>
  </section>
  <section>
    <h3>Verification</h3>
    <p>
      IBP/CROWN-style bounds, certificate checkers, VNN-COMP-style bundles, Bug
      Zoo contracts, 3D geometry certificates, and optional CUDA/native runtime
      paths with documented trust boundaries.
    </p>
  </section>
</div>

The README example is the shortest entry point; the Guide and Examples pages carry the longer
walkthroughs.

  </div>
</article>

</div>
