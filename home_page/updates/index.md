---
title: Updates
usemathjax: true
---

<nav class="timeline-nav" aria-label="TorchLean update timeline">
  <a href="#july-2026-refactor">July 2026 refactor</a>
  <a href="#june-2026-reliability">June 2026 reliability</a>
  <a href="#june-2026-lean-431">Lean 4.31</a>
  <a href="#may-2026-cleanup">Comment cleanup</a>
  <a href="#may-2026-runtime-api">Runtime API</a>
  <a href="#may-2026-cuda-stability">CUDA stability</a>
  <a href="#may-2026-data-note">Data note</a>
  <a href="#may-2026-release">Initial release</a>
</nav>

<div class="updates-timeline">

<article class="update-card" id="july-2026-refactor" markdown="1">
  <div class="update-date">July 2026</div>
  <div class="update-body" markdown="1">

## A Smaller Core and Explicit Runtime Boundaries

<p class="update-kicker">Public API, tensor semantics, floating point, backends, verification</p>
<p class="update-summary">
TorchLean now has a clearer library root, a more general tensor-facing API, and one vocabulary for
describing where an operation runs and what is trusted. At the same time, the numerical and
verification layers were tightened where executable behavior had drifted from the intended
mathematics.
</p>

### One Public Library

`import NN` is the ordinary entry point for model code. The old `NN.Library` and `NN.Entrypoint.*`
shells were removed, and the focused roots `NN.API`, `NN.Spec`, `NN.Runtime`, `NN.Floats`,
`NN.GraphSpec`, `NN.IR`, `NN.MLTheory`, `NN.Proofs`, `NN.Verification`, and `NN.Widgets` now import
their own canonical modules directly. Model implementations remain part of TorchLean, but the old
forwarding model-zoo facades and duplicate optimizer namespaces no longer define a second API.

Large implementation files were split without changing their ownership. Training, text utilities,
data handling, schedulers, CROWN propagation, graph compilation, runtime operations, normalization,
Muon, and floating-point semantics now live in smaller modules with narrow imports. After counting
the new split modules as well as the deleted forwarding layers, the `NN` Lean source tree is about
1,500 lines smaller than before.

<div class="update-grid">
  <section>
    <h3>General tensors</h3>
    <p>
      Batching and layout are expressed through tensor shapes and axes rather than separate image
      tensor types. Generic permutation, reduction, reshape, and global-average-pooling operations
      replace public CHW/NCHW and rank-specific convenience layers. Layout names remain only where
      an operation, such as channel-first batch normalization, genuinely depends on that layout.
    </p>
  </section>
  <section>
    <h3>Models</h3>
    <p>
      CNN, ResNet, ViT, FNO, transformer, GPT, Mamba, recurrent, generative, reinforcement-learning,
      and self-supervised models remain available. Their definitions now use the shared tensor and
      layer surface instead of model-specific forwarding stacks.
    </p>
  </section>
  <section>
    <h3>Training</h3>
    <p>
      Optimizers share one stateful tensor-optimizer interface. The retained laws describe stream
      composition and meaningful relations between update rules; generated certificate tables and
      definitional restatements that added no proof content were removed.
    </p>
  </section>
</div>

### Floating-Point Semantics

The rounded-real development under `NN.Floats.NeuralFloat` is now organized by format, rounding,
scalar operations, analysis, error bounds, and special execution policies. It includes radix and
exponent formats, directed and nearest rounding, round-to-odd, ULP and neighboring-value results,
double-rounding facts, Sterbenz subtraction, and absolute and relative error bounds. This is a
native Lean development influenced by Flocq's mathematical organization, not a claim to reproduce
the whole Coq library.

The layers now have distinct jobs:

- `NeuralFloat` and `NF` describe configurable rounded-real arithmetic used in proofs;
- `FP32` specializes the rounded-real model to binary32-sized parameters;
- `IEEE32Exec` models executable IEEE-754-style binary32 behavior, including special values;
- runtime bridges state how native values are interpreted by those models.

The new effective-rounding example follows a rounded value from its format and mode to an explicit
error statement. Runtime approximation documentation now begins from the ideal autograd theorems
and states the additional hypotheses required to bound an executable or rounded path.

### Backend Contracts

TorchLean now uses one backend vocabulary across planning, diagnostics, and documentation. A
`Device` says where work runs, a `Provider` names the implementation family, and a `BackendOp`
identifies the requested operation. A kernel capsule then records the operation, device, provider,
forward support, VJP ownership, shape and layout contracts, value contract, and trust level.

Named profiles keep choices that must agree in one object. The checked CPU and native-CUDA profiles
retain TorchLean tape ownership. The LibTorch-forward profile permits a registered external forward
provider while retaining the TorchLean graph and tape; the explicit LibTorch-autograd profile names
external backward ownership as a trusted boundary. The current LibTorch implementation remains the
optional scaled-dot-product-attention bridge, not a general PyTorch dispatcher.

macOS CPU is supported, Linux CPU and native CUDA are supported, and WSL2 is the recommended Windows
route. Metal, ROCm, WebGPU, TPU, Trainium, native Windows, and custom accelerators are represented as
targets for future capsules. Selecting an unavailable target fails with a diagnostic instead of
silently running on another device.

### Mathematical and Verification Corrections

<div class="update-grid">
  <section>
    <h3>Losses and masks</h3>
    <p>
      Huber loss and Smooth L1 now have their intended, distinct scaling. Hard attention masks use
      exact exclusion in the softmax semantics: a blocked entry contributes zero numerator. The old
      finite <code>-1000</code> masking convention was removed from attention paths and examples.
    </p>
  </section>
  <section>
    <h3>Bounds</h3>
    <p>
      Leaky-ReLU interval propagation handles a negative slope across the kink at zero. Logarithm
      interval checks reject nonpositive domains. Unsound or undocumented GELU and ELU candidate
      bounds were removed instead of being exposed under names that suggested certified enclosure.
    </p>
  </section>
  <section>
    <h3>Certificates</h3>
    <p>
      JSON certificate readers reject non-finite claims before array comparisons. CROWN, IBP,
      payload, graph-evaluation, robustness, PINN, and geometry checkers state their preconditions
      at the boundary where external artifacts become Lean objects.
    </p>
  </section>
</div>

The graph evaluator and verifier compiler were split into operation-level modules, while coverage
theorems continue to quantify over the operation vocabulary. Brittle theorems asserting a current
list length or merely restating a definition were removed. Definitional bridge lemmas remain where
later correctness proofs use them to avoid unfolding implementation details.

### Lean 4.32

TorchLean now builds with Lean 4.32, mathlib 4.32, DocGen 4.32, and Verso 4.32. The migration also
removed the deprecated `Lean.RBMap` dependency in favor of `Std.TreeMap`, adopted the stronger sine
remainder estimate now exposed by mathlib, and made several dependent eliminations explicit where
Lean 4.32 no longer inferred the required casts. Proof-valued runtime helpers are now declared as
theorems when they are opaque evidence; proof constructors that must compute remain reducible
`abbrev`s. The migration does not suppress the proposition or duplicate-namespace linters.

### Documentation and Validation

The Guide was rewritten around the mathematical objects used by the library: typed tensors,
autograd maps, graph denotations, rounded execution, kernel contracts, and checked certificates.
The installation page now gives separate Linux, macOS, WSL2, native-Windows, CUDA, and optional
LibTorch instructions. The API reference, module graph, examples, and site navigation were rebuilt
for the new module roots. Repository checks now build `NN` directly rather than the deleted
`NN.Library` shell.

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake lint`
- `lake build` (4,269 build jobs)
- `lake build NN NN.CI.All` (4,308 build jobs)
- `lake exe nn_tests_suite`
- `lake -R -K cuda=true exe nn_tests_suite`
- `scripts/checks/example_regression.sh` across 41 CLI/help paths and the default runtime examples
- `scripts/checks/example_regression.sh --cuda --extended-cuda --skip-help --skip-default`
- DocGen API generation
- Verso Guide generation
- dependency audit and interactive import-graph generation
- Jekyll production build
- `git diff --check`

The CPU and CUDA suites passed on the exercised Linux machine. The documentation builds completed,
and the repository linter reported no errors. These checks do not turn native CUDA, LibTorch, or
future accelerator implementations into Lean proofs; their trust and evidence levels remain visible
through the backend contracts and `TRUST_BOUNDARIES.md`.
</div>

  </div>
</article>

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
With `import NN.API`, users get `TorchLean.nn`, `TorchLean.optim`,
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
- `lake -R -K cuda=true build`
- `lake exe torchlean mlp --device cpu --steps 10 --log false`
- `lake exe torchlean mlp --steps 10 --dtype float --backend eager --log false`
- `lake -R -K cuda=true exe torchlean mlp --device cuda --steps 1000 --log false`
- `lake -R -K cuda=true exe torchlean cnn --device cuda --steps 1000 --log false`
- `lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1200 --generate 0 --log false`
- `lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 50 --log false`

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
