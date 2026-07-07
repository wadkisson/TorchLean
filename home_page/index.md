---
# layout: home
usemathjax: true
---

<div class="hero home-hero">
  <div class="hero-text">
    <p class="lede">
      TorchLean is a Lean 4 library for neural-network code that needs to run and be reasoned
      about. A model can train, lower to a graph, choose a backend, and still point to the Lean
      object a theorem or checker is talking about.
    </p>

    <div class="home-actions" aria-label="Primary links">
      <a class="primary-link" href="{{ '/blueprint/Introduction/' | relative_url }}">Start reading</a>
      <a class="secondary-link" href="{{ '/examples/' | relative_url }}">View examples</a>
      <a class="secondary-link" href="https://arxiv.org/abs/2602.22631">Read the paper</a>
    </div>
  </div>
</div>

<div class="home-overview">
  <img
    src="{{ '/assets/media/figures/torchlean-layout.png' | relative_url }}"
    alt="TorchLean overview: typed tensors, shared graph IR, selected reverse-mode autograd proofs, IEEE-754 semantics, certificate checking, PyTorch round trip, CUDA trust boundary, approximation theorems, and Lean verification."
    loading="lazy" />
</div>

## Working Paths

<div class="workflow-list">
  <a href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/Autograd-Walkthrough/' | relative_url }}">
    <span>01</span>
    <strong>Write and run models</strong>
    <em>Use Lean-native training loops, tensors, and autograd examples.</em>
  </a>
  <a href="{{ '/blueprint/Semantics-and-Graphs/Graphs-and-IR/' | relative_url }}">
    <span>02</span>
    <strong>Lower to graph IR</strong>
    <em>Inspect shapes, payloads, graph semantics, and executable traces.</em>
  </a>
  <a href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/Backend-Selection-and-Trust/' | relative_url }}">
    <span>03</span>
    <strong>Choose a backend</strong>
    <em>Keep one model while choosing eager, compiled, CUDA, or external kernel providers.</em>
  </a>
  <a href="{{ '/blueprint/Verification-and-Certificates/' | relative_url }}">
    <span>04</span>
    <strong>Check verification artifacts</strong>
    <em>Replay bounds and certificates, then read the Lean statements they support.</em>
  </a>
  <a href="{{ '/examples/bug-zoo/' | relative_url }}">
    <span>05</span>
    <strong>Turn bugs into contracts</strong>
    <em>Study small checked examples for masks, losses, normalization, caches, and runtime edges.</em>
  </a>
</div>
