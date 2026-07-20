---
usemathjax: true
---

<div class="hero home-hero">
  <div class="hero-text">
    <p class="lede">
      TorchLean is a Lean 4 library for neural-network code that needs to run and be reasoned
      about. Train, lower to a graph, pick a backend, and keep the Lean object a theorem or
      checker is talking about.
    </p>
    <p class="lede">
      The <a href="{{ '/blueprint/' | relative_url }}">Guide</a> is the main reference.
      For setup, see <a href="{{ '/installation/' | relative_url }}">Installation</a>.
      Paper:
      <a href="https://arxiv.org/abs/2602.22631">arXiv:2602.22631</a>.
    </p>
  </div>
</div>

<div class="home-overview">
  <img
    src="{{ '/assets/media/figures/torchlean-layout.png' | relative_url }}"
    alt="TorchLean overview: typed tensors, shared graph IR, autograd, IEEE-754 semantics, certificate checking, PyTorch round trip, CUDA trust boundary, and Lean verification."
    loading="lazy" />
</div>
