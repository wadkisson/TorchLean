import VersoManual

open Verso.Genre Manual

#doc (Manual) "Theory Map" =>
%%%
tag := "theory-survey"
%%%

TorchLean's theory modules sit beside the executable library. They are not a second product; they
are the claims you can currently state and, in some cases, prove about learning, optimization,
approximation, and scientific ML. This chapter is a map, not a textbook.

# What Is Proved Versus What Is Named

A useful distinction:

- _proved in Lean_: a theorem with an explicit hypothesis list and a proof term;
- _named scaffolding_: definitions and lemmas that fix notation so a later proof can attach;
- _executable check_: a program that rejects bad artifacts, without claiming completeness.

When reading a theory file, ask which of those three you are looking at before treating the
result as a guarantee about a trained model.

# Learning And Optimization

Start with:

- [`NN/MLTheory`](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory) for learning-theoretic
  vocabulary (generalization sketches, objective structure, self-supervised bookkeeping);
- optimizer contracts under the same tree for step rules and invariants that training code is
  expected to respect.

These modules make the shape of a claim precise. They do not, by themselves, certify that a
particular GPU run minimized a population risk.

# Approximation And Classical Models

Approximation theory and classical dynamical models (for example Hopfield-style updates) live under
`NN/MLTheory` and related proof directories. Use them when you need a finite approximant, a
contraction, or an energy argument—not as a substitute for end-to-end robustness certificates.

# Probability, Scientific ML, And Factorizations

Diffusion-style kernels, scientific forward models, and linear-algebra factorizations
(Cholesky / QR) each have dedicated modules. Treat them as specialized tools:

- probability pages fix Markov-kernel vocabulary for generative steps;
- scientific ML pages connect PDE / operator models to the graph and certificate workflow;
- factorization pages expose numerical linear algebra as checked Lean objects rather than opaque
  library calls.

If your project does not need that specialty, skip the module. The main verification path in the
previous chapters does not depend on reading every theory file first.

# Where To Go Next

For a property of a concrete network, return to the verification and certificate chapters. For a
property of an algorithm or objective in the abstract, open the matching `NN/MLTheory` file and
read the theorem statements before the commentary.
