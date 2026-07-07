import VersoManual

open Verso.Genre Manual

#doc (Manual) "Scientific Forward Models: Transcendentals and Affine Maps" =>
%%%
tag := "scientific-forward-models"
%%%

Many scientific models look less like a stack of neural network layers and more like an equation.
Radiative transfer, dielectric mixing, kinetics, and remote sensing models often combine affine
terms with `exp` or `log`. If those operations are missing from the functional API, users end up
leaving the model as external code, or they write a forward pass in one place and a hand derivative
in another.

TorchLean now includes the scalar building blocks these models need: `exp`, `log`, scaling,
shifting, and affine maps. A forward equation can be written once as an `autograd.func.Fn`, and the
runtime differentiates that same expression. The project no longer has to maintain one formula for
prediction and a second formula for gradients, waiting for a sign or scale factor to drift.

# Functional Operations

The operations live in the same functional namespace as `square` and `mean`. Each one wraps a
primitive with a registered backward rule, so reverse mode can pass through the expression:

- `nn.functional.exp`: elementwise `eˣ`, analogous to `torch.exp`;
- `nn.functional.log`: elementwise `ln x`, analogous to `torch.log`;
- `nn.functional.scale x c`: multiply by a scalar, `c · x`;
- `nn.functional.shift x c`: add a scalar, `x + c`;
- `nn.functional.affine x c k`: compute `c · x + k`.

For real valued reasoning about `log`, the intended domain is positive input. On floating point
backends, nonpositive input follows the backend behavior. Models that need a total log like
operation should use an epsilon protected form such as `safeLog`.

In the source tree, the implementation starts in
[NN/Runtime/Autograd/TorchLean/Functional/Core.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Functional/Core.lean).
As the public facade grows, beginner examples should prefer the `nn.functional.*` spelling; runtime
examples may still import the lower functional layer directly.

Here is a compact Lean shape for a one-dimensional forward equation:

```
def attenuated : autograd.func.Fn Shape.scalar Shape.scalar :=
  fun ndvi => do
    let neg ← nn.functional.scale ndvi (-2.0)
    let e ← nn.functional.exp neg
    nn.functional.affine e 0.8 0.1
```

Read it as:

$$`f(x) = 0.8 \cdot e^{-2x} + 0.1`

The equation is not hidden in a callback. It is the same expression the runtime differentiates.

The same style scales to vector inputs when the first step is to split or project features:

```
-- Sketch: x contains [ndvi, roughness, incidence].
def retrievalTerm : autograd.func.Fn (Shape.vec 3) Shape.scalar :=
  fun x => do
    let ndvi ← nn.functional.index1d x 0
    let rough ← nn.functional.index1d x 1
    let v ← nn.functional.scale ndvi (-2.0)
    let att ← nn.functional.exp v
    let scatter ← nn.functional.affine rough 0.4 0.05
    nn.functional.mul att scatter
```

When a snippet uses an operation not available in the current functional API, treat it as a sketch
and look for the corresponding example file. The design point is still the same: write the forward
equation once, then ask autograd for the derivative of that expression.

# Why The Gradient Should Come From The Model

The motivating case is a soil moisture retrieval model. Its surface term relates radar backscatter
to vegetation and dielectric response through an expression of the form

```
σ⁰ = a · NDVI + exp(-2 · b · NDVI) · c · |R|² + d
```

In a conventional implementation, that expression is often paired with a hand written analytic
Jacobian. If the Jacobian has a sign error or a missing factor, the program can still run. It may
only fit worse, or it may fail in a way that looks like noisy data. If the codebase also keeps a
separate compiled or JIT version of the derivative, the two formulas can drift apart.

With the functional operations above, the surface term is written once. Autograd differentiates the
same expression that produced the forward value. That removes a fragile transcription step and gives
the project a single object to inspect: the forward model.

# Executable gradient check (positive and negative controls)

The example `NN.Examples.Functional.Transcendentals` differentiates three small functions and
compares the autograd gradient to the closed form. Run it with `lake exe transcendentals_check`.

This example is executable evidence: users can run the code and see the gradients line up with
closed forms. The theorem backed autograd layer is separate and stronger. For supported graph and
tape fragments, TorchLean proves the reverse pass against the mathematical derivative. The
[Autograd Proofs](Verification-and-Certificates/Autograd-Proofs/) chapter points to the exact
declarations, including `Graph.backprop_correct`, the Fréchet-derivative bridge
`Graph.backpropVec_eq_adjoint_fderiv`, local `NodeFDerivCorrect` facts for supported primitive
nodes, and scalar-loss training algebra in `Graph.scalarLoss_grad_correct`.

```
def expNegativeTwoFn : autograd.func.Fn Spec.Shape.scalar Spec.Shape.scalar :=
  fun x => do
    let u ← nn.functional.scale x (-Numbers.two)   -- -2 · x
    nn.functional.exp u                              -- e^{-2x}
```

At `x = 0.5`, the check has two parts:

- the positive control checks that autograd gives `-2 · e^{-2x} = -0.735759`;
- the negative control checks that it does not give the wrong sign, `+2 · e^{-2x} = +0.735759`.

The negative control shows the example in miniature. A sign error in a handwritten derivative is
exactly the kind of defect that disappears when the gradient is derived from the forward model. The
companion checks for `exp` and the affine map cover the other new operations, and the executable
exits with a nonzero status on any regression.

# Domain Conditions Are Part Of The Model

Scientific formulas often contain partial functions. `log x` is the simplest example: the real
mathematical function has a positive-domain precondition. A runtime float backend may produce
`NaN`, `-inf`, or another backend-defined result outside that domain, but that is not the same as a
mathematical theorem about real logarithm.

TorchLean examples should make the choice explicit:

```
-- Mathematical model with a domain condition:
--   requires x > 0
let y ← nn.functional.log x

-- Total runtime model with an epsilon guard:
let xSafe ← nn.functional.shift x 1e-6
let y ← nn.functional.log xSafe
```

The first is appropriate when the theorem or data contract supplies positivity. The second is
appropriate when the executable model needs a total operation over noisy inputs. They are different
models and should not be silently swapped.

# What The Check Shows

The transcendental example is a runtime regression test. It checks that the implemented backward
rules for the selected scalar backend agree with closed-form calculations on chosen inputs. The
proof layer is stronger but narrower: it proves statements about the formal semantics and supported
primitive rules named in the theorem. Both are useful, and both should be cited precisely.

# References

- PyTorch `torch.exp`: https://docs.pytorch.org/docs/stable/generated/torch.exp.html
- PyTorch `torch.log`: https://docs.pytorch.org/docs/stable/generated/torch.log.html
- Baydin et al., automatic differentiation survey: https://arxiv.org/abs/1502.05767
