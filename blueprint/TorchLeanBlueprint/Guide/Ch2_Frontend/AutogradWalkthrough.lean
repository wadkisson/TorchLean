import VersoManual

open Verso.Genre Manual

#doc (Manual) "Autograd Walkthrough" =>
%%%
tag := "autograd-walkthrough"
%%%

Most ML users first meet automatic differentiation through `loss.backward()`. TorchLean keeps that
workflow, but it does not hide the objects involved. A gradient call names the function being
differentiated, the input shape, the output shape, and, for vector outputs, the cotangent seed that
chooses which linear combination of outputs to differentiate.

The objects to watch are:

- the function or model being differentiated,
- the input and parameter shapes,
- the seed for a vector-Jacobian product when one is needed,
- the gradient tensors returned by the runtime.

# A Scalar Function

The smallest case is a scalar valued tensor function.

```
import NN.API

open TorchLean

def sumsq : autograd.func.Fn (.dim 2 .scalar) Shape.scalar :=
  fun x => do
    let y ← nn.functional.square x
    nn.functional.mean y

def example : IO Unit := do
  let x : Tensor Float (.dim 2 .scalar) := tensorOfList! [2] [0.5, -1.2]
  let g ← autograd.func.grad (α := Float) sumsq x
  IO.println s!"grad = {Spec.pretty g}"
```

The shape tells us that `sumsq` consumes a vector of length two and returns a scalar. Because the
output is scalar, `autograd.func.grad` returns a tensor with the same shape as the input.

For `x = (x_1, x_2)`, the function is:

$$`\operatorname{mean}(x^2) = (x_1^2 + x_2^2)/2`

So the mathematical gradient is:

$$`\nabla_x \operatorname{mean}(x^2) = x`

so the returned gradient has the same two coordinates as `x`.

The example is a runtime computation. It is not a theorem about every possible scalar backend. The
mathematical calculation explains what should happen, and the executable call checks the selected
runtime implementation on this input.

# Value And Gradient Together

For debugging, compute the value and gradient in one call:

```
let (value, grad) ← autograd.func.valueAndGrad (α := Float) sumsq x
```

This mirrors the common workflow:

```
y = f(x)
dy_dx = grad(y, x)
```

TorchLean keeps both results ordinary values. There is no hidden `.grad` field that must be cleared
before the next step.

A common debugging pattern is to print both:

```
def debugGrad : IO Unit := do
  let x : Tensor Float (.dim 2 .scalar) := tensorOfList! [2] [0.5, -1.2]
  let (value, grad) ← autograd.func.valueAndGradScalar (α := Float) sumsq x
  IO.println s!"value = {value}"
  IO.println s!"grad = {Tensor.pretty grad}"
```

Use this when the function is small enough that a single value and gradient tell you what is going
on. Use the trainer when the function is really a model, a dataset, and an optimizer loop.

# Vector Outputs Need A Seed

For scalar outputs, there is a single gradient. For vector outputs, there is not. If
`f(x) = [f_1(x), f_2(x)]`, then “the gradient of `f`” could mean the gradient of `f_1`, the
gradient of `f_2`, their sum, or some weighted combination. A vector-Jacobian product supplies that
choice.

```
let dx ← autograd.func.vjp (α := Float) f x seedOut
```

Informally:

$$`\operatorname{vjp}(f,x,\bar y) = J_f(x)^\mathsf{T}\bar y`

Here `seedOut` is the output cotangent. It has the same shape as the output of `f`, and the returned
tensor has the same shape as `x`.

Reverse mode needs exactly this operation. Each local operation sends an output cotangent back to
input cotangents; composing those local rules gives the full gradient.

Here the shape discipline pays off: the derivative type tells you what kind of object to expect
back.

A tiny example is a diagonal scaling function:

```
def scale2 : autograd.func.Fn (.dim 2 .scalar) (.dim 2 .scalar) :=
  fun x => do
    let twoX ← nn.functional.scale x 2.0
    pure twoX

def vjpExample : IO Unit := do
  let x : Tensor Float (.dim 2 .scalar) := tensorOfList! [2] [3.0, 4.0]
  let seed : Tensor Float (.dim 2 .scalar) := tensorOfList! [2] [1.0, 10.0]
  let dx ← autograd.func.vjp (α := Float) scale2 x seed
  IO.println s!"dx = {Tensor.pretty dx}"
```

The seed says which output cotangent is flowing backward. For this function, the returned input
cotangent is twice the seed. In a larger graph, the same idea is applied node by node.

# Model Parameters

Training usually differentiates a loss with respect to parameters. The model API uses the same
reverse mode idea, but the returned gradient has the same parameter structure as the model.

The usual calls are:

- `autograd.model.gradParams` for gradients of the loss with respect to parameters,
- `autograd.model.gradInputs` for gradients with respect to input and target tensors,
- `autograd.model.valueAndGradParams` when the loss value and parameter gradients are both needed.

Those declarations live in [NN.API.Public](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean), while the runtime
implementation lives in [NN.API.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Runtime.lean) and
[NN.Runtime.Autograd.TorchLean.Autodiff](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Autodiff.lean).

The shape of a model-parameter call is:

```
let (lossValue, dparams) ←
  autograd.model.valueAndGradParamsScalar
    (α := Float) model mseLoss params x target
```

Read it carefully:

- `model` names the checked architecture;
- `mseLoss` turns prediction and target into a scalar loss;
- `params` is the current parameter bundle;
- `x` and `target` are the current input and label tensors;
- `dparams` has the same parameter structure as `params`.

This is the public analogue of PyTorch's `loss.backward()`, but the gradient bundle is returned as a
value. The optimizer step consumes `params` and `dparams` instead of looking for mutable `.grad`
fields on tensors.

# Jacobians, JVPs, And HVPs

TorchLean also exposes analysis tools beyond the basic training gradient:

- `autograd.func.jacfwd` for a forward mode Jacobian,
- `autograd.func.jacrev` for a reverse mode Jacobian,
- `autograd.func.hessian` for scalar valued functions,
- `autograd.model.jvpParams` for a directional derivative of a scalar loss,
- `autograd.model.hvpParams` for Hessian vector products over parameters.

Read the APIs this way:

- `grad` returns a scalar-output gradient, the common case for losses.
- `vjp` returns `J_f(x)^T seedOut`, the reverse-mode object for vector outputs.
- `jacfwd` works well when the input dimension is small.
- `jacrev` works well when the output dimension is small.
- `hvpParams` gives a Hessian-vector product for curvature diagnostics.

These tools appear again in chapters on sensitivity, curvature, verification, and second order
diagnostics. They also show why typed shapes matter: every returned object has the shape dictated by
the derivative it represents.

A Hessian-vector product is a good example of why TorchLean keeps parameter structure visible. The
vector is not a flat anonymous buffer; it has the same parameter tree as the model:

```
let hv ← autograd.model.hvpParams
  (α := Float) model mseLoss params x target vparams
```

Here `vparams` is the direction in parameter space, and `hv` is the Hessian applied to that
direction. Curvature diagnostics, influence-style calculations, and some optimizers want exactly
that object.

# From Local Rules To Global Backprop

The proof story for autograd starts from a familiar calculation.

1. Each primitive operation has a forward meaning.
2. Each primitive operation has a local VJP or JVP rule.
3. If those local rules match the derivative of the forward operation, then the composed reverse
   pass computes the adjoint derivative of the whole graph.

At the user level, that is the picture. The proof chapters point to the Lean theorems that make it
precise: supported primitive rules are linked to their forward meanings, and the tape theorem turns
those local facts into a statement about the whole reverse pass.

The main proof trail starts here:

- [NN.Proofs.Autograd.Overview API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Overview.lean)
- [NN.Proofs.Autograd.Tape.Algebra.Soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean)
- [NN.Proofs.Autograd.Runtime.Link API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Runtime/Link.lean)

# A Practical Rule

Use the smallest API that matches the question.

- For a gradient of `Tensor σ -> scalar`, use `autograd.func.grad`.
- For a value and gradient together, use `autograd.func.valueAndGrad`.
- For a vector output with a chosen cotangent, use `autograd.func.vjp`.
- For a model loss gradient with respect to parameters, use `autograd.model.gradParams`.
- For minibatch training, use `Trainer.Config`, `Trainer.TrainOptions`, `trainer.train`, and the
  quickstart examples.

For a runnable tour, open
[NN.Examples.Quickstart.AutogradBasics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean). It prints
gradients, VJPs, Jacobians, Hessian vector products, and parameter gradients in one place.

# References

- [PyTorch's autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
  is the closest conceptual comparison.
- [PyTorch's autograd reference](https://docs.pytorch.org/docs/stable/autograd.html)
  documents the corresponding Python API.
- For the mathematical background, see Baydin et al.,
  "Automatic differentiation in machine learning: a survey": https://arxiv.org/abs/1502.05767
