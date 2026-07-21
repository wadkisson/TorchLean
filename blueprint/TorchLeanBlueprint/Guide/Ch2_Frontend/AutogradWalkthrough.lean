import VersoManual

open Verso.Genre Manual

#doc (Manual) "Autograd And Runtime" =>
%%%
tag := "autograd-walkthrough"
%%%


Reverse-mode automatic differentiation is usually introduced as “call backward and read the
gradients.” TorchLean makes the derivative object explicit: a function, input, and output
cotangent determine a vector-Jacobian product. Parameter gradients are returned in a pack whose
shapes match the model's parameters.

We will begin with a two-coordinate function, then differentiate a linear model, stop a gradient
with `detach`, and finish with Jacobian and Hessian products.

# Run The Complete Tour

```
lake exe torchlean quickstart_autograd
```

The run prints several derivative views. The final portion is:

```
jacfwdInput(square) cols = 2
  col[0] = [1.000000, -0.000000]
  col[1] = [0.000000, -2.400000]

hessianInput(mean(x^2)) cols = 2
  H*e[0] = [1.000000, 0.000000]
  H*e[1] = [0.000000, 1.000000]

vjp(square, seed=ones) = [1.000000, -2.400000]
grad1(mean(x^2)) = [0.500000, -1.200000]
valueAndGradScalar(mean(x^2))
  value = 0.845000
  grad = [0.500000, -1.200000]
```

We will derive each value below. The source is
[`AutogradBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean).

# A Scalar Function

Define:

$$`
f(x_0,x_1)=\frac{x_0^2+x_1^2}{2}.
`

In the public functional API:

```
import NN.API
open TorchLean

def sumsq :
    autograd.func.Fn (shape![2]) Shape.scalar :=
  fun x => do
    let y ← nn.functional.square x
    nn.functional.mean y
```

The type states that `sumsq` maps a two-vector to a scalar. The `do` block builds a checked tensor
program; it is not using a special untyped scalar expression language.

The mathematical gradient is:

$$`
\nabla f(x_0,x_1)=(x_0,x_1).
`

At `x=(0.5,-1.2)`:

$$`
f(x)=\frac{0.25+1.44}{2}=0.845,
\qquad
\nabla f(x)=(0.5,-1.2).
`

The executable call is:

```
def example : IO Unit := do
  let x : Tensor.T Float (shape![2]) :=
    tensor! [0.5, -1.2]
  let (value, grad) ←
    autograd.func.valueAndGradScalar
      (alpha := Float) sumsq x
  IO.println s!"value={value}, grad={Tensor.pretty grad}"
```

The observed values match the hand calculation. That is a useful executable check. A theorem that
the autograd transform is correct for every input requires the semantic proof layer described
later; one agreeing sample is not that theorem.

# Why A VJP Is The Primitive Reverse Operation

Let:

$$`f:\mathbb R^n\to\mathbb R^m`

with Jacobian:

$$`J_f(x)\in\mathbb R^{m\times n}`.

Reverse mode accepts an output cotangent `ȳ ∈ ℝ^m` and returns:

$$`\bar x=J_f(x)^\mathsf{T}\bar y`.

TorchLean writes:

```
let dx ←
  autograd.func.vjp
    (alpha := Float)
    f x seedOut
```

The types enforce:

```
x       : Tensor α inputShape
seedOut : Tensor α outputShape
dx      : Tensor α inputShape
```

# What Actually Runs

The previous chapter used one public call:

```
autograd.model.valueAndGradParamsScalar ...
```

Behind that call, TorchLean may construct a tape, replay a compiled derivative graph, invoke native
CUDA kernels, and return a dependent gradient pack. Other parts of the repository also use explicit
IR graphs and backend execution plans. These objects are related, but they are not interchangeable.

This chapter identifies each artifact, its lifetime, and the claim it can support.

# Four Objects Commonly Called “The Graph”

| Object | Purpose | Contains |
| --- | --- | --- |
| eager tape | reverse-mode execution | values, parents, local VJPs |
| compiled derivative graph | repeated runtime execution | forward, JVP, VJP closures |
| `NN.IR.Graph` | inspection and verification | explicit operation tags and payload references |
| backend execution plan | provider selection | accepted capsules and audit metadata |

A fifth object, a CUDA Graph capture, is a device launch-replay mechanism. Selecting TorchLean's
`.compiled` backend does not mean CUDA Graph capture.

Confusing these artifacts leads to bad guarantees. For example, accepting a backend plan does not
prove that the compiled trainer executed it, and proving an IR semantics theorem does not certify a
native tape node whose provider was never related to that IR operation.

# Eager Execution

Run a short eager training job:

```
lake exe torchlean quickstart_mlp \
  --device cpu --backend eager --steps 2 --seed 2026
```

An eager session is created for the chosen scalar and profile. As the model runs, each operation:

1. reads one or more parent values;
2. computes and stores its output;
3. records parent identifiers;
4. records a local reverse rule.

For:

$$`y=\operatorname{ReLU}(Wx+b)`,

the tape has operations corresponding to matrix multiplication, bias addition, and ReLU. The loss
adds subtraction, squaring, and reduction nodes. Reverse traversal begins from cotangent one at the
scalar loss.

CPU eager values and VJPs live on the ordinary tape. CUDA eager values are device buffers and their
reverse actions live on the CUDA tape. Both obey the same high-level reverse traversal idea, but
their storage and primitive providers differ.

# Gradient Accumulation Is Part Of The Tape Semantics

Consider:

$$`z=x^2+x^2`.

The graph contains two paths from `x` to `z`. Each square contributes `2x`; the addition sends the
output seed to both parents. The final cotangent is:

$$`\bar x=2x+2x=4x`.

The runtime must add contributions associated with the same parent identifier. A correct local VJP
for square is insufficient if the tape traversal overwrites one contribution.

This is why TorchLean's autograd proofs have two layers:

- primitive derivative facts;
- global tape/traversal soundness.
