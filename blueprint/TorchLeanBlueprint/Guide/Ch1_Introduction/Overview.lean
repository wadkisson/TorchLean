import VersoManual

open Verso.Genre Manual

#doc (Manual) "Getting Started" =>
%%%
tag := "overview"
%%%


Suppose we train a small AI model, save its parameters, run it on a GPU, and later claim that its
output stays in a safe range for every input near a test point. During that one workflow, the phrase
"the model" can refer to at least six different things:

1. the source-level architecture;
2. the initialized parameter tensors;
3. the mutable parameters after training;
4. an exported operation graph;
5. the CPU or accelerator program that actually runs;
6. the graph and parameter payload analyzed by a verifier.

Usually these objects agree. When they do not, the resulting bug can be remarkably quiet. A
checkpoint loader may transpose a weight matrix. A verification script may forget that the model
expects normalized inputs. A compiler may replace separate multiplication and addition with an FMA.
The program still runs; it simply computes a different function from the one we had in mind.

TorchLean was designed around that problem. It is a neural-network library, but it is also a place
where the architecture, parameter payload, graph, arithmetic, and property can be named in the same
language. The point is not to force every numerical kernel through Lean's evaluator. The point is
to stop losing the identity of the computation as it moves from mathematics to execution.

# A First Program

Application code uses the focused `NN.API` import:

```
import NN.API

open TorchLean

def model :
    nn.M (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

The model accepts a length-two tensor and returns a length-one prediction. The dimensions occur in the model's
type, so the output of a layer with width seven cannot be fed to a layer expecting width eight.
`nn.M` means that this is a seeded model builder. It describes initialization but has not yet chosen
the random seed or produced concrete parameter tensors.

We can initialize it directly:

```
def initialized :=
  nn.run 2026 model
```

or ask the trainer to initialize and execute it:

```
def trainer :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.001 }
      seed := 2026 }
```

The notation is intentionally familiar to a PyTorch user, but Lean learns more from the declaration.
It checks the input and output dimensions while elaborating the file. It distinguishes the seeded
builder from the initialized model. Once initialized, the model exposes the exact order and shape of
its parameter tensors. That information is available later to the trainer, graph compiler, and
verifier without rediscovering it from runtime metadata.

# Opening The Model Up

The same initialized model can participate in several parts of TorchLean.

## The model description

An `nn.Sequential` stores layer definitions, parameter shapes, initialization values, gradient
flags, and a forward program. This is the object used by the public trainer. Training creates an
effectful runtime runner whose parameters change over time; it does not rewrite the source
definition.

## The equations

The specification layer defines tensors as shape-indexed objects and gives operations such as
matrix multiplication, softmax, normalization, and loss functions explicit mathematical meanings.
Proofs use these definitions rather than reverse-engineering an opaque native buffer.

## The running program

The eager runtime owns autograd tape nodes, parameter and optimizer state, and optional accelerator
buffers. A backend profile decides which device and provider supply an operation. The checked CPU
profile, native CUDA profile, and LibTorch-forward profile are execution choices. Selecting one does
not, by itself, prove its kernels satisfy the mathematical specification.

## The operation graph

`NN.IR.Graph` is a directed acyclic graph of operation-tagged nodes. Each node has parent ids and a
declared output shape. Constants, weights, convolution parameters, and similar data live in a
separate payload store. This separation lets the graph structure be inspected without assuming that
that two different payloads are the same trained model.

## The object checked by a verifier

The public verification compiler lowers supported forward programs and a concrete parameter payload
to a `CompiledIR`. IBP and CROWN operate on that graph. Numerical certificates can replay
bit-level ranges and backend policies over it. Other checkers consume external artifacts such as
alpha-beta-CROWN leaves or PINN residual certificates.

These are not rival implementations. They are views of the same computation made for different
jobs. The model description is pleasant to write. The runtime is built to execute. The graph is
easy to inspect. The equations are suitable for proofs. TorchLean's architecture is largely the
collection of translations that lets those views meet.

# Shapes Catch The Bug Where It Starts

A TorchLean tensor has a scalar type and a shape:

$$`\operatorname{Tensor.T}\;\alpha\;s`.

For example,

```
def predictions : Tensor.T Float (shape![32, 1]) :=
  tensorOfList! [32, 1] (List.replicate 32 0.0)

def labels : Tensor.T Float (shape![32]) :=
  tensorOfList! [32] (List.replicate 32 0.0)
```

# Why Running The Model Is Not The Whole Story

Suppose a ten-class image classifier returns logits
`f_θ(x₀) ∈ ℝ¹⁰` and the largest entry is at index `3`. Then the ordinary prediction is class `3`
on that one image `x₀`. That answers a pointwise question: what did the network output for this
exact input?

A local robustness claim asks something stronger. Fix a radius `ε > 0` and consider every image
`x` whose pixel values stay within `ε` of `x₀` in the `∞`-norm (each coordinate may move by at most
`ε`). For the predicted class to be stable on that whole box, class `3` must keep the largest logit
at every such `x`: for every competing class `j ≠ 3`,

$$`f_\theta(x)_3 - f_\theta(x)_j > 0`.

In other words, the margin of class `3` over every rival must stay positive on the entire
neighborhood, not merely at the original photograph. The difference is visible in the quantifiers.
A prediction is one computation:

$$`f_\theta(x_0)=y`.

A local robustness statement concerns every point in a region:

$$`\forall x,\quad \lVert x-x_0\rVert_\infty\leq\varepsilon
  \Longrightarrow
  f_\theta(x)_y-f_\theta(x)_j>0
  \quad\text{for every }j\ne y`.

The first line is a calculation. The second is a theorem about an uncountable set. No amount of
random sampling changes that quantifier. A verifier needs a description of the region and a way to
bound the network everywhere inside it.

# A Mask With The Right Shape And The Wrong Meaning

Some of the most important mistakes are perfectly well typed. Attention masking is a good example.
For query `i`, let `Aᵢ` be the keys that are allowed to receive attention. A hard mask means

$$`
\operatorname{attention}_{ij}
=
\begin{cases}
\dfrac{\exp(s_{ij})}
      {\sum_{k\in A_i}\exp(s_{ik})}, & j\in A_i,\\[1.2ex]
0, & j\notin A_i.
\end{cases}
`

Blocked entries never enter the denominator and receive exactly zero weight. A common numerical
shortcut instead adds a large negative constant `-C` to a blocked logit before softmax:

$$`
\widetilde{\operatorname{attention}}_{ij}
=
\frac{\exp(s_{ij}-C)}
     {\sum_{k\in A_i}\exp(s_{ik})
       +\sum_{k\notin A_i}\exp(s_{ik}-C)}.
`

For ordinary logits and a large `C`, this value may underflow to zero in a particular floating-point
run. Mathematically, however, it is positive for every finite `C`. Worse, the shortcut is not safe
for arbitrary logits. If a blocked score is `C+100`, then subtracting `C` leaves the very large score
`100`; the supposedly blocked key can dominate the softmax.

Both implementations have the same tensor shapes. Both run. Tests with moderate random logits may
make them look identical. The bug is in the definition of masking.

TorchLean's specification uses the hard-mask equation: blocked entries have zero numerator. Runtime
providers must implement that meaning or advertise a different operation. This is a useful example
of the library's larger design. Types settle structural questions; specifications settle semantic
ones.

# Coordinates Are Part Of The Claim

Now suppose the model consumes normalized vectors:

$$`N(x)_i=\frac{x_i-\mu_i}{\sigma_i}`.

If the raw input lies in a box

# A First Walk Through The API

The shortest useful TorchLean import is:

```
import NN.API

open TorchLean
```

It gives application code five main places to begin:

- `Tensor` for shape-indexed tensor values and constructors;
- `nn` for layers, blocks, model families, and seeded initialization;
- `Data` for in-memory datasets, loaders, checkpoints, and text helpers;
- `optim` for optimizer configurations;
- `Trainer` for prediction, training, summaries, and the public verification bridge.

We will take those names in order and use each one once. Every command below runs from the
repository root.

# First Contact: Print A Few Tensors

Run:

```
lake exe torchlean quickstart_tensors
```

The checked-in example prints:

```
== Quickstart: tensor basics ==
[Float] [0.100000, 0.200000, 0.300000, 0.400000]
[ℚ] [1/10, 1/5, 3/10, 2/5]
[Int] [1, 2, 3, 4]
[IEEE32Exec] [0.100000, 0.200000, 0.300000, 0.400000]
[Float] [[[1.000000, 2.000000], [3.000000, 4.000000]],
         [[5.000000, 6.000000], [7.000000, 8.000000]]]
Expected failure printing Tensor ℝ: Refusing to print `Tensor ℝ` ...
```

The same tensor structure can carry several scalar types. `Float` is Lean's executable host
floating type. `ℚ` is exact rational arithmetic. `IEEE32Exec` is TorchLean's executable bit-level
binary32 model. `ℝ` is useful in specifications and proofs, but arbitrary real values do not have a
general executable printer.

Open
[`NN/Examples/Quickstart/TensorBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean)
and find the definitions of `xF`, `xQ`, and `x32`. The values look alike when printed, but their
types select different arithmetic.

# Build A Small File Of Your Own

Create `Tour.lean` at the repository root:

```
import NN.API

open TorchLean

def point : Tensor.T Float (shape![2]) :=
  tensorOfList! [2] [0.25, -0.75]

def model : nn.M (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def initialized :=
  nn.run 2026 model

#eval Tensor.pretty point
#eval IO.println (nn.info initialized)
```
