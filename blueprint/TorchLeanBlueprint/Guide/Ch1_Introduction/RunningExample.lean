import VersoManual

open Verso.Genre Manual

#doc (Manual) "The Running Example" =>
%%%
tag := "running-example"
%%%


Our running example is the checked-in `quickstart_mlp` program. It is deliberately small, but it is
not pseudocode: it builds a dataset, initializes a model, runs autograd and Adam, prints predictions,
and can execute through the same runtime interfaces used by larger examples.

The task is to learn the piecewise-linear function

$$`y(x_1,x_2)
  =0.8\,\operatorname{ReLU}(x_1+x_2)
   -0.4\,\operatorname{ReLU}(x_2-x_1)+0.2`

on a grid in `[-1,1]^2`. This target is a useful first case for three reasons. It is nonlinear, so a
single affine layer cannot solve it. It is built from ReLUs, so a small ReLU network can represent it
without approximation-theory distractions. Finally, its two-dimensional input lets us inspect every
shape and parameter without pages of indices.

# The Source Program

The model is a two-layer MLP:

```
import NN.API

open TorchLean

def inDim : Nat := 2
def outDim : Nat := 1

def model :
    nn.M (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar)) :=
  nn.Sequential![
    nn.linear inDim 8,
    nn.relu,
    nn.linear 8 outDim
  ]
```

Read the type before the body. `nn.M` says that model construction consumes a deterministic seed
stream. `nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)` says that the initialized model will map a
length-two tensor to a length-one tensor. The hidden width `8` is checked through composition: the
first linear layer produces length eight, ReLU preserves the shape, and the second linear layer
expects length eight.

The model returns a one-element tensor rather than a scalar because the public layer API treats the
output feature dimension uniformly. A dataset target must therefore have shape `[1]`, not shape `[]`
and not shape `[batch]`.

# Why The Builder Is Seeded

Linear layers need initial weights and biases. Hiding randomness in a global generator would make a
model definition depend on ambient state. TorchLean instead represents initialization as a pure
state computation:

```
def initialized :=
  nn.run 2026 model
```

Running the same builder with the same seed produces the same initialization stream. The resulting
`nn.Sequential` contains four parameter tensors in layer order:

1. first-layer weights of shape `[8, 2]`;
2. first-layer bias of shape `[8]`;
3. second-layer weights of shape `[1, 8]`;
4. second-layer bias of shape `[1]`.

We can ask Lean for that layout:

```
#check nn.paramShapes initialized
#check nn.initParams initialized
```

The order is part of the forward-program interface. It is not a PyTorch-style dictionary of names.
Checkpoint adapters may use names at an external boundary, but they must eventually construct this
typed ordered payload.

# The Dataset

The target in the source example is:

```
def target (x1 x2 : Float) : Float :=
  let relu (x : Float) := if x < 0.0 then 0.0 else x
  (0.8 * relu (x1 + x2)) -
    (0.4 * relu (x2 - x1)) + 0.2
```

`Data.regressionGrid (-1.0) 1.0 5 target` samples five positions on each input axis, giving 25
examples. Each input has shape `[2]` and each target has shape `[1]`:

```
def dataset : Trainer.Dataset (.dim 2 .scalar) (.dim 1 .scalar) :=
  Data.regressionGrid (-1.0) 1.0 5 target
```

The tensor types settle the dimensional question: every row fits the model. Whether 25 points are
enough to learn the target is a different question, and this small run gives us a concrete place to
start asking it.

# Training It

The public trainer combines the model, task, optimizer, seed, and runtime choices:

```
def trainer :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      seed := 2026 }
```

# TorchLean and PyTorch

TorchLean looks familiar on purpose. It has tensors, modules, parameters, autograd, optimizers,
devices, and checkpoints because those ideas already work well in modern ML. But TorchLean is not
PyTorch rewritten in Lean, and it is not trying to catch PyTorch by accumulating the same number of
operators.

PyTorch's center of gravity is execution. A Python program can assemble a large model dynamically,
train it with highly tuned kernels, distribute the work over many devices, and deploy the result.
TorchLean's center of gravity is the connection between a running model and a mathematical
statement. It can execute models too, but it keeps asking questions that PyTorch normally leaves to
the surrounding project: What is the exact shape contract? Which parameter payload did the verifier
read? What equation does this graph node denote? Which arithmetic appears in the theorem?

That difference is easier to see in code, so we will build the same MLP in both
systems and follow it through initialization, autograd, compilation, and execution.

# The Same MLP In Both Systems

A PyTorch model commonly owns its parameters through `nn.Module`:

```
# Python / PyTorch
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
)

logits = model(x)
```

The corresponding TorchLean builder is:

```
import NN.API

open TorchLean

def model :
    nn.M (nn.Sequential (.dim 4 .scalar) (.dim 2 .scalar)) :=
  nn.Sequential![
    nn.linear 4 8,
    nn.relu,
    nn.linear 8 2
  ]
```

Both programs describe an affine map, ReLU, and another affine map. Their surrounding contracts are
different.

In PyTorch, an eager tensor carries its shape as runtime metadata. Calling a layer inspects those
dimensions while the program executes. PyTorch's export and compilation systems can add symbolic
shape constraints later, but an ordinary annotation such as `torch.Tensor` does not distinguish a
vector of length four from a matrix with four columns.

In TorchLean, the input and output shapes index the `nn.Sequential` type. Layer composition is
checked while Lean elaborates the definition. `nn.M` also records that `model` is a deterministic
seed-state computation waiting to initialize its parameters.

Dynamic shapes are convenient during exploration and for data-dependent programs. Shape-indexed
types require more information up front, but they make layer composition and later theorem
statements much cleaner. TorchLean still accepts runtime-loaded data; it checks the dimensions once
at the boundary and then works with the resulting typed tensor.

# Parameters: Object Fields Versus Explicit Payloads

A PyTorch `nn.Module` registers parameter objects. Calling `model(x)` reads the module's current
fields. An optimizer mutates those parameters, usually through gradient fields populated by
autograd.

TorchLean's public model description contains parameter shapes, initialization tensors, gradient
flags, and a forward program. The forward program receives the *live parameter payload* explicitly.
Initialization and execution are therefore related but distinguishable:

```
def initialized :=
  nn.run 2026 model

#check nn.paramShapes initialized
#check nn.initParams initialized
#check nn.forwardProgram (model := initialized)
```
