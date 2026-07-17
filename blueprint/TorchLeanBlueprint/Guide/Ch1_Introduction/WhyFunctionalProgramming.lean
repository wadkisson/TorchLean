import VersoManual

open Verso.Genre Manual

#doc (Manual) "Why Functional Programming?" =>
%%%
tag := "why_functional"
%%%

The reason TorchLean uses a functional style is not aesthetic. It is about making state visible.

In ordinary ML code, a forward pass may read parameters, update normalization buffers, consult a
random generator, depend on train/eval mode, and leave gradients in mutable fields. The convenience
is real, but it complicates the question a verifier eventually has to ask: which function did this
model compute?

Functional programming gives us a clean way to answer. A layer is a function from inputs and
parameters to outputs. A training step is a function from old parameters, data, gradients, optimizer
state, and random generator state to new values. Nothing essential is lost; the difference is that
every value that matters to a theorem has a name.

State does not disappear. It becomes an argument, a return value, or a named boundary. Once state is
visible, ordinary mathematical questions apply. Did this step use the old momentum or the new
momentum? Was this dropout mask sampled in
training mode or skipped in evaluation mode? Did this certificate check the payload before or after
an import conversion? A functional style gives those questions handles.

# The Problem with Mutable State

In an ML script, state is everywhere. BatchNorm has running statistics. Dropout depends on mode and
randomness. Optimizers carry momentum or Adam moments. Autoregressive models carry KV caches and
position counters. Tokenizers carry vocabularies and special-token conventions. Parameters can be
shared. These are normal features of practical systems.

The problem is that a theorem cannot reason about state that never appears in the object being
checked. In many frameworks, a model is an object with mutable fields. A call that looks like an
ordinary forward pass may update a buffer, consult a hidden random generator, write to gradient
storage, or depend on whether some parameter tensor is shared with another module. That style is
convenient for experimentation, but it makes the mathematical question less direct: which function
did the network compute?

A Python-style sketch makes the issue concrete:

```
class Layer:
    def __call__(self, x):
        self.calls += 1          # hidden state change
        self.running_mean *= 0.9 # another hidden state change
        return self.weight * x + self.bias
```

The return value is incomplete evidence. The next call can behave differently because this call
changed the object. If a proof, exporter, or verifier ignores those mutations, it may reason about a
different computation from the one that actually ran.

The same issue appears in more realistic layers. BatchNorm is not merely an affine expression; in
training mode it also updates running estimates, while in evaluation mode it reads stored estimates.
A useful specification must say which mode is active and which statistics are used. In a functional
presentation, that distinction can be made explicit:

```
inductive Mode where
  | train
  | eval

structure BatchNormState where
  runningMean : Float
  runningVar : Float

def batchNormSketch
    (mode : Mode) (state : BatchNormState) (x gamma beta : Float) :
    Float × BatchNormState :=
  match mode with
  | .eval =>
      let y := gamma * (x - state.runningMean) + beta
      (y, state)
  | .train =>
      let newState :=
        { runningMean := 0.9 * state.runningMean + 0.1 * x
          runningVar := state.runningVar }
      let y := gamma * x + beta
      (y, newState)
```

This is only the state-changing core of BatchNorm, not a production definition. The important part
is the interface: the state that changes is returned. A theorem about evaluation mode can
quantify over `state` without pretending a hidden object field stayed fixed.

# Pure Functions are Mathematical Functions

In a pure functional language such as Lean, ordinary functions have no side effects. A TorchLean
layer takes explicit inputs, including its parameters, and returns an explicit output. The simplest
version is affine arithmetic:

```
structure Affine1D where
  w : Float
  b : Float

def affine1D (p : Affine1D) (x : Float) : Float :=
  p.w * x + p.b
```

Here the mathematical reading and the executable reading coincide: `affine1D p x` computes
`p.w * x + p.b`. There is no hidden `.grad` field, no object identity, and no accidental parameter
mutation that a theorem has to account for later.

The same idea scales to tensors. In TorchLean, a layer is still read as

$$`\operatorname{forward}(\theta, x) = y`

where `θ` is the parameter payload, `x` is the input tensor, and `y` is the output tensor. The
values now carry tensor shapes, scalar semantics, and graph structure as needed. We can prove facts
about the same definitions that examples and checkers inspect.

# Explicit Effects And Explicit Randomness

Randomness is another place where ML code often hides state. A dropout layer is not a mathematical
function of its input alone during training; it also depends on a mask, seed, or generator state.
TorchLean's discipline is to represent that dependency instead of burying it inside an object.

```
structure DropoutSketchState where
  seed : Nat

def dropoutSketch
    (mode : Mode) (state : DropoutSketchState) (x : Float) :
    Float × DropoutSketchState :=
  match mode with
  | .eval => (x, state)
  | .train =>
      let keep := state.seed % 2 == 0
      let x' := if keep then 2.0 * x else 0.0
      (x', { seed := state.seed + 1 })
```

Again, the sketch is not a full RNG. It records the contract: training consumes state and returns
new state; evaluation does not sample a mask. This is the difference between an executable recipe
and a proof-ready object.

# Training Still Changes Things

Functional programming does not mean that training is static. It means that change is represented by
new values instead of silent updates to existing ones.

```
def sgdStep (eta gradW gradB : Float) (p : Affine1D) : Affine1D :=
  { w := p.w - eta * gradW
    b := p.b - eta * gradB }
```

The step is still an update, but now the update has a type and a result. An optimizer step takes an
old parameter bundle and returns a new parameter bundle. A logger takes an old log state and returns
an updated log state. A random generator takes an old seed or generator state and returns the next
one. The training loop remains inspectable because state changes appear at the places where the
program says they happen.

This makes optimizer claims more precise. A statement about SGD can talk about one update:

$$`\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)`

where `θ_t` is the parameter payload at step `t`, `η` is the learning rate, `L` is the loss, and
`∇L(θ_t)` is the gradient of that loss at the current parameters. A statement about Adam needs the
first-moment state, second-moment state, step counter, and bias-correction convention. If those
fields are implicit, a theorem about "the optimizer" is already underspecified. If they are data,
the statement can choose exactly the update rule it means.

This also clarifies trust boundaries. If a CUDA kernel, a PyTorch exporter, or an external
certificate producer contributes a value, TorchLean can name that imported value and state what is
assumed about it. The proof does not have to pretend that an external side effect was a Lean
definition.

# Reference Counting And Practical Execution

The usual worry about pure code is that it allocates too much. Lean 4 uses deterministic reference
counting, and values with a unique owner can often be updated in place under the hood. That means
the functional style does not require the runtime to behave naively.

For TorchLean, this matters because tensor code needs both a clean semantics and a realistic path to
performance. We can write programs as transformations of values, while the runtime is still allowed
to perform safe buffer reuse when uniqueness makes it possible.

# Where Purity Stops

Pure definitions are not the whole program. Loading a checkpoint, opening a socket to a Python
process, invoking a CUDA kernel, writing a JSON log, or calling Arb for interval arithmetic are
effects. Lean marks ordinary effects in the type with `IO`, and TorchLean uses documented runtime
and FFI boundaries for native or external systems.

That distinction is part of the trust story. A pure Lean definition can be unfolded in a theorem. An
`IO` action can be run, tested, and wrapped by a checker, but its external behavior is not proved
merely because the call is written in Lean syntax. "This action produced an artifact" is distinct
from "this theorem says accepted artifacts imply a semantic property."

# Related Design Ideas

The same design principle appears throughout formal methods: keep executable code, specifications,
and proof obligations close enough that they stay aligned. TorchLean applies that principle to
neural networks. State is data. Shapes are part of the interface. Semantics are named. Proof
artifacts are built around the same definitions that model examples use.

Functional programming matters here because it turns hidden context into data. Parameters, optimizer
state, random seeds, logs, masks, imported artifacts, and certificate payloads can all be passed,
saved, inspected, and mentioned in theorem statements. TorchLean does not remove state from ML. It
makes the state part of the computation we can talk about.

## References

- Lean 4 documentation: https://lean-lang.org/doc/reference/latest/
- Ullrich and de Moura, "Counting Immutable Beans: Reference Counting Optimized for Purely
  Functional Programming", IFL 2019. https://arxiv.org/abs/1908.05647
- George et al., "BRIDGE: Building Representations In Domain Guided Program Synthesis",
  arXiv:2511.21104.
