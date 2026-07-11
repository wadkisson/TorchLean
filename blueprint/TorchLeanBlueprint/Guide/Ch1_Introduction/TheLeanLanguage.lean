import VersoManual

open Verso.Genre Manual

#doc (Manual) "Lean as the Host Language" =>
%%%
tag := "lean_ecosystem"
%%%

Lean matters for TorchLean because it can play two roles at once. It is a theorem prover, so it can
check proofs about tensors, graphs, derivatives, bounds, and certificates. It is also a programming
language, so it can run examples, parse artifacts, call command-line tools, call native libraries
through FFI, and build executables.

That combination is the reason TorchLean can be more than a paper formalization. The same
repository can contain model code, graph semantics, executable checkers, CLI tools, and theorem
statements. Ordinary definitions remain pure, while effects such as printing, file access,
randomness, or native calls are marked in the type, usually through `IO`.

Deep-learning tooling often splits one artifact into several languages: Python for models, C++ or
CUDA for kernels, JSON or ONNX for interchange, and a separate logic for verification. Lean lets us
place the central objects in one typechecked language:

$$`\text{program} : \text{executable Lean definition}`

$$`\text{specification} : \text{mathematical denotation}`

$$`\text{theorem} : \text{kernel-checked proof connecting the two}`

TorchLean does not pull all numerical computing into Lean. It puts the semantic object in Lean,
then makes every handoff to Python, CUDA, Julia, Arb, or a verifier explicit enough to inspect.

# One Language, Several Kinds Of Object

Lean code can introduce ordinary executable definitions, propositions, and theorems. The distinction
matters throughout TorchLean.

```
def executableScore (x : Nat) : Nat :=
  x + 1

def scoreIsPositive (x : Nat) : Prop :=
  executableScore x > 0

theorem scoreIsPositive_zero : scoreIsPositive 0 := by
  decide
```

The first definition is a program. The second definition is a proposition about that program. The
third declaration proves one instance of the proposition. A verifier
this is the shape to remember: a checker can be executable code, while soundness is a theorem about
what acceptance by that checker means.

# Lean In Mathematics

Over the last few years, Lean has become increasingly visible in research mathematics. Projects such
as the *Liquid Tensor Experiment*, a large collaborative formalization effort around work of Scholze
and Clausen, demonstrate that Lean can scale to substantial mathematical developments.

Today, *Mathlib*, Lean's community-driven mathematical library, is among the largest formalized
mathematical repositories in existence. Lean also appears in theorem-proving research because its
kernel gives a small trusted base. When Lean accepts a theorem, the proof term has been checked by
that kernel.

That kernel check is narrower and stronger than a test. It does not say an external CUDA kernel is
correct, or that a dataset file is clean, or that a model is accurate. It says that the proof term
for a stated proposition typechecks in Lean's trusted core. TorchLean uses that strength where it
fits: semantic preservation, shape theorems, derivative statements, bound soundness, and
approximation results for supported fragments.

The relevance for TorchLean is not that neural networks are pure mathematics in the usual textbook
sense. It is that the same ecosystem that supports large formal libraries can also support reusable
theorem layers for tensors, graphs, derivatives, bounds, and numerical semantics.

# Lean In Scientific Computing

Lean has been very successful in pure mathematics, but it has been less common in high performance
scientific computing and machine learning. The usual reason is practical: theorem provers were built
first for proving, not for running large numerical workloads or calling GPU libraries.

Lean 4 made a few engineering choices that matter for TorchLean:

- A fast compiler and runtime, so Lean is usable as an application language rather than only as a proof scripting environment.
- A practical *C FFI (Foreign Function Interface)*, so Lean code can call C and CUDA libraries when needed.
- A reference-counting runtime, which avoids tracing-GC pauses and allows reuse of uniquely owned buffers.
- Lean’s *dependent type system*, which TorchLean uses to make shapes and other invariants part of the type.

TorchLean uses those properties to bring Lean's strengths into scientific computing. The same
artifact can support computation, gradients, graph extraction, and formal reasoning.

The FFI part matters, but it is also a boundary. A call into C, CUDA, Python, Julia, Arb, or another
numerical library can be part of a workflow; proof evidence appears when the result crosses a small
typed interface, Lean checks the relevant predicate, and the remaining external assumption is stated
clearly. The difference is between "we called a tool" and "we have a theorem about the artifact we
imported."

A Python test can tell us that a script behaved a certain way on a finite set of examples. A Lean
theorem can state a quantified property of the object being checked. TorchLean needs both kinds of
evidence. The reason to host the semantic core in Lean is that definitions, executable checkers, and
theorem statements can share names rather than being synchronized by prose.

External verifiers still belong in the story. Many optimizers and solvers belong outside Lean. The
host-language choice is about where the final claim is stated. A solver can produce a certificate;
Lean can parse it, check it against the graph semantics, and record the theorem or assumption that
follows.

# Executable Lean And Native Boundaries

Lean 4 can compile programs and run command-line tools, so TorchLean examples are not limited to
static proof scripts. But "Lean executed this command" and "Lean proved this native computation
correct" are different sentences.

For example, a command can load a file:

```
def readCheckpointManifest (path : System.FilePath) : IO String := do
  IO.FS.readFile path
```

The type says this is an effectful action. A later pure function may parse the string into a
structured payload and check names or shapes. A theorem may then talk about the parser or checker.
The file system itself remains an external source of bytes.

# Reading The Lean Snippets

Throughout this book, Lean snippets are used the way a textbook uses equations: not every snippet is
a complete tutorial, but each one pins down a concrete object. If you have never read a functional
language, the following pieces are enough to make the early chapters legible.

## `def` introduces a named value or function
In Lean, most everyday code is written in terms of definitions (`def`) and data
(`structure`/`inductive`), rather than Python-style mutable objects. We do not create
`class NeuralNet(nn.Module)` and then mutate fields; we define values and functions describing what
a model is.

```
-- A function named `addTwo` that takes a Nat and returns a Nat.
def addTwo (x : Nat) : Nat := x + 2

#eval addTwo 40
-- IDE Output: 42
```

Notice the syntax: `name (variable : Type) : ReturnType := logic`. The colon `:` is read as "is of type".

The same notation is used for small numerical definitions:

```
def scaleThenShift (a b x : Float) : Float :=
  a * x + b
```

This looks ordinary because it is ordinary executable code. For TorchLean, the extra value is that
Lean can also ask for its type, unfold its definition in a proof, or compile it as part of a program.

## `structure` names a payload
Model code usually needs bundles of related values. Lean structures make those bundles explicit.

```
structure LinearParams1D where
  weight : Float
  bias : Float

def runLinear1D (p : LinearParams1D) (x : Float) : Float :=
  p.weight * x + p.bias
```

This tiny structure is the same design idea as a parameter store: the numbers used by the forward
pass are named data. A theorem or checker can mention `p.weight` and `p.bias` directly, instead of
reconstructing them from a mutable object.

## Tensor shapes can appear in types
In Python, a tensor's type is usually just `Tensor`; the shape is checked dynamically, if it is
checked at all. In TorchLean, the tensor type can carry the scalar type and the exact dimensional
shape. This uses Lean's *dependent types*.

```
import NN

open TorchLean

-- A 4D NCHW tensor with batch 16, 3 channels, and 224x224 spatial dimensions.
def inputImageShape : Shape :=
  shape![16, 3, 224, 224]

#check Tensor.T Float inputImageShape
```

If a function demands a tensor of one shape and a value has another shape, the mismatch is visible to
Lean before the program runs. The mathematical reading is:

$$`\operatorname{Tensor}(\mathrm{Float}, [16,3,224,224])`

is a different type from

$$`\operatorname{Tensor}(\mathrm{Float}, [3,224,224])`.

If the program really wants to remove a batch dimension, add one, broadcast, or reshape, that move
should appear as a named operation. The choice is practical: deployment bugs often come from silent
shape conventions, so TorchLean tries to make those conventions inspectable.

The shape in the type is not a runtime benchmark. It is a contract. A typed tensor can still contain
bad values for a particular task: NaNs, out-of-range token ids, a misnormalized image, or labels
encoded under the wrong convention. Types catch dimensional mistakes; task-specific predicates and
checkers handle the next layer of meaning.

## `let` creates a local binding
By default, `let` bindings in Lean do not change. Once a value is bound, it is permanent for that
scope.

```
def localExample : Nat :=
  let x := 5
  -- Reassigning x is not how Lean code is written.
  x + 1
```

In deep learning, this means an optimizer step does not silently mutate gradients or parameter
weights. It takes old values and returns new values:

```
def updateWeight (eta grad w : Float) : Float :=
  w - eta * grad
```

The syntax matters less than the object it names. A theorem about the training step can say exactly
which weights were used before the update and which weights were produced after it.

## `#check` and `#eval` are your microscope
Lean is heavily interactive. Your IDE runs the Lean server continuously. If you type `#check`, Lean
tells you the type of an expression. If you type `#eval`, Lean executes an expression and prints the
result dynamically inside your editor.

```
import NN

open TorchLean

#check nn.Linear 128 64
-- Lean reports the input and output shapes in the type.

#eval 5 * 5
-- IDE Output: 25
```

For TorchLean, `#check` is often the fastest way to learn whether you are holding a tensor, a model
builder, a graph, a theorem, or an executable command. It gives a direct tutorial loop: write a
small expression, ask Lean what it is, and refine the expression until the type says the contract you
intended.

## Effects are explicit
Ordinary Lean definitions are pure. Effects such as printing or reading a file appear in the type:

```
def pureScore (x : Nat) : Nat :=
  x + 1

def printScore (x : Nat) : IO Unit :=
  IO.println s!"score = {x + 1}"
```

TorchLean relies on this distinction. A model denotation should be a pure object that can appear in a
theorem. Loading a checkpoint, calling a native kernel, or writing a log is an effectful action whose
boundary should be visible.

## `Prop` and theorem statements are ordinary citizens
A proposition is a type whose values are proofs. That is why a theorem can be used by later code and
later proofs as a named object.

```
def IsNonnegative (x : Int) : Prop :=
  x >= 0

theorem zero_nonnegative : IsNonnegative 0 := by
  decide
```

TorchLean theorem statements are larger, but they follow the same idea. A graph soundness theorem
might say that an accepted certificate encloses all values of a denotation. A runtime approximation
theorem might say that an executable float result lies inside a rounding envelope. The theorem's
type records the hypotheses; the proof term records why the conclusion follows.

## Further reading

- Lean documentation hub (including how to cite Lean 4): https://lean-lang.org/learn/
- Lean 4 system paper: https://lean-lang.org/papers/lean4.pdf
- Liquid Tensor Experiment blueprint: https://leanprover-community.github.io/liquid/
- Mathlib (Lean's main community library): https://github.com/leanprover-community/mathlib4
