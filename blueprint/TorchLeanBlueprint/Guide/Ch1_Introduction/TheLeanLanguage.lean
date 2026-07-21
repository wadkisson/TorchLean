import VersoManual

open Verso.Genre Manual

#doc (Manual) "Lean For TorchLean" =>
%%%
tag := "lean-language"
%%%


TorchLean is ordinary Lean code. Models are definitions, shapes are values that appear in types,
training entry points are `IO` programs, and proofs are terms checked by Lean's kernel. You do not
need to learn all of Lean before running a model, but a few language ideas make the rest of this
guide much easier to read.

This chapter is a working primer. Put the snippets in a file named `Primer.lean` at the TorchLean
repository root and run:

```
lake env lean Primer.lean
```

`lake env` selects the Lean version and dependencies declared by this checkout. When the file
contains only successful commands, Lean exits silently except for output requested by `#check` and
`#eval`.

# Imports And Namespaces

Begin the file with:

```
import NN.API

open TorchLean
```

`import NN.API` loads the focused application interface: tensors, model builders, datasets,
optimizers, and the public trainer. It does not import every proof or backend-internal module.

Most public names live in the `TorchLean` namespace. `open TorchLean` lets us write `Tensor.T`,
`nn.linear`, and `Trainer.new` instead of prefixing each name with `TorchLean.`. Lower-case `nn` and
`optim` are namespaces, not Python objects.

Lean's editor can show the type of any known name. Add:

```
#check Tensor.T
#check nn.linear 2 4
#check Trainer.new
```

The relevant parts of the output are:

```
TorchLean.Tensor.T (α : Type) : Spec.Shape → Type
nn.linear 2 4 : nn.M (nn.Sequential ...2... ...4...)
Trainer.new ... : Trainer.Handle σ τ
```

The full output contains qualified internal names and implicit arguments. That detail is useful when
debugging, but the second line already tells the main story: a `2 → 4` linear layer is a seeded model
builder whose input and output shapes occur in its type.

# Definitions Compute

`def` gives a name to a value or function:

```
def square (n : Nat) : Nat :=
  n * n

#eval square 12
```

The output is:

```
144
```

The first `Nat` is the input type and the second is the result type. Lean can infer many result
types, but explicit annotations are helpful at API boundaries.

Local values are introduced with `let`:

```
def centeredSquare (x center : Int) : Int :=
  let displacement := x - center
  displacement * displacement

#eval centeredSquare 7 3
```

This prints `16`. The name `displacement` does not mutate; it abbreviates the value `x - center`
inside the remaining expression.

# Structures Name The Pieces

A `structure` groups fields:

```
structure Affine where
  weight : Float
  bias : Float
deriving Repr

def Affine.forward (p : Affine) (x : Float) : Float :=
  p.weight * x + p.bias

#eval Affine.forward { weight := 1.5, bias := -0.5 } 2.0
```

Lean prints:

```
2.500000
```

TorchLean uses structures for configuration records, model summaries, backend contracts, graph
nodes, and certificate data. A value such as

```
{ task := .regression
  optimizer := optim.adam { lr := 0.03 }
  seed := 2026 }
```

is a structure literal. Expected types tell Lean which structure and constructors are intended.
