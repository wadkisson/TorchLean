import VersoManual

open Verso.Genre Manual

#doc (Manual) "CLI, BugZoo, And Labs" =>
%%%
tag := "tools"
%%%


This chapter collects the operator-facing tools: the CLI, BugZoo contracts, and small lab
exercises. Widgets remain available in the library (`NN.Widgets`) for Infoview inspection, but
they are not part of the narrative path.

TorchLean uses two command dispatchers:

```
lake exe torchlean <example> [flags...]
lake exe verify -- <tool> [args...]
```

The first runs examples, training applications, data checks, and numerical deep dives. The second
runs verifiers and certificate checkers. Small source-local programs can also be executed through
the example dispatcher:

```
lake exe torchlean quickstart_tensors
```

The dispatch tables are ordinary Lean definitions:

- [`NN.Examples.Models.Runner`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean)
  owns the `torchlean` subcommands;
- [`NN.Verification.CLI`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean)
  owns the `verify` tools.

When this chapter and the executable disagree, the executable is authoritative.

# Discovering Commands

Start with:

```
lake exe torchlean --help
```

The current help begins:

```
TorchLean runnable examples

Usage:
  lake exe torchlean <example> [flags...]
  lake exe torchlean --choose <example> [flags...]
  lake exe torchlean <example> --help

Start here:
  lake exe torchlean quickstart_tensors
  lake exe torchlean quickstart_autograd
  lake exe torchlean quickstart_mlp --steps 20
```

The shorter list in top-level help is a starting point, not the complete dispatch table. Verification
tools are listed independently:

```
lake exe verify -- list
```

That command currently includes in-memory TorchLean-to-IR workflows, LiRPA artifact checkers,
PINN and geometry certificate checkers, numerical ODE tools, and VNN-COMP-style applications. Use
the list printed by your checkout rather than copying a stale inventory from a paper or issue.

# Command Grammar

The normal form is:

```
lake exe torchlean <subcommand> [runtime flags] [command flags]
```

For example:

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 20 --seed 2026
```

Runtime flags may also precede the subcommand:

```
lake exe torchlean --device cpu quickstart_mlp \
  --steps 20 --seed 2026
```

Both forms reach the same parser. Documentation uses the first because the application name appears
before its options.

A leading separator is accepted for wrappers that require one:

```
lake exe torchlean -- quickstart_mlp --device cpu --steps 20
```

# BugZoo Catalog

BugZoo starts from failures that have appeared in frameworks, compilers, deployment tools, and LLM
serving systems. Each file asks a narrow question: what object or proposition would have made the
intended behavior explicit before the failure reached production?

The [BugZoo catalog API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/) and
[BugZoo overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/README.md) focus on the TorchLean fragment itself:
once a computation enters a typed TorchLean spec, shape changes, masks, token bounds, finite domain
choices, stateful normalization parameters, and backend semantics become named objects that can be
checked.

Every BugZoo file is small for a reason. A tiny theorem is often a better public contract than a
large example. It lets us say "this is the exact thing we checked" beside the paper or issue from the
real world that motivated the example.

# Compile The Catalog

All entries are imported by `NN/Examples/BugZoo/All.lean`. Compile them together with:

```
lake env lean NN/Examples/BugZoo/All.lean
```

A successful command is silent. It means every definition and theorem in the catalog elaborated;
it does not mean that every external framework implementation satisfies those contracts.

For a more interactive pass, create `BugZooAudit.lean`:

```
import NN.Examples.BugZoo.AttentionMask
import NN.Examples.BugZoo.ShapeAndBroadcast

#check NN.Examples.BugZoo.AttentionMask.exactMaskedLogit_blocked_exp_zero
#check NN.Examples.BugZoo.AttentionMask.trueInfinityMask_future_attention_weight_zero
#check NN.Examples.BugZoo.ShapeAndBroadcast.addSingletonBatch
#check NN.Examples.BugZoo.ShapeAndBroadcast.broadcastRowToMatrix_firstRow
```

Open the file in the Lean Infoview. The attention theorem quantifies over every strict-future
position `j > i`; the shape theorem exposes the singleton batch insertion and the proof-carrying
broadcast as different operations. These are the contracts. The motivating PyTorch snippets in the
source comments explain the bug family, but are not imported as trusted evidence.

# Example Anatomy

A good BugZoo example has four parts:

- the pattern in the framework that goes wrong;
- the TorchLean object that names the intended behavior;
- the theorem, structure, or definition that marks the checked boundary;
- the external conformance obligation or unsupported scope that remains outside the checked claim.

Most ML bugs here are semantic rather than syntactic. The program often still returns a tensor. The
loss may still be a scalar. A compiled graph may still run. An LLM server may still emit tokens.
BugZoo asks whether those tensors and tokens still mean what the user thought they meant.

The common contract shape is:

$$`\text{bug pattern}
\;\leadsto\;
\text{TorchLean object}
\;\leadsto\;
\text{checked claim}`

The examples stay short so the semantic invariant is visible. Each case isolates the condition that
would have made the original kind of bug harder to miss.

Some representative contract shapes:

| Example | Contract shape |
|---|---|
| Attention mask | $`j>i\Rightarrow A_{ij}=0` |
| Batch invariance | $`\operatorname{select}(\operatorname{mapBatch}(f,X),i)=f(X_i)` |
| Tokenizer boundary | token ids inhabit `Fin vocabSize` |
| KV cache | appended key/value appears at the final slot |
| Float boundary | runtime Float32 agrees with `IEEE32Exec` under a named agreement |
| Compiler boundary | target output equals source output |
| Stable loss | logits path uses log-softmax semantics |
| Ignored labels | inactive labels contribute zero |
| LayerNorm degenerate axis | zero-variance normalization follows an explicit epsilon policy |
| 3D projection | camera projection exposes depth and denominator preconditions |

# Worked Examples

The examples in this chapter form a progression. The first prints typed values. The second
differentiates a concrete linear map. The third trains a small network. The fourth lowers a model
to the verification IR and propagates an input box. The final example checks a numerical
certificate with explicit binary32 semantics.

Run them in order the first time. Each introduces one new object while keeping the earlier ones
visible.

# Lab One: A Tensor Has A Shape And A Scalar Meaning

Run:

```
lake exe torchlean quickstart_tensors
```

The current output is:

```
== Quickstart: tensor basics ==
[Float] [0.100000, 0.200000, 0.300000, 0.400000]
[ℚ] [1/10, 1/5, 3/10, 2/5]
[Int] [1, 2, 3, 4]
[IEEE32Exec] [0.100000, 0.200000, 0.300000, 0.400000]
[Float] [[[1.000000, 2.000000], [3.000000, 4.000000]],
         [[5.000000, 6.000000], [7.000000, 8.000000]]]
Expected failure printing Tensor ℝ: Refusing to print `Tensor ℝ` (proof-level);
cast to `Float`/`IEEE32Exec`/`ℚ` to display.
```

The first four tensors have the same vector shape and superficially similar entries, but their
scalar meanings differ:

- `Float` is Lean's native executable floating-point value;
- `ℚ` is exact rational arithmetic;
- `Int` is exact integer arithmetic;
- `IEEE32Exec` is TorchLean's explicit executable IEEE-754 binary32 model;
- `ℝ` is suitable for proofs but is not an object the runtime should pretend to print.

The shape of the final tensor is `2 × 2 × 2`. It is represented by nested, length-indexed
dimensions, so the eight entries cannot accidentally be interpreted as a `4 × 2` matrix without an
explicit reshape proof.

Read
[`TensorBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean)
beside the output. Change one scalar type at a time and let Lean show which operations require a
different algebraic context.
