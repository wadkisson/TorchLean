import VersoManual

open Verso.Genre Manual

#doc (Manual) "Floating Point" =>
%%%
tag := "floats"
%%%


Neural-network papers write equations over real numbers. Hardware evaluates a sequence of finite
operations. Most of the time we can ignore the difference. This chapter is about the times when we
cannot.

# Begin With A Calculation

Take the three exact real numbers

$$`a=2^{24},\qquad b=1,\qquad c=-2^{24}`.

Over the reals, both parenthesizations are equal:

$$`(a+b)+c=a+(b+c)=1`.

Binary32 has 24 bits of significand precision. Around `2²⁴`, adjacent representable numbers are two
units apart. The exact value `2²⁴+1` sits halfway between them, and ties-to-even rounds it back to
`2²⁴`. Therefore

$$`
\operatorname{fl}(\operatorname{fl}(a+b)+c)=0,
\qquad
\operatorname{fl}(a+\operatorname{fl}(b+c))=1.
`

TorchLean can run this example from the bit patterns themselves:

```
import NN.Floats

open TorchLean.Floats.IEEE754

def a : IEEE32Exec := IEEE32Exec.ofBits 0x4b800000  --  2^24
def b : IEEE32Exec := IEEE32Exec.ofBits 0x3f800000  --  1
def c : IEEE32Exec := IEEE32Exec.ofBits 0xcb800000  -- -2^24

#eval (IEEE32Exec.add (IEEE32Exec.add a b) c).bits
-- 0          (0x00000000, +0)

#eval (IEEE32Exec.add a (IEEE32Exec.add b c)).bits
-- 1065353216 (0x3f800000, 1)
```

Save the snippet as `FloatPlayground.lean` in the repository root and run:

```
lake env lean FloatPlayground.lean
```

There are two quick variations worth trying. Replace `b` by `2`; both parenthesizations now return
`2`, because `2²⁴+2` is representable. Replace it by the binary32 value `0.5`; both paths lose the
small addend. The interesting case `b=1` sits exactly at the tie between neighboring values.

This small calculation already raises three different questions.

First, *why* did the first addition round down? That is a theorem about the spacing of a binary
format and the nearest-even rule. Second, *which bits* should the complete operation return? That
requires an executable account of binary32, including signs, exponents, subnormals, infinities, and
NaNs. Third, will a CUDA reduction use either of these parenthesizations, a balanced tree, or a
fused instruction? That is a question about the selected backend.

Trying to answer all three with one definition makes every proof carry irrelevant machinery.
TorchLean instead builds a small ladder:

```
format and rounding theory
        ↓
rounded-real arithmetic
        ↓
finite binary32 specialization
        ↓
executable IEEE bit patterns
        ↓
CPU, CUDA, and external providers
```

The rest of the chapter constructs this ladder from the bottom of the mathematics upward.

# Where `NeuralFloat` Came From

TorchLean's generic floating-point theory was inspired by
[Flocq](https://flocq.gitlabpages.inria.fr/), the mature floating-point library developed in Rocq
(formerly Coq) by Sylvie Boldo, Guillaume Melquiond, and contributors. Flocq's central design
decision is wonderfully simple: a floating-point *format* and a *rounding rule* are different
objects.

A format tells us which real numbers are available. A rounding rule chooses one of those numbers.
The same binary32 format can round downward, upward, toward zero, or to nearest-even. Conversely, the
same nearest-even idea can be used with binary16, binary32, a fixed-point grid, or an experimental
low-precision format. Once those choices are separated, theorems about monotonicity, exactness,
neighboring values, and ULP error can be reused.

TorchLean follows that organization but is not a port of Flocq. The definitions and proofs are
native Lean. The library concentrates on the theory needed by tensor semantics, interval
enclosures, quantization, and runtime approximation. Flocq remains broader in areas such as its
effective-operation infrastructure and its collection of verified floating-point algorithms.
TorchLean adds a separate executable binary32 development and graph-level numerical machinery aimed
at machine learning.

The correspondence is useful when reading either library:

| Mathematical role | Flocq vocabulary | TorchLean vocabulary |
| --- | --- | --- |
| radix power `βᵉ` | `bpow` | `neuralBpow` |
| mantissa/exponent value | `F2R`-style representation | `NeuralFloat`, `neuralToReal` |
| exponent policy | `fexp` | `fexp` with `NeuralValidExp` |
| representable grid | `generic_format` | `neuralGenericFormat` |
| fixed, unbounded, gradual formats | `FIX`, `FLX`, `FLT` | `FIXExp`, `FLXExp`, `FLTExp` |
| rounding to the grid | `round` | `neuralRound` |

The `neural` prefix is not claiming that floating-point arithmetic is unique to neural networks.
It marks TorchLean's generic floating layer and avoids colliding with Lean's host `Float`. The
theorems themselves are ordinary numerical analysis and can be used independently of a model.

# Tracking Numerical Error Through A Network

The previous chapter explained one rounded operation. A neural network contains thousands or
billions of them. The useful question is no longer “how large can one rounding error be?” but:

> If the runtime starts near the ideal inputs, how far can its forward value, gradients, and next
> parameter state move from the corresponding real-valued computation?

TorchLean answers this compositionally rather than with one global rounding lemma for the whole
network.

1. _Local forward rules._ Each operator has a numerical contract: if its inputs approximate the
   ideal inputs within known tolerances, its output approximates the ideal operator within a
   (usually larger) tolerance that depends on the operator, the scalar semantics, and those input
   budgets.
2. _Forward composition on the graph._ A graph theorem walks the nodes in execution order and adds
   or otherwise combines those local budgets, so the final forward value carries an explicit
   end-to-end error relative to the real-valued network.
3. _Reverse / VJP composition._ Differentiating the same graph requires the same discipline in the
   other direction: each local VJP has an error rule, and a reverse-mode theorem accumulates those
   rules so the returned gradient approximates the ideal cotangent within a stated budget.
4. _Optimizer step._ Training does not stop at the gradient. An optimizer contract takes the
   approximate gradient (and any approximate moments or step-size logic) and bounds how far the
   next parameter state can drift from the ideal update.

The result is a chain of named tolerances—forward, backward, then update—rather than an informal
hope that “float32 is close enough.” If any link is missing (for example a native kernel with no
bridge theorem), that gap stays visible in the account of what was proved.

# Two Executions Of The Same MLP

Start with the checked-in example:

```
lake exe torchlean float32_modes
```

It evaluates

$$`y=W_2\operatorname{ReLU}(W_1x+b_1)+b_2`

twice. The first run uses Lean's host `Float`; the second uses `IEEE32Exec`. Both runs then compute
the parameter and input VJPs. A shortened output is:

```
== Float (runtime) ==
y = [2.080000]
hiddenBiasGrad = [0.700000, 0.800000, 0.900000]
inputGrad = [0.760000, 1.000000]

== Float32 (IEEE32Exec) ==
y = [2.080000]
hiddenBiasGrad = [0.700000, 0.800000, 0.900000]
inputGrad = [0.760000, 1.000000]

max_abs_diff(Float vs IEEE32Exec) =
  0.0000000762939453835542735760100185871124267578125
```

The values look identical at six decimal places. The final line shows that they are not. This is a
good experiment, but we want more than the distance observed at one input. We want a bound derived
from the operations in the model.

# The Approximation Relation

Let `x` be an ideal real tensor and `x̂` its rounded counterpart. TorchLean writes the basic
coordinatewise relation as

$$`\operatorname{approxT}(x,\widehat x,\varepsilon)
  \quad\Longleftrightarrow\quad
  \forall i,\;
  |\operatorname{toSpec}(\widehat x_i)-x_i|\leq\varepsilon`.

# Float32 Soundness

After naming the numerical objects, the next question is how a claim moves from real arithmetic to
Float32.

The basic situation is common. A verifier or proof establishes a real-valued inequality with some
margin. A floating-point analysis bounds how far the rounded computation can move from the real one.
If the margin is larger than the rounding error budget, the property survives finite precision.

Recurring vocabulary:

- `FP32`: TorchLean's proof model for binary32 style rounding on the reals.
- `IEEE32Exec`: the executable IEEE-754 kernel, used for checking actual bit behavior.
- `bound/approximation theorem`: a theorem of the form "the float32 execution stays within `eps`
  of the real spec".
- `soundness`: the claim that a bound proved in Lean actually covers the runtime behavior being
  modeled.

The guiding transfer lemma is simple:

- the decoded FP32 result is within `ε` of the real specification;
- the real specification is above the target threshold by more than `ε`;
- therefore the decoded FP32 result is still above the target threshold.

A real-valued margin of `10^-4` does not help if the Float32 error budget is `10^-3`. A real-valued
margin of `0.1` may survive the same rounding budget. The theorem has to say which case we are in.

For example, suppose a real-valued verifier proves a margin lower bound of `0.12`. Suppose the FP32
approximation theorem says each relevant logit can move by at most `0.03`. For a two-logit margin,
the margin can shrink by at most `0.06`. The FP32 margin is still at least `0.06`, so the
classification claim survives rounding.

# Run The Two Float32 Views

TorchLean includes a small forward-and-backward comparison using the same MLP parameters in host
`Float` arithmetic and in the executable `IEEE32Exec` semantics:

```
lake exe torchlean float32_modes
```

The command first names the available meanings:

```
Float32 mode: FP32: proof semantics (round-on-ℝ), finite-only; no NaN/Inf
Float32 mode: IEEE32Exec: executable IEEE-754 binary32 kernel (bit-level; includes NaN/Inf)
```

It then prints the output, parameter gradients, and input gradient for both executable paths. The
final comparison on the bundled example is:

```
max_abs_diff(Float vs IEEE32Exec) =
  0.0000000762939453835542735760100185871124267578125
```

This number is an observation about one input and one network. It is not a uniform error theorem.
The proof task is to derive a bound `ε` from input ranges, parameter ranges, and the sequence of
rounded operations, then prove that every execution covered by those hypotheses differs from the
real specification by at most `ε`.

Try changing the example's weights by a power of two and by a nearby non-power-of-two decimal. The
former often passes through binary arithmetic exactly; the latter exposes rounding earlier. The
experiment gives intuition for the formal representability and ULP theorems developed in the
floating-point chapters.
