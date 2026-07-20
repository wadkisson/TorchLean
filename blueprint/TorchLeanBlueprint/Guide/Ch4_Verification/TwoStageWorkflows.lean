import VersoManual

open Verso.Genre Manual

#doc (Manual) "Two-Stage Verification Workflows" =>
%%%
tag := "twostage"
%%%

A branch-and-bound verifier may spend hours splitting boxes, optimizing linear relaxations, and
running GPU kernels. Reimplementing that search inside Lean would make the trusted story simpler,
but it would also discard mature solvers and make large examples impractical. A two-stage workflow
separates the expensive search from the small object that must be checked.

The producer is allowed to be complicated:

```
trained model + input property
  -> external verifier
  -> branch-and-bound leaves and claimed lower bounds
```

The consumer should be narrow:

```
JSON artifact
  -> finite parser
  -> schema and local predicate checks
  -> accept or reject
```

This architecture is only valuable when the boundary is stated exactly. “Checked by Lean” may mean
anything from parsing a JSON file to replaying every bound computation and deriving a theorem.
TorchLean’s current α,β-CROWN leaf checker implements the former kind of boundary: it checks a
small structural leaf format and a local threshold predicate. It does not rerun α,β-CROWN.

# The Checked Artifact

The bundled artifact is:

```
{
  "format": "abcrown_leaf_artifact_v0_1",
  "input_dim": 2,
  "root": {
    "lo": [-1.0, -1.0],
    "hi": [1.0, 1.0]
  },
  "leaves": [
    {
      "lo": [-1.0, -1.0],
      "hi": [1.0, 1.0],
      "lb": [1.0],
      "threshold": [0.0],
      "witness_idx": 0,
      "witness_margin": 1.0
    }
  ]
}
```

The checker is
[`checkAbCrownLeafArtifact`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/AbCrownLeafCert.lean).
It performs the following operations:

1. require the exact format string `abcrown_leaf_artifact_v0_1`;
2. parse `input_dim`;
3. parse finite floating-point root bounds and require matching dimensions;
4. require `root.lo[i] ≤ root.hi[i]`;
5. require a nonempty leaf array;
6. parse each leaf’s finite `lo`, `hi`, `lb`, and `threshold` arrays;
7. require each leaf box to satisfy
   `root.lo ≤ leaf.lo ≤ leaf.hi ≤ root.hi`;
8. require matching `lb` and `threshold` lengths;
9. check either the supplied witness index or an existential coordinate with
   `threshold[i] < lb[i]`;
10. if a witness margin is supplied, compare it with `lb[i] - threshold[i]` at tolerance `1e-6`.

Every parsed numeric claim uses `expectFieldFiniteFloatArray` or
`optionalFieldFiniteFloat?`. A JSON number that converts to NaN or infinity is rejected before
ordered comparisons are used.

The local threshold predicate is implemented in
[`NN.Verification.Util.Array`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Util/Array.lean):

```
def refutesThreshold
    (lowerBound threshold : Array Float) : Bool :=
  anyPairwise lowerBound threshold
    (fun lb thr => decide (thr < lb))
```

If the vector represents output margins, this is the finite check

$$`\exists i,\quad \mathrm{threshold}_i<\mathrm{lb}_i.`

# Run The Bundled Check

From the TorchLean root:

```
lake exe verify -- abcrown-leaf
```

The current output is:

```
[artifact] Checked 1 leaves: ok=1, bad=0
```

The default path is
`NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json`. An explicit path can be
passed after the command name:

```
lake exe verify -- abcrown-leaf path/to/artifact.json
```

To see the available verification programs:

```
lake exe verify -- list
```

The registration describes this command as an “α,β-CROWN leaf artifact structural check.” That
wording is deliberate.

# Convert A Producer Dump

Vanilla α,β-CROWN does not emit TorchLean’s JSON schema. The adapter
[`export_leaf_artifact.py`](https://github.com/lean-dojo/TorchLean/blob/main/scripts/verification/abcrown/export_leaf_artifact.py)
accepts common raw names such as `x_L`, `x_U`, `lower_bounds`, and `thresholds`, normalizes them,
computes a positive witness coordinate, and writes the checked format.

Run the complete bundled producer-to-checker path:

```
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json \
  --out /tmp/torchlean-abcrown-artifact.json \
  --check
```

The verified output is:

```
Wrote TorchLean alpha-beta-CROWN-style leaf artifact to /tmp/torchlean-abcrown-artifact.json
[artifact] Checked 1 leaves: ok=1, bad=0
```

For integration inside a producer process, the same script exports a Python function:

```
from scripts.verification.abcrown.export_leaf_artifact import \
    write_abcrown_leaf_artifact

write_abcrown_leaf_artifact(
    root_lo=original_property_lo,
    root_hi=original_property_hi,
    leaves=terminal_verified_leaves,
    out_path="leaf_artifact.json",
)
```

The call belongs after the external verifier has collected terminal leaves. Each leaf must provide
an input box, lower-bound vector, and threshold vector. Supplying the original property box is
important. If no root is provided, the converter can infer the componentwise envelope of the
leaves, which is useful for fixtures but does not establish that the leaves cover the intended
property domain.

TorchLean does not vendor α,β-CROWN or the Two-Stage neural-controller repository. Their Python,
CUDA, solver, and model dependencies remain in separate environments. The core Lean build needs
only the exported artifact and checker.

# Make The Check Fail

Copy the bundled JSON and change

```
"lb": [1.0]
```

to

```
"lb": [-1.0]
```

while leaving the threshold at zero. Then run the checker on the modified file. The present output
is:

```
[artifact] Checked 1 leaves: ok=0, bad=1
uncaught exception: Artifact failed checks for 1 leaves
```

Other useful negative controls are:

- put a leaf upper bound above `root.hi`;
- use a witness index outside the lower-bound vector;
- make `witness_margin` disagree with `lb[i] - threshold[i]`;
- use a huge JSON number such as `1e999`, which is rejected as non-finite;
- make `lo[i] > hi[i]`.

Each variation reaches a different named check. This is often the quickest way to understand an
artifact schema: perturb one field and observe which invariant rejects it.

# What Acceptance Does Not Prove

The checker treats the numbers in `lb` as claims supplied by the producer. It verifies that one is
above the threshold; it does not prove that `lb[i]` is genuinely a lower bound for the network on
that leaf.

It also checks containment of every leaf in the root, not coverage of the root by the leaves.
Containment is

$$`B_\ell\subseteq B_{\mathrm{root}}.`

Coverage would be

$$`B_{\mathrm{root}}
\subseteq\bigcup_{\ell=1}^m B_\ell,`

which is a different and stronger condition. The current `abcrown-leaf` checker does not establish
it.

Finally, it does not parse a neural network, connect leaf bounds to shared IR semantics, or prove a
theorem of the form

$$`\operatorname{check}(artifact)=\texttt{true}
\Longrightarrow
\operatorname{Safe}(network,B_{\mathrm{root}}).`

The current result is an executable IO check over finite Float arrays. Calling it a complete
α,β-CROWN certificate checker would overstate the implementation.

# The Stronger Certificate We Want

A whole-region branch-and-bound certificate would contain enough information to establish three
independent facts:

1. *coverage*

   $$`B_{\mathrm{root}}=\bigcup_\ell B_\ell;`

2. *local bound soundness*

   $$`\forall x\in B_\ell,\quad
   g_i(x)\geq \mathrm{lb}_{\ell,i};`

3. *local safety*

   $$`\exists i,\quad
   \mathrm{lb}_{\ell,i}>\mathrm{threshold}_i.`

Then elementary set reasoning yields

$$`\left(
B_{\mathrm{root}}=\bigcup_\ell B_\ell
\;\land\;
\forall\ell,\operatorname{Safe}(B_\ell)
\right)
\Longrightarrow
\operatorname{Safe}(B_{\mathrm{root}}).`

The expensive producer may still choose splits and relaxation parameters. The artifact must carry
enough data for Lean to replay each local bound or check a proof object whose soundness theorem is
already established. This is the important research step between the present structural fixture
and a proof-producing verifier boundary.

# Neural Controllers

In a controller workflow, the producer may search for a policy `u_θ` and a Lyapunov candidate `V`.
The target inequalities often look like

$$`V(x)\geq 0,\qquad
\nabla V(x)\cdot f(x,u_\theta(x))
\leq-\alpha\|x\|^2.`

A trustworthy two-stage artifact must identify:

- the model and parameter hash;
- the state-space root region;
- the exact dynamics and scalar semantics;
- every partition leaf;
- replayable bounds for `V` and its Lie derivative;
- coverage and boundary conditions.

Merely exporting terminal boxes with positive numeric margins does not prove those analytic
conditions. TorchLean’s other `RealCert`-style and IR-based checkers illustrate stronger replay
patterns, but they should be evaluated command by command rather than transferred to
`abcrown-leaf` by association.

# Reproducible Evidence

A useful run record includes:

- TorchLean commit and Lean toolchain;
- external verifier repository and commit;
- Python and solver versions;
- model checkpoint hash;
- original property file and root box;
- dtype, device, and numerical flags;
- raw producer dump;
- normalized TorchLean artifact;
- exact checker command and output.

This is more than administrative detail. A lower-bound vector has no stable meaning if the model,
property, or output-margin convention changes.

The present leaf workflow is best used as a transparent integration fixture and schema boundary.
It proves that the finite data passed to TorchLean satisfies the checks listed above. Stronger
root-region claims require coverage and replayable local-bound evidence, and the guide keeps that
next step visible rather than hiding it behind the name of the external solver.
