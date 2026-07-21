import VersoManual

open Verso.Genre Manual

#doc (Manual) "Certificates And Two-Stage Workflows" =>
%%%
tag := "certificates"
%%%


A certificate is useful when producing an answer is expensive but checking the answer can be made
small. A branch-and-bound verifier may explore thousands of subdomains, optimize relaxation
parameters, and use GPU kernels. Lean need not replay that entire search if the producer emits
enough evidence for a compact checker.

For every certificate format, ask three questions:

1. What finite object did the producer return?
2. Which conditions does Lean recompute?
3. Which theorem turns acceptance into the final semantic claim?

The third question matters most. A perfectly parsed file can still contain a number that was never
proved to bound the network.

TorchLean does not vendor the Two-Stage / α,β-CROWN repository. The core Lean build does not
require that Python environment. Generating fresh α,β-CROWN leaf artifacts requires external verifier
output plus TorchLean's conversion helper.

During a branch-and-bound verification run, an instrumented external verifier can expose terminal
subdomains. TorchLean's helper converts that terminal-domain data into a small JSON *leaf artifact*
for each represented subdomain. TorchLean can parse that JSON and validate several
properties entirely inside Lean: every leaf box lies inside the declared root input region; every
leaf marked as verified satisfies the exported local prune test
(`∃ i, lb[i] > threshold[i]` in the exported fields); and the document is internally consistent
(dimensions, array lengths, and cross-references line up).

These checks focus on the JSON artifact itself: boxes nest correctly, verified leaves satisfy the
stated prune rule, and the fields fit together. The numeric bound propagation that produced each
`lb` belongs to the external producer unless a separate recompute-and-compare certificate path is
added.

The concrete path is:

- α,β-CROWN performs branch and bound outside Lean;
- the producer exports or exposes terminal leaf domains;
- TorchLean's converter writes those domains in `abcrown_leaf_artifact_v0_1.json`;
- TorchLean parses the JSON;
- Lean checks the structural predicate for each leaf;
- the checker accepts or rejects the artifact.

Here, `verified` or `pruned` means that a represented leaf passes the producer's exported local
test. It does not yet mean that Lean has proved the neural-network property on the root box.

# Run The Bundled Checker

The sample artifact contains one two-dimensional leaf. Its box equals the declared root, its
exported lower bound is `1.0`, and its threshold is `0.0`.

```
lake exe verify -- abcrown-leaf
```

Lean reports:

```
[artifact] Checked 1 leaves: ok=1, bad=0
```

To see what was actually checked, make a temporary copy whose threshold is larger than the
exported lower bound (requires `jq`), then re-run the checker on that file:

```
jq '.leaves[0].threshold=[2.0] | .leaves[0].witness_margin=-1.0' \
  NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json \
  > /tmp/torchlean_bad_leaf.json

lake exe verify -- abcrown-leaf /tmp/torchlean_bad_leaf.json
```

The second command is expected to exit unsuccessfully:

```
[artifact] Checked 1 leaves: ok=0, bad=1
uncaught exception: Artifact failed checks for 1 leaves
```

This is a useful failure: the JSON is valid, but the claimed witness no longer satisfies the prune
inequality. Try two other changes. Move a leaf coordinate outside the root box; then replace one
numeric field by an infinite value. Both should be rejected before the artifact can be treated as
checked evidence.

The checked predicate is small enough to write informally: a leaf is accepted when its box lies
inside the root box, its dimensions are coherent, and it has a witness index whose exported lower
bound is above the corresponding threshold.

The leaf prune test has the form:

$$`\exists i,\qquad lb_i>threshold_i.`

In Lean-facing pseudocode, the checked part is closer to:

```
def leafPruned (lb threshold : Array Float) : Bool :=
  any index i with lb[i] > threshold[i]

def leafInsideRoot (root leaf : Box) : Bool :=
  all coordinates k, root.lo[k] <= leaf.lo[k] && leaf.hi[k] <= root.hi[k]
```

Those checks are about exported numbers. They do not by themselves prove that the exported `lb`
values are lower bounds of the neural network. That stronger claim needs either recomputation in
Lean or a proof-backed certificate whose local transfer rules Lean can check.

Leaf nesting has the form:

# Two-Stage Verification Workflows

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
[`NN.Verification.Util.Array`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Util/Array.lean).
Use that helper when a certificate claims a coordinate-wise threshold crossing rather than a
hand-written comparison over JSON arrays.
