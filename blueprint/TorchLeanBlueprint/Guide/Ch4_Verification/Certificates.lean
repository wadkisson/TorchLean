import VersoManual

open Verso.Genre Manual

#doc (Manual) "Verification Certificates" =>
%%%
tag := "certificates"
%%%

A certificate is a finite artifact that lets Lean check a larger verification claim without relying
on the entire search procedure that produced it.

The producer may be α,β-CROWN, a branch-and-bound verifier, a scientific solver, or a Python
script. The checker asks three questions: what finite object was returned, what local conditions
does Lean validate, and what theorem turns acceptance into a semantic claim?

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

The path is:

- α,β-CROWN performs branch and bound outside Lean;
- the producer exports or exposes terminal leaf domains;
- TorchLean's converter writes those domains in `abcrown_leaf_artifact_v0_1.json`;
- TorchLean parses the JSON;
- Lean checks the structural predicate for each leaf;
- the checker accepts or rejects the artifact.

A few terms will help keep the certificate layers straight:

- `certificate`: an artifact produced outside Lean that Lean can parse and check.
- `leaf artifact`: exported per-subdomain data from a branch-and-bound run.
- `verified` / `pruned`: here, a leaf passes the solver's local check and can be removed
  from the search tree.
- `producer hypothesis`: the part supplied by the external solver when Lean checks only the exported
  artifact rather than recomputing the bound.

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

$$`B_\ell\subseteq B_{\mathrm{root}}.`

A stronger branch certificate would also check coverage:

$$`B_{\mathrm{root}}\subseteq\bigcup_\ell B_\ell.`

For a semantic margin property, the target shape is:

$$`\forall x\in B_\ell,\qquad c^\top f(x)\ge threshold.`

The current checker accepts the artifact when every represented leaf satisfies this predicate and
the top level metadata is coherent. It does not turn a set of leaves into a root-region proof unless
coverage of the root by those leaves is separately supplied and checked. In this fragment, the
certificate is structural.

# What The Checker Proves Today

This `v0.1` format stays small. Lean checks that every leaf domain sits inside the declared
root input box, that every leaf marked as verified really satisfies the solver's own prune test
(`∃ i, lb[i] > threshold[i]`), and that the JSON is internally consistent enough for downstream
tooling to rely on it.

The current artifact checks `lb_i > threshold_i` for an exported lower bound. A stronger artifact
would also check that `lb_i` is a sound lower bound for the graph on the leaf.

If we later add a recompute path, the stronger predicate would recompute the bound for the network
on the leaf box, compare that recomputed bound with the exported `lb`, and then check the same
margin witness.

That would move more work from the external solver into Lean's checker. The current JSON format was
kept focused so the first boundary is clear.

There are therefore three levels of confidence to keep straight:

- structural checking: the artifact is self-consistent.
- recompute-and-compare checking: Lean can reproduce the arithmetic.
- full soundness: Lean also checks the bound computation that produced the artifact.

Those levels are not a weakness of certificates. They let the project increase the amount checked by
Lean without changing the surrounding workflow.

# Lean Entry Points

The leaf checker is intentionally separate from CROWN node checkers:

```
#check NN.Verification.Cert.AbCrownLeafCert.checkAbCrownLeafArtifact
#check NN.Verification.CROWNNodeCertAlphaBeta.AlphaBetaCROWNNodeCertificate
#check NN.Verification.CROWNNodeCertAlphaBeta.checkAlphaBetaCROWNNodeCertificate
```

Use the first when the artifact is a branch-and-bound leaf summary. Use the second family when the
artifact contains per-node affine bound data that Lean can recompute against a graph and parameter
store.

This split is important for citations:

- `abcrown-leaf` checks a structural leaf artifact.
- `checkAlphaBetaCROWNNodeCertificate` checks per-node α,β-CROWN transfer data by recomputation and
  tolerance comparison.
- graph soundness theorems apply only when the certificate format and graph fragment supply the
  hypotheses those theorems demand.

# File Format: `abcrown_leaf_artifact_v0_1.json`

Top-level object:

```
{
  "format": "abcrown_leaf_artifact_v0_1",
  "input_dim": 2,
  "root": { "lo": [-4.8, -10.8], "hi": [4.8, 10.8] },
  "leaves": [
    {
      "lo": [...],
      "hi": [...],
      "lb": [...],
      "threshold": [...],
      "witness_idx": 0,
      "witness_margin": 0.123
    }
  ]
}
```

Semantics:

- `root` describes the input box being verified.
- For real verification runs, pass the original input-property box as `root`. If the raw dump does
  not contain a root, the exporter can infer the componentwise leaf envelope as a structural
  fallback, but that fallback is only the envelope of the represented leaves.
- Each `leaf` is a sub-box of `root`.
- `lb` and `threshold` are the lower bounds and thresholds reported by the external producer for
  that leaf at the moment it was pruned or verified.
- A leaf is considered "verified" iff `∃ i, lb[i] > threshold[i]`.
  (This matches how `complete_verifier/input_split/branching_domains.py` filters out verified domains.)
- `witness_idx` and `witness_margin` are a convenience witness for the check above:
  `witness_margin = lb[witness_idx] - threshold[witness_idx]`.

The schema deliberately does not contain a neural-network graph, α slopes, β phases, or per-node
affine forms. It is therefore a *leaf artifact*, not a full proof certificate.
The artifact records enough to check the terminal-domain bookkeeping exported by the producer; it
does not replay the producer's bound propagation.

# How To Generate

TorchLean now provides the producer-side conversion helper at
`scripts/verification/abcrown/export_leaf_artifact.py`. It converts a raw terminal-domain dump into the
`abcrown_leaf_artifact_v0_1.json` schema and can immediately run the Lean checker:

```
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json \
  --out _external/abcrown/leaf_artifact.json \
  --check
```

The helper accepts common raw field names such as `x_L`, `x_U`, `lower_bounds`, and `thresholds`.
For direct instrumentation of an external verifier, import
`write_abcrown_leaf_artifact` from that script and call it after the verifier has collected terminal
verified leaves. If no explicit output path is passed, that helper writes to `ABCROWN_ARTIFACT_OUT`.
That environment variable is a TorchLean helper convention; setting it alone does not modify an
unpatched α,β-CROWN run.

If you want to use an external α,β-CROWN producer, clone it separately:

```
git clone https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training.git \
  Two-Stage_Neural_Controller_Training
```

Run the external verifier, dump or instrument the terminal verified leaves, convert them with the
TorchLean helper, then pass the resulting artifact to TorchLean's checker.

# How To Check In Lean

Use the unified `verify` CLI tool `abcrown-leaf` to check the converted JSON artifact against
TorchLean's structural leaf predicate.

Example:

```
lake exe verify -- abcrown-leaf
```

The checker also runs on the bundled sample artifact when invoked with the tool's default input.

End-to-end Python plus Lean post-check: *Two-Stage Workflows*. What TorchLean proves vs imports:
*Verification*.

# References

- [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [β-CROWN / α,β-CROWN paper](https://arxiv.org/abs/2103.06624), covering branch and bound with
  optimized bound propagation.
- [LiRPA on general computational graphs](https://arxiv.org/abs/2002.12920), for automatic
  perturbation analysis.
- [Branch-and-bound for neural network verification](https://arxiv.org/abs/1907.10615), as one
  representative background entry.
