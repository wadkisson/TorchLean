# Alpha-Beta-CROWN Leaf Artifacts

This example is the boundary between an external alpha-beta-CROWN-style search and a Lean side
artifact checker. The external verifier is allowed to search, split domains, optimize relaxations,
and decide which terminal leaves it wants to report. TorchLean checks the finite leaf artifact that
the external run exports: the search procedure remains the named external producer.

The reusable Lean checker lives in `NN/Verification/Cert/AbCrownLeafCert.lean`. It checks the schema
and the local leaf conditions represented in the artifact: input lower/upper boxes, output lower
bounds, thresholds, labels, and metadata needed to interpret the terminal domain. The bundled files
are deliberately small enough to review in a pull request:

- `sample_abcrown_leaf_artifact_v0_1.json`: TorchLean's checked artifact format.
- `example_raw_leaf_dump.json`: a compact raw dump shaped like what an external verifier might
  produce before conversion.

Run the checked bundled artifact with:

```bash
lake exe verify -- abcrown-leaf
```

To exercise the producer/checker path, convert the raw dump into TorchLean's schema and then invoke
the Lean checker:

```bash
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json \
  --out _external/abcrown/leaf_artifact.json \
  --check
```

For a real external verifier run, instrument the verifier to dump the terminal verified leaves, then
convert that dump into `abcrown_leaf_artifact_v0_1`. The converter accepts common raw field names
such as `x_L`, `x_U`, `lower_bounds`, and `thresholds`. If the verifier is already running inside a
Python process, import `scripts.verification.abcrown.export_leaf_artifact.write_abcrown_leaf_artifact`;
that helper writes to `ABCROWN_ARTIFACT_OUT` when no explicit output path is passed.

The intended claim shape is:

```text
external search produced leaves
  -> TorchLean converted the leaf dump
  -> Lean checked the finite leaf artifact
  -> accepted leaves may be cited as checked evidence for the stated local property
```

The search strategy, GPU kernels, and alpha/beta optimization loop remain external unless they are
separately formalized. The checked object is the exported leaf artifact.
