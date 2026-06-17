# VNN-COMP Verification Assets

`NN.Verification.VNNComp.MnistFC` is the reusable checker for VNN-COMP style JSON exports.
This examples directory documents the local artifact layout used by the runnable verification CLI.

Expected local layout:

```text
_external/vnncomp/mnist_fc/model_weights.json
_external/vnncomp/mnist_fc/suite.json
_external/vnncomp/mnist_fc/alphas_crownobj.json   # optional
```

Run with:

```bash
lake exe verify -- vnncomp-mnistfc \
  --weights=_external/vnncomp/mnist_fc/model_weights.json \
  --suite=_external/vnncomp/mnist_fc/suite.json \
  --max=2
```

If you keep the artifacts somewhere else, pass `--weights=...`, `--suite=...`, and optionally
`--alphas=...`.

Full VNN-COMP model dumps and suite exports live in `_external/` or another local data directory.
