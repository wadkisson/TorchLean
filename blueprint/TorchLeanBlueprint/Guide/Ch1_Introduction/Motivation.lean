import VersoManual

open Verso.Genre Manual

#doc (Manual) "Why Verification Matters" =>
%%%
tag := "motivation"
%%%

The verification problem in ML has two parts. Neural networks can fail, and the object we check is
often not exactly the object we ran.

A trained model may live as Python code, a checkpoint, an exported graph, a verifier input, and a
CUDA execution path. Each representation can be reasonable on its own, but a guarantee becomes
fragile if no one can say precisely how these representations agree. TorchLean is built around that
question: when a model is checked, what exact computation is the claim about?

This is not only a philosophical worry. The adversarial-examples literature made clear that small
changes to an input can change a prediction even when the perturbation is meant to be harmless; a
canonical early reference is Szegedy et al.,
["Intriguing properties of neural networks"](https://arxiv.org/abs/1312.6199). Verification methods
such as IBP, CROWN, and LiRPA attack that problem by computing sound output bounds for input
regions. But a bound is only as meaningful as the object it bounds. If the verifier saw one graph
and deployment ran another graph, the certificate may be mathematically interesting and operationally
irrelevant.

# Why Neural Networks Are Awkward To Check

For ordinary software, a specification can often be crisp: a sorting routine returns a sorted list,
or a compiler preserves program meaning. Neural networks are less tidy. A serious claim usually
depends on a region of inputs, a trained parameter set, a numeric backend, and a property such as a
margin, an interval enclosure, a residual bound, a projection check, a Lyapunov condition, or a
robustness guarantee.

There is also a newer reason this matters. ML code is becoming easier to generate. A model, a
training loop, a verifier script, or an export wrapper can now be produced quickly by an LLM or
copied from a notebook. The bottleneck moves to meaning: what assumptions the code used, which
artifact it produced, and whether the artifact being checked is the artifact that will be used.

The distinction is sharpest in scientific ML. A classifier error is already serious, but a learned
PDE surrogate, a neural controller, or a 3D perception certificate may sit inside a larger
scientific or engineering argument. In those cases, a successful run is only the beginning. We want
to know whether a residual was bounded, whether a controller satisfied the checked condition,
whether a projection certificate was replayed, or whether a finite precision execution stayed inside
the intended envelope.

Common failure modes include:

- *Adversarial vulnerability:* a small input change can flip a prediction while remaining inside the
  perturbation model a user considers harmless.
- *Distributional shift:* a model can fail when the deployment distribution differs from the
  training and validation distributions.
- *Shape and state mistakes:* broadcasting, normalization state, dropout mode, masks, tokenizers,
  and cache layouts can change the intended computation without crashing.
- *Serialization and interop mistakes:* a checkpoint can be loaded with parameter names, transposes,
  or layout conventions that differ from the graph a tool analyzed.
- *Floating-point mismatch:* rounding, fused kernels, fast math, and device differences can change
  the exact value being checked.

A concrete example is a prediction tensor of shape `[batch, 1]` compared with labels of shape
`[batch]`. A dynamic runtime may broadcast the two tensors and compute a loss over a larger shape
than intended. The training curve can still move. The bug is not a crash; it is a plausible-looking
experiment that optimized a different objective. A verification claim has the same risk if it talks
about the clean objective while the deployed computation used the broadcasted one.

# The Semantic Gap

The ML verification literature has made real progress: interval bound propagation, CROWN/LiRPA
relaxations, branch-and-bound methods, and mixed-integer encodings can prove robustness
properties for supported model classes.

Those words also hide different kinds of evidence. A mixed-integer solver may return a certificate
or a status code. A bound-propagation implementation may compute intervals using floating point
arithmetic. A proof assistant theorem may show that a checker's accepted artifact implies an
enclosure over an idealized semantics. A runtime regression may compare CUDA output to a reference
implementation on selected inputs. All four are useful. They are not the same claim.

The difficult part is connecting those checks to the artifact that actually ran. A proof about an
idealized real-valued network does not automatically cover PyTorch modules, CUDA kernels, JSON
checkpoints, or float32 execution paths; those objects need explicit bridges or trust-boundary
statements before they inherit the claim.

That mismatch is the semantic gap. It can enter at many points:

- the architecture analyzed by a verifier may differ from the module used in training;
- the parameter payload may have been serialized under a different naming or layout convention;
- the theorem may quantify over real numbers while deployment uses float32 kernels;
- the graph checked by a certificate validator may omit preprocessing, masking, or mode-dependent
  state.

TorchLean is built around making those distinctions visible. The graph representation, the
floating point semantics, and the verifier infrastructure are ordinary Lean artifacts, not labels
attached to an experiment after it finishes.

# A Small But Typical Mismatch

Suppose a classifier is trained with preprocessing

$$`x \longmapsto (x-\mu)/\sigma`

but the verifier receives only the post-normalization network. That can be the right choice if the
input region is also transformed into post-normalization coordinates. It is the wrong choice if the
robustness claim is later written over raw pixels. The network did not become unsafe because of a
syntax error; the claim became ambiguous because a boundary was left implicit.

The same pattern appears in smaller bugs:

- a training script uses `model.train()` while the exported graph assumes evaluation mode;
- a mask is stored with `True` meaning "keep" in one library and "block" in another;
- a graph optimizer fuses operations under real arithmetic assumptions while the deployment path
  uses float32 rounding;
- a certificate stores a lower bound for one logit margin, while the report describes a different
  target label.

TorchLean's response is not to ban these workflows. It is to require the workflow to name the graph,
payload, mode, preprocessing convention, scalar semantics, and checker result that the claim uses.

# What Lean Statements Add

A Lean statement fixes the objects under discussion. It records the quantifiers, names the graph or
program being discussed, and can be reused by later proofs without reinterpreting the claim. If a
definition changes in a way that invalidates the theorem, Lean asks us to repair the proof or weaken
the statement explicitly.

For a robustness claim, the statement should name the graph, parameters, input region, label, and
margin:

$$`\forall x,\; x\in B
\;\Longrightarrow\;
\operatorname{margin}(\operatorname{denote}(g,\theta,x),y)>0`

Here the graph `g`, parameters `θ`, input region `B`, label `y`, scalar semantics, and denotation
are all part of the mathematical object being checked. A certificate checker can then prove that a
particular JSON artifact implies the bound predicate for that graph and payload pair.

For a compiler or runtime theorem, the statement has a different shape:

$$`\forall x,\quad
\operatorname{denoteIR}(g,payload,x)
=
\operatorname{denoteCompiled}(\operatorname{compile}(g,payload),x)`

That statement is about semantic preservation. Accuracy, robustness, compilation correctness, and
floating point approximation are different properties, and TorchLean keeps them as different
theorems.

# Checks Are Programs, Proofs Are Claims About Programs

It is useful to keep a simple ladder in mind:

```
run a model
  -> observe an output
  -> check a local property of an artifact
  -> prove that accepted artifacts imply a semantic property
  -> state which external boundaries remain
```

The lower rungs are still important. If a JSON artifact does not parse, or if inferred graph shapes
do not match the payload, there is no theorem to discuss. But the upper rungs are where a guide or
paper should be precise: "the checker accepted" is not automatically "the theorem applies." The
theorem applies when the checker, graph fragment, scalar model, and assumptions match the theorem's
hypotheses.

# Our Response

TorchLean keeps model code, graph semantics, executable float models, certificate checkers, and
proof statements in one Lean development. The project distinguishes checked Lean content from named
boundaries:

- the model is a Lean definition;
- the graph has a Lean denotation through the IR and specification layers;
- Float32 has explicit proof and executable models;
- CUDA and PyTorch sit behind documented boundaries;
- external certificates are parsed and checked according to stated predicates.

The recurring proof shape is:

$$`\text{checked artifact} \;+\; \text{named boundary assumptions}
\;\Longrightarrow\;
\text{semantic claim about the model}`

For a robustness verifier, that might mean that a checked certificate implies a positive margin on
all inputs in a box:

$$`\forall x\in B,\quad
\operatorname{margin}(\operatorname{denote}(model,\theta,x))>0`

For a runtime approximation theorem, it might mean that the executable float32 value remains inside
a proved interval around the real valued specification:

$$`\forall x,\quad
\operatorname{exec32}(model,\theta,x)\in
\operatorname{roundingEnvelope}(\operatorname{denote}_{\mathbb R}(model,\theta,x))`

The habit is the same in each case: the theorem should say what object was checked, what claim
follows, and which runtime or external assumptions remain. TorchLean keeps runnable code, graph
artifacts, and formal statements close together so verification remains connected to the object that
ran, rather than pasted into a report after the fact.

## References

- Szegedy et al., ["Intriguing properties of neural networks"](https://arxiv.org/abs/1312.6199),
  ICLR 2014.
- Gowal et al., ["On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust
  Models"](https://arxiv.org/abs/1810.12715), 2018.
- Zhang et al., ["Efficient Neural Network Robustness Certification with General Activation
  Functions"](https://arxiv.org/abs/1811.00866), NeurIPS 2018.
- Xu et al., ["Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond"](https://arxiv.org/abs/2002.12920), NeurIPS 2020.
- Wang et al., ["Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for
  Complete and Incomplete Neural Network Robustness Verification"](https://arxiv.org/abs/2103.06624),
  NeurIPS 2021.
- Jia and Rinard, ["Exploiting Verified Neural Networks via Floating Point Numerical
  Error"](https://arxiv.org/abs/2003.03021), 2020.
- de Moura et al., ["The Lean Theorem Prover"](https://lean-lang.org/papers/system.pdf), CADE 2015.
