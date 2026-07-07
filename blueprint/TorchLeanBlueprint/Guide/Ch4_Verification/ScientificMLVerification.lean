import VersoManual

open Verso.Genre Manual

#doc (Manual) "Scientific ML Verification" =>
%%%
tag := "scientific-ml-verification"
%%%

TorchLean also has verification bridges for scientific ML artifacts: ODE enclosures, PINN residual
certificates, and piecewise polynomial or spline certificates. These are not side examples. They show
the same producer/checker discipline when the artifact comes from numerical analysis, Julia,
Python, or a scientific ML workflow rather than from a standard neural network classifier.

The shared certificate pattern is:

- *ODE enclosure*: an external producer may integrate, search for a tube, or tune step sizes; the
  Lean checker insists on interval conditions for each segment of the claimed corridor.
- *PINN certificate*: Python may train a neural PDE surrogate; Lean checks the architecture,
  parameters, PDE expression, domain boxes, and residual bounds.
- *Spline or piecewise polynomial*: Julia or another system may fit the certificate; Lean checks
  explicit rational pieces, interval conditions, and the named serialization format.

For classifier verification, the artifact is often an input box and logit bounds. For scientific ML,
the artifact may be a time corridor, a residual bound, a polynomial certificate, or a derivative
enclosure. TorchLean makes all of these look like the same proof pattern: a producer proposes a
finite object, Lean checks the object, and a theorem states what follows.

# ODE Enclosures

An ODE enclosure certificate is a finite description of a corridor around a trajectory. The checker
does not depend on an external integrator's explanation of the run. It parses the ODE expression,
evaluates interval bounds over each segment, and checks that the claimed tube is closed under the
vector field with the required margins.

The mathematical object is an ODE

$$`\dot x(t)=f(t,x(t)).`

A corridor certificate gives boxes `X_i` over time intervals `[t_i,t_{i+1}]`. The theorem shape is:

$$`x(t_i)\in X_i
\quad\text{and}\quad
f([t_i,t_{i+1}],X_i)\subseteq \dot X_i
\quad\Longrightarrow\quad
x(t)\in X_i\ \text{for}\ t\in[t_i,t_{i+1}].`

The executable side is exposed through the
[ODE checker API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/ODE/Verify.lean). The core pieces are the expression AST,
the interval evaluator, the segment certificate, and the final checker result.

The concrete executable declarations are small enough to audit:

```
#check NN.Verification.ODE.Expr
#check NN.Verification.ODE.eval
#check NN.Verification.ODE.Verify.ODECertificateSegment
#check NN.Verification.ODE.Verify.ODECertificate
#check NN.Verification.ODE.Verify.checkSub
#check NN.Verification.ODE.Verify.checkSuper
```

The theorem side is the real mathematical statement. In the
[ODE enclosure API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Verification/ODE/Enclosure.lean), a corridor theorem says, in
plain language:

> If the initial state lies in the first set, and each segment satisfies the enclosure condition for
> the ODE vector field, then the true solution remains inside the certified corridor for the
> covered time interval.

The backend bridge in
[ODE enclosure backends](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Verification/ODE/EnclosureBackends.lean) explains how
backend valued trajectories, including FP32 and `IEEE32Exec` views, can be related back to the real
statement through explicit interpretation maps.

Lean has a local enclosure theorem, and the executable checker puts imported ODE or PINN artifacts
into the shape that theorem expects. Broader neural ODE and integrator claims need their own
enclosure conditions and agreement evidence.

The trusted boundary is therefore:

```
external integrator/search -> proposed tube JSON
Lean parser/checker        -> interval side conditions for the tube
ODE theorem                -> statement about true trajectories, if theorem hypotheses match
runtime bridge             -> needed for a claim about a concrete finite-precision integrator
```

# PINN Certificates

PINN verification is more than "run the neural network." The artifact has at least four claims:

1. the imported parameters match the architecture;
2. the PDE expression is the one being checked;
3. the residual is bounded over the domain;
4. boundary or dataset constraints are respected.

TorchLean gives each piece a small object. The
[PINN architecture API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/Architecture.lean) names sequential network
records and graph construction. The
[PDE expression API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/PdeAst.lean) names the PDE language. The
[PyTorch parameter store API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/PyTorch/ParamStore.lean) names imported
parameters instead of letting a raw tensor dictionary float around unchecked. The
[residual affine API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/ResidualAffine.lean) contains the bound helpers,
including McCormick style pieces and branch and bound support.

The certificate level combines the architecture, imported parameters, PDE expression, and domain
boxes, then checks residual bounds and produces an accepted certificate with a theorem-ready
residual proposition.

For a Burgers-style residual, the mathematical claim has the shape:

$$`R_\theta(t,x)
=
\partial_t u_\theta(t,x)
+u_\theta(t,x)\partial_x u_\theta(t,x)
-\nu\partial_{xx}u_\theta(t,x).`

The certificate target is a uniform bound over the domain:

$$`\forall (t,x)\in\Omega,\qquad |R_\theta(t,x)|\le\varepsilon.`

Boundary or data conditions have the same form:

$$`\forall z\in\partial\Omega,\qquad |u_\theta(z)-g(z)|\le\varepsilon_b.`

The [PINN certificate API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/Certificate.lean), the
[dataset checker API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/DatasetCheck.lean), and the
[PINN command API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/CLI.lean) are the user-facing pieces for that path.

Important Lean objects:

```
#check NN.Verification.PINN.SequentialPINNArch
#check NN.Verification.PINN.buildGraph
#check NN.Verification.PINN.PdeAst.Expr
#check NN.Verification.PINN.PdeAst.eval
#check NN.Verification.PINN.ResidualAffine.crownUBoundsForward
#check NN.Verification.PINN.DatasetCheck.DatasetCheckOpts
```

PINNs are a good stress test because the model is only part of the claim. The PDE residual, the
domain, the boundary data, and the imported parameters all matter. TorchLean's design makes those
pieces explicit in the certificate object.

The reference point for the application is Raissi, Perdikaris, and Karniadakis,
["Physics-informed neural networks"](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
(Journal of Computational Physics 2019; arXiv preprint
[1711.10561](https://arxiv.org/abs/1711.10561)). That paper motivates the residual objective.
TorchLean's checker makes a narrower claim: for an exported architecture, parameters, PDE
expression, and domain boxes, the certificate's residual and dataset checks pass the predicates
implemented in Lean.

# Piecewise Polynomial and Spline Certificates

The spline path is concentrated in
[NN.Verification.Splines.PiecewisePolyCert API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Splines/PiecewisePolyCert.lean).
It parses `piecewise_poly_v0` JSON, checks rational polynomial pieces, evaluates polynomials by
Horner's rule, and also has an `IEEE32Exec` exact conversion path.

A piecewise polynomial certificate names intervals `I_i` and polynomial pieces

$$`p_i(x)=\sum_k a_{ik}x^k,\qquad x\in I_i.`

The checker validates interval claims such as:

$$`\forall x\in I_i,\qquad p_i(x)\in[\ell_i,u_i].`

The example follows the external-tool pattern:

1. another system may generate a piecewise polynomial artifact;
2. the artifact is serialized into a small explicit certificate format;
3. Lean parses and checks that format;
4. any remaining producer hypothesis is named instead of hidden.

# Artifact Boundary Examples

The same scientific artifact can support different strengths of claim depending on what it exports.

- If a PINN JSON contains only sampled residuals, Lean can check those samples; it cannot infer a
  uniform residual bound over the domain.
- If a PINN certificate contains interval or affine residual bounds over domain boxes, Lean can
  check those box obligations and state a uniform residual claim for the boxes covered by the
  certificate.
- If an ODE artifact contains a proposed trajectory but no interval enclosure condition, Lean can
  parse the trajectory but does not get an enclosure theorem.
- If a piecewise polynomial artifact contains rational coefficients and interval bounds for each
  piece, Lean can check the finite polynomial obligations directly.

This is the same checked/proved/assumed distinction used for robustness certificates. The producer
may be a numerical solver; the theorem applies only to the artifact fields that Lean checked or to
producer hypotheses named in the statement.

# Scientific Artifacts In The Same Trust Story

Scientific ML often lives at the boundary between theorem proving and numerical tooling. TorchLean's
verification layer includes ODEs, PINNs, and splines alongside image classifiers and language
models. These sections give readers a clear place to start when their question is:

> How do I bring a scientific artifact into Lean without treating the external producer as
> already verified?

The answer is the same pattern we use elsewhere: small certificate formats, explicit parsers,
checked predicates, and theorem statements that say exactly which mathematical claim follows.

# References

- Maziar Raissi, Paris Perdikaris, and George Em Karniadakis,
  ["Physics-informed neural networks"](https://www.sciencedirect.com/science/article/pii/S0021999118307125),
  Journal of Computational Physics 2019; preprint
  [arXiv:1711.10561](https://arxiv.org/abs/1711.10561).
