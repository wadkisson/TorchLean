import VersoManual

open Verso.Genre Manual

#doc (Manual) "Probability and Gradient Utilities" =>
%%%
tag := "probability-and-gradients"
%%%

Model proofs are assembled from operator-level facts. TorchLean keeps several of those facts in
small modules: the Gaussian forward kernel used by diffusion models, linear-layer gradient
identities, and derivative rules for activations. They can be imported without committing a proof to
the entire runtime autograd development.

The reusable pieces are:

- *Diffusion noising*: a Gaussian law pushed through an affine map, so generative model proofs can
  cite the forward kernel directly.
- *Linear gradients*: derivatives with respect to input, weights, and bias, used when a local
  layer proof does not need the whole tape theorem.
- *Activation gradients*: scalar derivative rules with domain side conditions, so kink points and
  guarded domains stay visible.

# Probability Utilities

The [diffusion probability API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Probability/DiffusionForward.lean) states the
forward noising measure and kernel construction used by diffusion. The mathematical object is:

$$`\operatorname{forwardNoising}(a,b,x)
= \mathcal{L}\!\left(a\cdot x + b\cdot Z\right),
\qquad Z\sim\mathcal{N}(0,I)`

In DDPM notation, this is the familiar noising equation

$$`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.`

The corresponding Markov kernel is:

$$`\operatorname{forwardKernel}(a,b)(x)
= \operatorname{forwardNoising}(a,b,x)`

The exact schedule that chooses the coefficients belongs to the model spec. The probability proof
material records the measure and kernel facts that can be reused independently of one executable
diffusion example. The theorem names are:

- `forwardNoising_eq_map`: the measure is the map of Gaussian noise through the affine noising
  function.
- `forwardNoising_univ`: the measure assigns mass one to the whole space.
- `forwardKernel_apply`: applying the kernel to `x` gives the forward noising measure at `x`.
- `isGaussian_forwardKernel`: the forward kernel preserves the Gaussian law shape.

These facts complement the generative model theory layer. The runtime diffusion example trains an
epsilon predictor; the theory layer proves selected objective and sampler facts; the probability
utility API gives the forward noising construction a reusable theorem home.

# Local Gradient Theorems

The gradient pages contain direct facts for individual operators. They help when a proof only
needs a local derivative and pulling in a whole tape theorem would be too much machinery.

For a linear layer, the forward map has the familiar shape:

$$`y = Wx + b`

Given an upstream cotangent `\bar y`, the local reverse rules are:

$$`\bar x=W^\top\bar y,\qquad
\bar W=\bar y x^\top,\qquad
\bar b=\bar y.`

For a batched layer, the bias cotangent sums over the batch axis. That small convention is exactly
the kind of detail a local theorem should make explicit.

The [linear gradient API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Gradients/Linear.lean) records the three derivative
directions separately:

- `linear_weight_gradient_correct`: the weight gradient matches the outer product style derivative.
- `linear_input_gradient_correct`: the input gradient is obtained by multiplying by the transposed
  weight action.
- `linear_bias_gradient_correct`: the bias gradient is the upstream cotangent accumulated over the
  output coordinates.
- `linear_gradients_preserve_shapes`: every returned gradient has the intended tensor shape.
- `linear_gradients_mathematical_correctness`: the local pieces are packaged into one correctness
  statement.

The [activation gradient API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Gradients/Activation.lean) does the same for scalar
nonlinearities: `relu_deriv_correct`, `sigmoid_deriv_correct`, `tanh_deriv_correct`,
`gelu_deriv_correct`, `softplus_deriv_correct`, `silu_deriv_correct`, `safe_log_deriv_correct`,
and `smooth_abs_deriv_correct`. The ReLU and ELU style theorems state their differentiability away
from kink points; the smooth activations state the ordinary derivative law everywhere their domains
permit.

These local theorems are not a replacement for the autograd tape proof. They help in a
different way. If a theorem only needs a local statement about one layer, it can cite a small local
gradient theorem instead of pulling in the whole graph backprop layer.

# Where These Utilities Fit

These declarations serve three different proof tasks:

- diffusion probability facts support generative model proofs;
- local gradient facts support operator and model block reasoning;
- larger autograd proofs can coexist with smaller direct derivative theorems.

TorchLean does not need one proof style for every gradient claim. A whole-graph theorem is useful
when the tape or compiler is part of the statement; an operator theorem is clearer when the claim is
only about one derivative.

# References

- Ho, Jain, and Abbeel, ["Denoising Diffusion Probabilistic
  Models"](https://arxiv.org/abs/2006.11239), NeurIPS 2020.
- Baydin et al., ["Automatic Differentiation in Machine Learning: a
  Survey"](https://jmlr.org/papers/v18/17-468.html), JMLR 2018.
- Linnainmaa, ["Taylor Expansion of the Accumulated Rounding
  Error"](https://doi.org/10.1007/BF01931367), BIT Numerical Mathematics, 1976.
