import VersoManual

open Verso.Genre Manual

#doc (Manual) "Model Zoo" =>
%%%
tag := "modern-models"
%%%


A two-layer perceptron is enough to introduce typed tensors and reverse-mode differentiation. It is
not enough to test whether an ML library has found the right abstractions. Modern architectures add
branches, state, masks, patch grids, spectral transforms, latent variables, and interaction with an
environment. Each addition creates a new place where an informal convention can become a bug.

TorchLean therefore treats the model examples as applications of the same small set of ideas:

1. an architecture has a typed input and output shape;
2. its parameters have an explicit ordered layout;
3. its forward program runs through a selected runtime;
4. mathematical specifications and proofs refer to named objects, not to an opaque training log;
5. an external kernel, dataset, or environment remains visible at the boundary where it enters.

The public constructors live under
[`NN/API/Models`](https://github.com/lean-dojo/TorchLean/tree/main/NN/API/Models).
The runnable applications live under
[`NN/Examples/Models`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models).
That distinction matters. A constructor such as `nn.models.resnet` is reusable and
rank-polymorphic; the `resnet` command chooses a deliberately small CIFAR configuration so that a
reader can run the whole data and training path locally.

# From A Formula To A Typed Model

Consider an ordinary one-hidden-layer network

$$`f_\theta(x)=W_2\,\rho(W_1x+b_1)+b_2.`

For a batch of `B` vectors, the shapes are

$$`X\in\mathbb{R}^{B\times d_{\mathrm{in}}},\qquad
W_1\in\mathbb{R}^{d_{\mathrm{in}}\times d_h},\qquad
W_2\in\mathbb{R}^{d_h\times d_{\mathrm{out}}}.`

The TorchLean constructor records the outer contract as

```
nn.Sequential
  (shape![batch, inDim])
  (shape![batch, outDim])
```

and composes layers only when their intermediate shapes agree. The runtime still performs ordinary
matrix multiplication, bias addition, and activation evaluation. The type checker removes one
class of invalid programs before that runtime is reached.

The same pattern survives when the architecture becomes less linear. What changes is the shape
equation that must be made explicit.

# Residual Networks

A residual block computes

$$`y=F_\theta(x)+S_\theta(x),`

where `S` is either the identity or a projection. The addition is defined only if the two branches
have the same output shape. In
[`NN.API.Models.ResNet`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/ResNet.lean),
the reusable configuration is indexed by the number `d` of spatial axes:

```
structure ResNetConfig (d : Nat) where
  batch          : Nat
  inChannels     : Nat
  spatial        : Vector Nat d
  spatialNonzero : ∀ i, spatial.get i ≠ 0
  hiddenChannels : Nat
  numClasses     : Nat
```

Its input and output are

$$`\operatorname{input}
=B\times C_{\mathrm{in}}\times n_1\times\cdots\times n_d,`

$$`\operatorname{output}=B\times C_{\mathrm{class}}.`

The constructor uses a convolutional stem, two residual blocks, global average pooling over all
`d` spatial axes, and a linear classifier. The pooling operation is parameterized by the spatial
vector. The CIFAR example instantiates `d=2`; the model API itself is not tied to images or to two
dimensions.

Run the compact application after preparing CIFAR:

```
python3 scripts/datasets/download_example_data.py --cifar10
lake exe torchlean resnet --device cpu --n-total 1 --steps 1
```

On the current checkout with seed `0`, the final lines are:

```
dataset size = 1
mean_loss(before) = 2.333221
mean_loss(after) = 2.327496
steps=1 loss0=2.333221 loss1=2.327496
torchlean resnet: ok
```

This is a one-update integration run, not a CIFAR accuracy result. It confirms that the prepared
array, typed residual model, cross-entropy objective, autograd tape, optimizer, and CPU runtime
worked together.

# Three Complete Model Runs

This chapter runs three applications far enough to inspect the objects that move through
TorchLean. The first is a character-level Transformer, where sequence length, masking, and
generation matter. The second compares residual and patch-token vision models on the same prepared
dataset. The third is a Fourier neural operator, where the important boundary is the spectral
kernel rather than an image or token representation.

The commands below were run against the current checkout. The displayed losses are deterministic
for the stated seeds and one-example datasets, but they are not performance benchmarks. Their
purpose is to make the data path, model shape, runtime selection, and generated artifacts concrete.

# Case Study One: CharGPT

The Tiny Shakespeare experiment is the clearest sequence-model application because it begins with
a text file and ends with both a trained parameter file and generated text.

Prepare the corpus:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
```

Then run the two-update smoke configuration:

```
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke \
  --save-params /tmp/chargpt-params.json \
  --log /tmp/chargpt-trainlog.json
```

The current run reports:

```
torchlean chargpt: char-level GPT training
  trainable_parameters=30017
  step 0: val loss=4.207507
  step 1: val loss=4.195092
  step 2: val loss=4.174880
  wrote params: /tmp/chargpt-params.json
  vocab=65 (unique chars)
  architecture=width 32, heads 4, layers 2, dropout 0.000000
  sampled="First Citizen:ITJ?P,bduOc$Eaf'yjhYXGLHkR3vQq;V"
  wrote TrainLog JSON: /tmp/chargpt-trainlog.json
torchlean chargpt: ok
```

The sample is mostly noise because two optimizer updates are not language training. The useful
facts are elsewhere in the trace:

- the corpus produced a 65-character vocabulary;
- the model has 30,017 trainable scalar parameters;
- validation is computed on a disjoint ten-percent corpus suffix;
- the loss decreased at both reported updates;
- parameter and training-log artifacts were written explicitly.

## The Architecture

The command constructs a `CausalTransformerConfig` from the runtime options:

$$`D=32,\qquad H=4,\qquad D_h=D/H=8,\qquad L=2.`

For a batch of token windows `I ∈ Fin V`, the model first gathers embeddings

$$`E[I]\in\mathbb{R}^{B\times T\times D},`

adds a learned positional table, and applies two pre-normalized Transformer blocks. In one block,

$$`\begin{aligned}
Z_1&=X+\operatorname{Dropout}
  \left(\operatorname{MHA}(\operatorname{LN}(X))\right),\\
Z_2&=Z_1+\operatorname{Dropout}
  \left(W_2\,\rho(W_1\operatorname{LN}(Z_1)+b_1)+b_2\right).
\end{aligned}`

The final layer normalization and linear projection produce logits

$$`\operatorname{logits}\in\mathbb{R}^{B\times T\times V}.`

The training target is the same token window shifted by one position. Cross entropy at location
`t` therefore uses the target character `x_{t+1}`.

The mask is semantic, not merely a convenient floating-point bias:

$$`j>i\quad\Longrightarrow\quad
\operatorname{attentionWeight}_{i,j}=0.`

The source of the shared architecture is
[`NN/API/Models/Gpt2.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Gpt2.lean).
The corpus split, configuration presets, evaluation loop, generation, and checkpoint handling are
in
[`NN/Examples/Models/Sequence/CharGpt.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/CharGpt.lean).
