import VersoManual

open Verso.Genre Manual

#doc (Manual) "Generative Models and ML Theory" =>
%%%
tag := "generative-models"
%%%

Generative models stress the library because they mix runtime, probability, objectives, and
sampling. A diffusion example has a noising process, a denoising network, a timestep schedule, and
a sampler. A VAE has an encoder, decoder, latent distribution, KL term, and ELBO. A VQ-VAE has a
codebook and nearest-code objective. A GAN has two networks and two coupled losses.

TorchLean separates the runnable examples from the formal statements. The
[generative model examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/Generative/) show that the TorchLean API and
runtime can train compact programs shaped like diffusion models, autoencoders, VAEs, VQ-VAEs, and
GANs. The [generative theory API](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/Generative/) records the pieces we can presently
cite as checked Lean statements: Gaussian forward noising, sampler Lipschitz facts, ELBO algebra,
KL nonnegativity, nearest code facts, and generator/discriminator objective decompositions.

The split is deliberate. A generated image, reconstruction loss, or adversarial loss trace is a
runtime artifact. A theorem about a forward Gaussian law, a KL term, or a finite codebook is a
mathematical artifact. They should be connected in prose, but not collapsed into the same claim.

# Runtime Examples And Theory Statements

The generative material is split by role. The runnable examples cover the training and data
path; the theorem declarations state the objective or sampler facts that can be checked in Lean.

- *Diffusion*: the [diffusion command](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Diffusion.lean) has
  timestep sampling, epsilon prediction, logs, and PPM samples; the theory side records a forward
  Gaussian law plus sampler Lipschitz and contraction facts.
- *VAE*: the [compact VAE example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Vae.lean) runs over flattened
  image tensors; the theory side records ELBO algebra, beta weighting, and diagonal Gaussian KL
  nonnegativity.
- *VQ-VAE*: the [VQ-VAE example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/VqVae.lean) gives a
  reconstruction path; the theory side records codebook loss decomposition and nearest-code facts.
- *GAN / LSGAN*: the [GAN warmup](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Gan.lean) runs a small
  generator/discriminator objective; the theory side packages generator and discriminator losses.
- *Autoencoder / MAE*: the [autoencoder](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Autoencoder.lean) and
  [masked autoencoder](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Mae.lean) examples connect to reconstruction
  and masked objective algebra.

The formal layer currently proves objective and sampler facts. These are the pieces that future
stronger generative-model theorems can reuse.

Some objective shapes to keep in mind:

$$`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon`

$$`\mathcal L_{\mathrm{VAE}}
=
\mathbb E_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)]
+\beta\,D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))`

$$`L_{\mathrm{VQ}}
=
L_{\mathrm{recon}}
+\|\operatorname{sg}(z_e)-e_k\|^2
+\beta\|z_e-\operatorname{sg}(e_k)\|^2`

$$`L_D=(D(x)-1)^2+D(G(z))^2,\qquad
L_G=(D(G(z))-1)^2.`

The exact Lean declarations name the precise weights, reductions, and scalar choices used by the
corresponding theory file.

A useful source-reading pass is:

```
#check NN.MLTheory.Generative.Diffusion.forwardGaussian_isGaussian
#check NN.MLTheory.Generative.Diffusion.eulerStep_l2_lipschitz_of_rhs_lipschitz
#check NN.MLTheory.Generative.Latent.betaVae_loss_eq_weightedTwoTerm
#check NN.MLTheory.Generative.Latent.diagonalGaussianKlToStandardReal_nonneg
#check NN.MLTheory.Generative.Latent.nearestCode_minimizes_quantization_loss
#check NN.MLTheory.Generative.Latent.generatorLoss_eq_weightedTwoTerm
```

Those names are intentionally local. They do not claim that a trained image generator converges.
They state reusable facts about the objective, distribution, sampler, or finite choice that the
runtime examples are built around.

# Diffusion

Diffusion is the most visibly generative example in the current example set. The executable is exposed by
the [diffusion example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Diffusion.lean), and the API model
constructor is in [NN.API.Models.Diffusion API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Diffusion.lean). The example
uses a compact residual convolutional epsilon predictor because it gives users a real noising and
denoising training loop with an inspectable sampling artifact.

A compact CUDA runtime check looks like:

```
python3 scripts/datasets/download_example_data.py --cifar10
lake exe -K cuda=true torchlean diffusion --cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
```

For a longer image run, increase `--n-total`, `--steps`, `--hidden-c`, and the diffusion horizon:

```
python3 scripts/datasets/download_example_data.py --cifar10
lake exe -K cuda=true torchlean diffusion --cuda --fast-kernels \
  --dataset cifar10 --n-total 128 --steps 50 --hidden-c 8
```

For 64 by 64 local image folders, the example accepts converted arrays in the ImageNet style:

```
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/imagenet/train \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 --labels-from-dirs --limit 2000

lake exe -K cuda=true torchlean diffusion --cuda --fast-kernels \
  --dataset imagenet64 --n-total 800 --steps 200 --hidden-c 8 \
  --sample-ppm data/model_zoo/imagenet64_sample.ppm
```

The theory layer covers a different, more mathematical part of the subject:

- The [forward Gaussian API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion/ForwardGaussian.lean) defines
  `forwardGaussian` and proves `forwardGaussian_isGaussian`, so the forward noising distribution is
  recorded as a Gaussian law in Lean.
- The [diffusion samplers API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion/Samplers.lean) states base case identities
  for DDPM, DDIM, and probability flow Euler samplers, then proves Lipschitz and contraction style
  facts such as `eulerStep_l2_distance_bound`, `eulerStep_l2_lipschitz_of_rhs_lipschitz`,
  `ddimStepSystem_contracts_of_step_contracts`, and
  `pfOdeEulerSystem_contracts_of_step_lipschitz`.

Those are local sampler facts, not a full diffusion convergence theorem. The executable
still owns the empirical ML workflow: data loading, timestep sampling, optimizer steps, logs, and
image artifacts. The theory layer owns citeable statements about the mathematical objects that the
runtime example is named after.

The shortest honest diffusion claim therefore looks like:

```
The `diffusion` command ran a compact epsilon-prediction training loop and wrote a sample/log.
The forward noising law used in the mathematical account is represented by `forwardGaussian`.
The theorem `forwardGaussian_isGaussian` proves that this law is Gaussian.
```

Anything stronger, such as sample quality, FID, convergence of the learned denoiser, or equivalence
between a CUDA training run and a real-valued sampler theorem, needs a separate artifact.

The informal definition is:

$$`\operatorname{forwardGaussian}(c_0,c_1,x_0)
= \mathcal{L}\!\left(c_0x_0+c_1Z\right),
\qquad Z\sim\mathcal{N}(0,I)`

and the theorem `forwardGaussian_isGaussian` says that this law is Gaussian with the expected mean
and covariance transformation. Sampler facts then talk about one step maps:

$$`x_{t-1} = \operatorname{step}(t,x_t,\operatorname{scoreOrNoise})`

under Lipschitz or contraction hypotheses. The proof does not inspect image quality; it proves
properties of the mathematical step map.

The usual paper lineage is DDPM by Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models"
(NeurIPS 2020, https://arxiv.org/abs/2006.11239), DDIM by Song, Meng, and Ermon, "Denoising
Diffusion Implicit Models" (ICLR 2021, https://arxiv.org/abs/2010.02502), and score based SDEs by
Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR
2021, https://arxiv.org/abs/2011.13456).

# VAE And The ELBO Boundary

The VAE example is kept modest. In
[NN.Examples.Models.Generative.Vae source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Vae.lean), the model
trains on flattened CIFAR images with a supervised target that keeps reconstruction channels near
the input and latent mean/log variance proxy channels near zero. It is a runnable beta VAE style
path through the TorchLean model API; it is not a full stochastic variational inference experiment.

The proof statements are sharper:

- [NN.Spec.Models.Vae API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/Vae.lean) names the VAE forward pass and loss shape.
- [NN.MLTheory.Generative.Latent.VAE API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/VAE.lean) proves
  objective decompositions such as `betaVae_loss_eq_weightedTwoTerm`,
  `betaVae_loss_eq_reconstruction_of_zero_kl`, and `betaVae_loss_mono_beta_of_kl_nonneg`.
- The same file proves diagonal Gaussian KL facts, including
  `coordinateKlToStandard_nonneg`, `diagonalGaussianKlToStandardReal_nonneg`, and
  `diagonalGaussianKlToStandardReal_eq_zero_iff`.
- It also records the reparameterization side with `scalar_reparameterization_law` and
  `diagonal_reparameterization_coordinate_law`, plus named ELBO helpers `negativeElbo` and
  `betaNegativeElbo`.

The precise VAE claim is: the runtime example trains a compact
model shaped like a VAE, while the ML theory declarations prove algebraic and distributional facts about the
VAE objective and diagonal reparameterization. That distinction is healthier than claiming that a
tiny CIFAR runtime check formalizes all of variational inference.

The basic objective shape is:

$$`\operatorname{negativeElbo}(terms)
= \operatorname{reconstruction}(terms)+\operatorname{KL}(terms)`

$$`\operatorname{betaNegativeElbo}(\beta,terms)
= \operatorname{reconstruction}(terms)+\beta\operatorname{KL}(terms)`

The KL theorem `diagonalGaussianKlToStandardReal_nonneg` is the guarantee to remember: the diagonal
Gaussian KL to the standard normal is nonnegative, and it is zero exactly in the standard normal case
stated by `diagonalGaussianKlToStandardReal_eq_zero_iff`.

The paper anchor is Kingma and Welling, "Auto-Encoding Variational Bayes" (ICLR 2014,
https://arxiv.org/abs/1312.6114). For beta-VAE terminology, the common citation is Higgins et al.,
"beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (ICLR 2017,
https://openreview.net/forum?id=Sy2fzU9gl).

# VQ-VAE

The VQ-VAE runtime file,
[NN.Examples.Models.Generative.VqVae source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/VqVae.lean), uses a
small vector reconstruction model with a narrow bottleneck. That shape keeps the example runnable
while making the intended VQ-VAE connection explicit in the nearby spec and theory modules.

The theorem statements are in:

- [NN.Spec.Models.VqVae API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/VqVae.lean), which states quantization and loss
  equations for a codebook model.
- [NN.MLTheory.Generative.Latent.VQVAE API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/VQVAE.lean), which
  proves `vqvae_loss_eq_weightedThreeTerm`,
  `vqvae_loss_eq_reconstruction_of_zero_quantization`,
  `vqvae_loss_mono_beta_of_commitment_nonneg`, and nearest code facts such as
  `exactCodeMatch_isNearestCode` and `nearestCode_minimizes_quantization_loss`.

Here again, the boundary is precise. The runnable example exercises TorchLean's training API for
a reconstruction model shaped like a VQ-VAE. The Lean theory proves that the named VQ-VAE objective splits
into reconstruction, codebook, and beta weighted commitment terms, and that selected nearest codes
minimize the stated finite squared distance objective.

The nearest code predicate is the key mathematical object:

$$`\begin{aligned}
\operatorname{IsNearestCode}(codebook,z,k)
:= \forall j,\;&
\operatorname{squaredL2}(z,codebook(k))\\
&\le \operatorname{squaredL2}(z,codebook(j))
\end{aligned}`

From that, `nearestCode_minimizes_quantization_loss` turns the finite codebook choice into the
stated quantization loss minimum.

The paper anchor is van den Oord, Vinyals, and Kavukcuoglu, "Neural Discrete Representation
Learning" (NeurIPS 2017, https://arxiv.org/abs/1711.00937).

# GAN And LSGAN

GANs are especially easy to overstate. The
[NN.Examples.Models.Generative.Gan source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Gan.lean) is a small
LSGAN style executable path. It trains a generator from latent noise toward a CIFAR minibatch as a
stable warm up objective, and it trains a discriminator on real CIFAR images and deterministic noise
images. It is not a full alternating adversarial training recipe with all of the empirical care that
production GANs require.

The formal side is captured by two declaration groups:

- [NN.Spec.Models.Gan API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/Gan.lean), which names generation, fake scores, and
  discriminator losses.
- [NN.MLTheory.Generative.Latent.GAN API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/GAN.lean), which
  proves objective packaging lemmas such as `generatorLoss_eq_weightedTwoTerm` and
  `discriminatorLoss_eq_weightedThreeTerm`, plus zero loss facts
  `generatorLoss_zero_of_fake_score_real` and `discriminatorLoss_zero_of_perfect_scores`.

The citation line is clean: TorchLean contains checked algebra for LSGAN style scalar
objectives and a runnable generator/discriminator example. Convergence and distribution-matching
claims would require additional theorem layers.

The objective packaging is kept simple:

$$`\begin{aligned}
\operatorname{generatorLoss}
&= \text{weighted two-term objective},\\
\operatorname{discriminatorLoss}
&= \text{weighted three-term objective}
\end{aligned}`

$$`\begin{aligned}
\text{perfect discriminator scores}
&\Longrightarrow \operatorname{discriminatorLoss}=0,\\
\text{fake score marked real}
&\Longrightarrow \operatorname{generatorLoss}=0
\end{aligned}`

Those are compact theorems, but they matter because they pin down which scalar objective the example is
claiming to optimize.

The original GAN reference is Goodfellow et al., "Generative Adversarial Nets" (NeurIPS 2014,
[paper page](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)).
The least squares variant is Mao et al., "Least Squares Generative Adversarial Networks" (ICCV
2017, https://arxiv.org/abs/1611.04076).

# Autoencoders And Masked Autoencoders

The plain autoencoder in
[the autoencoder example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Autoencoder.lean) is the simplest
reconstruction baseline: load a CIFAR vector batch, run an encoder/decoder, and optimize MSE. With
the probabilistic and adversarial interpretation removed, this example exposes the runtime path
itself.

The masked autoencoder in [the MAE example source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Mae.lean) is more
architectural. It loads CIFAR image tensors, masks deterministic patches, embeds patch tokens with a
small ViT encoder, and trains a decoder head to reconstruct a flattened prefix of the image. The
example includes real masking and patch tokens, but it is not a large asymmetric MAE pretraining
run.

The self-supervised theory side lives under
[NN.MLTheory.SelfSupervised](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/SelfSupervised/).
For MAE, the useful finite-patch hooks are `maeLoss`, `maeLoss_append`, `maeLoss_reverse`, and
`exactReconstruction_identity`. They prove facts about a finite masked-patch objective, not about
large-scale representation quality.

The paper anchor for MAE is He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR
2022, https://arxiv.org/abs/2111.06377).

# Claim Scope

For generative models, the safest citation discipline is:

- For a command that trains or samples through TorchLean, cite the corresponding file in
  `NN/Examples/Models/Generative`.
- For the diffusion forward law being Gaussian, cite `forwardGaussian_isGaussian` in the forward
  Gaussian API.
- For a sampler step being Lipschitz or contractive under hypotheses, cite the named theorems in the
  diffusion samplers API.
- For a VAE objective decomposing into reconstruction plus beta-weighted KL, cite
  `betaVae_loss_eq_weightedTwoTerm`.
- For diagonal Gaussian KL nonnegativity, cite `diagonalGaussianKlToStandardReal_nonneg`.
- For a VQ-VAE loss splitting into reconstruction, codebook, and commitment terms, cite
  `vqvae_loss_eq_weightedThreeTerm`.
- For nearest-code minimization in a finite codebook, cite `nearestCode_minimizes_quantization_loss`.
- For LSGAN generator and discriminator losses as weighted objectives, cite
  `generatorLoss_eq_weightedTwoTerm` and `discriminatorLoss_eq_weightedThreeTerm`.

The current development focuses on objective and sampler facts, plus runnable examples. Global
distribution matching, convergence of SGD, image quality, FID, and complete training-step
equivalence are separate claims that would require additional theorem layers. That scope is part of
the discipline: TorchLean can run programs shaped like ML systems, and we can grow the formal
island around the parts whose meaning we are ready to state precisely.

# References

- Ho, Jain, and Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239),
  2020.
- Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502),
  2020/2021.
- Song et al., [*Score-Based Generative Modeling through Stochastic Differential
  Equations*](https://arxiv.org/abs/2011.13456), 2020/2021.
- Kingma and Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114),
  2013/2014.
- van den Oord, Vinyals, and Kavukcuoglu,
  [*Neural Discrete Representation Learning*](https://arxiv.org/abs/1711.00937), 2017.
- Goodfellow et al., [*Generative Adversarial Nets*](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html),
  2014.
- Mao et al., [*Least Squares Generative Adversarial Networks*](https://arxiv.org/abs/1611.04076),
  2016/2017.
- He et al., [*Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377),
  2021/2022.
