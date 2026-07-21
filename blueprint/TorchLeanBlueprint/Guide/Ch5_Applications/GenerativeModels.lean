import VersoManual

open Verso.Genre Manual

#doc (Manual) "Generative Models And RL" =>
%%%
tag := "generative-models"
%%%


Generative modeling forces several kinds of reasoning into one program. There is a trainable
network, but also a probability law, a noise source, an objective with several terms, and a
sampling procedure that may not resemble the training pass. TorchLean does not package all of this
under one vague claim of “verified generation.” It gives the pieces separate Lean definitions and
connects them where the current proofs justify the connection.

This chapter begins with one complete diffusion run. It then uses the VAE, VQ-VAE, GAN, and masked
autoencoder examples to show where executable training and formal objective theory currently meet.

# A Diffusion Run From Data To Sample

The maintained
[`diffusion` application](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Diffusion.lean)
supports prepared CIFAR-10 arrays and converted `64 × 64` image folders. Its compact CIFAR branch
crops images to `2 × 2` for a fast end-to-end check.

Prepare CIFAR and run one CPU update:

```
python3 scripts/datasets/download_example_data.py --cifar10

lake exe torchlean diffusion --device cpu \
  --dataset cifar10 --n-total 1 \
  --steps 1 --hidden-c 1 --T 2 \
  --log /tmp/diffusion-trainlog.json
```

The command prints the exact typed network before training:

```
torchlean diffusion: diffusion trainer (device=cpu)
model:
Sequential: [1, 4, 2, 2] -> [1, 3, 2, 2], layers=7, params=15
  [0] Conv2d(4, 1): [1, 4, 2, 2] -> [1, 1, 2, 2]
  [1] ReLU: [1, 1, 2, 2] -> [1, 1, 2, 2]
  [2] Conv2d(1, 1): [1, 1, 2, 2] -> [1, 1, 2, 2]
  [3] ReLU: [1, 1, 2, 2] -> [1, 1, 2, 2]
  [4] Conv2d(1, 1): [1, 1, 2, 2] -> [1, 1, 2, 2]
  [5] ReLU: [1, 1, 2, 2] -> [1, 1, 2, 2]
  [6] Conv2d(1, 3): [1, 1, 2, 2] -> [1, 3, 2, 2]
steps=1 loss0=1.093821 loss1=1.092744
  wrote TrainLog JSON: /tmp/diffusion-trainlog.json
torchlean diffusion: ok
```

Why does the input have four channels while the output has three? The clean image has three RGB
channels. The noised image receives one additional channel containing the normalized timestep.
For batch `B`, data channels `C`, and spatial extent `S`, the model contract is

$$`B\times(C+1)\times S
\longrightarrow
B\times C\times S.`

The public constructor
[`nn.models.epsConvNet`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Diffusion.lean)
is parameterized by arbitrary spatial rank. The runnable command instantiates two spatial axes and
uses four `1 × 1` convolutions. A stronger residual same-resolution denoiser also exists in the API,
but it is not the default command path.

# Forward Noising

Let `β_t` be the variance schedule, `α_t=1-β_t`, and

$$`\bar\alpha_t=\prod_{s=0}^{t}\alpha_s.`

The DDPM forward process can sample timestep `t` directly:

$$`x_t
=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).`

The training sample stores the noised image and timestep as input and the same `ε` as target. The
network therefore learns an epsilon predictor `ε_θ(x_t,t)` by mean squared error.

TorchLean keeps randomness outside the pure noising helper. In
[`NN.API.Models.Diffusion`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Diffusion.lean),
`noisedSampleFromEps` receives an explicit noise tensor. `noisedSample` obtains a reproducible
tensor from a `(seed, step)` pair and then calls the pure helper. This separation makes it possible
to state properties of the noising map without treating ambient randomness as an invisible global
effect.

The mathematical forward law is developed separately in
[`NN.MLTheory.Generative.Diffusion.ForwardGaussian`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion/ForwardGaussian.lean).
The theorem

```
NN.MLTheory.Generative.Diffusion.forwardGaussian_isGaussian
```

proves that the affine transformation of a Gaussian used by `forwardGaussian` is again Gaussian,
with the corresponding mean and covariance. It is a theorem about a probability law. It is not a
proof that the runtime RNG produced independent standard-normal bits, nor that a trained denoiser
matches the exact score.

# Reverse Sampling

The command uses deterministic DDIM updates for reconstruction and sample artifacts. Given adjacent
schedule values and a predicted noise tensor,

$$`\widehat x_0
=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\widehat\epsilon_t}
       {\sqrt{\bar\alpha_t}},`

# Reinforcement Learning

In supervised learning, the dataset is usually fixed before training starts. In reinforcement
learning, the current policy helps create its own future data. A complete application therefore
contains more than a neural network:

$$`\text{environment}
\longrightarrow\text{transition}
\longrightarrow\text{rollout or replay}
\longrightarrow\text{return and advantage}
\longrightarrow\text{policy/value update}.`

Each arrow carries assumptions. Is the observation shape correct? Is the action valid? Is the
reward finite? Does `done` mean termination, truncation, or either? Did the rollout keep its fields
aligned? TorchLean gives those questions explicit runtime and specification objects.

There are three complementary layers:

- [`NN.Spec.RL`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL.lean) defines
  environments, MDPs, Bellman operators, returns, and advantages;
- [`NN.Runtime.RL`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL.lean) implements
  checked transitions, replay, rollouts, PPO, and Gymnasium communication;
- [`NN.Proofs.RL`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Core.lean) proves
  structural, numerical, and dynamic-programming facts about named objects from the first two
  layers.

The easiest way to see the pieces together is a small GridWorld run.

# A Complete PPO GridWorld Run

The environment is a `4 × 4` GridWorld defined in Lean. The actor and critic use a fixed rollout
horizon of 64. Run one update on CPU and send the artifacts to temporary files:

```
lake exe torchlean ppo_gridworld --device cpu \
  --updates 1 \
  --eval-every 1 --eval-episodes 1 --eval-max-steps 8 \
  --log /tmp/ppo-gridworld-trainlog.json \
  --policy /tmp/ppo-gridworld-policy.json \
  --path /tmp/ppo-gridworld-path.json
```

The current checkout produces:

```
torchlean ppo_gridworld: PPO on Lean-native GridWorld (4x4, horizon=64) (device=cpu)
  env: pure Lean dynamics + boundary contract check + formal MDP validity proof available
  eval(step=0) avg_return=-0.400000
  update=0 avg_return=3.600000
  wrote TrainLog JSON: /tmp/ppo-gridworld-trainlog.json
torchlean ppo_gridworld: wrote policy snapshot to /tmp/ppo-gridworld-policy.json
torchlean ppo_gridworld: wrote path snapshot to /tmp/ppo-gridworld-path.json
torchlean ppo_gridworld: done
torchlean ppo_gridworld: ok
```

This trace contains three different results:

1. the PPO/autograd program completed one update;
2. a greedy-policy evaluation changed on this seeded run;
3. the command wrote a scalar curve, a policy table, and a decoded path.

Only the first is an optimizer execution fact. The second is an empirical observation from one
small run. The third gives inspectable evidence. The formal MDP facts described below are separate
theorems.

The implementation is
[`NN/Examples/Models/RL/PPOGridWorld.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOGridWorld.lean).
The pure environment is
[`NN.Spec.RL.Envs.GridWorld`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/Envs/GridWorld.lean).

# What Enters A Rollout

For a discrete action space of size `A`, one PPO step stores:

$$`(s_t,\;a_t,\;\log\pi_{\mathrm{old}}(a_t\mid s_t),\;
r_t,\;d_t,\;V(s_t),\;V(s_{t+1})).`

The Lean structure in
[`NN.Runtime.RL.PPO.Rollout`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/PPO/Rollout.lean)
uses:

- `state : Tensor α obsShape`;
- `action : Fin nActions`;
- scalar old log probability, reward, value, and next value;
- a Boolean episode-boundary marker.

A `Rollout α obsShape nActions horizon` contains an array plus a proof that its size is exactly
`horizon`. This avoids the “parallel arrays drifted out of sync” failure common in hand-written
buffers.
