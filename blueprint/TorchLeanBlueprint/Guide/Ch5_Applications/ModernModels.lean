import VersoManual

open Verso.Genre Manual

#doc (Manual) "Modern Models and Training" =>
%%%
tag := "modern-models"
%%%

The model examples are where the typed API meets the messier parts of ML systems: image patches,
attention masks, scan state, stochastic schedules, spectral kernels, and environment rollouts. A
two-layer MLP is enough to explain typed tensors and autograd. It is not enough to test whether the
framework handles the shapes that modern ML actually uses.

The examples below are best read as semantic stress tests. Each model family exercises a
different source of modeling complexity: KAN edge bases, residual branches, patch tokens, causal
masks, scan state, diffusion schedules, spectral kernels, RL trajectories, and CUDA runtime paths.

# What Each Family Stresses

The examples are compact, but each one touches a real source of ML complexity:

| Model family | What it stresses |
|---|---|
| MLP / KAN / CNN | typed tensors, edge bases, losses, optimizers, data loaders |
| Residual / ResNet specs | residual shape agreement and branch joins |
| ViT | image patches, token dimensions, attention blocks |
| GPT models | causal windows, token ids, masks, save/load |
| Mamba | recurrent state, selective scan, prefix causality |
| Diffusion | stochastic schedules, denoising objectives, sampling artifacts |
| FNO | spectral convolution, scientific data, cuFFT boundary |
| PPO / RL | trajectories, environment boundary, policy/value losses |

KANs are included as a model family rather than a task wrapper: the model supplies learned
one-dimensional edge functions, while the public trainer still chooses regression, classification,
or a custom loss. The current built-in basis is triangular and piecewise linear; future spline
families should fit the same edge family slot instead of adding task-specific KAN constructors.

A formal ML library that only works for one MLP is easy to make look clean. The harder test is
whether the same ideas survive edge basis models, residual sharing, attention masks, token windows,
recurrent state, stochastic sampling, spectral transforms, and external environments. The model
examples are where that test happens.

The useful question is not "does TorchLean include a fashionable acronym?" It is:

- what object enters the typed fragment;
- what state or parameter shapes must remain coherent;
- what runtime path is exercised;
- what later checker, theorem, widget, or explicit assumption can talk about the result.

# Runner Entry Point

Most model examples go through one command shape:

```
lake exe torchlean <example> [flags...]
```

The subcommands are registered in the [model runner API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean), and
the model implementations are collected by the [model examples API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/).

The runner is there so users can learn one command shape instead of remembering twenty separate
`lean --run` invocations.

# A Small Tour

The best first run is still a small one:

```
lake exe torchlean mlp --device cpu --steps 10
```

Once CUDA has been built, the same command shape works on GPU:

```
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 20
```

That one command runs through the public API, the eager tape, the optimizer, and the selected
runtime backend. The rest of the model examples grow from that same shape.

# Feedforward, Convolutional, And Vision Models

The classical supervised examples have easy-to-recognize behavior.

```
lake exe torchlean mlp --device cpu --steps 10
lake exe torchlean kan --device cpu --steps 10
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

These examples show:

- the same training loop works for vectors, images, residual blocks, and patch/attention
  vision models;
- KAN-style learned edge functions fit through the same trainer and loss vocabulary as ordinary
  supervised models;
- model parameters are ordinary TorchLean parameters, not hidden Python objects;
- the examples can use prepared real data under `data/real` when available.

The commands above are compact.  They are runtime checks, not leaderboard experiments.

The model code follows the same pattern as the running example:

```
import NN.API

open TorchLean

def smallVisionHead : nn.M (nn.Sequential (.dim 64 .scalar) (.dim 10 .scalar)) :=
  nn.Sequential![
    nn.Linear 64 32,
    nn.ReLU,
    nn.Linear 32 10
  ]
```

The larger CNN and ViT commands replace this small head with convolution or patch/attention blocks.
Residual blocks use the same typed model construction API, even though the maintained model example
runtime command set currently keeps vision training to CNN and ViT.

KAN changes the local layer shape instead of the training loop. A KAN edge model can be read as

$$`y_j=\sum_i \phi_{ij}(x_i)`

where each edge function `\phi_{ij}` is learned from a small basis family. In TorchLean's current
examples the basis is compact and piecewise linear, which is enough to test whether parameterized
edge functions, loss computation, and optimizer updates use the same public training surface as MLPs.

For residual models, the semantic shape is:

$$`\operatorname{block}(x)=x+F_\theta(x)`

The theorem burden, when we prove one, is to show both branches have the same shape and that the
runtime graph computes that sum, rather than producing some tensor with a compatible size.

# Sequence Models: RNN, LSTM, Transformer

The sequence examples give a gentle path from recurrent state to attention:

```
lake -R -K cuda=true exe torchlean rnn --device cuda --tiny-shakespeare --steps 1
lake -R -K cuda=true exe torchlean lstm --device cuda --tiny-shakespeare --steps 1
lake -R -K cuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
```

A one-step run does not learn language. It does exercise the full data path: text is loaded, token
windows are constructed, tensors are built in Lean, and the model trains through TorchLean's
runtime.

The three examples also separate three semantic burdens:

- RNN: a hidden state is updated at every time step;
- LSTM: hidden and cell state travel together, so the transition carries more state than the output;
- Transformer: the sequence is processed through attention, so mask layout and token position become
  part of the model meaning.

The core state signatures are:

$$`(h_t,x_t)\mapsto(h_{t+1},y_t)`

and

$$`(h_t,c_t,x_t)\mapsto(h_{t+1},c_{t+1},y_t).`

# GPT Language Models

TorchLean has two GPT examples:

- `gpt2`
  a small GPT model suitable for runtime checks;
- `text_gpt2`
  a file-backed corpus trainer, including a path that can use GPT-2 BPE vocabulary and merges.

Typical commands:

```
lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1
lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
  --data-file data/real/text/tinystories_valid.txt --allow-small-data --steps 100
```

The `text_gpt2` example is explicit about scale. It is a TorchLean-native miniature
language model. It can overfit small windows and exercise a real tokenizer/data path, but it is not
OpenAI GPT-2-small and should not be described as pretrained GPT-2.

The data path is:

- read a corpus file;
- tokenize into bounded ids;
- form causal windows;
- train next-token prediction with integer labels;
- optionally save parameters;
- reload the parameter file and prompt the saved model.

The boundary is the part worth studying. A tiny local run will not chat like a large pretrained
model, but it does exercise the same data, token, parameter, and loss conventions TorchLean needs
for serious sequence model work.

The language model objective is the familiar next token map:

$$`\operatorname{logits}_t =
\operatorname{model}_\theta(tokens_{<t}),
\qquad
L = -\sum_t \log p_\theta(tokens_t \mid tokens_{<t})`

The compact examples make token ids, causal windows, parameter files, and generation traces explicit
objects so later correctness work has something concrete to talk about.

The API supports two token representations. One-hot tokens are convenient for bounded examples and
teaching. Integer token ids are the representation used by file-backed language-model runs.
The GPT helper layer therefore separates the embedding boundary from the Transformer body:

```
nn.causalTransformerFromEmbeddings
nn.causalTransformerOneHot
nn.causalTransformerTokenScalarModuleDef
```

The integer-token loss uses row-wise class labels:

```
TorchLean.F.embeddingBatchSeqNat
TorchLean.Loss.crossEntropyRowsNat
```

So a language-model batch can store flattened token ids and targets instead of materializing a
large one-hot target tensor.  The model body is the same causal Transformer shape; only the input
boundary changes.

Use the generated text carefully. A sample can show that the tokenizer, model, save/load path, and
prompting loop are wired together. It does not show pretrained-model quality, benchmark
performance, or factual behavior. The relevant checked objects are token bounds, causal masks,
parameter files, and the specific runtime command that produced the log.

# Mamba State Space Models

The Mamba example exercises selective scan computation:

```
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```

Under the CUDA backend, TorchLean includes native selective-scan kernels for the float32 path.  The
model is small, but the architectural lesson is real: not every modern sequence model is attention,
and TorchLean's runtime has begun to cover that broader operator family.

The state space shape is:

$$`h_{t+1}=A_t h_t+B_t x_t,
\qquad
y_t=C_t h_t+D_t x_t`

That equation is why scan order and prefix causality matter. The state space proof chapters state
that future tokens do not change prefix outputs for the supported scan definitions.

Mamba belongs here because it keeps the sequence chapter from becoming attention-only. It also gives
CUDA a different kind of stressor: a scan-like operator whose correctness questions are about
prefixes, state layout, and broadcasted parameters rather than softmax weights.

# Diffusion

The diffusion example is a compact denoising training loop:

```
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
```

Diffusion is included because it stresses a different kind of ML program: a stochastic noise
schedule, a denoising objective, and a generative sampling path. It is a compact reference
implementation for the runtime and training API.

Read the diffusion example as a four-step pipeline:

1. load real image tensors or a small fallback dataset;
2. sample a timestep and add noise according to the schedule;
3. train a denoiser to predict the noise or clean signal;
4. run a sampler such as DDIM-style reverse steps to produce an image artifact.

Data and logging belong beside the tutorial because a poor sample usually raises ordinary ML
debugging questions first: dataset scale, resolution, schedule length, model capacity, training
time, and sampler settings.

The denoising objective is usually read as:

$$`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon_\theta(x_t,t)\approx\epsilon.`

The executable example samples timesteps and trains the predictor. The generative theory chapter
names which forward-process and sampler-step facts are presently proved in Lean.

# FNO And Burgers Operator Learning

The FNO example is the best current example of TorchLean training on a real scientific ML dataset.
The Python helper only downloads/converts data and plots the result; the model and training loop are
native TorchLean.

Prepare data with the [Burgers data helper](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/prepare_fno1d_burgers.py):

```
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
```

Train on CUDA:

```
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv
```

Plot one held-out prediction with the [plot helper](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/plot_fno1d_burgers.py):

```
python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
```

The generic FNO API defines a dense multidimensional real-split model. On CUDA, the Burgers command
uses the fused cuFFT-backed `spectralConv1dRfft` autograd primitive in place of dense transform
matrices.

The operator learning claim has a different type from classification:

$$`\mathcal{G}_\theta :
\operatorname{function\ samples\ on\ a\ grid}
\to
\operatorname{function\ samples\ on\ a\ grid}`

The FNO example exercises scientific data, spectral transforms, and CUDA interop while still using
the same typed tensor and runtime layer.

# Reinforcement Learning

The PPO examples show that TorchLean is not limited to supervised losses:

```
lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
```

The GridWorld example is written in Lean and has proof hooks.  The Gymnasium examples cross a Python
environment boundary and therefore use a runtime contract to check observations, actions, rewards,
and termination flags before they enter the learner.

For RL, the TorchLean idea is not "PPO exists." It is that an episode can be treated as data:
observations, actions, rewards, done flags, log probabilities, and value estimates are explicit
records that can be replayed, logged, checked, and visualized.

The rollout record gives a concrete object to inspect:

```
state, action, oldLogProb, reward, done, value, nextValue
```

From those fields, TorchLean computes returns, advantages, and policy/value losses. For Gymnasium
examples, the simulator remains external; the checked boundary is the transition record that enters
Lean.

# Data Is Part Of The Example

Several examples use real or prepared data paths:

- CSV and `.npy` loaders for tabular and tensor datasets;
- prepared CIFAR shards under `data/real`;
- Tiny Shakespeare and TinyStories text;
- Burgers `.mat` conversion into `.npy` tensors.

The [example-data helper](https://github.com/lean-dojo/TorchLean/blob/main/scripts/datasets/download_example_data.py) prepares common tiny datasets:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --tinystories-valid --cifar10
```

TorchLean examples should make the data source visible. If a file is generated, downloaded, or
converted, put the command beside the example or in the module header. Reproducibility starts before the
first tensor is allocated.

# Scope Of The Example Runs

A successful model example run establishes a concrete execution fact:

- the code builds;
- the selected runtime path executes;
- the loss and optimizer connect correctly;
- for training examples, the loss usually moves in the expected direction.

Architectural optimality, GPU-kernel agreement, and mathematical model theorems are separate
claims. Those belong in *Verification*, *Floating-Point Semantics*, and *GPU and CUDA*.

The most useful application prose therefore has the form:

```
Command C ran model family M on backend B and produced artifact A.
Artifact A can be inspected by widget/checker W.
The semantic claim is S, supported by theorem/checker T, or by assumption E.
```

That template is deliberately plain. It prevents a one-step runtime check from being read as a
model-quality result, and it prevents a local theorem from being stretched into a claim about an
unconnected external pipeline.

# References

- He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385),
  2015.
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [GPT-2 technical report / model card line](https://openai.com/index/better-language-models/),
  2019.
- Dosovitskiy et al., [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), 2020.
- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), 2020.
- Li et al., [*Fourier Neural Operator for Parametric Partial Differential
  Equations*](https://arxiv.org/abs/2010.08895), 2020/2021.
- Gu and Dao, [*Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*](https://arxiv.org/abs/2312.00752), 2023.
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347),
  2017.
