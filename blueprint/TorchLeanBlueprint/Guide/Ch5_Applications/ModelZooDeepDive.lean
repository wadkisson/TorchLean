import VersoManual

open Verso.Genre Manual

#doc (Manual) "Model Examples Deep Dive" =>
%%%
tag := "model-examples-deep-dive"
%%%

The model examples exercise TorchLean as a whole system. A tensor library can look good on
an MLP. A graph IR can look good on a two layer verifier fixture. A CUDA backend can look good on a
single matmul. The example suite asks a harder question: can the same API, runtime, graph boundary,
data helpers, and verification boundaries survive modern model shapes that users actually recognize?

We built the [model examples API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/) to answer that question
as an executable model suite rather than a benchmark leaderboard. The default runs are compact, but
the stress they apply is real: causal windows, residual branches, patch tokens, selective scan,
spectral convolution, stochastic denoising, replay buffers, PPO rollouts, Python environment
bridges, CUDA fast paths, and graph/verification boundaries that must remain explicit.

Read each model along five axes:

- *API*: does the public model API express the architecture?
- *Runtime*: can the example train, log, save, load, or sample?
- *Graph*: does the graph/IR story remain visible?
- *CUDA*: does the accelerated path match the intended runtime boundary?
- *Verification*: does the example point to a spec, theorem, checker, or explicit producer
  assumption?

# Baselines That Still Matter

The MLP, KAN, and CNN examples are not just warmups. They are where the public training API proves
it can handle ordinary user expectations before the architecture becomes exotic:

```
lake exe torchlean mlp --device cpu --steps 10
lake exe torchlean kan --device cpu --steps 10
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
```

The MLP checks the minimal tabular training path. The KAN command checks that learned edge-function
bases can fit the same trainer and optimizer surface. The CNN command checks image tensors,
convolution/pooling layers, and CUDA execution on a recognizable vision workload. If these commands
are not clear, the larger examples will be harder to interpret.

The useful reading pattern is:

```
data source -> typed tensors -> model parameters -> loss -> optimizer step -> log artifact
```

Every later model family keeps that line but adds a harder piece: masks, recurrence, spectral
transforms, latent variables, or an environment boundary.

The architecture equations are familiar, but the TorchLean reading is specific:

- causal language models estimate $`p_\theta(x_t\mid x_{<t})` and rely on the mask property
  $`j>i\Rightarrow A_{ij}=0`;
- residual blocks compute $`x\mapsto F_\theta(x)+S_\theta(x)`, so both branches must agree on
  output shape;
- state-space models use $`h_{t+1}=A_t h_t+B_t x_t` and $`y_t=C_t h_t+D_t x_t`, so prefix
  causality is a real contract;
- FNO layers use spectral convolution, informally
  $`u\mapsto \mathcal F^{-1}(R_\theta\cdot \mathcal F(u))`, which is why the cuFFT boundary matters.

# GPT-2-Style Models And CharGPT

The GPT APIs to look for are:

- [NN.Examples.Models.Sequence.Gpt2 source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Gpt2.lean), a small
  byte level causal language model with GPT-2 ingredients.
- [NN.Examples.Models.Sequence.TextGpt2 source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/TextGpt2.lean), a
  trainer over a text corpus that can use GPT-2 BPE assets.
- [NN.Examples.Models.Sequence.CharGpt source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/CharGpt.lean), a
  character model in the minGPT family with deterministic random windows and generation probes.
- [NN.Examples.Models.Sequence.Gpt2Saved source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Gpt2Saved.lean),
  which exercises saved parameter loading and prompting.
- [NN.Examples.Models.Sequence.GptAdder source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/GptAdder.lean), a
  minGPT-family algorithmic addition task that stresses CUDA runtime checking.

A typical small run is:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean gpt2 --device cuda \
  --tiny-shakespeare --steps 1 --windows 1 --generate 0 --prompt "ROMEO:"
```

For CharGPT:

```
lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare \
  --steps 1 --batch 1 --seq-len 1 --generate 0 --prompt "ROMEO:"
```

These examples stress the API in a way MLPs do not. The model shape depends on vocabulary, sequence
length, number of heads, head dimension, and transformer depth. The data path has to turn raw text
into bounded token windows. The runtime has to carry a long eager tape through embeddings, attention
blocks, layer style operations, and output logits. The logging path writes before/after losses and
decoded reports so generated text can be compared across training checkpoints.

The semantic core is still a typed next token map:

$$`\mathrm{tokens}_{0:T}
\longmapsto
\mathrm{logits}_{0:T,\;0:|\mathcal V|}`

Masking and positions are not decoration. A causal mask says token `i` may not attend to future
token `j>i`; RoPE or positional embeddings say which position each key and query belongs to. Those
are exactly the kinds of bugs the BugZoo and CUDA chapters keep naming.

They also stress the boundary between "GPT-family model" and "GPT-2". TorchLean's `gpt2` command is not
OpenAI's pretrained GPT-2-small. It is a small causal Transformer with GPT-2-like ingredients,
chosen so the Lean side runtime, tokenizer path, parameter handling, and CUDA hooks can be exercised
locally. The right citations for the architecture lineage are Radford et al., "Language Models are
Unsupervised Multitask Learners" (OpenAI technical report, 2019,
[PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)),
Vaswani et al., "Attention Is All You Need" (NeurIPS 2017, https://arxiv.org/abs/1706.03762), and
Karpathy's minGPT educational implementation (https://github.com/karpathy/minGPT) for the compact
training example style.

The command writes a training log when requested:

```
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
  --steps 10 --windows 1 --generate 64 --prompt "ROMEO:" \
  --log data/model_zoo/gpt2_trainlog.json
```

Open that artifact with the GPT widget rather than reading generated text out of a terminal scroll:

```
#gpt2_train_log_file_view "data/model_zoo/gpt2_trainlog.json"
```

That widget is still an inspection tool. The semantic claim remains about the causal model, token
ids, parameter store, and runtime command that produced the log.

# Mamba

[NN.Examples.Models.Sequence.Mamba source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Mamba.lean) keeps the
sequence coverage broader than attention. The Mamba command trains a byte level language model through
the public Mamba API:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean mamba --device cuda \
  --tiny-shakespeare --steps 1 --windows 1 --generate 0 --prompt "ROMEO:"
```

The specification side includes state space material that later theorems can use in
[NN.Spec.Models.Mamba API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/Mamba.lean), including run-list length facts for
compact Mamba blocks. The API side in [NN.API.Models.Mamba API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Mamba.lean)
builds a trainable model with a gated recurrent core. The runtime side can use CUDA fast paths for
the float32 selective scan family when built with CUDA.

This example stresses a different runtime shape: recurrent state, scan computation, gating, and
generation. Attention bugs are not the only sequence bugs. Mamba models make us care about state
update conventions, scan order, and assumptions for specific kernels.

The paper anchor is Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(2023, https://arxiv.org/abs/2312.00752).

The recurrent scan has the form:

$$`h_{t+1}=A_t h_t + B_t x_t,\qquad
y_t=C_t h_t + D_t x_t`

The details differ by implementation, but the proof pressure is the same: state shape, scan order,
and parameter broadcasting must be explicit.

# RNN, LSTM, And The GRU Gap

The current runnable recurrent examples are:

- [NN.Examples.Models.Sequence.Rnn source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Rnn.lean), a vanilla RNN
  text-window runtime check.
- [NN.Examples.Models.Sequence.Lstm source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Lstm.lean), an LSTM
  text-window runtime check.
- [NN.Examples.Models.Supervised.LstmRegression source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Supervised/LstmRegression.lean),
  a more natural LSTM forecasting tutorial over household power windows.

The small text commands are short:

```
lake -R -K cuda=true exe torchlean rnn --device cuda --tiny-shakespeare --steps 1
lake -R -K cuda=true exe torchlean lstm --device cuda --tiny-shakespeare --steps 1
```

The forecasting run is the better tutorial:

```
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
lake -R -K cuda=true exe torchlean lstm_regression --device cuda --steps 200 --windows 96
```

These examples stress recurrent state handling: hidden state shape, application at each time step,
heads distributed across time, and optimizer updates through unfolded computation. The LSTM regression
case also stresses real time series data and a loss that readers can interpret as forecasting
error rather than token reconstruction.

A recurrent cell has the shape:

$$`(h_t,x_t)\longmapsto(h_{t+1},y_t)`

An LSTM cell carries more state:

$$`(h_t,c_t,x_t)\longmapsto(h_{t+1},c_{t+1},y_t)`

That extra carried state is why recurrent examples are good documentation tests: framework code may
store the state implicitly, while a proof needs to reason about the actual transition that ran.

There is no `gru` subcommand today, even though "LSTM/GRU/RNN" is often used as one category in ML
writing. TorchLean covers vanilla RNN and LSTM executables; GRU is a natural extension point rather
than a file currently cited as present. The classic recurrent
citations are Hochreiter and Schmidhuber, "Long Short-Term Memory" (Neural Computation 1997,
https://www.bioinf.jku.at/publications/older/2604.pdf) and Cho et al., "Learning Phrase
Representations using RNN Encoder Decoder for Statistical Machine Translation" (2014,
https://arxiv.org/abs/1406.1078) for GRU style gating.

# ResNet

The public [ResNet API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/ResNet.lean)
constructs residual networks over an arbitrary number of spatial dimensions. A configuration fixes
the channel widths, block counts, convolution parameters, and class count; the resulting model uses
the same typed layer API as other TorchLean models.

ResNet stresses the graph boundary because residual connections create branching structure that later rejoins.
A purely sequential chain can postpone many mistakes; a residual add makes shape agreement explicit. If the
shortcut branch and the convolution branch disagree, the typed construction has to confront the
mismatch before training or verification treats the block as valid.

The residual block contract is:

$$`x \longmapsto F_\theta(x)+S_\theta(x)`

Both branches must land in the same shape before the addition exists. That short condition is one of
the main reasons typed architectures help.

The paper anchor is He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016,
https://arxiv.org/abs/1512.03385).

# Vision Transformers

[NN.Examples.Models.Vision.Vit source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Vit.lean) trains a compact
ViT style CIFAR classifier:

```
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

The corresponding spec material in [NN.Spec.Models.Vit API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/Vit.lean) names
ViT configurations and proves that standard patch 16 config records are well formed. The executable
uses much smaller dimensions, but it stresses the same class of concerns: converting images to
patches, token embeddings, dimensions compatible with attention, and classifier heads.

ViT connects the vision and sequence parts of TorchLean. It asks the image loader
to produce CIFAR shaped tensors, then asks the model API to treat patches as tokens. That crossover
is a common source of shape bugs in ordinary frameworks.

The shape contract is:

$$`\mathrm{image}(C,H,W)
\longmapsto
\mathrm{patches}(N_{\mathrm{patch}},d_{\mathrm{model}})
\longmapsto
\mathrm{tokens}`

The verifier and runtime should not have to guess which image layout or patch convention was used.

The paper anchor is Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale" (ICLR 2021, https://arxiv.org/abs/2010.11929).

# FNO

[NN.Examples.Models.Operators.Fno1dBurgers source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean)
is the main scientific ML example. It trains a Fourier neural operator on a prepared one dimensional
Burgers dataset:

```
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32

lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda \
  --steps 700 --lr 0.003 --plot-csv data/real/fno/predictions.csv
```

FNO stresses a runtime boundary that language and vision models do not: spectral convolution. The
public model applies a dense tensor-product DFT over an arbitrary number of spatial axes while
carrying real and imaginary components separately. CUDA execution replaces that dense calculation
with the fused `spectralConv1dRfft` autograd primitive backed by cuFFT.

The operator learning map is:

$$`u_0(\cdot)\longmapsto u_T(\cdot)`

The FNO does not predict one label for one image; it learns an operator between function-like
signals. The example belongs next to Scientific ML verification, not only next to ordinary supervised
classification.

The example also stresses artifact hygiene. A helper prepares `.npy` arrays, Lean trains the model,
and a plotting helper renders a prediction CSV. The Python scripts are data/artifact utilities, not
the owner of the model semantics.

The model stressor is spectral convolution:

```
input grid -> lift -> Fourier modes -> learned spectral multiplier -> inverse transform -> projection
```

That line is why FNO is a good scientific ML example. It tests tensor layouts, real data, operator
learning, and the CPU/CUDA split in one small command. The trained result should be read as an
operator-learning runtime artifact. PINN residual certificates, ODE enclosures, and Lyapunov checks
are the places where scientific claims become checker artifacts.

The paper anchor is Li et al., "Fourier Neural Operator for Parametric Partial Differential
Equations" (ICLR 2021, https://arxiv.org/abs/2010.08895).

# Diffusion And Other Generative Commands

The generative commands are:

```
lake -R -K cuda=true exe torchlean autoencoder --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean mae --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean vae --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean vqvae --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean gan --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
```

These examples stress stochastic and reconstruction oriented training. Diffusion adds timestep
schedules, noising, denoising, and sampling artifacts. VAE, VQ-VAE, and GAN examples connect a
runnable training API to objective definitions used by theorems in the
[generative theory API](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/Generative/). The companion generative guide explains
that boundary in more detail.

These examples are more than larger MLP variants. They force the API to express latent shapes,
reconstruction targets, score targets, generator/discriminator pairs, patch masks, and image-valued
sampling outputs.

The expected artifacts differ by family:

| Command | Primary artifact | Typical claim |
|---|---|---|
| `autoencoder` | reconstruction loss/log | encoder-decoder runtime path ran |
| `mae` | masked reconstruction log | patch mask and reconstruction objective ran |
| `vae` | beta-VAE style loss/log | latent-objective shaped path ran |
| `vqvae` | reconstruction/codebook-shaped loss/log | finite-codebook objective path ran |
| `gan` | generator/discriminator loss log | LSGAN-shaped scalar objectives ran |
| `diffusion` | train log and optional PPM sample | noising/denoising/sampler path ran |

The generative theory modules then give stronger local facts about objectives and sampler steps.
Do not read image quality, distribution matching, or convergence into a short command unless a
separate experiment and theorem actually state those claims.

# RL Workflows

The RL examples are different from the supervised and generative examples because the data is produced
by an environment boundary rather than read from a static tensor file:

- [NN.Examples.Models.RL.PPOCartPole source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOCartPole.lean) trains an
  actor critic on Gymnasium `CartPole-v1` through a Python subprocess bridge.
- [NN.Examples.Models.RL.PPOGridWorld source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOGridWorld.lean) gives a
  smaller controlled PPO environment.
- [NN.Examples.Models.RL.PPOPongRam source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOPongRam.lean) exercises a
  larger observation/action setting.
- [NN.Examples.Models.RL.DQNReplay source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/DQNReplay.lean) focuses on the
  off policy replay and minibatch update pieces.

For CartPole:

```
lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda --updates 50
```

For the DQN replay mini example:

```
lake exe torchlean dqn_replay
```

RL stresses producer/checker boundaries harder than ordinary supervised learning. TorchLean can
typecheck the tensors that enter training, define the policy/value losses, and write logs that
widgets can read. When Gymnasium is the environment, raw observations are checked against a runtime
boundary contract before becoming training data. TorchLean checks the imported tensors and payloads
that enter the Lean side, while the external simulator remains a named producer.

The paper anchors are Mnih et al., "Human-level control through deep reinforcement learning"
(Nature 2015, https://www.nature.com/articles/nature14236) for DQN and Schulman et al., "Proximal
Policy Optimization Algorithms" (2017, https://arxiv.org/abs/1707.06347) for PPO. The environment
API reference is Gymnasium by the Farama Foundation (https://gymnasium.farama.org/).

The interaction loop is:

$$`s_t \xrightarrow{\pi_\theta} a_t
\xrightarrow{\mathrm{env}} (r_t,s_{t+1})`

TorchLean can type and check the policy, rollout buffers, returns, and PPO losses. If the
environment is Gymnasium, the environment transition itself remains an external producer step.

For Pong RAM, the command name follows the runner:

```
lake -R -K cuda=true exe torchlean ppo_pong_ram --device cuda --updates 1 \
  --eval-every 1 --eval-episodes 1 --eval-max-steps 8
```

This example is useful because the observation boundary is larger but still inspectable: a fixed RAM
vector crosses from Gymnasium/ALE into typed TorchLean rollout data.

# Additional Formulas And Snippets

```
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 20
```

# From Application Walkthroughs

```
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 10 --dtype float32 --backend eager
lake exe verify -- torchlean-ibp
lake exe torchlean --help
lake exe verify --help
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=true exe torchlean diffusion --device cuda \
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare \
lake exe verify -- torchlean-ibp --dtype float
lake exe verify -- torchlean-crown-ops --dtype float
lake exe verify -- torchlean-robustness --dtype float
lake exe verify -- torchlean-mlp-workflow
lake exe verify -- digits-train-certify --epochs=50 --eps=0.02 --max=100
lake exe verify -- margin-cert
lake exe verify -- vnncomp-mnistfc
lake exe verify -- camera-box3d-cert
lake exe verify -- abcrown-leaf
lake exe verify -- lirpa-mlp
lake exe verify -- lirpa-cnn
lake exe verify -- lirpa-attention
lake exe verify -- lirpa-gru
lake exe verify -- lirpa-encoder
lake exe verify -- all
lake exe verify -- abcrown-leaf \
python3 NN/Examples/Data/prepare_fno1d_burgers.py \
lake -R -K cuda=true exe torchlean fno1d_burgers \
python3 NN/Examples/Data/plot_fno1d_burgers.py \
python3 scripts/verification/regenerate_assets.py --group pinn-small --run
lake exe verify -- pinn-cert
lake exe verify -- pinn-cli -- "u_t + u*u_x - 0.01*u_xx" 0.0 0.5 0.01
lake exe verify -- pinn-dataset-check
python3 scripts/verification/pinn/train_pinn_1d.py \
python3 scripts/verification/regenerate_assets.py --group geometry3d-wilddet3d --run
```

```
def vocab : Nat := text.Tokenizer.byte.vocabSize
```

```
abbrev σ : Shape := shape![batch, seqLen, vocab]
abbrev τ : Shape := σ
```

```
def mkSampleFromTokenIds (toks : List Nat) : SupervisedSample Float σ τ :=
  Data.causalLmOneHotSample (α := Float) batch seqLen vocab toks (padId := 32)
```

```
abbrev σ : Shape := nn.models.mambaTokenMat cfg seqLen
abbrev τ : Shape := nn.models.mambaLogitMat cfg seqLen

def model : nn.M (nn.Sequential σ τ) :=
  nn.models.MambaTextLM cfg seqLen
```

```
nn.causalTransformerFromEmbeddings
nn.causalTransformerOneHot
nn.causalTransformerTokenScalarModuleDef
```

```
let trainer := Trainer.new model <|
  Trainer.Config.fromRunConfig run (.crossEntropy)
let trained ← trainer.train data train.options probes
trained.printSummary
```

