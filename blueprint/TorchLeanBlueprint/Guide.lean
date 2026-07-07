import VersoManual
import TorchLeanBlueprint.Guide.Ch1_Introduction.Overview
import TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation
import TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour
import TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming
import TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage
import TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch
import TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample
import TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes
import TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels
import TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders
import TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch
import TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI
import TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes
import TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection
import TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough
import TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels
import TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd
import TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip
import TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR
import TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer
import TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec
import TorchLeanBlueprint.Guide.Ch3_Backend.Floats
import TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature
import TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA
import TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI
import TorchLeanBlueprint.Guide.Ch4_Verification.Verification
import TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems
import TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs
import TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation
import TorchLeanBlueprint.Guide.Ch4_Verification.LearningTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.OptimizationTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.SelfSupervisedTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.ApproximationTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.ClassicalMLProofs
import TorchLeanBlueprint.Guide.Ch4_Verification.ProbabilityAndGradients
import TorchLeanBlueprint.Guide.Ch4_Verification.ScientificMLVerification
import TorchLeanBlueprint.Guide.Ch4_Verification.FactorizationsCholeskyQR
import TorchLeanBlueprint.Guide.Ch4_Verification.Certificates
import TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness
import TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows
import TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels
import TorchLeanBlueprint.Guide.Ch5_Applications.ModelZooDeepDive
import TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels
import TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning
import TorchLeanBlueprint.Guide.Ch5_Applications.Widgets
import TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog
import TorchLeanBlueprint.Guide.Ch5_Applications.Examples
import TorchLeanBlueprint.Guide.Ch5_Applications.CLI
import TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion

open Verso.Genre Manual

#doc (Manual) "TorchLean Guide" =>
%%%
shortTitle := "TorchLean"
tag := "torchlean"
%%%

TorchLean is a Lean 4 library for neural-network programs that are meant to survive scrutiny. You
can still begin in the ordinary ML way: define a model, give it tensors, run a training loop, and
inspect the result. The difference is what happens next. The same program can be lowered to a graph,
connected to a specification, executed through a chosen backend, or paired with a certificate while
remaining recognizably the object you started with.

Machine-learning work changes form as it moves through a project. An equation becomes a model; the
model becomes a training script; the script calls kernels; the run produces checkpoints, plots, and
claims. The fragile step is not any one conversion. It is the accumulation of small conventions:
which tensor layout was used, which mask was meant, which graph was exported, which scalar semantics
the verifier saw. TorchLean is built to keep those conventions close to Lean objects, so the code,
graph, runtime artifact, certificate, and theorem can be compared without pretending they are all the
same thing.

This guide is a tour of that discipline. It starts with small models because every later claim is
easier to read when there is a concrete network on the page. From there the same object passes
through training, graph lowering, floating-point execution, CUDA and PyTorch boundaries,
certificates, and proofs. You do not need to memorize the module tree on a first pass. The useful
habit is simpler: when a page makes a claim, ask which Lean object carries it.

The question we keep asking is simple:

> Which object is this statement about?

For a classifier, the object might be a typed model with a fixed input shape. For autograd, it might
be the reverse-mode computation used by a training step. For a verifier, it might be a graph plus an
input region and a bound certificate. For CUDA, it might be a fast native forward path tied back to a
TorchLean graph and backward rule. For scientific ML, it might be a residual check over a grid or an
interval certificate for a trajectory.

Lean does not make any of this automatic, and TorchLean does not try to hide that. A theorem is not
a smoke test. A CUDA regression is not a proof. A JSON certificate is not trusted merely because a
tool emitted it. The point of the library is to make those distinctions precise enough that a reader
can see what was run, what was checked, and what was proved.

Here is the kind of ledger the rest of the guide keeps returning to:

```
-- Graph semantics: what does the lowered graph mean?
#check NN.IR.Graph.denote
#check NN.IR.Graph.denoteAll

-- Autograd correctness: when does the reverse pass compute the intended derivative?
#check NN.IR.Graph.backprop_correct
#check NN.IR.Graph.backpropVec_eq_adjoint_fderiv
#check NN.IR.Graph.scalarLoss_grad_correct

-- Float32 transfer: when does a real-valued statement survive rounded execution?
#check NN.Proofs.RuntimeApprox.FP32.approxT_linear_fp32
#check NN.Proofs.RuntimeApprox.FP32.ibpBound_contains_reluTwoLayerMlp_float32

-- Imported verifier artifacts: what is checked before an external certificate is used?
#check NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaCrown_transfer_sound
#check NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaBetaCrown_transfer_sound
```

Some of those declarations are theorems. Some are denotational functions used by theorems. Some are
transfer lemmas saying how one layer supports a claim at another layer. The important habit is to
name the declaration rather than leaving the word "verified" floating by itself.

![TorchLean guide map](Guide/Assets/torchlean-guide-map.png)

Read the chapters in order once, then treat them as a map of the repository. The early chapters show
how to write models and training loops. The middle chapters explain the graph and specification
layers. The later chapters show how numerical backends, scientific ML examples, optimizer laws,
learning theory statements, and verifier certificates attach to those same objects.

When an example writes a JSON certificate, the guide shows which checker reads it. When an example
uses CUDA, the guide separates the native computation from the TorchLean object it is meant to
implement. When a theorem mentions a graph denotation, the guide explains where that graph came from.

If you are new to Lean, keep the official Lean material nearby:
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/). TorchLean uses Lean as
both a programming language and a proof language, so the examples introduce proof-oriented notation
when it first becomes necessary.

The guide also points outside the repository when the outside tool or paper matters to the claim.
PyTorch is the reference point for the everyday training interface and for comparison with
[torch.autograd](https://docs.pytorch.org/docs/stable/autograd.html). VNN-COMP and
[VNN-LIB](https://www.vnnlib.org/) explain the verifier ecosystem around ONNX networks and input
specifications. The α,β-CROWN family, including
[β-CROWN](https://arxiv.org/abs/2103.06624), is the main external verifier line behind several
certificate and branch-and-bound examples. Scientific ML pages cite the relevant PDE, PINN, neural
operator, and residual-checking literature when the example depends on that background rather than
on TorchLean alone.

# Introduction

The first chapter explains the basic promise of TorchLean: neural-network code should remain
runnable while becoming more precise about what it means. It starts with the everyday model construction
interface, then introduces the reason the same classifier will later reappear as a program, a graph,
a floating point computation, a runtime artifact, and a verification target.

The chapter is deliberately practical. It does not begin with a theorem prover slogan. It begins
with the question a user actually has: if I write a neural network, where does the mathematical
meaning live, and how much work is required before Lean can say something about it?

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Overview}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample}


# Building Models

Before there is anything to verify, there has to be a model worth talking about. This part stays
close to everyday ML work: tensors with shapes, layer builders, datasets, loaders, losses,
optimizers, and short training runs. The runnable path should feel ordinary while still carrying
enough structure for later chapters to inspect the same model rather than reconstruct it.

This is where TorchLean has to earn trust from programmers, not just from proof engineers. The
examples show the small details that matter in practice: which dimensions are fixed in the type,
which values are runtime data, where parameters live, how a batch is represented, how a loss is
chosen, and how an optimizer step changes a parameter bundle. Those details are ordinary ML
engineering details, but they become proof-relevant as soon as a later page talks about gradients,
graph semantics, or certificates.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI}


# Runtime, Autograd, and Interop

Once the model exists, it has to execute. This part follows the runtime path: eager execution,
compiled graph execution, reverse-mode autograd, optimizers, checkpoints, and PyTorch interop.
Faster execution should not secretly change the object being trained or checked. Backends
are choices about how to compute; they are not new mathematical models.

The runtime chapter therefore treats execution modes as interfaces with contracts. Eager TorchLean
execution, compiled graph execution, CUDA kernels, ATen/libtorch calls, PyTorch export, and imported
checkpoints can all be useful, but they must preserve enough structure for the proof side to know
what happened. This is also where the guide explains the difference between an executable check
against a closed form and an autograd correctness theorem.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip}


# Semantics and Graphs

The same network can be read at more than one level. A specification gives the mathematical meaning
of tensors, layers, losses, masks, modes, and scalar choices. GraphSpec describes architectures with
an explicit parameter interface. `NN.IR.Graph` is the lower-level DAG consumed by widgets,
exporters, runtime bridges, and verification passes.

Those levels are connected, but they are not interchangeable. A spec definition is a reference
meaning. A GraphSpec model is a structured architecture. An IR graph is an artifact with op tags,
parent ids, shapes, and payloads. Keeping the roles separate is what lets a later theorem or
checker say exactly what it used.

A good way to read this part is to keep three questions in mind. What is the mathematical
denotation? What is the artifact stored in the repository or produced by a run? What is the bridge
between them? Most mistakes in ML verification come from treating those as the same object too
early.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec}


# Floating Point and Native Boundaries

Neural-network claims often cross numerical worlds. A model may be specified over the reals,
executed in Float32, accelerated by CUDA, bounded by a verifier, and checked through an imported
certificate. Those worlds are related, but they are not the same.

TorchLean keeps those numerical worlds separate. Real-valued specs are clean mathematical
references. `FP32` is a rounded-real proof model for finite-precision error budgets. `IEEE32Exec`
is executable binary32 behavior inside Lean. Native execution through Lean `Float32`, CUDA, cuBLAS,
cuFFT, Python, Julia, Arb, or external verifiers enters through explicit producer/checker
boundaries. Practical numerical tools stay in the workflow, with a named role in each claim.

This part also explains why TorchLean does not simply pretend that a real-valued proof is a GPU
proof. A real proof may be the right starting point, but deployment usually passes through rounded
arithmetic, library kernels, scheduling choices, and finite buffers. The guide names those steps so
they can be checked, bounded, or left as explicit trusted boundaries instead of being smuggled into
the theorem statement.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.Floats}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness}


# Verification and Certificates

Verification in TorchLean is not one trick. For a robustness example, it may be an IBP or
CROWN bound on one graph and one input region. For a compiler pass, it may be a theorem that
two graph denotations agree. For scientific ML, it may be a residual certificate, an ODE enclosure,
or a dataset artifact whose fields are checked in Lean. For optimizer or learning theory
work, it may be a lemma about the update rule itself.

The common pattern is simple: every claim has an object and a support. The object may be a graph, a
payload of parameters, a tensor program, a dataset, a trajectory, a residual, or an external JSON
certificate. The support may be a Lean theorem, a small checker, a replayed artifact, a runtime
diagnostic, or an explicitly named external producer.

A run is evidence about execution. A certificate is a finite artifact with a stated checker. A
real-valued theorem needs a numerical bridge before it becomes a Float32 claim. When the right
bridges are present, concrete artifacts can become precise mathematical statements rather than
screenshots of successful runs.

The verification chapter is intentionally broader than classifier robustness. Robustness is the
standard benchmark language, but the same idea appears in optimizer laws, autograd rules, scientific
ML residuals, ODE enclosures, dataset checks, and imported verifier leaves. The unifying principle
is that a certificate should say what finite object was checked, and a theorem should say which
mathematical statement follows.

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Verification}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.LearningTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.OptimizationTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.SelfSupervisedTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ApproximationTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ClassicalMLProofs}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ProbabilityAndGradients}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ScientificMLVerification}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FactorizationsCholeskyQR}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Certificates}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows}


# Examples and Applications

The examples are where the abstractions earn their keep. Small MLPs show the basic training path.
GPT models bring token ids, positions, masks, and caches. ResNets and ViTs bring residual
branches and attention structure. FNOs bring spectral convolutions and scientific data. Diffusion
brings sampling schedules. Reinforcement learning brings environment boundaries and trajectory
data. BugZoo turns common ML mistakes into explicit contracts.

Each example is intentionally small enough to inspect, but it touches a real source of complexity.
A finished command is the start of the story. The better question is what object it produced, what
TorchLean can inspect about that object, and what kind of claim the object can support.

Read this chapter as a catalog of workflows rather than a gallery of demos. Each example should
answer three concrete questions: how do I run it, what artifact do I get, and what can Lean say
about that artifact today?

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModelZooDeepDive}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Widgets}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Examples}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.CLI}

{include 2 TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion}
