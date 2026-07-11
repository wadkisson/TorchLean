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

#doc (Manual) "TorchLean" =>
%%%
shortTitle := "TorchLean"
tag := "torchlean"
%%%

TorchLean brings neural-network programming and formal reasoning into the same Lean 4 project. You
can define a model, train it, inspect its parameters, and run it on the CPU or an accelerated backend.
Because the model is written in Lean, its tensor shapes, graph structure, specifications, and theorem
statements can share the same definitions.

A model is more than its forward function. Its behavior also depends on parameter layouts, masks,
training mode, scalar arithmetic, and the runtime selected to execute each operation. TorchLean makes
these choices explicit. A model written through the public API can be lowered to `NN.IR.Graph`; the
graph has a Lean denotation; and runtime or verification code can refer to that graph together with a
specific parameter payload.

The execution backend is a separate choice. TorchLean can evaluate supported operations through its
Lean runtime, native CUDA kernels, or registered external providers such as LibTorch. Changing the
backend should not silently change the model. Backend contracts record which operation is being
implemented, which device and layout it expects, how gradients are supplied, and which parts are
proved, checked, or trusted.

Verification follows the same principle. A robustness bound concerns a graph, parameters, and an
input region. A scientific ML claim may concern a residual over a domain. An imported certificate is
accepted only by a checker for a stated artifact format. A successful program run is useful evidence,
but it is not a theorem; a theorem states its own semantics and hypotheses.

![TorchLean guide map](Guide/Assets/torchlean-guide-map.png)

We begin with tensors, models, and training. We then make the graph semantics explicit, examine
floating-point and backend boundaries, and finally connect executable artifacts to verification
claims. To run a first model, begin with *Building Models*. For graph or certificate work, begin with
*Semantics and Graphs* or *Verification and Certificates*.

No prior theorem-proving experience is required for the executable examples. For Lean language and
proof background, the standard references are
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/). Proof notation is
introduced alongside the first examples that use it.

# Introduction

We begin with an ordinary classifier and ask what happens as its meaning becomes more precise. The
same classifier can appear as a program, a graph, a floating-point computation, a runtime artifact,
and a verification target. At each representation, we identify what Lean can establish and what
remains an execution or trust assumption.

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
enough structure for graph lowering, numerical analysis, and verification to inspect the same model
rather than reconstruct it.

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

Execution modes are interfaces with contracts. Eager TorchLean
execution, compiled graph execution, CUDA kernels, ATen/libtorch calls, PyTorch export, and imported
checkpoints can all be useful, but they must preserve enough structure for the proof side to know
what happened. An executable check against a closed form and an autograd correctness theorem are
different forms of evidence.

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

Three questions keep the levels distinct: What is the mathematical denotation? What artifact is
stored or produced by a run? What bridge relates them? Many ML verification mistakes come from
treating these as the same object too early.

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

A real-valued proof is not automatically a GPU proof. Deployment usually passes through rounded
arithmetic, library kernels, scheduling choices, and finite buffers. Those steps must be checked,
bounded, or left as explicit trusted boundaries instead of being smuggled into the theorem statement.

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

Verification is broader than classifier robustness. Robustness is the standard benchmark language,
but the same idea appears in optimizer laws, autograd rules, scientific
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

Each workflow answers three concrete questions: how to run it, which artifact it produces, and what
Lean can establish about that artifact today.

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModelZooDeepDive}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Widgets}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Examples}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.CLI}

{include 2 TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion}
