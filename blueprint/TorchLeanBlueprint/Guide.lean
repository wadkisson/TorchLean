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

Most machine-learning projects begin in a familiar way: define a model, prepare some data, train,
and inspect the result. That is enough to answer many practical questions. It tells us whether the
loss decreased, how the model behaved on a test set, and whether a larger architecture or a longer
run might help.

The difficulty begins when a numerical result is turned into a mathematical claim. By that point,
the model may have changed form several times. The source program became initialized parameters;
the parameters moved into runtime buffers; the runtime constructed an autograd tape; an exporter
produced a graph; and a verifier read that graph together with its own assumptions about shapes,
arithmetic, and the input domain. Each representation is useful, but they are not interchangeable.
A successful training run alone does not tell us that every later tool analyzed the intended
function.

TorchLean brings these stages into Lean 4. Models are written with shape-typed tensors, trained
through an executable runtime, and lowered to an operation graph with explicit parameter payloads.
The mathematical specification of an operator lives alongside the code that uses it. When a
backend delegates work to native CUDA or LibTorch, the selected provider and its trust boundary
remain part of the account of what ran.

This does not require every numerical kernel to be reimplemented inside the theorem prover.
Established accelerator libraries can still do the expensive work. Lean is used to state the model,
record the translation into executable forms, check structural conditions, and prove the parts of
the argument for which a theorem is available. Where an external implementation remains trusted,
the boundary is named rather than hidden behind a generic call to a GPU.

The result is a library that can be used at several levels. A new user can write and train a small
network without first developing a proof. A verification project can inspect the lowered graph and
establish a property over an input region. A numerical analysis can compare ideal real-valued
semantics with finite-precision execution. These uses share definitions, but they ask different
questions and therefore provide different kinds of evidence.

We will learn the library by following one small nonlinear regression model. First we write and
train it. Then we open it up: parameters, autograd tape, graph IR, floating-point behavior, and
verification bounds. Once that example is familiar, we will use the same ideas for transformers,
ResNets, Fourier neural operators, generative models, reinforcement learning, and scientific ML.

The examples are meant to be run from the repository root. No theorem-proving background is needed
to begin. Readers new to Lean may also use
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/).

# Introduction

We begin with the problem TorchLean is trying to solve, then write the regression model that will
stay with us through the rest of the book.

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Overview}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample}


# Building Models

Now we turn the model definition into a training program: tensors, datasets, initialization,
forward evaluation, loss, backward, and optimizer updates.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI}


# Runtime, Autograd, and Interop

Training introduces state and hardware. Parameters change, tape nodes save intermediate values, and
optimizer buffers accumulate history. None of that appears in the clean equation

$$`f_\theta(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2`.

We will follow one step through the runtime, then move the same operation to compiled execution,
CUDA, and LibTorch.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip}


# Semantics and Graphs

Autograd records one execution. Verification needs a meaning that survives after the tape is gone.
We first write the model as a mathematical function, then preserve its architecture in `GraphSpec`,
and finally lower it to the ordinary node array used by importers and verification passes.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR}


# Floating Point and Native Boundaries

Our graph equations use real numbers, but the program does not. This part starts with that mismatch
and develops TorchLean's floating-point stack. The story begins with Flocq's influential separation
of formats from rounding, continues through TorchLean's generic `NeuralFloat` theory, and ends with
executable binary32 and native kernels.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.Floats}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness}


# Verification and Certificates

We can now state the verification question precisely:

$$`\forall x\in B,\qquad P(\operatorname{denote}(g,\theta,x))`.

This part develops several ways to answer it: interval and affine bounds, compiler-correctness
proofs, autograd theorems, numerical error bounds, optimizer laws, and replayed certificates. We
will also feed bad artifacts to the checkers and see exactly where they fail.

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

Finally we leave the two-layer MLP. ResNets and vision transformers add spatial and attention
layouts; GPT adds token streams and causal masks; Fourier neural operators connect learned maps to
PDE data; diffusion and reinforcement learning add stochastic state. Each example includes the
actual command, the architecture it builds, and the checks currently available for it.

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModelZooDeepDive}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Widgets}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Examples}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.CLI}

{include 2 TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion}
