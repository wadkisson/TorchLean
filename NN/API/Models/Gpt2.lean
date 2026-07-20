/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.Runtime.Autograd.TorchLean.NN

/-!
# GPT-2-Style Model Helpers (API)

This module collects compact, reusable GPT-2-style building blocks for TorchLean examples:

- a single “causal LM over one-hot tokens” model constructor, and
- a small configuration record that keeps the hyperparameter inventory explicit.

These helpers live in the API layer so runnable examples can stay focused on:
data prep, training loops, and text decoding, rather than repeating the same
`embedding → positional embedding → Transformer stack → LayerNorm → linear` boilerplate.

Important scope note:
- This is *not* a pretrained checkpoint loader.
- These are compact example architectures shaped like GPT-2 blocks.
- Tokenizers live under `NN.API.text` / `NN.API.text.Gpt2Bpe`.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/--
Configuration shared by TorchLean's GPT-style causal language models.

The model has the common GPT-2 “shape”:

`embedding → learned positional embedding → (masked self-attention + FFN)×layers → LayerNorm → linear`

The configuration is independent of how token ids enter the model. One-hot, integer-token, and
pre-embedded constructors reuse the same Transformer width, depth, and output vocabulary.
-/
structure CausalTransformerConfig where
  batch : Nat
  seqLen : Nat
  vocab : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
  layers : Nat
  /-- Feed-forward activation used in every Transformer block. -/
  activation : nn.blocks.Activation := .gelu
  /-- Dropout probability for attention and feed-forward outputs. -/
  dropout? : Option Float := none
  /-- Use pre-normalized Transformer blocks. -/
  normFirst : Bool := false
  /-- Add a trainable bias after each attention output projection. -/
  attentionOutputBias : Bool := false
  /-- Shared initialization for embedding and projection weights. `none` keeps layer defaults. -/
  parameterInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /-- Seed stride used when initializing repeated blocks. -/
  seedStride : Nat := 100
deriving Repr

/-- Transformer width implied by `numHeads * headDim`. -/
def CausalTransformerConfig.dModel (cfg : CausalTransformerConfig) : Nat :=
  cfg.numHeads * cfg.headDim

/-- Vocabulary-grid shape `(batch × seqLen × vocab)` used by one-hot inputs and output logits. -/
abbrev causalVocabularyShape (cfg : CausalTransformerConfig) : Spec.Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.vocab]

/-- Embedded-token tensor shape `(batch × seqLen × dModel)`. -/
abbrev causalEmbeddingShape (cfg : CausalTransformerConfig) : Spec.Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.dModel]

/--
GPT-2-style causal Transformer body after token embeddings have already been computed.

This is the shared body used by both one-hot-token models and indexed-token experiments.  Keeping
it separate avoids duplicating the Transformer stack when callers use a different token
representation: the input boundary changes, while positional embeddings, masked self-attention
blocks, layer norm, and the
language-model head stay the same.
-/
def causalTransformerFromEmbeddings (cfg : CausalTransformerConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let encCfg : nn.blocks.TransformerEncoderStack :=
    { layers := cfg.layers
      block :=
        { numHeads := cfg.numHeads
          headDim := cfg.headDim
          ffnHidden := cfg.ffnHidden
          activation := cfg.activation
          dropout? := cfg.dropout?
          normFirst := cfg.normFirst
          attentionOutputBias := cfg.attentionOutputBias
          weightInit? := cfg.parameterInit? }
      seedStride := cfg.seedStride }
  let posInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  nn.Sequential![
    nn.learnedPositionalEmbedding (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel)
      { posInit := posInit },
    nn.transformerEncoderStack (batch := cfg.batch) (n := cfg.seqLen) (dModel := dModel) encCfg
      (mask := some (text.causalMask cfg.seqLen)),
    nn.layerNorm (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel),
    nn.linearWith dModel cfg.vocab { weightInit? := cfg.parameterInit? }
      (pfx := .dim cfg.batch (.dim cfg.seqLen .scalar))
  ]

/--
Build a GPT-2-style causal language model over one-hot tokens.

This is the shared constructor used by the runnable GPT-2 examples. It stays in `nn.M` so it
composes with the rest of the API-layer model-building interface.
-/
def causalTransformerOneHot (cfg : CausalTransformerConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalVocabularyShape cfg) (causalVocabularyShape cfg)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  nn.embedding cfg.vocab dModel { wInit := embeddingInit }
    (pfx := .dim cfg.batch (.dim cfg.seqLen .scalar)) >>= fun embed =>
  causalTransformerFromEmbeddings cfg (h_seqLen := h_seqLen) (h_dModel := h_dModel) >>= fun body =>
  pure (embed >>> body)

/--
Scalar loss for causal language modeling with integer token ids.

The public one-hot constructor above is useful for small teaching examples because the input is an
ordinary Float tensor.  File-backed tokenized datasets use the representation found in
language-model training systems: token ids are `Nat`s, the embedding table is a trainable Float
parameter, and the loss gathers the target classes directly instead of building one-hot targets.

`tokens` and `targets` are flattened `(batch * seqLen)` vectors.  This matches the backend gather
ops and keeps dataset storage simple; the embedding helper reshapes gathered rows back to
`(batch, seqLen, dModel)` before running the Transformer body.
-/
def causalTransformerTokenScalarModuleDefWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg))
    (tokens targets : Spec.Tensor Nat (.dim (cfg.batch * cfg.seqLen) .scalar))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body) [] :=
  { initParams :=
      .cons (Spec.zeros Float (.dim cfg.vocab (.dim cfg.dModel .scalar))) (initParams body)
    runtimeInit :=
      match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? body with
      | some bodyPlan => some (.cons .zeros bodyPlan)
      | none => none
    initRequiresGrad := List.replicate (((.dim cfg.vocab (.dim cfg.dModel .scalar)) ::
      paramShapes body).length) true
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
          (ss := ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body) ++
            ([] : List Spec.Shape))
          (β := m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α)
            Spec.Shape.scalar))
          (fun args => do
            let (ps, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
                (ss₁ := (.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body)
                (ss₂ := ([] : List Spec.Shape)) args
            let .nil := empty
            let .cons tokenEmbedding bodyParams := ps
            let x ← _root_.Runtime.Autograd.TorchLean.F.embeddingBatchSeqNat (m := m) (α := α)
              (vocab := cfg.vocab) (dim := cfg.dModel) (batch := cfg.batch)
              (seqLen := cfg.seqLen) tokenEmbedding tokens
            let logits ← _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardParams
              (model := body) (α := α) (m := m) mode bodyParams x
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := .dim cfg.batch (.dim cfg.seqLen (.dim cfg.vocab .scalar)))
              (s₂ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.vocab .scalar))
              logits (by
                simp [_root_.Spec.Shape.size, Nat.mul_assoc])
            _root_.Runtime.Autograd.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := cfg.batch * cfg.seqLen) (classes := cfg.vocab)
              logitsRows targets (reduction := reduction)) }

/-- Training-mode wrapper for integer-token causal language modeling. -/
def causalTransformerTokenScalarModuleDef (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg))
    (tokens targets : Spec.Tensor Nat (.dim (cfg.batch * cfg.seqLen) .scalar))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body) [] :=
  causalTransformerTokenScalarModuleDefWithMode .train cfg body tokens targets
    (reduction := reduction)

/--
Flattened token-id input shape for causal-LM training.

The sequence batch is represented as one vector of length `batch * seqLen`. The values are token
ids encoded as floats by the data loader, then checked and converted back to `Nat` inside the eager
runtime. Keeping this as a flat vector matches the existing scalar-module input convention.
-/
abbrev causalTokenIdLmInputShape (cfg : CausalTransformerConfig) : Spec.Shape :=
  .dim (cfg.batch * cfg.seqLen) .scalar

/--
Embedding lookup for a runtime batch of float-encoded integer token ids.

The input is flat because batches are assembled dynamically by data loaders. The layer validates
that every value is an integer token id, gathers the corresponding rows from its trainable table,
and returns the usual `(batch, seqLen, dModel)` embedding tensor.
-/
def causalTokenIdEmbedding (cfg : CausalTransformerConfig) (seed : Nat) :
    nn.Sequential (causalTokenIdLmInputShape cfg) (causalEmbeddingShape cfg) :=
  let tableShape : Spec.Shape := .dim cfg.vocab (.dim cfg.dModel .scalar)
  let tableInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  let table0 : Spec.Tensor Float tableShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor tableInit (seed := seed)
  nn.of
    { kind := s!"TokenEmbedding({cfg.vocab}, {cfg.dModel})"
      paramShapes := [tableShape]
      initParams := .cons table0 .nil
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme tableInit seed)
        .nil)
      paramRequiresGrad := [true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun table tokenValues =>
            ((do
              let tokens ← _root_.Runtime.Autograd.TorchLean.F.tokenIdsFromFloatVec
                (m := m) (α := α) (k := cfg.batch * cfg.seqLen) tokenValues
              _root_.Runtime.Autograd.TorchLean.F.embeddingBatchSeqNat
                (m := m) (α := α) (vocab := cfg.vocab) (dim := cfg.dModel)
                (batch := cfg.batch) (seqLen := cfg.seqLen) table tokens) :
              m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
                (causalEmbeddingShape cfg))) }

/-- GPT model with dynamic integer-token batches and a gathered embedding table. -/
def causalTransformerTokenId (cfg : CausalTransformerConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalTokenIdLmInputShape cfg) (causalVocabularyShape cfg)) := do
  let embed ← nn.withSeed (causalTokenIdEmbedding cfg)
  let body ← causalTransformerFromEmbeddings cfg h_seqLen h_dModel
  pure (embed >>> body)

/--
Scalar loss for causal language modeling with per-step float-encoded token ids as inputs.

`xTokens` and `yTokens` are flattened `(batch * seqLen)` float vectors holding integer token ids.
The float representation is a runtime transport format, not a mathematical relaxation of token
ids: the eager backend rejects negative or fractional values before indexing the embedding table.

This gives the PyTorch-shaped training path, `nn.Embedding` followed by row-wise cross entropy,
without rebuilding the module at every text window. Optimizer state therefore stays attached to the
same persistent parameter session.
-/
def causalTransformerTokenIdLmScalarModuleDefWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (model : nn.Sequential (causalTokenIdLmInputShape cfg) (causalVocabularyShape cfg))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      (paramShapes model)
      [causalTokenIdLmInputShape cfg, causalTokenIdLmInputShape cfg] :=
  { initParams := initParams model
    runtimeInit := _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? model
    initRequiresGrad := paramRequiresGrad model
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
          (ss := paramShapes model ++
            [causalTokenIdLmInputShape cfg, causalTokenIdLmInputShape cfg])
          (β := m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α)
            Spec.Shape.scalar))
          (fun args => do
            let (ps, ins) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
                (ss₁ := paramShapes model)
                (ss₂ := [causalTokenIdLmInputShape cfg, causalTokenIdLmInputShape cfg]) args
            let .cons xFloat (.cons yFloat .nil) := ins
            let targets ← _root_.Runtime.Autograd.TorchLean.F.tokenIdsFromFloatVec (m := m) (α := α)
              (k := cfg.batch * cfg.seqLen) yFloat
            let logits ← _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardParams
              (model := model) (α := α) (m := m) mode ps xFloat
            -- The transformer returns `(batch, seqLen, vocab)`. Cross entropy expects one row per
            -- prediction site, so the batch and time axes are flattened together.
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := .dim cfg.batch (.dim cfg.seqLen (.dim cfg.vocab .scalar)))
              (s₂ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.vocab .scalar))
              logits (by
                simp [_root_.Spec.Shape.size, Nat.mul_assoc])
            _root_.Runtime.Autograd.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := cfg.batch * cfg.seqLen) (classes := cfg.vocab)
              logitsRows targets (reduction := reduction)) }

/--
Training-mode wrapper for float-encoded token-id causal language modeling.

Use this when the body consumes embeddings and emits logits, while the dataset supplies changing
integer token-id windows.
-/
def causalTransformerTokenIdLmScalarModuleDef (cfg : CausalTransformerConfig)
    (model : nn.Sequential (causalTokenIdLmInputShape cfg) (causalVocabularyShape cfg))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      (paramShapes model)
      [causalTokenIdLmInputShape cfg, causalTokenIdLmInputShape cfg] :=
  causalTransformerTokenIdLmScalarModuleDefWithMode .train cfg model (reduction := reduction)

end models
end nn

end API
end NN
