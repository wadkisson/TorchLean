/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Models.Common.RealData

/-!
# Char-GPT (minGPT-style) Example

This example mirrors the classic "character-level GPT on a single text file" walkthrough popularized
by Andrej Karpathy's minGPT/nanoGPT teaching material:

- build an alphabet (`itos`) from the training text,
- build a `stoi` tokenizer from that alphabet,
- train a compact causal Transformer to predict the next character,
- sample text continuations from a prompt.

It uses TorchLean's one-hot token interface (`batch × seqLen × vocab`) so the whole example stays in
the same typed tensor world as the rest of the codebase.

Implementation note: training draws a fresh deterministic random window each step, following the
minGPT/nanoGPT batching pattern. The `--windows` flag is accepted as a corpus-scale hint for shared
scripts, but this command does not precompute a fixed window table.

Quick run:

```bash
lake build -R -K cuda=true torchlean:exe
lake exe -K cuda=true torchlean chargpt --cuda --tiny-shakespeare --steps 1 --batch 1 --seq-len 1 --generate 0
```

`chargpt` is the character-tokenizer teaching path. It rebuilds deterministic training windows from
the corpus, so it is not part of the 10-step CUDA check tier. Use `gpt2` or `text_gpt2` for the
compact GPT-style 10-step checks.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.CharGpt

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean chargpt"

/-- Parse corpus flags and return the UTF-8 training text plus remaining CLI arguments. -/
def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName RealData.tinyShakespearePath
    [("--tiny-shakespeare", RealData.tinyShakespearePath)] RealData.missingTinyShakespeareHint args

/-- Build a deterministic character alphabet from the corpus. -/
def buildAlphabet (s : String) : Array Char :=
  let chars : List Char := s.toList.eraseDups
  -- Deterministic order: sort by codepoint.
  let sorted := List.mergeSort chars (fun a b => decide (a.toNat ≤ b.toNat))
  sorted.toArray

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "chargpt"

/-- Help text for character-level GPT training. -/
def usage : String :=
  String.intercalate "\n"
    [ "torchlean chargpt: character-level GPT training"
    , ""
    , "Usage:"
    , "  lake exe -K cuda=true torchlean chargpt --cuda --tiny-shakespeare [flags]"
    , ""
    , "Quick check:"
    , "  lake exe -K cuda=true torchlean chargpt --cuda --tiny-shakespeare --steps 1 --batch 1 --seq-len 1 --generate 0"
    , ""
    , "Notes:"
    , "  - CUDA is required; CPU eager mode is too slow for this path."
    , "  - This command is the character-tokenizer walkthrough, not the 10-step CUDA check target."
    , "  - Use `torchlean gpt2` or `torchlean text_gpt2` for compact GPT-style 10-step checks."
    ]

/-- Decode token ids for terminal output with control characters escaped. -/
def escapeCharIdsForDisplay (t : text.Tokenizer) (ids : List Nat) : String :=
  text.escapeForDisplay (t.decode ids)

/-- Printable-ASCII generation filter used by `--ascii-only`. -/
def asciiAllowed (c : Char) : Bool :=
  c = '\n' || (32 ≤ c.toNat && c.toNat ≤ 126)

/-- Fitted predictor for a runtime-sized character GPT model. -/
abbrev Predictor (batch seqLen vocab : Nat) :=
  Tensor.T Float (shape![batch, seqLen, vocab]) → IO (Tensor.T Float (shape![batch, seqLen, vocab]))

/-- Autoregressively extend character token ids using a trained CharGPT model. -/
partial def generateSampledFromIds
    (batch seqLen vocab : Nat)
    (predict : Predictor batch seqLen vocab)
    (promptIds : List Nat)
    (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (allowId : Nat → Bool := fun _ => true)
    (padId : Nat := 0) : IO (List Nat) := do
  let gen : text.GenerationOptions :=
    { prompt := ""
      generate := steps
      temperature := temperature
      topK := topK
      repeatPenalty := repeatPenalty
      repeatWindow := repeatWindow
      seed := seed
      asciiOnly := false }
  if seqLen = 0 then
    -- The CLI rejects this case, but keeping the definition total makes the stream reusable.
    pure promptIds
  else if hBatch : batch = 0 then
    -- Likewise, `--batch 0` is rejected by the CLI parser.
    pure promptIds
  else
  let b0 : Fin batch := ⟨0, Nat.pos_of_ne_zero hBatch⟩
  text.autoregressiveTokenIds seqLen padId promptIds gen
    (fun padded predPos => do
        let x : Tensor.T Float (shape![batch, seqLen, vocab]) :=
          text.causalLmXOneHotBatch (α := Float) batch seqLen vocab padded (padId := padId)
        let logits ← predict x
        pure (text.batchLogitScoresAt logits b0 predPos))
    (allowId := allowId)

/-- CLI entrypoint for character-level GPT training and sampling. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  ModelZoo.runFloat exeName args
    (banner := fun _ => s!"{exeName}: char-level GPT training")
    (k := fun opts rest => do
      if !opts.useGpu then
        throw <| IO.userError s!"{exeName}: use --cuda (CPU char-gpt is extremely slow in eager mode)"
      let (corpus, rest) ← takeInputText rest
      let defaultSteps : Nat := if opts.useGpu then 1 else 0
      let (train, rest) ← ModelZoo.orThrow exeName <|
        text.BatchedCheckpointedWindowedTrainGenerationOptions.parse
          exeName rest defaultLogJson defaultSteps 0.0005 1 1 1
          { prompt := "First Citizen:"
            generate := 0
            temperature := 0.9
            topK := 12
            repeatPenalty := 1.15
            repeatWindow := 64
            seed := 7
            asciiOnly := false }
          (allowZeroSteps := true)
      CLI.requireNoArgs exeName rest
      let alphabetFull := buildAlphabet corpus
      -- Keep the compact model's output head small. Characters outside this prefix map to `unkId`.
      let alphabet := alphabetFull.extract 0 (Nat.min 16 alphabetFull.size)
      let tok := text.Tokenizer.ofAlphabet alphabet (unkId := 0) (unkChar := '?')
      let vocab := tok.vocabSize
      let batch := train.batch
      let seqLen := train.seqLen

      let σ : Shape := shape![batch, seqLen, vocab]
      let τ : Shape := σ
      let cfg : nn.models.CausalOneHotConfig :=
        { batch := batch
          seqLen := seqLen
          vocab := vocab
          numHeads := 1
          headDim := 1
          ffnHidden := 2
          layers := 1 }
      let mkModel : nn.M (nn.Sequential σ τ) :=
        if hSeq : seqLen = 0 then
          nn.Linear vocab vocab (Shape.mat batch seqLen)
        else
          have h_dModel : nn.models.CausalOneHotConfig.dModel cfg ≠ 0 := by
            rw [nn.models.CausalOneHotConfig.dModel_eq]
            simp [cfg]
          nn.models.CausalTransformerOneHot cfg (h_seqLen := hSeq) (h_dModel := h_dModel)

      let toksList := tok.encode corpus
      let toks := toksList.toArray
      let usableStarts : Nat := text.Corpus.usableTokenStarts toks.size seqLen

      let mkBatchSample (step : Nat) : SupervisedSample Float σ τ :=
        Data.causalLmOneHotSampleRowsFromTokenArray
          (α := Float) batch seqLen vocab toks train.seed step (padId := 0)

      /-
      CharGPT trains on deterministic random-looking windows.  We spell those windows out as a
      finite dataset so the example remains easy to inspect: the seed controls the window schedule,
      while `trainer.train` owns checkpointing, optimizer stepping, and prediction.
      -/
      let windowCount := Nat.max 1 train.steps
      let samples : List (SupervisedSample Float σ τ) :=
        (List.range windowCount).map mkBatchSample
      let run := Trainer.runConfig opts { optimizer := optim.adam { lr := train.lr } }
      let trainer := Trainer.new mkModel <|
        Trainer.Config.fromRunConfig run .crossEntropy
      trainer.printInfo
      let trained ← trainer.train
        (Data.floatSamples samples)
        { steps := train.steps
          log := .disabled
          loadParams? := train.loadParams?
          saveParams? := train.saveParams? }
      let (L0, L1) ←
        Trainer.TrainSummary.requireAndPrintFloatLosses exeName trained.report
          (steps? := some train.steps) (lr? := some train.lr)
      let promptIds := tok.encode train.prompt
      let allowId : Nat → Bool :=
        if train.asciiOnly then
          fun i => asciiAllowed (alphabet.getD i '?')
        else
          fun _ => true
      let outIds ←
        generateSampledFromIds batch seqLen vocab trained.eval promptIds
          train.generate train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty
          (allowId := allowId) (padId := 0)
      let sampled := escapeCharIdsForDisplay tok outIds
      IO.println s!"  vocab={vocab} (unique chars)"
      IO.println s!"  sampled={sampled}"
      text.writeGenerationTrainLog
        train.log "CharGPT (minGPT-style)" train.steps L0 L1
        train.toGenerationOptions sampled
        #[ModelZoo.deviceNote opts,
          s!"vocab={vocab}",
          s!"usable_windows={usableStarts}",
          ModelZoo.cudaMemWatchNote opts train.steps train.cudaMemWatch]
      pure 0)

end NN.Examples.Models.Sequence.CharGpt
