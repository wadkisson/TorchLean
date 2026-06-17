/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Models.Sequence.Gpt2

/-!
# GPT-2 Saved-Weights Example

This is the load-and-sample half of the byte-level GPT example.

1. Train and save parameters:

```bash
lake build -R -K cuda=true torchlean:exe
lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare --steps 1 --windows 1 \
  --prompt "First Citizen:" --generate 0 \
  --save-params data/model_zoo/gpt2_shakespeare.params.json
```

2. Load the saved weights and sample text (no training loop, no optimizer state):

```bash
lake exe -K cuda=true torchlean gpt2_saved --cuda --fast-kernels \
  --params data/model_zoo/gpt2_shakespeare.params.json \
  --prompt "First Citizen:" --generate 0
```

## What A Checkpoint Is Here

This example uses the simplest TorchLean checkpoint format:

- a shape-indexed pack of model parameters,
- stored as exact `Float.toBits` values in JSON, and
- checked against the model's parameter shapes before inference starts.

So save/load is model-agnostic: if we can name the model, TorchLean can compute the expected
parameter shapes and reject stale or mismatched checkpoint files.

## Why This Is A Separate Example

The inference-only workflow is direct: load a checkpoint, convert it into runtime
parameter handles, and sample text without building a training loop.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Gpt2Saved

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean gpt2_saved"

/-- Help text for checkpoint-only GPT-2 sampling. -/
def usage : String :=
  String.intercalate "\n"
    [ "Usage:"
    , "  lake exe -K cuda=true torchlean gpt2_saved --cuda --params PATH [generation flags]"
    , ""
    , "Required:"
    , "  --params PATH        JSON parameter checkpoint written by `torchlean gpt2 --save-params`"
    , ""
    , "Common generation flags:"
    , "  --prompt TEXT        prompt prefix"
    , "  --generate N         number of bytes to generate"
    , "  --temperature X      sampling temperature"
    , "  --top-k N            top-k cutoff"
    , "  --seed N             sampling seed"
    ]

/--
Load parameters from disk and run sampling with the fixed byte-level GPT architecture.

Important: the checkpoint must match `Gpt2.model`'s parameter shapes. If the model configuration
in `Gpt2.lean` changes (heads, width, layers, etc.), mismatched checkpoints fail the shape check
before sampling starts.
-/
def sampleWithSavedParams
    (load : text.SavedParamsGenerationOptions) :
    IO String := do
  nn.withModel NN.Examples.Models.Sequence.Gpt2.model fun model => do
    -- The checkpoint boundary is shape-indexed: stale files fail before sampling starts.
    let paramsBits ← Checkpoint.loadModelParamBits model load.paramsPath
    let compiled ← nn.compileOut model (α := Float)
    let predict : NN.Examples.Models.Sequence.Gpt2.Predictor :=
      fun x => pure <| nn.predict1 model compiled paramsBits x
    let outIds ←
      NN.Examples.Models.Sequence.Gpt2.generateSampled predict load.prompt load.generate
        load.temperature load.topK load.seed load.repeatWindow load.repeatPenalty load.asciiOnly
    let txt := text.escapeByteIdsForDisplay outIds
    IO.println s!"  loaded={load.paramsPath}"
    IO.println s!"  prompt={text.escapeForDisplay load.prompt}"
    IO.println s!"  sampled={txt}"
    pure txt

/-- CLI entrypoint for saved-parameter sampling. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  ModelZoo.runFloat exeName args
    (banner := fun _ => s!"{exeName}: sample from saved params")
    (k := fun _opts rest => do
      let (load, rest) ← ModelZoo.orThrow exeName <|
        text.SavedParamsGenerationOptions.parse exeName rest
          { prompt := "First Citizen:"
            generate := 96
            temperature := 0.85
            topK := 12
            repeatPenalty := 1.25
            repeatWindow := 24
            seed := 7
            asciiOnly := false }
      CLI.requireNoArgs exeName rest
      let _ ← sampleWithSavedParams load)

end NN.Examples.Models.Sequence.Gpt2Saved
