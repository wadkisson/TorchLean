/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import LeanProfiler

/-!
# GPT-2 (~500M) TorchLean twin of `benchmark/train_gpt2.py`

Lab harness for measuring the stock CUDA eager path (no product hot-path patches):

* `--device cuda` + `-K cuda=true`
* default CUDA MHA capsule (native fused flash on current `main`)
* integer token-id embedding gather (`embeddingBatchSeqNat`), not one-hot matmul
* storage-first `instantiateFloat` at this call site (existing API; not a runtime default change)

Same hyperparams / print scheme as the Python script: sample `"ROMEO:"` every 25 steps,
print loss every 10.

```bash
lake -R -K cuda=true build benchmark_gpt2
lake -R -K cuda=true exe benchmark_gpt2 --device cuda

# profiled run (LeanProfiler):
LEAN_PROFILE=1 BENCH_MAX_ITERS=10 LEAN_PROFILE_OUT=benchmark/out/torchlean-trace.json \
  lake -R -K cuda=true exe benchmark_gpt2 --device cuda
```

Env knobs: `BENCH_MAX_ITERS`, `BENCH_ACCUM`, `BENCH_SAMPLE`.

Known API gaps vs the PyTorch script (not available in TorchLean yet):
pre-norm blocks, weight tying, bf16 AMP, true grad-accumulation before `opt.step`.
-/

@[expose] public section

open TorchLean
open Spec
open LeanProfiler

/--
Profile an `IO` action under `name`.

Bypasses LeanProfiler's `span`/`Spannable` layer: the pure instance is higher priority than the
`IO` one, so `span "n" ioAction` can type as `IO (IO α)`. Call `recordSpanWith` directly instead.
-/
@[inline] def spanIO {α : Type} (name : String) (body : IO α) : IO α :=
  if profilingEnabled then
    recordSpanWith name {} body
  else
    body

/--
Mirror of the ~7× `gpt2-500m` timing config:
vocab 50257, seq 128, 32L × 1024d × 16H, batch 1 → ~506M params.
-/
def vocab : Nat := 50257
def blockSize : Nat := 128
def nLayer : Nat := 32
def nHead : Nat := 16
def nEmbd : Nat := 1024
/-- Match the 7×-era harness (batch 1). -/
def batch : Nat := 1
def accum : Nat := 8
def lr : Float := 0.0003
def maxItersDefault : Nat := 2000
def sampleEvery : Nat := 25
def maxNew : Nat := 120
def seed : Nat := 1337
def headDim : Nat := nEmbd / nHead
def ffnHidden : Nat := 4 * nEmbd

def dataPath : System.FilePath := "benchmark/data/tinyshakespeare.txt"

/-- Override with `BENCH_MAX_ITERS` for short profiled comparison runs. -/
def resolveMaxIters : IO Nat := do
  match ← IO.getEnv "BENCH_MAX_ITERS" with
  | none => pure maxItersDefault
  | some s =>
      match s.toNat? with
      | some n => pure n
      | none => throw <| IO.userError s!"BENCH_MAX_ITERS must be a Nat, got {s}"

/--
Short profile runs use `accum=1` so "10 iters" means 10 optimizer steps, not 80.

Full GPT-2 runs keep `accum=8` (still not true grad-accum — each call is fwd+bwd+AdamW).
Override with `BENCH_ACCUM=N`.
-/
def resolveAccum (maxIters : Nat) : IO Nat := do
  match ← IO.getEnv "BENCH_ACCUM" with
  | some s =>
      match s.toNat? with
      | some n =>
          if n = 0 then
            throw <| IO.userError "BENCH_ACCUM must be ≥ 1"
          else
            pure n
      | none => throw <| IO.userError s!"BENCH_ACCUM must be a Nat, got {s}"
  | none =>
      pure (if maxIters < sampleEvery then 1 else accum)

/--
Skip autoregressive sampling on short profile runs.

At small `BENCH_MAX_ITERS`, sampling at iter 0 burns ~120 full 32-layer forwards before train timing.
Set `BENCH_SAMPLE=1` to force sampling anyway.
-/
def resolveDoSample (maxIters : Nat) : IO Bool := do
  match ← IO.getEnv "BENCH_SAMPLE" with
  | some "1" | some "true" => pure true
  | some "0" | some "false" => pure false
  | _ => pure (maxIters ≥ sampleEvery)

/-- Monotonic milliseconds for always-on step timing (independent of LeanProfiler). -/
def monoMs : IO Float := do
  let ns ← IO.monoNanosNow
  pure ((Float.ofNat ns) / 1000000.0)

local instance : NeZero blockSize := ⟨by decide⟩
local instance : NeZero nEmbd := ⟨by decide⟩
local instance : NeZero vocab := ⟨by decide⟩

def cfg : nn.models.CausalOneHotConfig :=
  { batch := batch
    seqLen := blockSize
    vocab := vocab
    numHeads := nHead
    headDim := headDim
    ffnHidden := ffnHidden
    layers := nLayer }

abbrev embσ : Spec.Shape := nn.models.causalEmbeddingShape cfg
abbrev logitτ : Spec.Shape := nn.models.causalOneHotShape cfg
abbrev tokσ : Spec.Shape := NN.API.nn.models.causalTokenIdLmInputShape cfg

/-- Transformer body after token embeddings (shared with the token-id LM module). -/
def bodyM : nn.M (nn.Sequential embσ logitτ) :=
  nn.models.causalTransformerFromEmbeddings cfg

/-- Load Tiny Shakespeare bytes (run the Python script once if missing). -/
def loadTokens : IO (Array Nat) := do
  if !(← dataPath.pathExists) then
    throw <| IO.userError
      s!"missing {dataPath}; run `python3 benchmark/train_gpt2.py` once to download it"
  let text ← IO.FS.readFile dataPath
  pure (text.toByteArray.toList.map (fun b => b.toNat)).toArray

/-- Deterministic window start (same spirit as the Python RNG batching). -/
def windowStart (nToks step row : Nat) : Nat :=
  if nToks ≤ blockSize + 1 then
    0
  else
    (seed + step * 9973 + row * 7919) % (nToks - blockSize - 1)

/-- Flattened float-encoded token ids of length `batch * blockSize`. -/
def mkFlatTokenTensor (ids : Array Nat) : Tensor.T Float tokσ :=
  Tensor.dim fun i =>
    Tensor.scalar (Float.ofNat (ids.getD i.val 0 % vocab))

/-- One train batch: `x` = tokens `[0..T)`, `y` = tokens `[1..T]` (flattened over batch). -/
def mkBatch (toks : Array Nat) (step : Nat) : Tensor.T Float tokσ × Tensor.T Float tokσ :=
  Id.run do
    let mut xIds : Array Nat := Array.mkEmpty (batch * blockSize)
    let mut yIds : Array Nat := Array.mkEmpty (batch * blockSize)
    for row in [0:batch] do
      let start := windowStart toks.size step row
      for t in [0:blockSize] do
        xIds := xIds.push (toks.getD (start + t) 0 % vocab)
        yIds := yIds.push (toks.getD (start + t + 1) 0 % vocab)
    pure (mkFlatTokenTensor xIds, mkFlatTokenTensor yIds)

/-- Host gather of embedding rows → `(batch, seq, dModel)` (for sampling only). -/
def gatherEmbed
    (W : Tensor.T Float (.dim vocab (.dim nEmbd .scalar)))
    (ids : Array Nat) : Tensor.T Float embσ :=
  Tensor.dim fun b =>
    Tensor.dim fun t =>
      let flat := b.val * blockSize + t.val
      let id0 := ids.getD flat 0
      let id := id0 % vocab
      W.get ⟨id, Nat.mod_lt id0 (Nat.pos_of_ne_zero (by decide : vocab ≠ 0))⟩

/-- Decode token ids as UTF-8 bytes (same as the Python `decode`). -/
def decode (ids : List Nat) : String :=
  text.Tokenizer.byte.decode (ids.map (· % 256))

/-- Pad/truncate token ids so the last position is the newest token (left-pad). -/
def fitBlock (ids : Array Nat) : Array Nat :=
  if ids.size ≥ blockSize then
    ids.extract (ids.size - blockSize) ids.size
  else
    Array.replicate (blockSize - ids.size) 0 ++ ids

/-- Cosine+warmup schedule from the Python script (logged; AdamW uses constant `lr`). -/
def lrAt (it maxIters : Nat) : Float :=
  if it < 50 then
    lr * (Float.ofNat (it + 1)) / 50.0
  else
    let progress := (Float.ofNat (it - 50)) / Float.ofNat (Nat.max (maxIters - 50) 1)
    let coeff := 0.5 * (1.0 + Float.cos (3.141592653589793 * progress))
    lr * 0.1 + coeff * (lr - lr * 0.1)

def fmtNat5 (n : Nat) : String :=
  let s := toString n
  let pad := 5 - s.length
  String.ofList (List.replicate (Nat.max pad 0) ' ') ++ s

/-- Pack a token-id batch for the scalar LM module. -/
def packXY (x y : Tensor.T Float tokσ) : TensorPack Float [tokσ, tokσ] :=
  tensorpack.pair x y

/--
Autoregressive sample: gather embed on host, run Transformer body on GPU via live `ParamList`
(shared with the training module — no second copy of the ~500M body).
-/
partial def generate
    (opts : Options)
    (body : nn.Sequential embσ logitτ)
    (trainMod :
      Module.ScalarModule Float
        ((.dim vocab (.dim nEmbd .scalar)) :: nn.paramShapes body) [tokσ, tokσ])
    (prompt : String) : IO String := do
  let genOpts : text.GenerationOptions :=
    { prompt := prompt
      generate := maxNew
      temperature := 0.8
      topK := 50
      repeatPenalty := 0.0
      repeatWindow := 0
      seed := seed
      asciiOnly := false }
  -- Sync only the embedding table for host gather; body params stay device-resident.
  match trainMod.trainer.params with
  | .cons embedP bodyParams => do
      _root_.Runtime.Autograd.Torch.Internal.syncParamCudaToHost (α := Float) embedP
      let embedW ← embedP.value.get
      let mut ids : Array Nat := (text.Tokenizer.byte.encode prompt).toArray
      for stepIdx in [0:maxNew] do
        let window := fitBlock ids
        let mut batched : Array Nat := Array.mkEmpty (batch * blockSize)
        for _r in [0:batch] do
          batched := batched ++ window
        let xEmb ← span "sample.embed_gather" (gatherEmbed embedW batched)
        let logits ← spanIO "sample.forward" (nn.predict (α := Float) body opts bodyParams xEmb)
        let predPos : Fin blockSize := ⟨blockSize - 1, by decide⟩
        let scores := text.batchLogitScoresAt logits ⟨0, by decide⟩ predPos.val
        let next := text.chooseNextToken scores genOpts stepIdx
        ids := ids.push (next % 256)
      pure (decode ids.toList)

profiled def main (args : List String) : IO UInt32 :=
  Runtime.runFloat "benchmark_gpt2" args
    (banner := fun _ =>
      "benchmark_gpt2: gpt2-500m config (32L/seq128/vocab50257) + stock CUDA MHA + token-id gather")
    (k := fun opts rest => do
      CLI.requireNoArgs "benchmark_gpt2" rest
      if !opts.usesCuda then
        throw <| IO.userError "benchmark_gpt2: need --device cuda (same as the PyTorch script)"
      let maxIters ← resolveMaxIters
      let runAccum ← resolveAccum maxIters
      let doSample ← resolveDoSample maxIters
      let tAll0 ← monoMs
      IO.println "building model graph..."
      let toks ← spanIO "load.data" loadTokens
      let nTrain := (toks.size * 9) / 10
      let trainToks := toks.extract 0 nTrain
      nn.withModel bodyM fun body => do
        let tGraph ← monoMs
        IO.println s!"model graph ready in {tGraph - tAll0} ms"
        let embedShape : Spec.Shape := .dim vocab (.dim nEmbd .scalar)
        -- Cheap host placeholders: storage-first `runtimeInit` allocates real CUDA buffers.
        -- Do NOT call `Seq.initParams body` here — that forces host materialization of ~500M floats.
        let initParams :=
          _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.zeroFloatTList
            (ss := (.dim vocab (.dim nEmbd .scalar)) :: nn.paramShapes body)
        let base :=
          NN.API.nn.models.causalTransformerTokenIdLmScalarModuleDef cfg body initParams
        let runtimeInit :
            Option
              (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.Plan
                ((.dim vocab (.dim nEmbd .scalar)) :: nn.paramShapes body)) :=
          match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? body with
          | some bodyPlan =>
              some
                (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.Plan.cons
                  (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.uniform
                    (-0.02) 0.02 seed)
                  bodyPlan)
          | none => none
        let defn := { base with runtimeInit := runtimeInit }
        let nParams :=
          nn.paramCount (embedShape :: nn.paramShapes body)
        IO.println
          s!"GPT-2 on cuda: {(Float.ofNat nParams) / 1000000.0}M params | train {maxIters} iters | accum {runAccum}"
        IO.println "MHA: stock CUDA capsule (native_cuda.flash_attention unless product defaults change)"
        if !doSample then
          IO.println "sampling OFF for short run (set BENCH_SAMPLE=1 to force)"
        IO.println "initializing params on CUDA (token-id LM + storage-first instantiateFloat)..."
        let tInit0 ← monoMs
        let trainMod ← spanIO "init.model"
          (_root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloat defn opts)
        let tInit1 ← monoMs
        IO.println s!"CUDA param init done in {tInit1 - tInit0} ms (embedding = gather, not one-hot)"
        let step ← Module.optimizerInputs (α := Float) trainMod <|
          optim.adamw { lr := lr, weightDecay := 0.1, beta1 := 0.9, beta2 := 0.95 }
        for it in [0:maxIters + 1] do
          if doSample && it % sampleEvery == 0 then
            let out ← spanIO "sample.generate" (generate opts body trainMod "ROMEO:")
            IO.println s!"\n=== iter {it} ===\n{out}\n"
          if it != maxIters then
            let mut lastXY ← span "train.batch" (mkBatch trainToks it)
            let tStep0 ← monoMs
            for a in [0:runAccum] do
              lastXY ← span "train.batch" (mkBatch trainToks (it * runAccum + a))
              let (x, y) := lastXY
              spanIO "train.step" (step (packXY x y))
            let tStep1 ← monoMs
            IO.println
              s!"iter {fmtNat5 it}  train {tStep1 - tStep0} ms  ({runAccum}× step)  lr {lrAt it maxIters}"
            if maxIters ≥ sampleEvery && it % 10 == 0 then
              let (lastX, lastY) := lastXY
              let lossT ← spanIO "train.loss_eval"
                (_root_.Runtime.Autograd.TorchLean.Module.ScalarModule.forward
                  (α := Float) trainMod (packXY lastX lastY))
              IO.println s!"iter {fmtNat5 it}  loss {lossT.toScalar}"
        let tAll1 ← monoMs
        IO.println s!"total wall {tAll1 - tAll0} ms")
