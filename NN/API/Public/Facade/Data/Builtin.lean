/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Data.Text

/-!
# TorchLean Built-In Public Data Samples

Small built-in tutorial datasets, sample constructors, and tensor-source operations.
-/

@[expose] public section

namespace TorchLean

namespace Data

namespace Bands

@[inherit_doc NN.API.Samples.Bands.shape]
abbrev shape := NN.API.Samples.Bands.shape

/-- Small vertical-vs-horizontal band dataset, encoded as one-hot class targets. -/
def dataset : Trainer.Dataset shape (.dim 2 .scalar) :=
  { build := fun {α} _ =>
      pure <| NN.API.Data.labeled (α := α) (σ := shape) 2 NN.API.Samples.Bands.trainFloat }

/-- Named vertical/horizontal probes for public CNN examples. -/
def probes : List (Trainer.ClassProbe shape) :=
  NN.API.Samples.Bands.probesFloat.map (fun (name, xF, expected) =>
    { name := name
      input := fun {α} _ _ => NN.API.Common.castTensor (NN.API.Runtime.ofFloat (α := α)) xF
      expected := expected })

/-- Concrete Float probe samples for post-training prediction demos in public classifier quickstarts. -/
def probeSamples : List (String × Tensor.T Float shape × Nat) :=
  NN.API.Samples.Bands.probesFloat

end Bands

namespace TensorSource

export NN.API.Data.TensorSource
  (loadCsvTensorND loadFloatAs loadFloatLeadingPrefixAs loadFloat)

end TensorSource

end Data

end TorchLean
