/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Runtime Tensor-Pack Facade

Public tensor-pack names used by runtime-facing examples.
-/

@[expose] public section

namespace TorchLean

namespace tensorpack

@[inherit_doc TorchLean.TensorPack]
abbrev TensorPack := TorchLean.TensorPack

export NN.API.tensorpack
  (mk1 mk2 mk3 mk4 map zipWith append split unpack1 unpack2 unpack3 unpack4 get0 get1 get2 get3)

end tensorpack


end TorchLean
