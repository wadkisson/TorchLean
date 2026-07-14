/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN.Layers
public import NN.API.Models

/-!
# Model Zoo

The model zoo is exposed under `TorchLean.nn.models`. Each architecture has one public constructor
and one configuration type.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace models

export NN.API.nn.models
  (MlpConfig mlpInShape mlpOutShape mlpRelu
   KANEdgeFamily KANConfig kanInShape kanOutShape kanLayer KAN
   CnnConfig cnnInShape cnnOutShape cnn
   ResNetConfig resnetInShape resnetHiddenShape resnetOutShape resnet
   VitConfig vitInShape vitConvOutShape vitTokensShape vitOutShape spatialToTokens vit
   SeqRnnHeadConfig seqRnnHeadInShape seqRnnHeadOutShape rnnWithLinearHead lstmWithLinearHead
   TransformerEncoderConfig transformerEncoderShape transformerEncoder
   MambaTextConfig mambaTokenMat mambaLogitMat mambaTextLm mambaTrainingOffsets
   CausalOneHotConfig causalOneHotShape causalEmbeddingShape
   causalTransformerFromEmbeddings causalTransformerOneHot
   VectorGenerativeConfig vectorGenerativeConfig compactImageConfig
   vectorDataShape vectorLatentShape vectorVaeOutShape flattenBatchPrefix reconstructionSample
   vaeSample latentNoise dataNoise onesScore zerosScore
   vectorAutoencoder vectorVae vectorVqVae vectorGanGenerator vectorGanDiscriminator
   VitMaeConfig vitMaeInShape vitMaeOutShape vitMaskedAutoencoder vectorMaskedAutoencoder
   EpsConvNetConfig epsConvNetInShape epsConvNetOutShape epsConvNet epsResidualConvNet
   FNOConfig fnoInShape fnoOutShape fno
   PPOActorCriticConfig ppoActorInShape ppoActorOutShape ppoCriticOutShape ppoActor ppoCritic)

namespace KANPiecewiseLinear

export NN.API.nn.models.KANPiecewiseLinear (basisLayer edgeFamily)

end KANPiecewiseLinear

namespace CausalOneHotConfig

export NN.API.nn.models.CausalOneHotConfig (dModel)

end CausalOneHotConfig

namespace VitConfig

export NN.API.nn.models.VitConfig (patchSpatial seqLen flatDim)

end VitConfig

namespace VitMaeConfig

export NN.API.nn.models.VitMaeConfig (seqLen flatDim)

end VitMaeConfig

end models
end nn
end TorchLean
