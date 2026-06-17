/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN.Layers

/-!
# TorchLean NN Model Constructors

User-facing model-zoo constructors under `TorchLean.nn.models`.
-/

@[expose] public section

namespace TorchLean

namespace nn

namespace models

@[inherit_doc NN.API.nn.models.Mlp1Config]
abbrev Mlp1Config := NN.API.nn.models.Mlp1Config

@[inherit_doc NN.API.nn.models.mlp1InShape]
abbrev mlp1InShape := NN.API.nn.models.mlp1InShape

@[inherit_doc NN.API.nn.models.mlp1OutShape]
abbrev mlp1OutShape := NN.API.nn.models.mlp1OutShape

@[inherit_doc NN.API.nn.models.mlp1Relu]
abbrev Mlp1ReLU := NN.API.nn.models.mlp1Relu

@[inherit_doc NN.API.nn.models.KANEdgeFamily]
abbrev KANEdgeFamily := NN.API.nn.models.KANEdgeFamily

@[inherit_doc NN.API.nn.models.KANPiecewiseLinear]
abbrev KANPiecewiseLinear := NN.API.nn.models.KANPiecewiseLinear

namespace KANPiecewiseLinear

@[inherit_doc NN.API.nn.models.KANPiecewiseLinear.edgeFamily]
abbrev edgeFamily := NN.API.nn.models.KANPiecewiseLinear.edgeFamily

end KANPiecewiseLinear

@[inherit_doc NN.API.nn.models.KANConfig]
abbrev KANConfig := NN.API.nn.models.KANConfig

@[inherit_doc NN.API.nn.models.kanInShape]
abbrev kanInShape := NN.API.nn.models.kanInShape

@[inherit_doc NN.API.nn.models.kanOutShape]
abbrev kanOutShape := NN.API.nn.models.kanOutShape

@[inherit_doc NN.API.nn.models.kanLayer]
abbrev KANLayer := NN.API.nn.models.kanLayer

@[inherit_doc NN.API.nn.models.KAN]
abbrev KAN := NN.API.nn.models.KAN

@[inherit_doc NN.API.nn.models.CnnConfig]
abbrev CnnConfig := NN.API.nn.models.CnnConfig

@[inherit_doc NN.API.nn.models.cnnInShape]
abbrev cnnInShape := NN.API.nn.models.cnnInShape

@[inherit_doc NN.API.nn.models.cnnOutShape]
abbrev cnnOutShape := NN.API.nn.models.cnnOutShape

@[inherit_doc NN.API.nn.models.cnn]
def CNN (cfg : CnnConfig)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_kH : cfg.conv.kH ≠ 0 := by decide)
    (h_kW : cfg.conv.kW ≠ 0 := by decide)
    (h_poolKH : cfg.pool.kH ≠ 0 := by decide)
    (h_poolKW : cfg.pool.kW ≠ 0 := by decide) :
    M (Sequential (cnnInShape cfg) (cnnOutShape cfg)) :=
  NN.API.nn.models.cnn cfg h_inC h_kH h_kW h_poolKH h_poolKW

@[inherit_doc NN.API.nn.models.ResnetConfig]
abbrev ResnetConfig := NN.API.nn.models.ResnetConfig

@[inherit_doc NN.API.nn.models.resnetInShape]
abbrev resnetInShape := NN.API.nn.models.resnetInShape

@[inherit_doc NN.API.nn.models.resnetOutShape]
abbrev resnetOutShape := NN.API.nn.models.resnetOutShape

@[inherit_doc NN.API.nn.models.resnet]
def ResNet (cfg : ResnetConfig)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_inH : cfg.inH ≠ 0 := by decide)
    (h_inW : cfg.inW ≠ 0 := by decide)
    (h_stemC : cfg.stemC ≠ 0 := by decide)
    (h_stage2C : cfg.stage2C ≠ 0 := by decide) :
    M (Sequential (resnetInShape cfg) (resnetOutShape cfg)) :=
  NN.API.nn.models.resnet cfg h_batch h_inC h_inH h_inW h_stemC h_stage2C

@[inherit_doc NN.API.nn.models.VitConfig]
abbrev VitConfig := NN.API.nn.models.VitConfig

@[inherit_doc NN.API.nn.models.vitInShape]
abbrev vitInShape := NN.API.nn.models.vitInShape

@[inherit_doc NN.API.nn.models.vitOutShape]
abbrev vitOutShape := NN.API.nn.models.vitOutShape

@[inherit_doc NN.API.nn.models.vit1]
def ViT (cfg : VitConfig)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_patchH : cfg.patchH ≠ 0 := by decide)
    (h_patchW : cfg.patchW ≠ 0 := by decide)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    M (Sequential (vitInShape cfg) (vitOutShape cfg)) :=
  NN.API.nn.models.vit1 cfg h_inC h_patchH h_patchW h_seqLen h_dModel

@[inherit_doc NN.API.nn.models.SeqRnnHeadConfig]
abbrev SeqRnnHeadConfig := NN.API.nn.models.SeqRnnHeadConfig

@[inherit_doc NN.API.nn.models.seqRnnHeadInShape]
abbrev seqRnnHeadInShape := NN.API.nn.models.seqRnnHeadInShape

@[inherit_doc NN.API.nn.models.seqRnnHeadOutShape]
abbrev seqRnnHeadOutShape := NN.API.nn.models.seqRnnHeadOutShape

@[inherit_doc NN.API.nn.models.rnnWithLinearHead]
abbrev RNNWithLinearHead := NN.API.nn.models.rnnWithLinearHead

@[inherit_doc NN.API.nn.models.lstmWithLinearHead]
abbrev LSTMWithLinearHead := NN.API.nn.models.lstmWithLinearHead

@[inherit_doc NN.API.nn.models.TransformerEncoderConfig]
abbrev TransformerEncoderConfig := NN.API.nn.models.TransformerEncoderConfig

@[inherit_doc NN.API.nn.models.transformerEncoderShape]
abbrev transformerEncoderShape := NN.API.nn.models.transformerEncoderShape

@[inherit_doc NN.API.nn.models.transformerEncoder]
def TransformerEncoder (cfg : TransformerEncoderConfig)
    (h_seqLen : cfg.seqLen ≠ 0)
    (h_dModel : cfg.dModel ≠ 0) :
    M (Sequential (transformerEncoderShape cfg) (transformerEncoderShape cfg)) :=
  NN.API.nn.models.transformerEncoder cfg h_seqLen h_dModel

@[inherit_doc NN.API.nn.models.MambaTextConfig]
abbrev MambaTextConfig := NN.API.nn.models.MambaTextConfig

@[inherit_doc NN.API.nn.models.mambaTokenMat]
abbrev mambaTokenMat := NN.API.nn.models.mambaTokenMat

@[inherit_doc NN.API.nn.models.mambaLogitMat]
abbrev mambaLogitMat := NN.API.nn.models.mambaLogitMat

@[inherit_doc NN.API.nn.models.mambaTextLm]
abbrev MambaTextLM := NN.API.nn.models.mambaTextLm

@[inherit_doc NN.API.nn.models.mambaTrainingOffsets]
abbrev mambaTrainingOffsets := NN.API.nn.models.mambaTrainingOffsets

@[inherit_doc NN.API.nn.models.CausalOneHotConfig]
abbrev CausalOneHotConfig := NN.API.nn.models.CausalOneHotConfig

@[inherit_doc NN.API.nn.models.CausalOneHotConfig.dModel]
abbrev CausalOneHotConfig.dModel := NN.API.nn.models.CausalOneHotConfig.dModel

/-- The public `dModel` projection is the product of heads and per-head width. -/
theorem CausalOneHotConfig.dModel_eq (cfg : CausalOneHotConfig) :
    CausalOneHotConfig.dModel cfg = cfg.numHeads * cfg.headDim := by
  rfl

@[inherit_doc NN.API.nn.models.causalOneHotShape]
abbrev causalOneHotShape := NN.API.nn.models.causalOneHotShape

@[inherit_doc NN.API.nn.models.causalEmbeddingShape]
abbrev causalEmbeddingShape := NN.API.nn.models.causalEmbeddingShape

@[inherit_doc NN.API.nn.models.causalTransformerFromEmbeddings]
def CausalTransformerFromEmbeddings (cfg : CausalOneHotConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : NN.API.nn.models.CausalOneHotConfig.dModel cfg ≠ 0 := by decide) :
    M (Sequential (causalEmbeddingShape cfg) (causalOneHotShape cfg)) :=
  NN.API.nn.models.causalTransformerFromEmbeddings cfg
    (h_seqLen := h_seqLen) (h_dModel := h_dModel)

@[inherit_doc NN.API.nn.models.causalTransformerOneHot]
def CausalTransformerOneHot (cfg : CausalOneHotConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : NN.API.nn.models.CausalOneHotConfig.dModel cfg ≠ 0 := by decide) :
    M (Sequential (causalOneHotShape cfg) (causalOneHotShape cfg)) :=
  NN.API.nn.models.causalTransformerOneHot cfg
    (h_seqLen := h_seqLen) (h_dModel := h_dModel)

@[inherit_doc NN.API.nn.models.VectorGenerativeConfig]
abbrev VectorGenerativeConfig := NN.API.nn.models.VectorGenerativeConfig

@[inherit_doc NN.API.nn.models.vectorGenerativeConfig]
abbrev vectorGenerativeConfig := NN.API.nn.models.vectorGenerativeConfig

@[inherit_doc NN.API.nn.models.compactImageConfig]
abbrev compactImageConfig := NN.API.nn.models.compactImageConfig

@[inherit_doc NN.API.nn.models.vectorDataShape]
abbrev vectorDataShape := NN.API.nn.models.vectorDataShape

@[inherit_doc NN.API.nn.models.vectorLatentShape]
abbrev vectorLatentShape := NN.API.nn.models.vectorLatentShape

@[inherit_doc NN.API.nn.models.vectorVaeOutShape]
abbrev vectorVaeOutShape := NN.API.nn.models.vectorVaeOutShape

@[inherit_doc NN.API.nn.models.flattenBatchPrefix]
def flattenBatchPrefix {α : Type} [Inhabited α]
    (cfg : VectorGenerativeConfig) {source : Shape}
    (hData : cfg.dataDim ≤ Shape.size source)
    (x : Tensor.T α (.dim cfg.batch source)) :
    Tensor.T α (vectorDataShape cfg) :=
  NN.API.nn.models.flattenBatchPrefix cfg hData x

@[inherit_doc NN.API.nn.models.reconstructionSample]
abbrev reconstructionSample {α : Type} (cfg : VectorGenerativeConfig)
    (x : Tensor.T α (vectorDataShape cfg)) :
    SupervisedSample α (vectorDataShape cfg) (vectorDataShape cfg) :=
  NN.API.nn.models.reconstructionSample (α := α) cfg x

@[inherit_doc NN.API.nn.models.vaeSample]
abbrev vaeSample (cfg : VectorGenerativeConfig)
    (x : Tensor.T Float (vectorDataShape cfg)) :
    SupervisedSample Float (vectorDataShape cfg) (vectorVaeOutShape cfg) :=
  NN.API.nn.models.vaeSample cfg x

@[inherit_doc NN.API.nn.models.latentNoise]
abbrev latentNoise := NN.API.nn.models.latentNoise

@[inherit_doc NN.API.nn.models.dataNoise]
abbrev dataNoise := NN.API.nn.models.dataNoise

@[inherit_doc NN.API.nn.models.onesScore]
abbrev onesScore := NN.API.nn.models.onesScore

@[inherit_doc NN.API.nn.models.zerosScore]
abbrev zerosScore := NN.API.nn.models.zerosScore

@[inherit_doc NN.API.nn.models.vectorAutoencoder]
abbrev VectorAutoencoder := NN.API.nn.models.vectorAutoencoder

@[inherit_doc NN.API.nn.models.vectorVae]
abbrev VectorVAE := NN.API.nn.models.vectorVae

@[inherit_doc NN.API.nn.models.vectorVqVae]
abbrev VectorVQVAE := NN.API.nn.models.vectorVqVae

@[inherit_doc NN.API.nn.models.vectorGanGenerator]
abbrev VectorGANGenerator := NN.API.nn.models.vectorGanGenerator

@[inherit_doc NN.API.nn.models.vectorGanDiscriminator]
abbrev VectorGANDiscriminator := NN.API.nn.models.vectorGanDiscriminator

@[inherit_doc NN.API.nn.models.VitMaeConfig]
abbrev VitMaeConfig := NN.API.nn.models.VitMaeConfig

@[inherit_doc NN.API.nn.models.vitMaeInShape]
abbrev vitMaeInShape := NN.API.nn.models.vitMaeInShape

@[inherit_doc NN.API.nn.models.vitMaeOutShape]
abbrev vitMaeOutShape := NN.API.nn.models.vitMaeOutShape

@[inherit_doc NN.API.nn.models.vitMaskedAutoencoder]
def VitMAE (cfg : VitMaeConfig)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_patchH : cfg.patchH ≠ 0 := by decide)
    (h_patchW : cfg.patchW ≠ 0 := by decide)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    M (Sequential (vitMaeInShape cfg) (vitMaeOutShape cfg)) :=
  NN.API.nn.models.vitMaskedAutoencoder cfg h_inC h_patchH h_patchW h_seqLen h_dModel

@[inherit_doc NN.API.nn.models.EpsConvNetConfig]
abbrev EpsConvNetConfig := NN.API.nn.models.EpsConvNetConfig

@[inherit_doc NN.API.nn.models.epsConvNetInShape]
abbrev epsConvNetInShape := NN.API.nn.models.epsConvNetInShape

@[inherit_doc NN.API.nn.models.epsConvNetOutShape]
abbrev epsConvNetOutShape := NN.API.nn.models.epsConvNetOutShape

@[inherit_doc NN.API.nn.models.epsConvNet]
def EpsConvNet (cfg : EpsConvNetConfig)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_dataC : cfg.dataC ≠ 0 := by decide)
    (h_inC : (cfg.dataC + 1) ≠ 0 := by decide)
    (h_h : cfg.h ≠ 0 := by decide)
    (h_w : cfg.w ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenC ≠ 0 := by decide) :
    M (Sequential (epsConvNetInShape cfg) (epsConvNetOutShape cfg)) :=
  NN.API.nn.models.epsConvNet cfg h_batch h_dataC h_inC h_h h_w h_hiddenC

@[inherit_doc NN.API.nn.models.epsResidualConvNet]
def EpsResidualConvNet (cfg : EpsConvNetConfig)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_dataC : cfg.dataC ≠ 0 := by decide)
    (h_inC : (cfg.dataC + 1) ≠ 0 := by decide)
    (h_h : cfg.h ≠ 0 := by decide)
    (h_w : cfg.w ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenC ≠ 0 := by decide) :
    M (Sequential (epsConvNetInShape cfg) (epsConvNetOutShape cfg)) :=
  NN.API.nn.models.epsResidualConvNet cfg h_batch h_dataC h_inC h_h h_w h_hiddenC

@[inherit_doc NN.API.nn.models.Fno1dConfig]
abbrev Fno1dConfig := NN.API.nn.models.Fno1dConfig

@[inherit_doc NN.API.nn.models.fno1dInShape]
abbrev fno1dInShape := NN.API.nn.models.fno1dInShape

@[inherit_doc NN.API.nn.models.fno1dOutShape]
abbrev fno1dOutShape := NN.API.nn.models.fno1dOutShape

@[inherit_doc NN.API.nn.models.fno1dReal]
def Fno1dReal (cfg : Fno1dConfig)
    (hModesFit : 2 * cfg.modes ≤ cfg.grid := by decide) :
    M (Sequential (fno1dInShape cfg) (fno1dOutShape cfg)) :=
  NN.API.nn.models.fno1dReal cfg hModesFit

@[inherit_doc NN.API.nn.models.PPOActorCriticConfig]
abbrev PPOActorCriticConfig := NN.API.nn.models.PPOActorCriticConfig

@[inherit_doc NN.API.nn.models.ppoActorInShape]
abbrev ppoActorInShape := NN.API.nn.models.ppoActorInShape

@[inherit_doc NN.API.nn.models.ppoActorOutShape]
abbrev ppoActorOutShape := NN.API.nn.models.ppoActorOutShape

@[inherit_doc NN.API.nn.models.ppoCriticOutShape]
abbrev ppoCriticOutShape := NN.API.nn.models.ppoCriticOutShape

@[inherit_doc NN.API.nn.models.ppoActor]
abbrev PPOActor := NN.API.nn.models.ppoActor

@[inherit_doc NN.API.nn.models.ppoCritic]
abbrev PPOCritic := NN.API.nn.models.ppoCritic

end models

end nn

end TorchLean
