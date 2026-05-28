"""Hyperparameters aligned with the runnable TorchLean examples."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "real"

# NN.Examples.Models.Supervised.Mlp
MLP_BATCH = 5
MLP_IN_DIM = 7
MLP_HID_DIM = 32
MLP_OUT_DIM = 1
MLP_LR = 1e-3
MLP_SEED = 0
AUTO_MPG_CSV = DATA_DIR / "auto_mpg" / "auto_mpg.csv"

# NN.Examples.Models.Vision.Cnn
CNN_BATCH = 4
CNN_IN_C = 3
CNN_IN_H = 32
CNN_IN_W = 32
CNN_OUT_DIM = 10
CNN_CONV_OUT_C = 16
CNN_CONV_K = 3
CNN_CONV_STRIDE = 2
CNN_CONV_PADDING = 1
CNN_POOL_K = 2
CNN_POOL_STRIDE = 2
CNN_LR = 1e-3
CNN_SEED = 0
CNN_N_ROWS = 200
CIFAR_X = DATA_DIR / "cifar10" / "cifar10_train_X.npy"
CIFAR_Y = DATA_DIR / "cifar10" / "cifar10_train_y.npy"

# NN.Examples.Models.Sequence.Gpt2
GPT2_BATCH = 2
GPT2_SEQ_LEN = 64
GPT2_VOCAB = 256
GPT2_NUM_HEADS = 2
GPT2_HEAD_DIM = 16
GPT2_D_MODEL = GPT2_NUM_HEADS * GPT2_HEAD_DIM
GPT2_FFN_HIDDEN = 128
GPT2_LAYERS = 2
GPT2_LR = 1e-3
GPT2_WINDOWS = 128
GPT2_PROMPT = "First Citizen:"
GPT2_PAD_ID = 32
GPT2_SEED_STRIDE = 100
TINY_SHAKESPEARE = DATA_DIR / "text" / "tiny_shakespeare.txt"

STEP_COUNTS = (1000, 10000)
