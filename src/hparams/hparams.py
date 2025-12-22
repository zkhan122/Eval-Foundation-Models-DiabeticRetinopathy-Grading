import torch

# inference loop params
NUM_CLASSES = 5
BATCH_SIZE = [4, 8]
NUM_EPOCHS = 50
LEARNING_RATE = [1e-4, 1e-2] # min, max
WEIGHT_DECAY = [0.01, 0.001, 0.005]


# LoRA params
RANK_OPTIONS = [2, 4, 6, 8]
ALPHA_OPTIONS = [8, 16, 32, 64]
DROPOUT_OPTIONS = [0.0, 0.05, 0.1, 0.2]


# Optuna params
NUM_TRIALS = 3

# GPU params
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print(f"Using device: {DEVICE}")
