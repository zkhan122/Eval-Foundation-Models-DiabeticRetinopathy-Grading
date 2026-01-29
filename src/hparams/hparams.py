import math
import torch
from torch.optim import AdamW

NUM_EPOCHS = 120
WARMUP_EPOCHS = 10
COOLDOWN_EPOCHS = 20
LR_MAX = 5e-5
LR_MIN = 5e-9
BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4

def make_param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no weight decay on biases; common practise
        if name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at_epoch(epoch):
    # warmup: linear LR_MIN -> LR_MAX
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / WARMUP_EPOCHS
        return LR_MIN + t * (LR_MAX - LR_MIN)

    # cooldown: last COOLDOWN_EPOCHS epochs stay at LR_MIN
    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN

    # cosine over the middle
    t = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


