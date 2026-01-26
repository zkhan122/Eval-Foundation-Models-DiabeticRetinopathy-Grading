import sys
import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn

from data_processing.dataset import CombinedDRDataSet
from utilities.utils import (
    identity_transform,
    train_one_epoch_retfound,
    validate_retfound,
    weighted_class_imbalance,
    validate_retfound_with_metrics
)
from peft import get_peft_model, LoraConfig
from torch.cuda.amp import GradScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from collections import Counter

# -------------------------
# Paper-aligned training hparams
# -------------------------
NUM_CLASSES = 5

NUM_EPOCHS = 120
WARMUP_EPOCHS = 10
COOLDOWN_EPOCHS = 20

LR_MAX = 5e-5
LR_MIN = 5e-9

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4

# Batch + accumulation (effective batch = 128)
MICRO_BATCH_SIZE = 8   # set to what fits on your GPU (e.g., 4, 8, 16)
EFFECTIVE_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)

NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LoRA (fixed baseline; tune later once stable)
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Optional stability knobs
MAX_GRAD_NORM = 1.0
SEED = 42

print(f"Using device: {DEVICE}")
print(f"Micro batch: {MICRO_BATCH_SIZE} | Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS} "
      f"| Grad accum steps: {GRAD_ACCUM_STEPS}")


# -------------------------
# Reproducibility (helps a lot when debugging)
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)


# RETFOUND LR schedule: warmup -> cosine -> cooldown according to paper
def lr_at_epoch(epoch: int) -> float:
    # warmup: linear LR_MIN -> LR_MAX over WARMUP_EPOCHS
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / WARMUP_EPOCHS
        return LR_MIN + t * (LR_MAX - LR_MIN)

    # cooldown: keep LR at LR_MIN for the last COOLDOWN_EPOCHS
    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN

    # cosine decay in the middle
    mid_total = NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, mid_total)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


# AdamW param groups: no weight decay on bias/norm
def make_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no WD on bias; commonly also no WD on LayerNorm/BatchNorm params
        if name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main():
    DATA_DIR = "../../datasets"
    SRC_DIR = "../"

    train_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "IDRID": f"{DATA_DIR}/IDRID",
        "MESSIDOR": f"{DATA_DIR}/MESSIDOR",
    }
    val_root_directories = dict(train_root_directories)

    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # NOTE: this is ImageNet normalization; keep for now, but consider matching RETFound's expected preprocessing if specified.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    validation_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CombinedDRDataSet(
        root_directories=train_root_directories,
        split="train",
        img_transform=train_transformations,
        label_transform=identity_transform
    )
    validation_dataset = CombinedDRDataSet(
        root_directories=val_root_directories,
        split="val",
        img_transform=validation_transformations,
        label_transform=identity_transform
    )

    train_csv_paths = {
        "IDRID": f"{train_root_directories['IDRID']}/B-Disease-Grading/Disease-Grading/2-Groundtruths/IDRiD_Disease_Grading_Training_Labels.csv",
        "DEEPDRID": f"{train_root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv",
        "MESSIDOR": f"{train_root_directories['MESSIDOR']}/messidor_data.csv",
    }
    val_csv_paths = {
        "IDRID": f"{val_root_directories['IDRID']}/B-Disease-Grading/Disease-Grading/2-Groundtruths/IDRiD_Disease_Grading_Training_Labels.csv",
        "DEEPDRID": f"{val_root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv",
        "MESSIDOR": f"{val_root_directories['MESSIDOR']}/messidor_data.csv",
    }

    train_dataset.load_labels_from_csv(train_csv_paths)
    validation_dataset.load_labels_from_csv(val_csv_paths)

    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights_np = 1.0 / np.clip(class_counts, a_min=1, a_max=None)

    sample_weights = class_weights_np[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    classes = list(range(NUM_CLASSES))
    counts = class_counts.tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts)
    plt.xlabel("Class label")
    plt.ylabel("Number of samples")
    plt.title("Training samples per class")
    plt.xticks(classes)
    plt.tight_layout()
    plt.savefig("../train_class_distribution.jpg")


    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # Model + RETFound weights
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )

    checkpoint_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()

    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    # LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv", "proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        modules_to_save=["head"]
    )
    model = get_peft_model(model, peft_config)
    model = model.to(DEVICE)

    # Loss
    class_weights = weighted_class_imbalance(train_dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Optimizer (paper-ish)
    optimizer = torch.optim.AdamW(
        make_param_groups(model, WEIGHT_DECAY),
        lr=LR_MAX,
        betas=BETAS
    )

    scaler = GradScaler()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        # set epoch LR per paper schedule
        lr = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss, train_acc = train_one_epoch_retfound(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            scaler=scaler,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
        )

        val_loss, val_acc, val_bal_acc, val_macro_f1, report = validate_retfound_with_metrics(model, validation_loader, criterion, DEVICE, NUM_CLASSES)

        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
              f"lr={lr:.2e} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%")

        print(f"val_acc={val_acc:.2f}% | bal_acc={val_bal_acc:.2f}% | macro_f1={val_macro_f1:.2f}%")
        print(report)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs("../best_models", exist_ok=True)
    save_path = "../best_models/best_retfound_model.pth"
    torch.save({
        "val_acc": best_val_acc,
        "model_state_dict": best_state,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "train": {
            "epochs": NUM_EPOCHS,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "warmup_epochs": WARMUP_EPOCHS,
            "cooldown_epochs": COOLDOWN_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
            "micro_batch": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch": MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
        }
    }, save_path)

    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    main()

