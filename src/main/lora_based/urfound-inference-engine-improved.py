import sys
import os
import time
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from torch.cuda.amp import GradScaler
from peft import get_peft_model, LoraConfig

from models.UrFound.finetune import models_vit
from models.UrFound.util import pos_embed
from timm.models.layers import trunc_normal_

from data_processing.dataset import CombinedDRDataSet
from utilities.utils import (
    identity_transform,
    train_one_epoch_urfound,
    class_balanced_weights,
    validate_urfound_with_metrics,
    subsample_dataset,
    save_metric_plot,
    plot_epoch_time,
    plot_gpu_memory,
    plot_throughput,
    plot_benchmark_summary,
    plot_all_benchmark
)



NUM_CLASSES = 5

NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
COOLDOWN_EPOCHS = 10

LR_MIN = 1e-6
LR_MAX = 5e-4

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4

MICRO_BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)

NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

MAX_GRAD_NORM = 1.0

print(f"Using device: {DEVICE}")
print(f"Micro batch: {MICRO_BATCH_SIZE} | Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(42)

def lr_at_epoch(epoch: int) -> float:

    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / WARMUP_EPOCHS
        return LR_MIN + t * (LR_MAX - LR_MIN)

    if epoch >= NUM_EPOCHS - COOLDOWN_EPOCHS:
        return LR_MIN

    mid_total = NUM_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, mid_total)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))

def make_param_groups(model, weight_decay):

    decay, no_decay, lora = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        name_l = name.lower()
        if "lora_" in name_l:
            lora.append(p)
        elif name.endswith(".bias") or "norm" in name_l:
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if lora:
        groups.append({"params": lora, "weight_decay": 0.0})

    return groups

def create_balanced_sampler(dataset, num_classes=NUM_CLASSES):
    """
    Create a WeightedRandomSampler that balances classes during training.
    Each sample gets weighted inversely to its class frequency.
    """
    labels = np.array(dataset.labels, dtype=np.int64)

    class_counts = np.bincount(labels, minlength=num_classes)

    class_weights = class_balanced_weights(class_counts, beta=0.9999, device=DEVICE)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow oversampling minority classes
    )

    print(f"Created balanced sampler with {len(sample_weights)} samples")
    print(f"Sample weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    return sampler


def main():

    DATA_DIR = "../../../datasets"
    SRC_DIR = "../../"

    train_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "EYEPACS": f"{DATA_DIR}/EYEPACS",
        "DDR": f"{DATA_DIR}/DDR",
    }
    val_root_directories = dict(train_root_directories)

    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
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
        "DEEPDRID": f"{train_root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv",
        "EYEPACS": f"{train_root_directories['EYEPACS']}/all_labels.csv",
        "DDR": f"{train_root_directories['DDR']}/DR_grading.csv"
    }

    val_csv_paths = {
        "DEEPDRID": f"{train_root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv",
        "EYEPACS": f"{train_root_directories['EYEPACS']}/all_labels.csv",
        "DDR": f"{train_root_directories['DDR']}/DR_grading.csv"
     }

    train_dataset.load_labels_from_csv(train_csv_paths)
    validation_dataset.load_labels_from_csv(val_csv_paths)
    
    train_dataset.prune_unlabeled()
    validation_dataset.prune_unlabeled()
    
    print("\n" + "="*60)
    print("SUBSAMPLING LARGE DATASETS")
    print("="*60)
    print(f"Before subsampling: {len(train_dataset)} samples")

    train_dataset = subsample_dataset(train_dataset, max_samples_per_class=3000)

    print(f"After subsampling: {len(train_dataset)} samples")
    print("="*60 + "\n")


    # Get class distribution
    labels = np.array(train_dataset.labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)

    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION:")
    print(f"{'='*60}")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count:5d} samples ({count/len(labels)*100:.1f}%)")
    print(f"{'='*60}\n")

    # Calculate class weights with capping
    class_weights_np = len(labels) / (NUM_CLASSES * class_counts.astype(float))
    
    # IMPORTANT: Cap weights to prevent extreme values that cause model collapse
    max_weight = 12.0
    class_weights_np = np.clip(class_weights_np, None, max_weight)
    class_weights_np = class_weights_np / class_weights_np.sum() * NUM_CLASSES  # Re-normalize

    class_weights_tensor = torch.FloatTensor(class_weights_np).to(DEVICE)

    print(f"CLASS WEIGHTS (Balanced, capped at {max_weight}x):")
    for i, weight in enumerate(class_weights_np):
        print(f"Class {i}: {weight:.4f}")
    print()

    classes = list(range(NUM_CLASSES))
    counts = class_counts.tolist()


    print("\n" + "="*60)
    print("CREATING BALANCED SAMPLER FOR TRAINING")
    print("="*60)
    
    
    sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS // 2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )



    model = models_vit.__dict__["vit_base_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )

    checkpoint_path = f"{SRC_DIR}/models/UrFound/weights/urfound_model_weights.pth"
    print(f"Loading pretrained weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()

    # removing head weights (as they dont match number of classes for severity grading 0-4)
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
            del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)

    # Loading the weights
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    # verifying we're only missing the classification head
    assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}

    # initializing the new classification head
    trunc_normal_(model.head.weight, std=2e-5)

    print("\nWrapping model with LoRA adapters...")


    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv", "proj"],  # adjust if needed
        lora_dropout=LORA_DROPOUT,
        bias="none",
        modules_to_save=["head"]
    )

    model = get_peft_model(model, peft_config)

    for name, param in model.named_parameters():
        if "lora" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        make_param_groups(model, WEIGHT_DECAY),
        lr=LR_MAX,
        betas=BETAS
    )

    scaler = GradScaler()

    best_val_bal_acc = 0.0
    best_state = None
        
    history_epochs = []
    history_acc = []
    history_bal_acc = []
    history_macro_f1 = []
    history_macro_auc = []

    benchmark = {
        "epoch_times_s":   [],
        "peak_gpu_mb":     [],
        "train_throughput": [],
    }

    for epoch in range(NUM_EPOCHS):

        lr = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch_urfound(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            scaler=scaler
        )

        epoch_time = time.perf_counter() - t0
        benchmark["epoch_times_s"].append(epoch_time)
        benchmark["train_throughput"].append(len(train_dataset) / epoch_time)
        benchmark["peak_gpu_mb"].append(
            torch.cuda.max_memory_allocated(DEVICE) / 1024**2
            if torch.cuda.is_available() else float("nan")
        )

        if (epoch + 1) % 10 != 0:
            print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
                  f"lr={lr:.2e} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                  f"epoch_time={epoch_time:.1f}s | "
                  f"peak_gpu={benchmark['peak_gpu_mb'][-1]:.0f}MiB | "
                  f"[Skipping validation]")
            continue

        val_loss, val_acc, val_bal_acc, val_macro_f1, val_macro_auc, report =  validate_urfound_with_metrics(
                model, validation_loader, criterion, DEVICE, NUM_CLASSES
            )
        
        history_epochs.append(epoch + 1)
        history_acc.append(val_acc)
        history_bal_acc.append(val_bal_acc)
        history_macro_f1.append(val_macro_f1)
        history_macro_auc.append(val_macro_auc)

        
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
              f"lr={lr:.2e} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%")

        print(f"val_acc={val_acc:.2f}% | bal_acc={val_bal_acc:.2f}% | macro_f1={val_macro_f1:.2f}% | macro_auc={val_macro_auc:.2f}%")
        print(report)

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"★ New best balanced accuracy: {best_val_bal_acc:.2f}%")

    
    plot_save_dir = "./plots/urfound"
    save_metric_plot(history_epochs, history_acc, "Validation Accuracy", plot_save_dir)
    save_metric_plot(history_epochs, history_bal_acc, "Balanced Accuracy", plot_save_dir)
    save_metric_plot(history_epochs, history_macro_f1, "Macro F1", plot_save_dir)
    save_metric_plot(history_epochs, history_macro_auc, "Macro AUROC", plot_save_dir)


    c_times = np.array(benchmark["epoch_times_s"][1:])
    c_gpu   = np.array([v for v in benchmark["peak_gpu_mb"][1:] if not math.isnan(v)])
    c_thru  = np.array(benchmark["train_throughput"][1:])

    benchmark["summary"] = {
        "epochs_measured":    len(c_times),
        "avg_epoch_time_s":   float(np.mean(c_times)),
        "std_epoch_time_s":   float(np.std(c_times)),
        "min_epoch_time_s":   float(np.min(c_times)),
        "max_epoch_time_s":   float(np.max(c_times)),
        "total_train_time_s": float(sum(benchmark["epoch_times_s"])),
        "avg_peak_gpu_mb":    float(np.mean(c_gpu)) if len(c_gpu) else None,
        "max_peak_gpu_mb":    float(np.max(c_gpu))  if len(c_gpu) else None,
        "avg_throughput_sps": float(np.mean(c_thru)),
    }

    s = benchmark["summary"]
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY (epochs 2–{NUM_EPOCHS})")
    print(f"{'='*60}")
    print(f"  Avg epoch time  : {s['avg_epoch_time_s']:.1f}s ± {s['std_epoch_time_s']:.1f}s")
    print(f"  Total train time: {s['total_train_time_s']/60:.1f} min")
    print(f"  Avg peak GPU mem: {s['avg_peak_gpu_mb']:.0f} MiB")
    print(f"  Max peak GPU mem: {s['max_peak_gpu_mb']:.0f} MiB")
    print(f"  Avg throughput  : {s['avg_throughput_sps']:.1f} samples/s")
    print(f"{'='*60}\n")

    os.makedirs("../best_models", exist_ok=True)
    
    with open("../best_models/urfound-benchmark.json", "w") as f:
        json.dump(benchmark, f, indent=4)

    plot_all_benchmark(
        source     = benchmark,
        output_dir = "./plots/retfound/urfound-benchmark-plots",
        skip       = 1,
        model_name = "URFound LoRA",

    )
    save_path = "../best_models/best_urfound_model.pth"

    torch.save({
        "val_bal_acc": best_val_bal_acc,
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

    print(f"\nBest Balanced Accuracy: {best_val_bal_acc:.2f}%")
    print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    main()

