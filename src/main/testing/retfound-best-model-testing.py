import sys
import os
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

from data_processing.dataset import CombinedDRDataSet
from utilities.utils import (
    identity_transform,
    test_retfound,
    weighted_class_imbalance,
    json_to_csv,
)
from peft import get_peft_model, LoraConfig

# ---- minimal hparams: avoid importing anything that creates optimizers etc.
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../../../datasets"
SRC_DIR = "../../"

def load_retfound_backbone(model):
    """Load RETFound pretrained weights into the base ViT (head removed if shape mismatch)."""
    checkpoint_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = ckpt["model"]
    state_dict = model.state_dict()

    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)
    return model

def main():
    test_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "IDRID": f"{DATA_DIR}/IDRID",
        "MESSIDOR": f"{DATA_DIR}/MESSIDOR",
        "MFIDDR": f"{DATA_DIR}/MFIDDR",
    }

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = CombinedDRDataSet(
        root_directories=test_root_directories,
        split="test",
        img_transform=test_transformations,
        label_transform=identity_transform
    )

    test_csv_paths = {
        "IDRID": f"{test_root_directories['IDRID']}/B-Disease-Grading/Disease-Grading/2-Groundtruths/IDRiD_Disease_Grading_Testing_Labels.csv",
        "DEEPDRID": f"{test_root_directories['DEEPDRID']}/regular_fundus_images/Online-Challenge1&2-Evaluation/Challenge1_labels.csv",
        "MESSIDOR": f"{test_root_directories['MESSIDOR']}/messidor_data.csv",
        "MFIDDR": f"{test_root_directories['MFIDDR']}/sample/test_fourpic_label.csv",
    }

    test_dataset.load_labels_from_csv_for_test(test_csv_paths)

    # --- load best model checkpoint (either Optuna-style or baseline-style)
    best_path = f"{SRC_DIR}/best_models/best_retfound_model.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

    print("Checkpoint keys:", checkpoint.keys())

    # Extract LoRA + batch size from either checkpoint format
    if "params" in checkpoint:
        # old optuna style
        params = checkpoint["params"]
        batch_size = params.get("batch_size", 8)
        lora_r = params["lora_r"]
        lora_alpha = params["lora_alpha"]
        lora_dropout = params["lora_dropout"]
        train_val_acc = checkpoint.get("val_acc", None)
        trial_number = checkpoint.get("trial_number", None)
        hyperparams_out = params
    else:
        # new baseline style
        lora_cfg = checkpoint.get("lora", {"r": 8, "alpha": 32, "dropout": 0.05})
        train_cfg = checkpoint.get("train", {})
        batch_size = train_cfg.get("micro_batch", 8)  # you used MICRO_BATCH_SIZE in training
        lora_r = lora_cfg["r"]
        lora_alpha = lora_cfg["alpha"]
        lora_dropout = lora_cfg["dropout"]
        train_val_acc = checkpoint.get("val_acc", None)
        trial_number = None
        hyperparams_out = {"lora": lora_cfg, "train": train_cfg}

    print("\nUsing parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  LoRA r: {lora_r}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  LoRA dropout: {lora_dropout}")
    print(f"  Device: {DEVICE}")

    # --- build model, load RETFound backbone, then apply LoRA, then load finetuned weights
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=NUM_CLASSES,
        drop_path_rate=0.2,
        global_pool=True
    )
    model = load_retfound_backbone(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj"],
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["head"],
    )
    model = get_peft_model(model, peft_config)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # For test: do NOT use label smoothing; weights are optional
    class_weights = weighted_class_imbalance(test_dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loss, test_acc, precision, recall, f1, quadratic_weighted_kappa = test_retfound(
        model, test_loader, criterion, DEVICE
    )

    print(f"\nFINAL TEST RESULTS:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Quadratic Weighted Kappa: {quadratic_weighted_kappa:.4f}")

    if train_val_acc is not None:
        print(f"\nCOMPARISON WITH TRAINING:")
        print(f"Validation accuracy during training: {train_val_acc:.2f}%")
        print(f"Test accuracy on unseen data: {test_acc:.2f}%")
        print(f"Difference: {test_acc - train_val_acc:+.2f}%")

    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa),
        "training_validation_accuracy": float(train_val_acc) if train_val_acc is not None else None,
        "hyperparameters": hyperparams_out,
        "trial_number": int(trial_number) if trial_number is not None else None,
    }

    os.makedirs("results/retfound", exist_ok=True)
    results_path = "results/retfound/retfound_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    json_to_csv(results_path, "results/retfound", "retfound_results")

    print(f"\nResults saved to: {results_path}")
    print("RETFound csv saved")
    print("=" * 70)

if __name__ == "__main__":
    main()

