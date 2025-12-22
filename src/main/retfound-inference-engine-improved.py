import sys
import os 
import torch
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, show_images, train_one_epoch_retfound, validate_retfound, weighted_class_imbalance
from hparams.hparams import NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, RANK_OPTIONS, ALPHA_OPTIONS, DROPOUT_OPTIONS, NUM_TRIALS, NUM_WORKERS, DEVICE
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig, TaskType

def objective(trial):
    print(f"\n{'='*50}")
    print(f"Starting Trial {trial.number + 1}")
    print(f"{'='*50}")
    # Optuna hparams optimization
    lr = trial.suggest_float("lr", LEARNING_RATE[0], LEARNING_RATE[1], log=True)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE)
    weight_decay = trial.suggest_categorical("weight_decay", WEIGHT_DECAY)
    lora_r = trial.suggest_categorical("lora_r", RANK_OPTIONS)
    lora_alpha = trial.suggest_categorical("lora_alpha", ALPHA_OPTIONS)
    lora_dropout = trial.suggest_categorical("lora_dropout", DROPOUT_OPTIONS)

    DATA_DIR = "../../datasets"
    SRC_DIR = "../"
    
    train_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "IDRID": f"{DATA_DIR}/IDRID",
        "MESSIDOR": f"{DATA_DIR}/MESSIDOR",
    }
    val_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "IDRID": f"{DATA_DIR}/IDRID",
        "MESSIDOR": f"{DATA_DIR}/MESSIDOR",
    }

    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    validation_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CombinedDRDataSet(root_directories=train_root_directories, split="train", img_transform=train_transformations, label_transform=identity_transform)
    validation_dataset = CombinedDRDataSet(root_directories=val_root_directories, split="val", img_transform=validation_transformations, label_transform=identity_transform)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


    model = models_vit.__dict__["vit_large_patch16"](num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True)
    checkpoint_path = f"{SRC_DIR}/models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"\n{'='*60}")
    print(f"Starting Trial {trial.number + 1}")
    print(f"{'='*60}\n")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()


    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj"],
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["head"]
    )

    model = get_peft_model(model, peft_config)
    model = model.to(DEVICE)

    class_weights = weighted_class_imbalance(train_dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    scaler = GradScaler()

    best_trial_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        train_one_epoch_retfound(model, train_loader, criterion, optimizer, DEVICE, epoch, scaler)
        val_loss, val_acc = validate_retfound(model, validation_loader, criterion, DEVICE)
        scheduler.step()

        if val_acc > best_trial_acc:
            best_trial_acc = val_acc
            trial.set_user_attr("best_model_state", model.state_dict())

    return best_trial_acc

def save_best_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        torch.save({
            'trial_number': trial.number,
            'params': trial.params,
            'val_acc': trial.value,
            'model_state_dict': trial.user_attrs["best_model_state"],
        }, '../best_models/best_retfound_model.pth')

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3, callbacks=[save_best_model_callback])
    
    print(f"Best Accuracy: {study.best_trial.value}")
    print(f"Best Params: {study.best_trial.params}")
