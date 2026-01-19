import sys
import os 
import json
import torch
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.UrFound.finetune import models_vit
from models.UrFound.util import pos_embed
from timm.models.layers import trunc_normal_
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, show_images, test_urfound, weighted_class_imbalance, calculate_metrics
from hparams.hparams import NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, RANK_OPTIONS, ALPHA_OPTIONS, DROPOUT_OPTIONS, NUM_TRIALS, NUM_WORKERS, DEVICE
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig, TaskType



DATA_DIR = "../../../datasets"
SRC_DIR = "../../"

test_root_directories = {
    "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
    "IDRID": f"{DATA_DIR}/IDRID",
    "MESSIDOR": f"{DATA_DIR}/MESSIDOR",
    "MFIDDR": f"{DATA_DIR}/MFIDDR"
}

test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_dataset = CombinedDRDataSet(root_directories=test_root_directories, split="test", img_transform=test_transformations, label_transform=identity_transform)

test_csv_paths = {
        "IDRID": f"{test_root_directories['IDRID']}/B-Disease-Grading/Disease-Grading/2-Groundtruths/IDRiD_Disease_Grading_Testing_Labels.csv",
        "DEEPDRID": f"{test_root_directories['DEEPDRID']}/regular_fundus_images/Online-Challenge1&2-Evaluation/Challenge1_labels.csv",
        "MESSIDOR": f"{test_root_directories['MESSIDOR']}/messidor_data.csv",
        "MFIDDR": f"{test_root_directories['MFIDDR']}/sample/test_fourpic_label.csv"
}

test_dataset.load_labels_from_csv_for_test(test_csv_paths)


# note to self: THIS IS THE MODEL (during testing phase we only load weights)
checkpoint_path = f"{SRC_DIR}/best_models/best_urfound_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)



print("Checkpoint keys:", checkpoint.keys())
print("\nCheckpoint structure:")
for key in checkpoint.keys():
    print(f"  {key}: {type(checkpoint[key])}")
    if isinstance(checkpoint[key], dict):
        print(f"    Subkeys: {list(checkpoint[key].keys())[:5]}...")  


params = checkpoint['params']
batch_size = params['batch_size']
lr = params['lr']
weight_decay = params['weight_decay']
lora_r = params['lora_r'] 
lora_alpha = params['lora_alpha']
lora_dropout = params['lora_dropout']


print(f"\nUsing parameters from training:")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {lr}")
print(f"  LoRA r: {lora_r}")
print(f"  LoRA alpha: {lora_alpha}")



model = models_vit.__dict__["vit_base_patch16"](
    num_classes=NUM_CLASSES,
    drop_path_rate=0.2,
    global_pool=True
)

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

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



# param count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


class_weights = weighted_class_imbalance(test_dataset).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


test_loss, test_acc, precision, recall, f1, quadratic_weighted_kappa = test_urfound(
    model, test_loader, criterion, DEVICE
)

print(f"\nFINAL TEST RESULTS:")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Quadratic Weighted Kappa: {quadratic_weighted_kappa:.4f}")

print(f"\nCOMPARISON WITH TRAINING:")
print(f"Validation accuracy during training: {checkpoint['val_acc']:.2f}%")
print(f"Test accuracy on unseen data: {test_acc:.2f}%")
print(f"Difference: {test_acc - checkpoint['val_acc']:+.2f}%")


results = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "quadratic_weighted_kappa": float(quadratic_weighted_kappa),
    "training_validation_accuracy": float(checkpoint['val_acc']),
    "hyperparameters": params,
    "trial_number": int(checkpoint['trial_number'])
}

results_path = "results/urfound_test_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {results_path}")
print("="*70)
