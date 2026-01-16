import sys
import os 
import torch
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, show_images, train_one_epoch_retfound, test_retfound, weighted_class_imbalance, calculate_metrics
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



model = models_vit.__dict__["vit_large_patch16"](num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True)
checkpoint_path = f"{SRC_DIR}/best_models/best_retfound_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("Checkpoint keys:", checkpoint.keys())
print("\nCheckpoint structure:")
for key in checkpoint.keys():
    print(f"  {key}: {type(checkpoint[key])}")
    if isinstance(checkpoint[key], dict):
        print(f"    Subkeys: {list(checkpoint[key].keys())[:5]}...")  

checkpoint_model = checkpoint["model_state_dict"]
state_dict = model.state_dict()


for k in ["head.weight", "head.bias"]:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        del checkpoint_model[k]

pos_embed.interpolate_pos_embed(model, checkpoint_model)
model.load_state_dict(checkpoint_model, strict=False)
model.eval()
trunc_normal_(model.head.weight, std=2e-5)


model = model.to(DEVICE)

# param count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")



# SETTING STATIC paramas and LOADING params from the state_dict
learning_rate = checkpoint["params"]["lr"]
weight_decay = checkpoint["params"]["weight_decay"]
batch_size = checkpoint["params"]["batch_size"]
NUM_EPOCHS = 50


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

class_weights = weighted_class_imbalance(test_dataset).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

scaler = GradScaler()

best_acc = 0.0
best_loss = float("inf")
for epoch in range(NUM_EPOCHS):
    test_loss, test_acc, precision, recall, f1,  quadratic_weighted_kappa  = test_retfound(model, test_loader, criterion, DEVICE)
    scheduler.step()    

    if test_acc > best_acc:
        best_acc = test_acc
    
    if test_loss < best_loss:
        best_loss = test_loss
    
    print(f"METRICS = current_acc: {test_acc}, current_loss: {test_loss}, precision: {precision}, recall: {recall}, f1: {f1}, quadratic_weighted_kappa: {quadratic_weighted_kappa}")
    
    print(f"best_acc: {best_acc}, best_loss: {best_loss}")

