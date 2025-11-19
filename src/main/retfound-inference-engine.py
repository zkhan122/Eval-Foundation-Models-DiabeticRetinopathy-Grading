import sys
import os 
import torch
# ensure parent path added so imports work when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.layers import trunc_normal_
import torchvision
from data_processing.dataloader import load_idrid_grading_labels
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, show_images, train_one_epoch, validate
from hparams.hparams import NUM_CLASSES,BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_WORKERS, DEVICE
from torch import nn
from torch import optim

# sys.path already adjusted above

if __name__ == "__main__":

    print(f"Using device: {DEVICE}")

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # loading the data 

    BASE_PATH = "../../datasets/IDRID/B-Disease-Grading/Disease-Grading/2-Groundtruths"

    root_directories = {
        "DEEPDRID": "D:/Zayaan/D_git/Eval-Foundation-Models-DiabeticRetinopathy-Grading/datasets/DEEPDRID",
        "IDRID": "D:/Zayaan/D_git/Eval-Foundation-Models-DiabeticRetinopathy-Grading/datasets/IDRID",
        "MFIDDR": "D:/Zayaan/D_git/Eval-Foundation-Models-DiabeticRetinopathy-Grading/datasets/MFIDDR" 
    }

    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


    # train_labels_df = load_idrid_grading_labels("train", f"{BASE_PATH}/IDRiD_Disease_Grading_Training_Labels.csv")
    # test_labels_df = load_idrid_grading_labels("test", f"{BASE_PATH}/IDRiD_Disease_Grading_Testing_Labels.csv")

    train_dataset = CombinedDRDataSet(root_directories=root_directories, split="train", img_transform=train_transformations, label_transform=identity_transform)
    # test_dataset = CombinedDRDataSet(root_directories=root_directories, split="test", img_transform=test_transformations)


    print("Labels", train_dataset.get_labels())

    # loading csv_paths
    train_csv_paths = {
        "IDRID": f"{root_directories['IDRID']}/B-Disease-Grading/Disease-Grading/2-Groundtruths/IDRiD_Disease_Grading_Training_Labels.csv",
        "DEEPDRID": f"{root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv",
        "MFIDDR": f"{root_directories['MFIDDR']}/sample/train_fourpic_label.csv"
    }

    # test_csv_paths = {
    #     "IDRID": f"{root_directories['IDRID']}/B-Disease-Grading/Disease-Grading/2-Groundtruths/IDRiD_Disease_Grading_Testing_Labels.csv",
    #     "DEEPDRID": f"{root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv",
    #     "MFIDDR": f"{root_directories['MFIDDR']}/sample/test_fourpic_label.csv"
    # }

    train_labels = train_dataset.load_labels_from_csv(train_csv_paths)
    # train_dataset.load_labels_from_csv(test_csv_paths)


    # print("TRAIN DATASET LENGTH:", train_dataset.__len__()) # 11/11/25 len is 0 for both hence there is error in data preprocessing 
    # print("TEST DATASET LENGTH:", test_dataset.__len__())



    # printing dataset statistics
    # print("Training Set Statistics:")
    # print(train_dataset.get_dataset_statistics())

    
    # print("\nTest Set Statistics:")
    # print(test_dataset.get_dataset_statistics())

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    print(train_loader.batch_size)
    
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=4
    # )


    print(train_loader)

    # checking inside dataset
    # for idx in range(len(train_dataset)):
    #     image, label, metadata = train_dataset[idx]
    #     print(f"Image: {image} -- Label: {label} -- Metadata: {metadata}")

    dataset_length = len(train_dataset)
    print("total samples: ", dataset_length)
    
    show_images(train_dataset, train_labels, num_images=60, start_idx=0)

# mfiddr -> idrid -> deepdrid
# 413 idrid
# 1661 total 

    # print(f"\nTrain batches: {len(train_loader)}")
    # # print(f"Test batches: {len(test_loader)}")
    
    # # ============ Initialize Model ============
    # print("\n" + "="*50)
    # print("Initializing RETFound Model")
    # print("="*50)

    # model = models_vit.__dict__["vit_large_patch16"](
    #     num_classes=NUM_CLASSES,
    #     drop_path_rate=0.2,
    #     global_pool=True
    # )

    # checkpoint_path = "../models/RETFound_MAE/weights/RETFound_cfp_weights.pth"
    # print(f"Loading pretrained weights from: {checkpoint_path}") 

    # checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # checkpoint_model = checkpoint["model"]
    # state_dict = model.state_dict()

    #     # Remove head weights (they're for different number of classes)
    # for k in ["head.weight", "head.bias"]:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
    #         del checkpoint_model[k]
    
    # # Interpolate position embeddings if needed
    # pos_embed.interpolate_pos_embed(model, checkpoint_model)
    
    # # Load the weights
    # msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(f"Missing keys: {msg.missing_keys}")
    # print(f"Unexpected keys: {msg.unexpected_keys}")
    
    # # Verify we're only missing the classification head
    # assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
    
    # # Initialize the new classification head
    # trunc_normal_(model.head.weight, std=2e-5)
    
    # # Move model to device
    # model = model.to(DEVICE)
    
    # # Count parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\nTotal parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")
    
    # # ============ Setup Training ============
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # # ============ Training Loop ============
    # print("\n" + "="*50)
    # print("Starting Training")
    # print("="*50)
    
    # best_val_acc = 0.0
    
    # for epoch in range(NUM_EPOCHS):
    #     print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    #     print("-" * 50)
        
    #     # Train
    #     train_loss, train_acc = train_one_epoch(
    #         model, train_loader, criterion, optimizer, DEVICE, epoch
    #     )
        
    #     # Validate
    #     # val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
        
    #     # Update learning rate
    #     scheduler.step()
        
    #     # Print epoch summary
    #     print(f"\nEpoch {epoch+1} Summary:")
    #     print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    #     # print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")\
    #     print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
    #     # Save best model
    #     # if val_acc > best_val_acc:
    #     #     best_val_acc = val_acc
    #     #     print(f"  ✓ New best validation accuracy: {val_acc:.2f}%")
    #     #     torch.save({
    #     #         'epoch': epoch,
    #     #         'model_state_dict': model.state_dict(),
    #     #         'optimizer_state_dict': optimizer.state_dict(),
    #     #         'val_acc': val_acc,
    #     #         'val_loss': val_loss,
    #     #     }, 'best_retfound_model.pth')
    
    # print("\n" + "="*50)
    # print("Training Complete!")
    # print("="*50)
    # print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
