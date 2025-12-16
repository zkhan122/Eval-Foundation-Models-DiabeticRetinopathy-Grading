import sys
import os 
import torch
# ensure parent path added so imports work when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.RETFound_MAE import models_vit
from models.RETFound_MAE.util import pos_embed
from timm.models.layers import trunc_normal_
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing.dataset import CombinedDRDataSet
from utilities.utils import identity_transform, show_images, train_one_epoch_clip, validate_clip, get_specific_layer_names
from hparams.hparams import NUM_CLASSES,BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_WORKERS, DEVICE
from torch import nn
from torch import optim
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from peft import get_peft_model, LoraConfig, TaskType
# sys.path already adjusted above

if __name__ == "__main__":

    print(f"Using device: {DEVICE}")


    # loading the data 

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

    test_root_directories = {
        "DEEPDRID": f"{DATA_DIR}/DeepDRiD",
        "IDRID": f"{DATA_DIR}/IDRID",
        "MESSIDOR": f"{DATA_DIR}/MESSIDOR",
        "MFIDDR": f"{DATA_DIR}/MFIDDR", # only use for test if necessary
    }


    # train_labels_df = load_idrid_grading_labels("train", f"{BASE_PATH}/IDRiD_Disease_Grading_Training_Labels.csv")
    # test_labels_df = load_idrid_grading_labels("test", f"{BASE_PATH}/IDRiD_Disease_Grading_Testing_Labels.csv")

    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    validation_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
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

    train_dataset = CombinedDRDataSet(root_directories=train_root_directories, split="train", img_transform=train_transformations, label_transform=identity_transform)
    validation_dataset = CombinedDRDataSet(root_directories=val_root_directories, split="val", img_transform=validation_transformations, label_transform=identity_transform)
    test_dataset = CombinedDRDataSet(root_directories=test_root_directories, split="test", img_transform=test_transformations)


    print("Labels", train_dataset.get_labels())

    # loading csv_paths (removed MFIDDR as too little samples , but possibly can use in testing)
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

    # test_csv_paths = {
    #     "IDRID": f"{root_directories['IDRID']}/B-Disease-Grading/Disease-Grading/2-Groundtruths/IDRiD_Disease_Grading_Testing_Labels.csv",
    #     "DEEPDRID": f"{root_directories['DEEPDRID']}/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv",
    #     "MFIDDR": f"{root_directories['MFIDDR']}/sample/test_fourpic_label.csv"
    # }

    train_labels = train_dataset.load_labels_from_csv(train_csv_paths)
    validation_labels = validation_dataset.load_labels_from_csv(val_csv_paths)
    # test_labels = test_dataset.load_labels_from_csv(test_csv_paths)
    # train_dataset.load_labels_from_csv(test_csv_paths)


    print("TRAIN DATASET LENGTH:", train_dataset.__len__()) # 11/11/25 len is 0 for both hence there is error in data preprocessing
    print("VALIDATION DATASET LENGTH:", validation_dataset.__len__())
    # print("TEST DATASET LENGTH:", test_dataset.__len__())



    # printing dataset statistics
    print("Training Set Statistics:")
    print(train_dataset.get_dataset_statistics())

    print("\n\n")

    print("Validation Set Statistics:")
    print(validation_dataset.get_dataset_statistics())

    # print()

    # print("Test Set Statistics:")
    # print(test_dataset.get_dataset_statistics())


    # print("\nTest Set Statistics:")
    # print(test_dataset.get_dataset_statistics())

    # --- FIX: Set num_workers=0 to prevent freezing on Drive ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print(train_loader.batch_size)


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print(validation_loader.batch_size)

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=4
    # )


    print(f"Train loader: {train_loader}")
    print("Validation loader: {validation_loader}")

    # checking inside dataset
    # for idx in range(len(train_dataset)):
    #     image, label, metadata = train_dataset[idx]
    #     print(f"Image: {image} -- Label: {label} -- Metadata: {metadata}")

    train_dataset_length = len(train_dataset)
    print("TRAIN total samples: ", train_dataset_length)


    validation_dataset_length = len(validation_dataset)
    print("VALIDATION total samples: ", validation_dataset_length)

    # show_images(train_dataset, train_labels, num_images=60, start_idx=0)

    # mfiddr -> idrid -> deepdrid
    # 413 idrid
    # 1661 total

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"\nValidation batches: {len(validation_loader)}")
    # print(f"Test batches: {len(test_loader)}")

    
    # ============ Initialize Model ============
    # show_images(train_dataset, train_labels, num_images=60, start_idx=0)

    # mfiddr -> idrid -> deepdrid
    # 413 idrid
    # 1661 total

    print(f"\nTrain batches: {len(train_loader)}")
    # print(f"Test batches: {len(test_loader)}")


    print("\n" + "="*50)
    print("Initializing CLIP Model")
    print("="*50)

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print(model)
    print(processor)


    print("CLIP model loaded successfully")

    num_classes = 5
    embedding_dim = model.config.projection_dim  # 768 for CLIP ViT-Large
    model.classifier = nn.Linear(embedding_dim, num_classes)

    print(f"Added classification head: {embedding_dim} -> {num_classes} classes")

    print("LAYER NAMES:", list(set(get_specific_layer_names(model))))
    print()
    print("\nWrapping model with LoRA adapters...")
    peft_config = LoraConfig(
        r=16,                  # rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj"], # target attention layers in ViT
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model = model.to(DEVICE)

    # param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    
    # # ============ Training Loop ============
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch_clip(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )

    #   Validate
        val_loss, val_acc = validate_clip(model, validation_loader, criterion, DEVICE)

    #   lr update
        scheduler.step()

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    #     # Saving best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✓ New best validation accuracy: {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_clip_model.pth')

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
