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
from utilities.utils import show_img
from data_processing.dataset import CombinedDRDataSet


# sys.path already adjusted above

if __name__ == "__main__":

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

    train_dataset = CombinedDRDataSet(root_directories=root_directories, split="train", img_transform=train_transformations)
    test_dataset = CombinedDRDataSet(root_directories=root_directories, split="test", img_transform=test_transformations)


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

    train_dataset.load_labels_from_csv(train_csv_paths)
    # train_dataset.load_labels_from_csv(test_csv_paths)


    print("TRAIN DATASET LENGTH:", train_dataset.__len__()) # 11/11/25 len is 0 for both hence there is error in data preprocessing 
    # print("TEST DATASET LENGTH:", test_dataset.__len__())



    # printing dataset statistics
    print("Training Set Statistics:")
    print(train_dataset.get_dataset_statistics())
    # print("\nTest Set Statistics:")
    # print(test_dataset.get_dataset_statistics())

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=4
    # )
    





    # model = models_vit.__dict__["vit_large_patch16"]( # getting model args and params
    #     num_classes=2,
    #     drop_path_rate=0.2,
    #     global_pool=True
    # )

    # # loading RETFound weights
    # checkpoint = torch.load("../models/RETFound_MAE/weights/RETFound_cfp_weights.pth", map_location="cpu", weights_only=False) # running on GPU
    # checkpoint_model = checkpoint["model"]
    # state_dict = model.state_dict()

    # for k in ["head.weight", "head.bias"]:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoints")
    #         del checkpoint_model[k]

    # # interpolated position embedding
    # pos_embed.interpolate_pos_embed(model, checkpoint_model)

    # msg = model.load_state_dict(checkpoint_model, strict=False)
    # assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}

    # # initializing fc layer
    # trunc_normal_(model.head.weight, std=2e-5)
    # print("Model = %s" % str(models_vit))
