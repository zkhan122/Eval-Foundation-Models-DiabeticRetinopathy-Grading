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
from utilities.utils import show_img

# sys.path already adjusted above

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# loading the data 

BASE_PATH = "../../datasets/IDRID/B-Disease-Grading/Disease-Grading/2-Groundtruths"

train_labels_df = load_idrid_grading_labels("train", f"{BASE_PATH}/IDRiD_Disease_Grading_Training_Labels.csv")
test_labels_df = load_idrid_grading_labels("test", f"{BASE_PATH}/IDRiD_Disease_Grading_Testing_Labels.csv")

print(train_labels_df)
print(test_labels_df)

# using split on training set -> 70-30 for train-val

BATCH_SIZE = 4

train_df, val_df = train_test_split(train_labels_df, test_size=0.3, random_state=42)

trainloader = torch.utils.data.DataLoader(train_df.head(4), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# will define testloader later, avoid touching test set

dataiterator = iter(trainloader)
images, labels = next(dataiterator)

show_img(torchvision.utils.make_grid(images))


model = models_vit.__dict__["vit_large_patch16"]( # getting model args and params
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True
)

# loading RETFound weights
checkpoint = torch.load("../models/RETFound_MAE/weights/RETFound_cfp_weights.pth", map_location="cpu", weights_only=False) # running on GPU
checkpoint_model = checkpoint["model"]
state_dict = model.state_dict()

for k in ["head.weight", "head.bias"]:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoints")
        del checkpoint_model[k]

# interpolated position embedding
pos_embed.interpolate_pos_embed(model, checkpoint_model)

msg = model.load_state_dict(checkpoint_model, strict=False)
assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}

# initializing fc layer
trunc_normal_(model.head.weight, std=2e-5)
print("Model = %s" % str(models_vit))


