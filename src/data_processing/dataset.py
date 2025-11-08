import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

valid_file_extensions = ["jpg", "jpeg", "png"]

class CombinedDRDataSet(Dataset):

    def __init__(self, 
            root_directories: Dict[str, str],
            split: str="train", 
            img_transform: Optional[transforms.Compose] = None,
            label_transform: Optional[transforms.Compose] = None):
        
        self.root_directories = root_directories # dictionary containing dataset name : dataset path
        self.split = split 
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_paths = []
        self.labels = []
        # to track which dataset each image comes from
        self.sources = []

        def __len__(self) -> int:
            return len(self.image_paths)

        def load_MFIDDR(self):
            MFIDDR_ROOT = Path(self.root_directories["mfiddr"])

            if self.split == "train":
                image_dir = MFIDDR_ROOT / 'sample' / 'train-examples'
            else:
                image_dir = MFIDDR_ROOT / 'sample' / 'test-examples'

            if not image_dir.exists():
                print(f"ERROR: MFIDDR path not found at {image_dir}")

            # collecting the images

            for image_file_path in os.listdir(image_dir):
                filename, file_extension = os.path.splitext(image_file_path)
                if file_extension in valid_file_extensions:
                    self.image_paths.append(str(image_file_path))
                    self.labels.append(filename)
                    self.sources.append("mfiddr")

        def load_IDRID(self):
            IDRID_root = Path(self.root_dirs["idrid"])
            base_path = IDRID_root / 'B-Disease-Grading' / 'Disease-Grading' / '1-Original-Images'

            if self.split == "train":
                image_dir = base_path / 'Training_Set'
            else:
                image_dir = base_path / 'Testing_Set'

            if not image_dir.exists():
                print(f"Warning: IDRID {self.split} directory could not be found")

            for image_file_path in os.listdir(image_dir):
                filename, file_extension = os.path.splitext(image_file_path)
                if file_extension in valid_file_extensions:
                    self.image_paths.append(str(image_file_path))
                    self.labels.append(filename)
                    self.sources.append("idrid")

        def load_DEEPDRID(self):
            DEEPDRID_root = Path(self.root_dirs["deepdrid"])

            if self.split == "train":
                image_dir = DEEPDRID_root / "regular_fundus_images" / "regular_fundus_training"
                # image_dir2 = DEEPDRID_root / "regular_fundus_images" / "regular_fundus_validation"
            else:
                image_dir = DEEPDRID_root / "regular_fundus_images" / "Online-Challenge1&2-Evaluation"

            if not image_dir.exists():
                print(f"ERROR: MFIDDR path not found at {image_dir}")

            for image_file_path in os.listdir(image_dir):
                filename, file_extension = os.path.splitext(image_file_path)
                if file_extension in valid_file_extensions:
                    self.image_paths.append(str(image_file_path))
                    self.labels.append(filename)
                    self.sources.append("deepdrid")

        def extract_label_deepdr(self, filename: str):
            return 0

        def extract_labels_mfiddr(self, filename: str):
            return 0
        
        def extract_label_idrid(self, filename: str):
            return 0
        
        def load_labels_from_csv(self, csv_paths_dict: Dict[str, str]): # {'dataset name': 'path_to_labels.csv'}
            for dataset_name, csv_path in csv_paths_dict.items():
                if not os.path.exists(csv_path):
                    print(FileNotFoundError, f"at path {csv_path}")
                labels_df = pd.read_csv(csv_path)

                for index, (img_path, source) in enumerate(zip(self.image_paths, self.sources)):
                    if source == dataset_name:
                        filename = Path(img_path).stem
                        # matching filename to the csv and updating the model
                        if filename in labels_df["image_id"].values:
                            label = labels_df[labels_df["image_id"]] == filename["grade"].values[0]
                            self.labels[index] = label


        def __getitem__(self, index) -> Tuple[torch.Tensor, int, str]:
            img_path = self.image_paths[index]
            label = self.labels[index]
            source = self.sources[index]

            # loading the image 
            image = Image.open(img_path).convert("RGB")
            # applying transformations to images and labels
            if self.img_transform is None:
                print("Image transform -> None")
            
            if self.label_transform is None:
                print("Label trasnform -> None")
            
            image_trans = self.img_transform(image)
            label_trans = self.label_transform(label)
            return image_trans, label_trans, source
            
