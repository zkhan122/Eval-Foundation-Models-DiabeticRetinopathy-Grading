import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter

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

        if "MFIDDR" in self.root_directories:
            self.load_MFIDDR()
        if "IDRID" in self.root_directories:
            self.load_IDRID()
        if "DEEPDRID" in self.root_directories:
            self.load_DEEPDRID()

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_MFIDDR(self):
        MFIDDR_ROOT = Path(self.root_directories["MFIDDR"])

        print(f"\nMFIDDR_ROOT: {MFIDDR_ROOT}")
        print(f"MFIDDR_ROOT exists: {MFIDDR_ROOT.exists()}")

        if self.split == "train":
            image_dir = MFIDDR_ROOT / 'sample' / 'train-examples'
        else:
            image_dir = MFIDDR_ROOT / 'sample' / 'test-examples'

        if not image_dir.exists():
            print(f"ERROR: MFIDDR path not found at {image_dir}")

        # collecting the images

        for image_file_path in os.listdir(image_dir):
            filename, file_extension = os.path.splitext(image_file_path)
            file_extension = file_extension.lstrip('.').lower()
            if file_extension in valid_file_extensions:
                self.image_paths.append(str(image_dir / image_file_path)) 
                self.labels.append(filename)
                self.sources.append("MFIDDR")

    def load_IDRID(self):
        IDRID_root = Path(self.root_directories["IDRID"])
        base_path = IDRID_root / 'B-Disease-Grading' / 'Disease-Grading' / '1-Original-Images'

        
        print(f"\nIDRID: {IDRID_root}")
        print(f"IDRID exists: {IDRID_root.exists()}")

        if self.split == "train":
            image_dir = base_path / 'Training_Set'
        else:
            image_dir = base_path / 'Testing_Set'

        if not image_dir.exists():
            print(f"Warning: IDRID {self.split} directory could not be found")

        for image_file_path in os.listdir(image_dir):
            filename, file_extension = os.path.splitext(image_file_path)
            file_extension = file_extension.lstrip('.').lower()
            if file_extension in valid_file_extensions:
                self.image_paths.append(str(image_dir / image_file_path))
                self.labels.append(filename)
                self.sources.append("IDRID")

    def load_DEEPDRID(self):
        DEEPDRID_root = Path(self.root_directories["DEEPDRID"])
        
        print(f"\nDEEPDRID_root: {DEEPDRID_root}")
        print(f"DEEPDRID_root exists: {DEEPDRID_root.exists()}")
        
        if self.split == "train":
            base_dir = DEEPDRID_root / "regular_fundus_images" / "regular-fundus-training" / "Images"
        else:
            base_dir = DEEPDRID_root / "regular_fundus_images" / "Online-Challenge1&2-Evaluation"
        
        print(f"Looking for images in: {base_dir}")
        print(f"Directory exists: {base_dir.exists()}")
        
        if not base_dir.exists():
            print(f"ERROR: DEEPDRID path not found at {base_dir}")
            return
        
        loaded_count = 0
        # Iterate through numbered subdirectories (1, 2, 3, ...)
        for subdir in os.listdir(base_dir):
            subdir_path = base_dir / subdir
            
            # each image is inside a nested folder
            if not subdir_path.is_dir():
                continue
            
            # Now iterate through images in each subdirectory
            for image_file_path in os.listdir(subdir_path):
                filename, file_extension = os.path.splitext(image_file_path)
                file_extension = file_extension.lstrip('.').lower()
                
                if file_extension in valid_file_extensions:
                    full_image_path = subdir_path / image_file_path
                    self.image_paths.append(str(full_image_path))
                    self.labels.append(filename)
                    self.sources.append("DEEPDRID")
                    loaded_count += 1

    def extract_label_deepdr(self, filename: str):
        return 0

    def extract_labels_mfiddr(self, filename: str):
        return 0
    
    def extract_label_idrid(self, filename: str):
        return 0
    

    def load_labels_from_csv(self, csv_paths_dict: Dict[str, str]):
        if len(self.labels) == 0:
            self.labels = [None] * len(self.image_paths)

        for dataset_name, csv_path in csv_paths_dict.items():
            print(dataset_name)
            if not os.path.exists(csv_path):
                print(f"FileNotFoundError: CSV not found at {csv_path}")
                continue
        
            labels_df = pd.read_csv(csv_path)
            print(f"Loaded labels for {dataset_name}: {len(labels_df)} rows")
        
            # Create lookup dictionary based on dataset format (INDENTED INSIDE LOOP)
            if dataset_name == "IDRID":
                label_dict = dict(zip(labels_df["Image name"], labels_df["Retinopathy grade"]))
           
            elif dataset_name == "DEEPDRID":
                label_dict = dict(zip(labels_df["image_id"], labels_df["patient_DR_Level"])) 
            
            elif dataset_name == "MFIDDR":
                label_dict = dict(zip(labels_df["id"], labels_df["level"]))
                print(f"Sample CSV keys: {list(label_dict.keys())[:5]}")
                mfiddr_images = [Path(img_path).stem for img_path, source in zip(self.image_paths, self.sources) if source == "MFIDDR"]
                print(f"Sample image filenames: {mfiddr_images[:5]}")
            
            matched_count = 0
            for index, (img_path, source) in enumerate(zip(self.image_paths, self.sources)):
                if source == dataset_name:
                    filename = Path(img_path).stem
                    if dataset_name == "MFIDDR":
                        # splitting by underscore and taking first 3 parts for mfiddr to remove suffix
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            filename = '_'.join(parts[:3])  # e.g. "20_28096452_left" for mfiddr
                    
                    if filename in label_dict:
                        self.labels[index] = label_dict[filename]
                        matched_count += 1
                    else:
                        print(f"Warning: No label found for {filename}")
            
            print(f"Matched {matched_count} images from {dataset_name}")

            return csv_paths_dict



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
            print("Label transform -> None")
        
        image_trans = self.img_transform(image)
        label_trans = self.label_transform(label)
        return image_trans, label_trans, source
    
    def get_dataset_statistics(self) -> Dict:
        count_sources = Counter(self.sources)
        count_labels = Counter(self.labels)

        return {
            "total_images": len(self.image_paths),
            "images_per_source": dict(count_sources),
            "distribution_of_labels": dict(count_labels),
            "split": self.split
        }

    def get_labels(self):
        return self.labels
