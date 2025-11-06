import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

valid_file_extensions = ["jpg", "jpeg", "png"]

class CombinedDRDataSet(Dataset):

    def __init__(self, 
            root_directories: Dict[str, str],
            split: str="train", 
            img_transform: Optional[transforms.Compose] = None,
            label_transform: Optional[transforms.Compose] = None):
        
        self.root_direcories = root_directories
        self.split = split 
        self.img_transform = img_transform
        self.image_paths = []
        self.labels = []
        # to track which dataset each image comes from
        self.sources = []

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


