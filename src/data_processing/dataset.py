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
from utilities.utils import normalize_stem, _is_image_valid
from concurrent.futures import ThreadPoolExecutor, as_completed

valid_file_extensions = ["jpg", "jpeg", "png"]

class CombinedDRDataSet(Dataset):

    def __init__(self, 
            root_directories: Dict[str, str],
            split: str, 
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

        if "IDRID" in self.root_directories: # pre usage
            self.load_IDRID()
        
        if "DEEPDRID" in self.root_directories:
            self.load_DEEPDRID()

        if "MESSIDOR" in self.root_directories: # pre usage
            self.load_MESSIDOR()

        if "MFIDDR" in self.root_directories:
            self.load_MFIDDR()
        
        if "APTOS" in self.root_directories:
            self.load_APTOS()

        if "DDR" in self.root_directories:
            self.load_DDR()

        if "EYEPACS" in self.root_directories:
            self.load_EYEPACS()



    def __len__(self) -> int:
        return len(self.image_paths)

    
    def prune_unlabeled(self):
        filtered = [
            (p, l, s)
            for p, l, s in zip(self.image_paths, self.labels, self.sources)
            if l is not None
        ]

        self.image_paths, self.labels, self.sources = map(list, zip(*filtered))
        print(f"Pruned dataset to remove unlabeled samples -- {len(self.labels)} labeled samples.")


    def prune_corrupted_images(self, num_workers: int = 16):
        print("Pruning corrupted images (threaded)...")

        valid_indices = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_is_image_valid, path): idx
                for idx, path in enumerate(self.image_paths)
            }

            for future in as_completed(futures):
                idx = futures[future]
                if future.result():
                    valid_indices.append(idx)
                else:
                    print(f"Removing corrupted image: {self.image_paths[idx]}")

        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels      = [self.labels[i] for i in valid_indices]
        self.sources     = [self.sources[i] for i in valid_indices]

        print(f"Dataset size after corruption prune: {len(self.labels)}")


    def cache_images_to_memory(self, max_workers: int = 8):
        """Cache all images to RAM for faster training"""
        print(f"Caching {len(self.image_paths)} images to memory...")
        self.cached_images = [None] * len(self.image_paths)
    
        def load_single(idx):
            try:
                img = Image.open(self.image_paths[idx]).convert('RGB')
                return idx, img
            except Exception as e:
                print(f"Failed to load {self.image_paths[idx]}: {e}")
                return idx, None
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single, i) for i in range(len(self.image_paths))]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Caching"):
                idx, img = future.result()
                if img is not None:
                    self.cached_images[idx] = img
    
        valid_indices = [i for i, img in enumerate(self.cached_images) if img is not None]
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.sources = [self.sources[i] for i in valid_indices]
        self.cached_images = [self.cached_images[i] for i in valid_indices]
    
        print(f"Cached {len(self.cached_images)} images successfully")

    def load_MFIDDR(self):
        MFIDDR_ROOT = Path(self.root_directories["MFIDDR"])

        print(f"\nMFIDDR_ROOT: {MFIDDR_ROOT}")
        print(f"MFIDDR_ROOT exists: {MFIDDR_ROOT.exists()}")

        # no split for val since not enough samples in this dataset
        if self.split == "train":
            image_dir = MFIDDR_ROOT / "sample" / "train-examples"
        elif self.split == "test":    
            image_dir = MFIDDR_ROOT / "sample" / "test-examples"
        else: 
            raise ValueError(f"Invalid argument for split identified in {MFIDDR_ROOT}") 


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
            image_dir = base_path / "Training_Set"
        
        elif self.split == "val":
            image_dir = base_path / "Validation_Set"

        elif self.split == "test":
            image_dir = base_path / "Testing_Set"

        else:
            raise ValueError(f"Invalid argument for split identified in {IDRID_root}") 

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
            image_dir = DEEPDRID_root / "regular_fundus_images" / "regular-fundus-training" / "Images"
        elif self.split == "val":
            image_dir = DEEPDRID_root / "regular_fundus_images" / "regular-fundus-validation" / "Images"
        elif self.split == "test":
            image_dir = DEEPDRID_root / "regular_fundus_images" / "Online-Challenge1&2-Evaluation" /  "Images"
        else:
            raise ValueError(f"Invalid argument for split identified in {DEEPDRID_root}")

        print(f"Looking for images in: {image_dir}")
        print(f"Directory exists: {image_dir.exists()}")
        
        if not image_dir.exists():
            print(f"ERROR: DEEPDRID path not found at {image_dir}")
            return
        
        loaded_count = 0
        # Iterate through numbered subdirectories (1, 2, 3, ...)
        for subdir in os.listdir(image_dir):
            subdir_path = image_dir / subdir
            
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
                    # self.labels.append(filename)
                    self.sources.append("DEEPDRID")
                    loaded_count += 1

    def load_MESSIDOR(self):
        MESSIDOR_ROOT = Path(self.root_directories["MESSIDOR"])

        print(f"\nMESSIDOR_ROOT: {MESSIDOR_ROOT}")
        print(f"MESSIDOR_ROOT exists: {MESSIDOR_ROOT.exists()}")


        image_dir = None
        if self.split == "train":
            image_dir = MESSIDOR_ROOT / "messidor-2" / "messidor-2" / "preprocess"
        elif self.split == "val":
            image_dir = MESSIDOR_ROOT / "messidor-2" / "messidor-2" / "validation"
        elif self.split == "test":
            image_dir = MESSIDOR_ROOT / "messidor-2" / "messidor-2" / "test"
        else:
            raise ValueError(f"Invalid argument for split identified in {MESSIDOR_ROOT}")

        
        if not image_dir.exists():
            print(f"Error: MESSIDOR path not found at {image_dir}")

        # collecting the images

        for image_file_path in os.listdir(image_dir):
            filename, file_extension = os.path.splitext(image_file_path)
            file_extension = file_extension.lstrip('.').lower()
            if file_extension in valid_file_extensions:
                self.image_paths.append(str(image_dir / image_file_path)) 
                self.labels.append(filename)
                self.sources.append("MESSIDOR")
    
    def load_APTOS(self):

        APTOS_ROOT = Path(self.root_directories["APTOS"])

        print(f"\nAPTOS_ROOT: {APTOS_ROOT}")
        print(f"APTOS_ROOT exists: {APTOS_ROOT.exists()}")

        image_dir = None
        if self.split == "train":
            image_dir = APTOS_ROOT / "train_images"
        elif self.split == "val":
            image_dir = APTOS_ROOT / "val_images"
        elif self.split == "test":
            image_dir = APTOS_ROOT / "test_images"
        else:
            raise ValueError(f"Invalid argument for split identified in {APTOS_ROOT}")


        if not image_dir.exists():
            print(f"Error: APTOS path not found at {image_dir}")

        # collecting the images

        for image_file_path in os.listdir(image_dir):
            filename, file_extension = os.path.splitext(image_file_path)
            file_extension = file_extension.lstrip('.').lower()
            if file_extension in valid_file_extensions:
                self.image_paths.append(str(image_dir / image_file_path))
                self.labels.append(filename)
                self.sources.append("APTOS") 


    def load_DDR(self):
        DDR_ROOT = Path(self.root_directories["DDR"])

        print(f"\nDDR_ROOT: {DDR_ROOT}")
        print(f"DDR_ROOT exists: {DDR_ROOT.exists()}")


        image_dir = None
        if self.split == "train":
            image_dir = DDR_ROOT / "train"
        elif self.split == "val":
            image_dir = DDR_ROOT / "val"
        elif self.split == "test":
            image_dir = MESSIDOR_ROOT / "test"
        else:
            raise ValueError(f"Invalid argument for split identified in {DDR_ROOT}")


        if not image_dir.exists():
            print(f"Error: DDR path not found at {image_dir}")

        # collecting the images

        for image_file_path in os.listdir(image_dir):
            filename, file_extension = os.path.splitext(image_file_path)
            file_extension = file_extension.lstrip('.').lower()
            if file_extension in valid_file_extensions:
                self.image_paths.append(str(image_dir / image_file_path))
                self.labels.append(filename)
                self.sources.append("DDR")


    def load_EYEPACS(self):
        EYEPACS_ROOT = Path(self.root_directories["EYEPACS"])

        print(f"\nEYEPACS_ROOT: {EYEPACS_ROOT}")
        print(f"EYEPACS_ROOT exists: {EYEPACS_ROOT.exists()}")


        image_dir = None
        if self.split == "train":
            image_dir = EYEPACS_ROOT / "train"
        elif self.split == "val":
            image_dir = EYEPACS_ROOT / "val"
        elif self.split == "test":
            image_dir = MESSIDOR_ROOT / "test"
        else:
            raise ValueError(f"Invalid argument for split identified in {EYEPACS_ROOT}")


        if not image_dir.exists():
            print(f"Error: MESSIDOR path not found at {image_dir}")

        # collecting the images

        for image_file_path in os.listdir(image_dir):
            filename, file_extension = os.path.splitext(image_file_path)
            file_extension = file_extension.lstrip('.').lower()
            if file_extension in valid_file_extensions:
                self.image_paths.append(str(image_dir / image_file_path))
                self.labels.append(filename)
                self.sources.append("EYEPACS")



    def load_labels_from_csv(self, csv_paths_dict: Dict[str, str]):
            self.labels = [None] * len(self.image_paths)

            for dataset_name, csv_path in csv_paths_dict.items():
                print(f"\n--- Processing {dataset_name} ---")
                if not os.path.exists(csv_path):
                    print(f"FileNotFoundError: CSV not found at {csv_path}")
                    continue
            
                labels_df = pd.read_csv(csv_path)
                print(f"Loaded CSV: {len(labels_df)} rows")
            
                label_dict = {}
                
                if dataset_name == "IDRID":
                    label_dict = {str(k).strip(): int(v) for k, v in zip(labels_df["Image name"], labels_df["Retinopathy grade"])}
                
                elif dataset_name == "DEEPDRID":
                    label_dict = {
                            normalize_stem(k): int(v)
                            for k, v in zip(labels_df["image_id"], labels_df["patient_DR_Level"])
                            if pd.notna(v)
                    }

                    print("DEEPDRID CSV columns:", labels_df.columns.tolist())
                    print("DEEPDRID CSV sample image_id:", labels_df["image_id"].head(10).tolist())

                    print("DEEPDRID dict size:", len(label_dict))
                    print("DEEPDRID dict sample keys:", list(label_dict.keys())[:10])

                    deepdrid_samples = [p for p, s in zip(self.image_paths, self.sources) if s == "DEEPDRID"][:10]
                    print("DEEPDRID sample file stems:", [normalize_stem(Path(p).name) for p in deepdrid_samples])



                elif dataset_name == "MESSIDOR":
                    label_dict = {normalize_stem(k).strip().lower(): int(v) for k, v in zip(labels_df["id_code"], labels_df["diagnosis"])}

                elif dataset_name == "MFIDDR":
                    print("Building MFIDDR dictionary from columns id1-id4...")
                    for _, row in labels_df.iterrows():
                        try:
                            grade = int(row['level'])
                            # Check columns id1, id2, id3, id4
                            for col in ['id1', 'id2', 'id3', 'id4']:
                                if col in row and pd.notna(row[col]):
                                    image_name = str(row[col]).strip()
                                    label_dict[image_name] = grade
                        except ValueError:
                            continue

                elif dataset_name == "APTOS":
                    label_dict = {str(k).strip().lower(): int(v) for k, v in zip(labels_df["id_code"], labels_df["diagnosis"])}

                elif dataset_name == "DDR":
                    label_dict = {normalize_stem(k): int(v) for k, v in zip(labels_df["id_code"], labels_df["diagnosis"])}
                
                elif dataset_name == "EYEPACS":                    
                    label_dict = {normalize_stem(k).strip().lower(): int(v) for k, v in zip(labels_df["image"], labels_df["level"])}

                print(f"DEBUG: First 3 keys in {dataset_name} dict: {list(label_dict.keys())[:3]}")

                matched_count = 0
                
                for index, (img_path, source) in enumerate(zip(self.image_paths, self.sources)):
                    
                    if source != dataset_name:
                        continue
                    
                    # Default: Use stem and original case (works for IDRID/DEEPDRID)
                    filename_for_lookup = normalize_stem(Path(img_path).name) 




                    # --- Lookup ---
                    if filename_for_lookup in label_dict:
                        self.labels[index] = int(label_dict[filename_for_lookup])
                        matched_count += 1
                    else:
                        if matched_count == 0: 
                            print(f"Mismatch Debug: File '{filename_for_lookup}' not in {dataset_name} CSV dict.")

                print(f"Successfully matched {matched_count} images from {dataset_name}.")
                
                if matched_count == 0:
                    print(f"Error: No images matched for {dataset_name}. Check filename parsing.")

            return csv_paths_dict


    def load_labels_from_csv_for_test(self, csv_paths_dict: Dict[str, str]):
            if len(self.labels) == 0:
                self.labels = [None] * len(self.image_paths)

            for dataset_name, csv_path in csv_paths_dict.items():
                print(f"\n--- Processing {dataset_name} ---")
                if not os.path.exists(csv_path):
                    print(f"FileNotFoundError: CSV not found at {csv_path}")
                    continue

                labels_df = pd.read_csv(csv_path)
                print(f"Loaded CSV: {len(labels_df)} rows")

                label_dict = {}

                if dataset_name == "IDRID":
                    label_dict = {str(k).strip(): int(v) for k, v in zip(labels_df["Image name"], labels_df["Retinopathy grade"])}

                elif dataset_name == "DEEPDRID":
                    label_dict = {normalize_stem(k): int(v) for k, v in zip(labels_df["image_id"], labels_df["patient_DR_Level"]) if pd.notna(v)}

                elif dataset_name == "MESSIDOR":
                    label_dict = {str(k).strip().lower(): int(v) for k, v in zip(labels_df["id_code"], labels_df["diagnosis"])}

                elif dataset_name == "MFIDDR":
                    print("Building MFIDDR dictionary from columns id1-id4...")
                    for _, row in labels_df.iterrows():
                        try:
                            grade = int(row['level'])
                            # Check columns id1, id2, id3, id4
                            for col in ['id1', 'id2', 'id3', 'id4']:
                                if col in row and pd.notna(row[col]):
                                    image_name = str(row[col]).strip()
                                    label_dict[image_name] = grade
                        except ValueError:
                            continue

                print(f"DEBUG: First 3 keys in {dataset_name} dict: {list(label_dict.keys())[:3]}")

                matched_count = 0

                for index, (img_path, source) in enumerate(zip(self.image_paths, self.sources)):

                    if source != dataset_name:
                        continue

                    # Default: Use stem and original case (works for IDRID/DEEPDRID)
                    filename_for_lookup = Path(img_path).stem

                    # CONDITION: Apply specific logic only for MESSIDOR
                    if source == "MESSIDOR":
                        # For MESSIDOR, we need the full name and lowercase matching the dictionary keys
                        filename_for_lookup = Path(img_path).name.strip().lower()

                    # CONDITION: Apply specific logic for MFIDDR if its CSV contains stems
                    elif source == "MFIDDR":
                        # MFIDDR uses stem and we must convert it to lowercase to match the dict
                        filename_for_lookup = Path(img_path).stem.strip().lower()

                    # --- Lookup ---
                    if filename_for_lookup in label_dict:
                        self.labels[index] = int(label_dict[filename_for_lookup])
                        matched_count += 1
                    else:
                        if matched_count == 0:
                            print(f"Mismatch Debug: File '{filename_for_lookup}' not in {dataset_name} CSV dict.")

                print(f"Successfully matched {matched_count} images from {dataset_name}.")

                if matched_count == 0:
                    print(f"Error: No images matched for {dataset_name}. Check filename parsing.")

            return csv_paths_dict


    def __getitem__(self, idx):
        if hasattr(self, 'cached_images') and self.cached_images:
            image = self.cached_images[idx].copy()
        else:
            image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)

        if self.label_transform:
            label = self.label_transform(self.labels[idx])

        else:
            label = self.labels[idx]

        return image, label
   


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
