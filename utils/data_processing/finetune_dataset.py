import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from dotenv import load_dotenv

load_dotenv(override=True)

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", input_size=256, dataset_name="metaldam"):
        """
        Segmentation dataset for fine-tuning
        
        Args:
            root_dir (str): Path to dataset root directory
            split (str): Dataset split ('train' or 'test')
            input_size (int): Input image size for resizing
            dataset_name (str): Name of the dataset for specific preprocessing
        """
        self.root_dir = os.path.join(root_dir, split)
        self.input_dir = os.path.join(self.root_dir, "inputs")
        self.target_dir = os.path.join(self.root_dir, "targets")
        self.dataset_name = dataset_name.lower()
        self.split = split
        
        # Get image and mask file lists
        self.images = sorted(os.listdir(self.input_dir))
        self.masks = sorted(os.listdir(self.target_dir))
        
        # Validate dataset integrity
        assert len(self.images) == len(self.masks), f"Mismatch in number of images ({len(self.images)}) and masks ({len(self.masks)})"
        
        # Setup transforms based on dataset and split
        if self.dataset_name == "ebc" and self.split == "train":
            # Use random crop for EBC training data
            self.transform = A.Compose([
                A.Resize(512, 512, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
                A.ToTensorV2(),
            ], is_check_shapes=False)
        elif self.dataset_name == "ebc" and (self.split == "val" or self.split == "test"):
            # For EBC test data: keep original size, only normalize
            self.transform = A.Compose([
                A.Resize(512, 512, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
                A.ToTensorV2(),
            ], is_check_shapes=False)
        else:
            # Use resize for all other cases (other datasets or test split)
            self.transform = A.Compose([
                
                A.Resize(input_size, input_size, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
                A.ToTensorV2(),
            ], is_check_shapes=False)
        
        print(f"Loaded {len(self.images)} {split} samples from {dataset_name} dataset")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.input_dir, self.images[idx])
        mask_path = os.path.join(self.target_dir, self.masks[idx])
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask, dtype=np.int64)
        
        # Apply dataset-specific preprocessing
        if self.dataset_name == "aachen" or self.dataset_name == "default":
            mask[mask > 0] = 1  # Convert Aachen mask to binary (0, 1)
        # Apply dataset-specific preprocessing
        if self.dataset_name == "ebc":
            mask[mask > 0] = 1  # Convert ebc mask to binary (0, 1) 
            
        # Apply transforms
        augmented = self.transform(image=np.array(image), mask=mask)
        
        return augmented['image'], augmented['mask'].long()


def create_finetune_dataloaders(dataset_root, dataset_name="metaldam", batch_size=128, num_workers=4, input_size=256, set_split='val'):
    """
    Create train and test dataloaders for fine-tuning
    
    Args:
        dataset_root (str): Path to dataset root directory
        dataset_name (str): Name of the dataset
        batch_size (int): Batch size for data loading
        num_workers (int): Number of workers for data loading
        input_size (int): Input image size
        disable_ebc_processing (bool): If True, disable special processing for EBC dataset
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    
    # Create datasets
    train_dataset = SegmentationDataset(
        root_dir=dataset_root,
        split="train",
        input_size=input_size,
        dataset_name=dataset_name 
    )
    
    test_dataset = SegmentationDataset(
        root_dir=dataset_root,
        split="val" if (dataset_name=="ebc" and set_split!="test") else "test",
        input_size=input_size,
        dataset_name=dataset_name 
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders - Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    
    return train_loader, test_loader


def get_dataset_config(dataset_name):
    """
    Get dataset-specific configuration
    
    Args:
        dataset_name (str): Name of the dataset
    
    Returns:
        dict: Dataset configuration
    """
    configs = {
        "metaldam": {
            "num_epochs": 200,
            "lr": 1e-4,
            "n_classes": 5,
        },
        "aachen": {
            "num_epochs": 50,
            "lr": 1e-3,
            "n_classes": 2,  
        },
        "ebc": {
            "num_epochs": 200,
            "lr": 1e-4,
            "n_classes": 2,
        },
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in configs:
        # Default configuration for unknown datasets
        print(f"Warning: Unknown dataset '{dataset_name}'. Using default configuration.")
        return {
            "num_epochs": 50,
            "lr": 1e-4,
            "n_classes": 5,
        }
    
    return configs[dataset_name]
