#!/usr/bin/env python3
"""
Fine-tuning Script for Segmentation Models with SSL Pre-trained Encoders
"""

import argparse
import os
import torch
import segmentation_models_pytorch as smp
import re
from pathlib import Path
from dotenv import load_dotenv

# Import custom modules
from utils.finetuning_trainer import FinetuningTrainer
from utils.data_processing.finetune_dataset import create_finetune_dataloaders, get_dataset_config

# Set deterministic behavior
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

# Load environment variables
load_dotenv(override=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune segmentation models with SSL pre-trained encoders')
    
    parser.add_argument(
        '--finetune_dataset', 
        type=str, 
        required=True,
        help='Path to fine-tuning dataset directory'
    )
    
    parser.add_argument(
        '--ssl_weights_path', 
        type=str, 
        default=None,
        help='Path to SSL weights'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--gpu_id', 
        type=int, 
        default=0,
        help='GPU ID to use for training'
    )
    
    parser.add_argument(
        '--input_size', 
        type=int, 
        default=256,
        help='Input image size'
    )
    
    parser.add_argument(
        '--note', 
        type=str, 
        default='',
        help='Additional note for experiment naming'
    )
    
    parser.add_argument(
        '--custom_lr', 
        type=float, 
        default=None,
        help='Custom learning rate (overrides dataset default)'
    )
    
    parser.add_argument(
        '--custom_epochs', 
        type=int, 
        default=None,
        help='Custom number of epochs (overrides dataset default)'
    )
    
    return parser.parse_args()

def extract_ssl_info_from_path(ssl_weights_path):
    """
    Extract SSL model type, dataset, and epoch from SSL weights path
    
    Args:
        ssl_weights_path (str): Path to SSL weights file
        
    Returns:
        tuple: (ssl_model, ssl_dataset, ssl_epoch) or (None, None, None) if extraction fails
    """
    if not ssl_weights_path or not os.path.exists(ssl_weights_path):
        return None, None, None
    
    try:
        path_obj = Path(ssl_weights_path)
        
        # Extract from path structure: experiments/ssl/{model}/{dataset}/{filename}
        path_parts = path_obj.parts
        
        # Find 'ssl' in path and get model and dataset
        ssl_idx = None
        for i, part in enumerate(path_parts):
            if part == 'ssl':
                ssl_idx = i
                break
        
        if ssl_idx is None or ssl_idx + 2 >= len(path_parts):
            print(f"Warning: Could not parse SSL path structure: {ssl_weights_path}")
            return None, None, None
        
        ssl_model = path_parts[ssl_idx + 1]  # Model type (densecl, matssl, mocov2)
        ssl_dataset = path_parts[ssl_idx + 2]  # Dataset name (e.g., aachen-uhcs)
        
        # Extract epoch from filename using regex
        filename = path_obj.name
        epoch_match = re.search(r'epoch(\d+)\.pth$', filename)
        
        if epoch_match:
            ssl_epoch = int(epoch_match.group(1))
        else:
            print(f"Warning: Could not extract epoch from filename: {filename}")
            return ssl_model, ssl_dataset, None
        
        print(f"Extracted SSL info - Model: {ssl_model}, Dataset: {ssl_dataset}, Epoch: {ssl_epoch}")
        return ssl_model, ssl_dataset, ssl_epoch
        
    except Exception as e:
        print(f"Error extracting SSL info from path: {e}")
        return None, None, None

def create_model(n_classes, use_ssl=True):
    """Create UNet++ model with ResNet50 encoder"""
    if use_ssl:
        # No ImageNet pre-training when using SSL weights
        encoder_weights = None
        print("Creating UNet++ model without ImageNet weights (will load SSL weights)")
    else:
        # Use ImageNet pre-training when not using SSL
        encoder_weights = "imagenet"
        print("Creating UNet++ model with ImageNet pre-trained weights")
    
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=n_classes,
        
        
    )
    return model

def main():
    """Main fine-tuning function"""
    args = parse_arguments()
    
    # Extract dataset name from path
    dataset_name = os.path.basename(args.finetune_dataset.rstrip('/'))
    # Clean up dataset name (remove common suffixes)
    dataset_name = dataset_name.replace('-finetuning-data', '').replace('_finetuning_data', '').lower()
    
    if "aachen" in dataset_name:
        dataset_name = "aachen"
    if "ebc" in dataset_name:
        dataset_name = "ebc"
    # Extract SSL information from weights path if provided
    ssl_model, ssl_dataset, ssl_epoch = "ssl", "default", 50
    use_ssl = bool(args.ssl_weights_path)
    
    print("="*60)
    print(f"Fine-tuning Configuration:")
    print(f"Fine-tune Dataset Path: {args.finetune_dataset}")
    print(f"Dataset Name: {dataset_name.upper()}")
    if use_ssl:
        print(f"SSL Weights Path: {args.ssl_weights_path}")
        print(f"SSL Model: {ssl_model.upper()}")
        print(f"SSL Dataset: {ssl_dataset}")
        print(f"SSL Epoch: {ssl_epoch}")
        print("Using SSL pre-trained weights")
    else:
        print("Using ImageNet pre-trained weights")
    print(f"Batch Size: {args.batch_size}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Input Size: {args.input_size}")
    if args.note:
        print(f"Note: {args.note}")
    print("="*60)
    
    # Check CUDA availability
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    
    try:
        # Get dataset configuration using extracted dataset name
        dataset_config = get_dataset_config(dataset_name)
        
        # Override dataset_root with provided path
        dataset_config['dataset_root'] = args.finetune_dataset
        
        # Override with custom parameters if provided
        lr = args.custom_lr if args.custom_lr else dataset_config['lr']
        num_epochs = args.custom_epochs if args.custom_epochs else dataset_config['num_epochs']
        n_classes = dataset_config['n_classes']
        dataset_root = dataset_config['dataset_root']
        
        print(f"Dataset Config - LR: {lr}, Epochs: {num_epochs}, Classes: {n_classes}")
        
        # Create model with appropriate weights
        model = create_model(n_classes, use_ssl=use_ssl)
        print(f"Created UNet++ model with {n_classes} classes")
        
        # Create data loaders
        train_loader, test_loader = create_finetune_dataloaders(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_size=args.input_size,
        )
        
        # Create trainer with SSL usage flag
        if use_ssl:
            note = f"{ssl_model.upper()}"
            if args.note:
                note += f"_{args.note}"
        else:
            note = "ImageNet"
            if args.note:
                note += f"_{args.note}"
            
        trainer = FinetuningTrainer(
            model=model,
            dataset_name=dataset_name,
            ssl_dataset_name=ssl_dataset if use_ssl else None,
            ssl_epoch=ssl_epoch if use_ssl else None,
            note=note,
            gpu_id=args.gpu_id,
            lr=lr,
            num_epochs=num_epochs,
            n_classes=n_classes,
            use_ssl=use_ssl,
        )
        
        # Load SSL weights if using SSL
        if use_ssl:
            if not os.path.exists(args.ssl_weights_path):
                print(f"Warning: SSL weights not found: {args.ssl_weights_path}")
                print("Falling back to ImageNet pre-trained weights...")
                use_ssl = False
                trainer.use_ssl = False
                # Recreate model with ImageNet weights
                model = create_model(n_classes, use_ssl=False)
                trainer.model = model.to(trainer.device)
            else:
                trainer.load_ssl_weights(args.ssl_weights_path)
        
        # Start training
        training_type = "SSL fine-tuning" if trainer.use_ssl else "ImageNet fine-tuning"
        print(f"\nStarting {training_type}...")
        trainer.train(train_loader, test_loader)
        
        print(f"\n{training_type} completed successfully!")
        
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise e

if __name__ == "__main__":
    main()