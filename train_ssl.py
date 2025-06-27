#!/usr/bin/env python3
"""
SSL Training Script for DenseCL, MatSSL, and MoCoV2 models
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset
from dotenv import load_dotenv
from pathlib import Path
# Import custom modules
from utils.ssl_trainer import SSLTrainer
from utils.models.matSSL import MatSSL
from utils.models.denseCL import DenseCL
from utils.models.mocoV2 import MoCoV2
from utils.data_processing.ssl_dataset import custom_collate_fn

# Load environment variables
load_dotenv(override=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SSL models (DenseCL, MatSSL, or MoCoV2)')
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['densecl', 'matssl', 'mocov2'], 
        required=True,
        help='Model type to train: densecl, matssl, or mocov2'
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='./processed_dataset/for_ssl/combination',
        help='Path to SSL training data directory'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
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
        help='GPU ID to use for training (default: 0)'
    )
    
    parser.add_argument(
        '--dense_output_dim', 
        type=int, 
        default=128,
        help='Output dimension for dense features (DenseCL only)'
    )
    
    parser.add_argument(
        '--global_output_dim', 
        type=int, 
        default=128,
        help='Output dimension for global features'
    )
    
    parser.add_argument(
        '--dense_weight', 
        type=float, 
        default=0.5,
        help='Weight for dense loss in DenseCL (0.0-1.0)'
    )
    
    return parser.parse_args()

def create_model(model_type, dense_output_dim=128, global_output_dim=128):
    """Create and return the specified model"""
    if model_type == 'densecl':
        print(f"Creating DenseCL model with dense_dim={dense_output_dim}, global_dim={global_output_dim}")
        return DenseCL()
    elif model_type == 'matssl':
        print("Creating MatSSL model")
        return MatSSL()
    elif model_type == 'mocov2':
        print("Creating MoCoV2 model")
        return MoCoV2()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_dataloader(data_path, batch_size, num_workers):
    """Create and return the data loader"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    print(f"Loading data from: {data_path}")
    
    # Create LightlyDataset
    dataset = LightlyDataset(input_dir=data_path)
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True,
        pin_memory=True
    )
    
    return dataloader

def main():
    """Main training function"""
    args = parse_arguments()
    
    print("="*50)
    print(f"SSL Training Configuration:")
    print(f"Model: {args.model.upper()}")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"GPU ID: {args.gpu_id}")
    if args.model == 'densecl':
        print(f"Dense output dim: {args.dense_output_dim}")
        print(f"Global output dim: {args.global_output_dim}")
        print(f"Dense weight: {args.dense_weight}")
    print("="*50)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    try:
        # Create model
        model = create_model(
            args.model, 
            dense_output_dim=args.dense_output_dim,
            global_output_dim=args.global_output_dim
        )
        
        # Create data loader
        train_loader = create_dataloader(
            args.data_path,
            args.batch_size,
            args.num_workers
        )
        
        # Create trainer
        trainer = SSLTrainer(model, model_type=args.model, saved_model=f"{args.model}/{os.path.basename(args.data_path)}", gpu_id=args.gpu_id)
        
        # Update dense weight for DenseCL if specified
        if args.model == 'densecl' and hasattr(trainer.criterion, 'dense_weight'):
            trainer.criterion.dense_weight = args.dense_weight
            print(f"Updated dense weight to: {args.dense_weight}")
        
        # Start training
        print("\nStarting training...")
        trainer.train(train_loader)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()