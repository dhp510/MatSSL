#!/usr/bin/env python3
"""
Evaluation Script for Trained Segmentation Models
"""

import argparse
import os
import torch
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# Import custom modules
from utils.data_processing.finetune_dataset import create_finetune_dataloaders, get_dataset_config
from utils.evaluations.evaluate import IOU
import torch.nn.functional as F

# Set deterministic behavior
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

# Load environment variables
load_dotenv(override=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained segmentation models')
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to the trained model weights'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Name of the dataset (e.g., ebc, aachen, metaldam). If not provided, will be extracted from dataset path.'
    )
    
    parser.add_argument(
        '--n_classes',
        type=int,
        default=None,
        help='Number of classes in the dataset. If not provided, will use default from dataset config.'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128,
        help='Batch size for evaluation'
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
        help='GPU ID to use for evaluation'
    )
    
    parser.add_argument(
        '--input_size', 
        type=int, 
        default=256,
        help='Input image size (for datasets that require resizing)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save evaluation results. Defaults to results_<dataset_name> if not specified.'
    )
    
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Save visualization of predictions'
    )

    return parser.parse_args()

def create_model(n_classes):
    """Create UNet++ model with ResNet50 encoder"""
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,  # No pre-training needed for evaluation
        in_channels=3,
        classes=n_classes,
    )
    return model

def create_error_visualization(prediction, ground_truth):
    """
    Create a visualization of TP, FP, and FN areas
    
    Args:
        prediction: Predicted mask (B, H, W)
        ground_truth: Ground truth mask (B, H, W)
        
    Returns:
        numpy array: Visualization of TP, FP, FN areas
    """
    prediction = prediction.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    
    # Create error visualization (0: background, 1: TP, 2: FP, 3: FN)
    error_map = np.zeros_like(prediction)
    
    # True Positives (prediction=1 and ground_truth=1)
    error_map[np.logical_and(prediction == 1, ground_truth == 1)] = 1  # TP
    
    # False Positives (prediction=1 and ground_truth=0)
    error_map[np.logical_and(prediction == 1, ground_truth == 0)] = 2  # FP
    
    # False Negatives (prediction=0 and ground_truth=1)
    error_map[np.logical_and(prediction == 0, ground_truth == 1)] = 3  # FN
    
    return error_map

def evaluate_model(model, test_loader, dataset_name, n_classes, device, save_dir=None, save_visualizations=False):
    """Evaluate the model on test set"""
    model.eval()
    class_ious = {i: 0.0 for i in range(n_classes)}
    class_ious["mean_IOU_no_minority"] = 0.0
    class_ious["mean_IOU"] = 0.0
    
    # For pixel-wise metrics tracking
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    all_iou_results = []
    
    # Create visualization directory if needed
    if save_visualizations and save_dir:
        vis_dir = os.path.join(save_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    else:
        vis_dir = None
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            preds = torch.argmax(model(images), dim=1)
            
            iou_dict = IOU(preds, masks, n_classes, dataset_name)
            
            # Add batch results to tracking
            all_iou_results.append({
                'batch_idx': batch_idx,
                'iou': iou_dict
            })
            
            # Compute TP, FP, FN, TN (assuming binary segmentation)
            if n_classes == 2:
                for i in range(images.size(0)):
                    pred = preds[i].cpu().numpy()
                    mask = masks[i].cpu().numpy()
                    
                    tp = np.sum(np.logical_and(pred == 1, mask == 1))
                    fp = np.sum(np.logical_and(pred == 1, mask == 0))
                    fn = np.sum(np.logical_and(pred == 0, mask == 1))
                    tn = np.sum(np.logical_and(pred == 0, mask == 0))
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_tn += tn
            
            # Save visualization if requested
            if save_visualizations and vis_dir:
                for i in range(images.size(0)):
                    # Create figure with 4 subplots
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    # Original image
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    axes[0].imshow(img)
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    
                    # Ground truth mask
                    axes[1].imshow(masks[i].cpu().numpy(), cmap='gray')
                    axes[1].set_title(f'Ground Truth')
                    axes[1].axis('off')
                    
                    # Prediction mask
                    axes[2].imshow(preds[i].cpu().numpy(), cmap='gray')
                    axes[2].set_title(f'Prediction\nIoU: {iou_dict["mean_IOU_no_minority"]:.4f}')
                    axes[2].axis('off')
                    
                    # Error visualization (TP/FP/FN)
                    error_map = create_error_visualization(preds[i], masks[i])
                    
                    # Create custom colormap for error visualization
                    colors = np.array([
                        [0, 0, 0, 0],           # Background: transparent
                        [0, 1, 0, 0.7],         # TP: green (semi-transparent)
                        [1, 0, 0, 0.7],         # FP: red (semi-transparent)
                        [0, 0, 1, 0.7]          # FN: blue (semi-transparent)
                    ])
                    
                    cmap = ListedColormap(colors)
                    
                    # Show error map on top of original image
                    axes[3].imshow(img)
                    axes[3].imshow(error_map, cmap=cmap, interpolation='nearest')
                    axes[3].set_title('Error Analysis\nGreen: TP, Red: FP, Blue: FN')
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'vis_batch{batch_idx}_img{i}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
            for cls_str, iou in iou_dict.items():
                if cls_str.startswith("IOU_"):
                    cls = int(cls_str.split("_")[1])
                    class_ious[cls] += iou
                elif cls_str == "mean_IOU_no_minority":
                    class_ious["mean_IOU_no_minority"] += iou
                elif cls_str == "mean_IOU":
                    class_ious["mean_IOU"] += iou
    
    # Average IoUs across all batches
    avg_class_ious = {cls: iou / len(test_loader) for cls, iou in class_ious.items()}
    
    # Calculate pixel-wise metrics if binary segmentation
    pixel_metrics = {}
    if n_classes == 2:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
        
        pixel_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': int(total_tp),
            'false_positives': int(total_fp),
            'false_negatives': int(total_fn),
            'true_negatives': int(total_tn)
        }
    
    print(f"Evaluation Results:")
    print(f"Test IoU: {avg_class_ious['mean_IOU_no_minority']:.4f}")
    print(f"Classes IoU: {avg_class_ious}")
    
    if pixel_metrics:
        print(f"Pixel-wise metrics:")
        print(f"  Precision: {pixel_metrics['precision']:.4f}")
        print(f"  Recall: {pixel_metrics['recall']:.4f}")
        print(f"  F1-Score: {pixel_metrics['f1_score']:.4f}")
        print(f"  Accuracy: {pixel_metrics['accuracy']:.4f}")
        print(f"  TP: {pixel_metrics['true_positives']}, FP: {pixel_metrics['false_positives']}")
        print(f"  FN: {pixel_metrics['false_negatives']}, TN: {pixel_metrics['true_negatives']}")
    
    # Save detailed results to JSON if save directory provided
    if save_dir:
        result_json = {
            'avg_class_ious': avg_class_ious,
            'batch_results': all_iou_results
        }
        
        if pixel_metrics:
            result_json['pixel_metrics'] = pixel_metrics
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(result_json, f, indent=2)
    
    return avg_class_ious

def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Extract dataset name from path if not provided
    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = os.path.basename(args.dataset.rstrip('/'))
        # Clean up dataset name (remove common suffixes)
        dataset_name = dataset_name.replace('-finetuning-data', '').replace('_finetuning_data', '').lower()
        
    # Map common dataset substrings
    if "aachen" in dataset_name:
        dataset_name = "aachen"
    if "ebc" in dataset_name:
        dataset_name = "ebc"
    if "metaldam" in dataset_name:
        dataset_name = "metaldam"
        
    # Setup output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = f"results_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print(f"Evaluation Configuration:")
    print(f"Model Path: {args.model_path}")
    print(f"Dataset Path: {args.dataset}")
    print(f"Dataset Name: {dataset_name.upper()}")
    print(f"Output Directory: {output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Save Visualizations: {args.save_visualizations}")
    print("="*60)
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    
    try:
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        # Override n_classes if provided
        n_classes = args.n_classes if args.n_classes is not None else dataset_config['n_classes']
        print(f"Number of classes: {n_classes}")
        
        # Create model and load weights
        model = create_model(n_classes)
        print(f"Created UNet++ model with {n_classes} classes")
        
        # Load model weights
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        print(f"Loaded model weights from {args.model_path}")
        
        # Create data loaders
        _, test_loader = create_finetune_dataloaders(
            dataset_root=args.dataset,
            dataset_name=dataset_name,
            batch_size=128,
            num_workers=args.num_workers,
            input_size=args.input_size,
            set_split='test'
        )
        
        # Run evaluation
        print(f"\nEvaluating model...")
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            dataset_name=dataset_name,
            n_classes=n_classes,
            device=device,
            save_dir=output_dir,
            save_visualizations=args.save_visualizations,
        )
        
        # Print final results
        print("\nEvaluation complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Mean IoU: {results['mean_IOU']:.4f}")
        if args.save_visualizations:
            print(f"Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise e

if __name__ == "__main__":
    main()