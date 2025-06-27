import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from utils.evaluations.evaluate import IOU
from dotenv import load_dotenv
import math
import torch.nn.functional as F

load_dotenv(override=True)

class FinetuningTrainer:
    def __init__(self, model, dataset_name, ssl_dataset_name=None, ssl_epoch=None, note="", gpu_id=0, lr=1e-4, num_epochs=50, n_classes=5, use_ssl=True):
        self.model = model
        self.dataset_name = dataset_name
        self.ssl_dataset_name = ssl_dataset_name
        self.ssl_epoch = ssl_epoch
        self.note = note
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_classes = n_classes
        self.use_ssl = use_ssl
        
        # Setup device
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup experiment path based on SSL usage
        if self.use_ssl and ssl_dataset_name and ssl_epoch is not None:
            self.exp_path = f"experiments/{dataset_name}_ssl({ssl_dataset_name}-epoch{ssl_epoch})_finetune_unetplusplus_{note}"
        else:
            self.exp_path = f"experiments/{dataset_name}_imagenet_finetune_unetplusplus_{note}"
            
        if os.path.exists(self.exp_path):
            shutil.rmtree(self.exp_path)
        os.makedirs(self.exp_path, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss and optimizer
        self.criterion = smp.losses.DiceLoss(mode="multiclass", classes=n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize tracking variables
        self.best_iou = 0.0
        self.all_ious = []
        
    def load_ssl_weights(self, ssl_weights_path):
        """Load SSL pre-trained weights into the model encoder"""
        if not self.use_ssl:
            print("Warning: SSL weights loading skipped. Using ImageNet pre-trained weights.")
            return
            
        print(f"Loading SSL weights from: {ssl_weights_path}")
        ssl_weights = torch.load(ssl_weights_path, map_location=self.device)
        
        # Clean up SSL weights to remove backbone_momentum keys
        clean_ssl_weights = {k: v for k, v in ssl_weights.items() if not k.startswith('backbone_momentum')}
        
        clean_ssl_weights = {k: v for k, v in clean_ssl_weights.items() if not k.startswith('key_')}
        
        
        # Load SSL weights into encoder
        encoder_weights = {k.replace('backbone.', ''): v for k, v in clean_ssl_weights.items() if 'backbone.' in k}
        
        try:
            self.model.encoder.load_state_dict(encoder_weights, strict=True)
            print("SSL weights loaded successfully!")
            
            # Print encoder parameter statistics
            for name, param in self.model.encoder.named_parameters():
                if 'conv1.weight' in name or 'layer4' in name:  # Sample key layers
                    print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                    break
        except Exception as e:
            print(f"Error loading SSL weights: {e}")
            print("Falling back to ImageNet pre-trained weights...")
            self.use_ssl = False
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {avg_loss:.4f}")
        return avg_loss
        
    def evaluate(self, test_loader, epoch):
        """Evaluate the model on test set"""
        self.model.eval()
        class_ious = {i: 0.0 for i in range(self.n_classes)}
        class_ious["mean_IOU"] = 0.0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                preds = torch.argmax(self.model(images), dim=1)
       
                iou_dict = IOU(preds, masks, self.n_classes, self.dataset_name)
                
                for cls_str, iou in iou_dict.items():
                    if cls_str.startswith("IOU_"):
                        cls = int(cls_str.split("_")[1])
                        class_ious[cls] += iou
                class_ious["mean_IOU"] += iou_dict["mean_IOU"]
        
        # Average IoUs across all batches
        avg_class_ious = {cls: iou / len(test_loader) for cls, iou in class_ious.items()}
        
        print(f"Epoch [{epoch+1}/{self.num_epochs}] - Test IoU: {avg_class_ious['mean_IOU']:.4f}")
        print(f"Classes IoU: {avg_class_ious}")
        
        return avg_class_ious
    
    def save_best_model(self, avg_class_ious, epoch):
        """Save model if it achieves best IoU"""
        current_iou = avg_class_ious["mean_IOU"]
        self.all_ious.append(current_iou)
        
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"New best IOU: {self.best_iou:.4f}")
            model_path = f"{self.exp_path}/unetplusplus_best_ssl_finetuned_{self.best_iou:.4f}_epoch{epoch}.pth"
            torch.save(self.model.state_dict(), model_path)
    
    def save_results(self):
        """Save training results and plots"""
        # Save IoU plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.all_ious) + 1), self.all_ious, marker='o')
        plt.title('Mean IOU (no minority) over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IOU (no minority)')
        plt.grid()
        plt.savefig(f"{self.exp_path}/mean_iou_over_epochs.png")
        plt.close()
        
        # Save IoU values
        np.save(f"{self.exp_path}/mean_iou_values.npy", np.array(self.all_ious))
        
        print(f"Results saved to: {self.exp_path}")
        print(f"Best IOU achieved: {self.best_iou:.4f}")
    
    def train(self, train_loader, test_loader):
        """Main training loop"""
        print("Starting fine-tuning from SSL pre-trained encoder...")
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluation
            avg_class_ious = self.evaluate(test_loader, epoch)
            
            # Save best model
            self.save_best_model(avg_class_ious, epoch)
        
        # Save final results
        self.save_results()
        print("Fine-tuning complete!")