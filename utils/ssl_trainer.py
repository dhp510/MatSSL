import torch
from utils.models.matSSL import MatSSL
from utils.models.denseCL import DenseCL, DenseCLLoss
from utils.models.mocoV2 import MoCoV2, MoCoV2Loss
from dotenv import load_dotenv
import os
from lightly.loss import NTXentLoss
import shutil
from loguru import logger

load_dotenv(override=True)

input_size = int(os.getenv("SSL_INPUT_SIZE", "224"))
lr = float(os.getenv("SSL_LR", "0.1"))
momentum = float(os.getenv("SSL_MOMENTUM", "0.9"))
num_epochs = int(os.getenv("SSL_EPOCHS", "100"))
weight_decay = float(os.getenv("SSL_WEIGHT_DECAY", "1e-6"))
exp_path = os.getenv("SSL_EXP_PATH", "./experiments/ssl")

class SSLTrainer:
    def __init__(self, model, saved_model, model_type="matssl", gpu_id=0):
        self.saved_model = saved_model
        if os.path.exists(os.path.join(exp_path, saved_model)):
            shutil.rmtree(os.path.join(exp_path, saved_model))
        os.makedirs(os.path.join(exp_path, saved_model), exist_ok=True)
        self.model = model
        self.model_type = model_type.lower()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Choose loss function based on model type
        if self.model_type == "densecl":
            self.criterion = DenseCLLoss()
        elif self.model_type == "mocov2":
            self.criterion = MoCoV2Loss()
        else:
            self.criterion = NTXentLoss(temperature=0.07)
            
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self, train_loader):
        self.model.train()
        self.model.to(self.device)
                
        # Training loop
        print(f"Starting {self.model_type.upper()} training...")
        for epoch in range(num_epochs):
            total_loss = 0
            total_global_loss = 0
            total_dense_loss = 0
            
            for batch_idx, images in enumerate(train_loader):
                view_1, view_2 = images[0].to(self.device), images[1].to(self.device)
                
                # Debug shapes
                if batch_idx == 0 and epoch == 0:
                    print(f"View 1 shape: {view_1.shape}")
                
                # Forward pass
                if self.model_type == "densecl":
                    # DenseCL forward pass for both views
                    query_features, query_global, query_local = self.model(view_1)
                    with torch.no_grad():
                        key_features, key_global, key_local = self.model.forward_momentum(view_2)
                    
                    # Debug output shape for DenseCL
                    if batch_idx == 0 and epoch == 0:
                        print(f"Global output shape: {query_global.shape}")
                        print(f"Dense output shape: {query_local.shape}")
                    
                    # Compute DenseCL loss
                    loss_dict = self.criterion(query_global, key_global, query_local, key_local)
                    loss = loss_dict['total_loss']
                    
                    total_global_loss += loss_dict['global_loss'].item()
                    total_dense_loss += loss_dict['local_loss'].item()
                elif self.model_type == "mocov2":
                    # MoCoV2 forward pass
                    query_projections = self.model(view_1)
                    with torch.no_grad():
                        key_projections = self.model.forward_momentum(view_2)
                    
                    # Debug output shape for MoCoV2
                    if batch_idx == 0 and epoch == 0:
                        print(f"MoCoV2 output shape: {query_projections.shape}")
                    
                    # Compute MoCoV2 loss
                    loss = self.criterion(query_projections, key_projections)
                else:
                    # MatSSL forward pass
                    z1 = self.model(view_1)
                    z2 = self.model(view_2)
                    
                    # Debug output shape for MatSSL
                    if batch_idx == 0 and epoch == 0:
                        print(f"Projection output shape: {z1.shape}")
                    
                    # Compute MatSSL loss
                    loss = self.criterion(z1, z2)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update momentum for DenseCL and MoCoV2
                if self.model_type == "densecl" or self.model_type == "mocov2":
                    self.model.update_momentum()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    if self.model_type == "densecl":
                        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], "
                              f"Total Loss: {loss.item():.4f}, "
                              f"Global Loss: {loss_dict['global_loss'].item():.4f}, "
                              f"Dense Loss: {loss_dict['local_loss'].item():.4f}")
                    else:
                        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
            
            self.scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            # Save model checkpoint
            if epoch % 1 == 0:
                checkpoint_name = f"{self.model_type}_loss_{avg_loss:.4f}_epoch{epoch}.pth"
                torch.save(self.model.state_dict(), os.path.join(exp_path, self.saved_model, checkpoint_name))
            
            # Print epoch summary
            if self.model_type == "densecl":
                avg_global_loss = total_global_loss / len(train_loader)
                avg_dense_loss = total_dense_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Avg Total Loss: {avg_loss:.4f}, "
                      f"Avg Global Loss: {avg_global_loss:.4f}, "
                      f"Avg Dense Loss: {avg_dense_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")