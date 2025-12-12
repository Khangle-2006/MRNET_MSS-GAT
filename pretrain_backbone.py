import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn.functional as F
import time
import os
import json

# Import dataset
from dataset_pretrain import RadImageNetDataset

# --- CẤU HÌNH ---
ROOT_DIR = './radiology_ai' 
CHECKPOINT_PATH = './resnet18_modan_mulsupcon_1ch_2.pth'

BATCH_SIZE = 128 
EPOCHS = 100 
LEARNING_RATE = 0.05 
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TEMP = 0.07       
TAU_THRESHOLD = 0.3 
EMBED_DIM = 128   
# --------------------------

# --- 1. TWO CROP TRANSFORM ---
class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# --- 2. LOSS FUNCTION ---
class ModAnMulSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, threshold=0.3):
        super(ModAnMulSupConLoss, self).__init__()
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        intersection = torch.matmul(labels, labels.T) 
        labels_sum = labels.sum(dim=1, keepdim=True)
        union = labels_sum + labels_sum.T - intersection
        jaccard_sim = intersection / (union + 1e-8)
        
        mask_threshold = (jaccard_sim >= self.threshold).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_threshold * (1 - mask_self)
        
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        weighted_log_prob = jaccard_sim * log_prob * mask_pos
        num_positives = mask_pos.sum(1)
        
        loss = - (weighted_log_prob.sum(1) / (num_positives + 1e-6))
        loss = loss[num_positives > 0].mean()
        
        return loss

# --- 3. MODEL WRAPPER ---
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super(ResNetSimCLR, self).__init__()
        self.encoder = base_model
        self.feature_dim = 512 
        self.encoder.fc = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x) 
        z = self.projection(h)
        return h, F.normalize(z, dim=1)

# --- 4. MAIN ---
if __name__ == '__main__':
    # --- (SỬA ĐỔI QUAN TRỌNG) ---
    # Thêm transforms.Grayscale(1) để ép ảnh về 1 kênh bất kể đầu vào là gì
    train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])   # hoặc mean/std dataset
])

    
    dataset = RadImageNetDataset(root_dir=ROOT_DIR, transform=TwoCropTransform(train_transform))
    
    num_modalities = len(dataset.modalities)
    num_anatomies = len(dataset.anatomies)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=16, pin_memory=True, drop_last=True)

    # Khởi tạo Model 1-Channel
    base_resnet = resnet18(weights=None) 
    
    # Thay thế conv1 chuẩn paper: 3x3, stride 1
    base_resnet.conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    model = ResNetSimCLR(base_resnet, out_dim=EMBED_DIM).cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = ModAnMulSupConLoss(temperature=TEMP, threshold=TAU_THRESHOLD)
    
    print(f"Start Pre-training (Forced 1-Channel): SGD, LR={LEARNING_RATE}, Batch={BATCH_SIZE}")
    
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        start = time.time()
        
        for i, (images, mod_labels, anat_labels) in enumerate(dataloader):
            # Images sẽ tự động là (2B, 1, 224, 224) nhờ Grayscale transform
            images = torch.cat(images, dim=0).cuda()
            
            B = mod_labels.shape[0]
            mod_onehot = F.one_hot(mod_labels, num_classes=num_modalities).float().cuda()
            anat_onehot = F.one_hot(anat_labels, num_classes=num_anatomies).float().cuda()
            labels_single = torch.cat([mod_onehot, anat_onehot], dim=1) 
            labels = torch.cat([labels_single, labels_single], dim=0)   
            
            optimizer.zero_grad()
            _, features = model(images)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Step {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        epoch_time = time.time() - start
        print(f"=== Epoch {epoch+1} Done | Avg Loss: {running_loss/len(dataloader):.4f} | Time: {epoch_time:.0f}s ===")
        
        torch.save(model.encoder.state_dict(), CHECKPOINT_PATH)

    print("PRE-TRAINING FINISHED.")