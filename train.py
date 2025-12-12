import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn as nn
import time
import os
import torch.nn.functional as F

# --- 1. CẤU HÌNH ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from src.dataset import MRNetDataset 
from src.model import MultiViewSRRNet_V9

ROOT_DIR = './MRNet-v1.0' 
CHECKPOINT_PATH = './best_model_v9_meniscus_final.pth' 
BACKBONE_WEIGHTS = './resnet18_modan_mulsupcon_1ch.pth' 

BATCH_SIZE = 14            
ACCUMULATION_STEPS = 1    
EPOCHS = 350              
MIXUP_ALPHA = 0.4         

# Zoom settings cho Meniscus và ACL
ZOOM_MENISCUS = 0.75 
ZOOM_ACL = 0.55 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 2. GPU PROCESSOR (SHIFTING LOGIC) ---
class GPUProcessor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        self.geo_transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5)
        )
        self.eraser = transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0)

    def create_shifted_multi_scale(self, tensor_batch):
        B, C, D, H, W = tensor_batch.shape
        
        # 1. Global View
        global_v = F.interpolate(tensor_batch, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # 2. Shifted Mid View (Meniscus)
        crop_h, crop_w = int(H * ZOOM_MENISCUS), int(W * ZOOM_MENISCUS)
        max_dy, max_dx = H - crop_h, W - crop_w
        
        start_y = torch.randint(0, max_dy + 1, (B,), device=self.device)
        start_x = torch.randint(0, max_dx + 1, (B,), device=self.device)
        
        mid_crops = []
        for i in range(B):
            sy, sx = start_y[i], start_x[i]
            crop = tensor_batch[i:i+1, :, :, sy:sy+crop_h, sx:sx+crop_w]
            mid_crops.append(crop)
            
        mid_batch = torch.cat(mid_crops, dim=0)
        mid_v = F.interpolate(mid_batch, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # 3. Center Close View (ACL)
        crop_h_acl, crop_w_acl = int(H * ZOOM_ACL), int(W * ZOOM_ACL)
        sh, sw = (H - crop_h_acl)//2, (W - crop_w_acl)//2
        close_crop = tensor_batch[:, :, :, sh:sh+crop_h_acl, sw:sw+crop_w_acl]
        close_v = F.interpolate(close_crop, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # Stack
        multi_scale_tensor = torch.cat([global_v, mid_v, close_v], dim=1)
        return multi_scale_tensor

    def process_batch_list(self, batch_list, is_training=True):
        views_list = [item[0] for item in batch_list]
        labels_list = [item[1] for item in batch_list]
        labels_tensor = torch.stack(labels_list).to(self.device, non_blocking=True)
        
        final_views = {}
        for view_name in ['sagittal', 'coronal', 'axial']:
            batch_tensors = []
            for i in range(len(views_list)):
                raw_t = views_list[i][view_name].to(self.device, non_blocking=True)
                if raw_t.ndim == 3: raw_5d = raw_t.unsqueeze(0).unsqueeze(0)
                elif raw_t.ndim == 4: raw_5d = raw_t.unsqueeze(0)
                else: raw_5d = raw_t
                resized = F.interpolate(raw_5d, size=(32, 256, 256), mode='trilinear', align_corners=False)
                batch_tensors.append(resized.squeeze(0))
            
            batch_tensor_gpu = torch.stack(batch_tensors)
            
            ms_tensor = self.create_shifted_multi_scale(batch_tensor_gpu)
            
            B, C, D, H, W = ms_tensor.shape
            flat = ms_tensor.view(B*D, C, H, W) 
            flat = self.normalize(flat)
            
            if is_training:
                flat = self.geo_transforms(flat)
                flat = self.eraser(flat)
            
            final_views[view_name] = flat.reshape(B, C, D, H, W)
            
        return final_views, labels_tensor

# --- 3. WEIGHT LOADING (ĐÃ SỬA LỖI 3x3 vs 7x7) ---
def load_and_inflate_weights(model, weight_path, device):
    print(f"--> [Smart Weight Loading] {weight_path}")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
        
        key_mapping = {
            'conv1.': 'features_2d.0.', 'bn1.': 'features_2d.1.',
            'layer1.': 'features_2d.4.', 'layer2.': 'features_2d.5.',
            'layer3.': 'features_2d.6.', 'layer4.': 'features_2d.7.'
        }
        
        new_state_dict = {}
        # Lấy shape hiện tại của conv1 trong model mới để so sánh
        current_conv1_shape = model.backbone_sag.features_2d[0].weight.shape # [64, 3, 7, 7]
        
        for k, v in state_dict.items():
            new_key = k
            for old, new in key_mapping.items():
                if k.startswith(old):
                    new_key = k.replace(old, new)
                    break
            
            # XỬ LÝ LỖI KERNEL SIZE (3x3 vs 7x7)
            if 'features_2d.0.weight' in new_key:
                # Kiểm tra kích thước không gian (Spatial Dimensions)
                if v.shape[2:] != current_conv1_shape[2:]:
                    print(f"   !!! Skipping {new_key}: Kernel mismatch ({v.shape[2:]} vs {current_conv1_shape[2:]}). Using ImageNet.")
                    continue # Bỏ qua, dùng weight ImageNet mặc định
                
                # Nếu Kernel khớp (7x7) nhưng Channel lệch (1 vs 3) -> Inflate
                if v.shape[1] == 1 and current_conv1_shape[1] == 3:
                    print(f"   >>> Inflating {new_key} (1->3 channels)")
                    v = torch.cat([v, v, v], dim=1) / 3.0
            
            new_state_dict[new_key] = v

        # Load với strict=False
        model.backbone_sag.load_state_dict(new_state_dict, strict=False)
        model.backbone_cor.load_state_dict(new_state_dict, strict=False)
        model.backbone_axi.load_state_dict(new_state_dict, strict=False)
        print("--> Loaded successfully! (Layer 1 skipped if mismatch)")
    except Exception as e:
        print(f"!!! Error loading weights: {e}")
        print("!!! Continuing with ImageNet weights.")
    return model

# --- 4. LOSS & MAIN ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

def custom_collate(batch): return batch
device = torch.device('cuda')
gpu_processor = GPUProcessor(device).to(device)

print("--> Loading Dataset...")
train_dataset = MRNetDataset(ROOT_DIR, 'train', transform=None, cache_to_ram=True) 
valid_dataset = MRNetDataset(ROOT_DIR, 'valid', transform=None, cache_to_ram=True) 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate, persistent_workers=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate, persistent_workers=True)

model = MultiViewSRRNet_V9(pretrained_path=None, n_clusters=256, n_neighbors=8, node_features=128, gnn_out_features=128, num_heads=8, dropout=0.5).to(device)
if os.path.exists(BACKBONE_WEIGHTS): model = load_and_inflate_weights(model, BACKBONE_WEIGHTS, device)

optimizer = optim.AdamW([{'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5}, {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 5e-5}])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
criterion = FocalLoss(alpha=0.75, gamma=2.0)
scaler = torch.amp.GradScaler('cuda')

def mixup_criterion(criterion, pred, y_a, y_b, lam): return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

best_target_auc = 0.0
best_meniscus_auc = 0.0

print(f"--> START SHIFTING STRATEGY (Random Crop for Meniscus)...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, batch_list in enumerate(train_loader):
        views, labels = gpu_processor.process_batch_list(batch_list, is_training=True)
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        index = torch.randperm(views['sagittal'].size(0)).to(device)
        
        sag_mix = lam * views['sagittal'] + (1 - lam) * views['sagittal'][index]
        cor_mix = lam * views['coronal'] + (1 - lam) * views['coronal'][index]
        axi_mix = lam * views['axial']    + (1 - lam) * views['axial'][index]
        labels_a, labels_b = labels, labels[index]

        with torch.amp.autocast('cuda'):
            logits, aux_logits, view_embs = model(sag_mix, cor_mix, axi_mix)
            loss_main = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss_aux = mixup_criterion(criterion, aux_logits, labels_a, labels_b, lam)
            h_sag, h_cor, h_axi = view_embs
            loss_cons = (F.mse_loss(h_sag, h_cor) + F.mse_loss(h_cor, h_axi) + F.mse_loss(h_sag, h_axi)) / 3.0
            loss = loss_main + 0.4 * loss_aux + 0.1 * loss_cons
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    scheduler.step()
    
    # VALIDATION
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_list in valid_loader:
            views, labels = gpu_processor.process_batch_list(batch_list, is_training=False)
            with torch.amp.autocast('cuda'):
                logits, _, _ = model(views['sagittal'], views['coronal'], views['axial'])
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    try:
        auc_abn = roc_auc_score(all_labels[:, 0], all_preds[:, 0])
        auc_acl = roc_auc_score(all_labels[:, 1], all_preds[:, 1])
        auc_men = roc_auc_score(all_labels[:, 2], all_preds[:, 2])
        target_metric = (0.3 * auc_acl) + (0.7 * auc_men)
        
        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"   >>> Val AUC:  Abn={auc_abn:.4f} | ACL={auc_acl:.4f} | Men={auc_men:.4f}")
        
        if auc_men > best_meniscus_auc:
            best_meniscus_auc = auc_men
            torch.save({'model_state_dict': model.state_dict(), 'best_meniscus': auc_men}, './best_model_v9_meniscus_final.pth')
            print(f"   [SAVE] ⭐ NEW BEST MENISCUS! ({auc_men:.4f})")

        if target_metric > best_target_auc:
            best_target_auc = target_metric
            torch.save({'model_state_dict': model.state_dict(), 'best_target_auc': best_target_auc}, CHECKPOINT_PATH)
            print(f"   [SAVE] ⭐ NEW BEST AVG! ({target_metric:.4f})")
    except: pass