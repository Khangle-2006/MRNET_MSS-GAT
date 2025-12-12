import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

# Import source
from src.dataset import MRNetDataset
from src.model import MultiViewSRRNet_V9

# --- CẤU HÌNH ---
ROOT_DIR = './MRNet-v1.0'
OUTPUT_DIR = './vis_results_Static_MultiScale' # Folder mới
CHECKPOINT_PATH = './best_model_v9_multi_scale.pth' # Model Multi-Scale mới nhất

# [QUAN TRỌNG] Khớp với file train_v9_rgb_multi_scale.py
ZOOM_MID = 0.75   # Kênh G: Cho Meniscus
ZOOM_CLOSE = 0.55 # Kênh B: Cho ACL

TARGET_TASK_NAME = 'acl' 
MAX_SAMPLES = 20 

# Chuẩn ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

TASK_MAP = {'abnormal': 0, 'acl': 1, 'meniscus': 2}
TARGET_CLASS_IDX = TASK_MAP[TARGET_TASK_NAME]

# --------------------------------------------------------
# 1. PREPARE MULTI-SCALE INPUT (R-G-B)
# --------------------------------------------------------
def prepare_multi_scale_input(raw_tensor, device):
    raw_t = raw_tensor.to(device)
    # Đảm bảo 5D (Batch, Channel, Depth, H, W)
    if raw_t.ndim == 3: raw_t = raw_t.unsqueeze(0).unsqueeze(0)
    elif raw_t.ndim == 4: raw_t = raw_t.unsqueeze(0)
    
    # Resize Depth về 32 chuẩn
    if raw_t.shape[2] != 32:
        raw_t = F.interpolate(raw_t, size=(32, 256, 256), mode='trilinear', align_corners=False)
    
    B, C, D, H, W = raw_t.shape

    # 1. CHANNEL R: GLOBAL VIEW
    global_v = F.interpolate(raw_t, size=(32, 224, 224), mode='trilinear', align_corners=False)
    
    # 2. CHANNEL G: MID VIEW (Zoom 0.75)
    crop_h_mid, crop_w_mid = int(H * ZOOM_MID), int(W * ZOOM_MID)
    sh_mid, sw_mid = (H - crop_h_mid)//2, (W - crop_w_mid)//2
    mid_crop = raw_t[:, :, :, sh_mid:sh_mid+crop_h_mid, sw_mid:sw_mid+crop_w_mid]
    mid_v = F.interpolate(mid_crop, size=(32, 224, 224), mode='trilinear', align_corners=False)
    
    # 3. CHANNEL B: CLOSE VIEW (Zoom 0.55)
    crop_h_close, crop_w_close = int(H * ZOOM_CLOSE), int(W * ZOOM_CLOSE)
    sh_close, sw_close = (H - crop_h_close)//2, (W - crop_w_close)//2
    close_crop = raw_t[:, :, :, sh_close:sh_close+crop_h_close, sw_close:sw_close+crop_w_close]
    close_v = F.interpolate(close_crop, size=(32, 224, 224), mode='trilinear', align_corners=False)
    
    # Stack (R, G, B)
    ms_tensor = torch.cat([global_v, mid_v, close_v], dim=1)
    
    # Normalize
    B_out, C_out, D_out, H_out, W_out = ms_tensor.shape
    flat = ms_tensor.view(B_out*D_out, C_out, H_out, W_out)
    norm = normalize(flat)
    return norm.view(B_out, C_out, D_out, H_out, W_out)

# --------------------------------------------------------
# 2. GRAD-CAM WRAPPER
# --------------------------------------------------------
class MultiViewGradCAM:
    def __init__(self, model, target_layer_name='adapter_3d'):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = dict(self.model.backbone_sag.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, views_raw, target_class_idx):
        self.model.zero_grad()
        
        sag = prepare_multi_scale_input(views_raw['sagittal'], device)
        cor = prepare_multi_scale_input(views_raw['coronal'], device)
        axi = prepare_multi_scale_input(views_raw['axial'], device)
        sag.requires_grad = True
        
        logits, _, _ = self.model(sag, cor, axi)
        score = logits[0, target_class_idx]
        score.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        cam_vol = cam[0, 0].detach().cpu().numpy()
        cam_vol = (cam_vol - cam_vol.min()) / (cam_vol.max() - cam_vol.min() + 1e-8)
        
        return cam_vol, torch.sigmoid(score).item()

# --------------------------------------------------------
# 3. OVERLAY VỚI 2 KHUNG HÌNH (MID & CLOSE)
# --------------------------------------------------------
def overlay_multi_scale(raw_slice, heatmap_slice):
    H, W = raw_slice.shape
    
    # Resize Heatmap về kích thước ảnh gốc
    hm_resized = cv2.resize(heatmap_slice, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Filter
    full_hm = hm_resized
    full_hm[full_hm < 0.2] = 0 
    full_hm = cv2.GaussianBlur(full_hm, (15, 15), 0)
    
    hm_uint8 = np.uint8(255 * full_hm)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    
    # Background
    img_min, img_max = raw_slice.min(), raw_slice.max()
    img_norm = (raw_slice - img_min) / (img_max - img_min + 1e-8)
    img_rgb = np.stack([np.uint8(255 * img_norm)]*3, axis=-1)
    
    # Blend
    alpha = full_hm[:, :, np.newaxis]
    alpha = np.clip(alpha * 0.7, 0, 0.6) 
    output = img_rgb.astype(np.float32) * (1 - alpha) + hm_color.astype(np.float32) * alpha
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # --- VẼ 2 KHUNG HÌNH ---
    # 1. Khung Vàng (Mid View - Meniscus)
    h_mid, w_mid = int(H * ZOOM_MID), int(W * ZOOM_MID)
    sh_mid, sw_mid = (H - h_mid)//2, (W - w_mid)//2
    cv2.rectangle(output, (sw_mid, sh_mid), (sw_mid+w_mid, sh_mid+h_mid), (0, 255, 255), 2) # Vàng
    
    # 2. Khung Xanh (Close View - ACL)
    h_close, w_close = int(H * ZOOM_CLOSE), int(W * ZOOM_CLOSE)
    sh_close, sw_close = (H - h_close)//2, (W - w_close)//2
    cv2.rectangle(output, (sw_close, sh_close), (sw_close+w_close, sh_close+h_close), (0, 255, 0), 2) # Xanh Lá
    
    return output

# --------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--> Loading Model (Multi-Scale RGB)...")
    
    model = MultiViewSRRNet_V9(
        pretrained_path=None, n_clusters=256, n_neighbors=8,
        node_features=128, gnn_out_features=128, num_heads=8, dropout=0.0
    ).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--> Loading Checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
        model.eval()
    else:
        print(f"!!! Checkpoint not found: {CHECKPOINT_PATH}")
        exit()
    
    grad_cam = MultiViewGradCAM(model)
    dataset = MRNetDataset(ROOT_DIR, 'valid', transform=None, cache_to_ram=False)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    count = 0
    
    print(f"--> Generating Multi-Scale Visualization for {TARGET_TASK_NAME.upper()}...")
    
    for i in tqdm(range(len(dataset))):
        views, labels = dataset[i]
        
        if labels[TARGET_CLASS_IDX] == 1.0:
            case_id = dataset.patient_list[i]
            
            # Heatmap
            cam_vol_32, prob = grad_cam(views, TARGET_CLASS_IDX)
            
            # Load Raw
            path_sag = os.path.join(ROOT_DIR, 'valid', 'sagittal', f'{case_id}.npy')
            vol_raw = np.load(path_sag).astype(np.float32)
            
            # Resize
            real_depth = vol_raw.shape[0]
            cam_vol_real = F.interpolate(
                torch.tensor(cam_vol_32).unsqueeze(0).unsqueeze(0), 
                size=(real_depth, 224, 224), 
                mode='trilinear', align_corners=False
            ).squeeze().numpy()
            
            # Plot
            mid = real_depth // 2
            indices = [mid-2, mid, mid+2]
            
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Case {case_id} | {TARGET_TASK_NAME} | Pred: {prob:.2%}", fontsize=14)
            
            for plot_idx, slice_idx in enumerate(indices):
                if slice_idx < real_depth:
                    img_slice = vol_raw[slice_idx]
                    hm_slice = cam_vol_real[slice_idx]
                    res = overlay_multi_scale(img_slice, hm_slice)
                    
                    plt.subplot(1, 3, plot_idx+1)
                    plt.imshow(res)
                    plt.axis('off')
                    plt.title(f"Slice {slice_idx}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"MultiScale_{case_id}.png"), bbox_inches='tight', dpi=150)
            plt.close()
            
            count += 1
            if count >= MAX_SAMPLES: break
            
    print(f"DONE. Images saved to: {OUTPUT_DIR}")