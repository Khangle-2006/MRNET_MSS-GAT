import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import DenseGATConv
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# --- 1. BACKBONE CHUẨN (TRẢ VỀ RESNET GỐC) ---
class HybridBackbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Load ResNet18 chuẩn (ImageNet Weights)
        # KHÔNG dùng dilated, trả về 7x7 để tận dụng tối đa pre-train weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.features_2d = nn.Sequential(*list(resnet.children())[:-2])
        
        # --- [FIX QUAN TRỌNG CHO MENISCUS] ---
        # Thay AvgPool3d bằng MaxPool3d
        # Lý do: Vết rách sụn chêm là tín hiệu nhỏ (high intensity). 
        # AvgPool sẽ làm mờ nó. MaxPool sẽ bắt được nó dù nó chỉ xuất hiện ở 1-2 lát cắt.
        self.adapter_3d = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(512), 
            nn.ReLU(inplace=True),
            # Dùng MaxPool để giữ lại tín hiệu bệnh lý mạnh nhất
            nn.MaxPool3d(kernel_size=(32, 1, 1), stride=(32, 1, 1)) 
        )
        
    def forward(self, x):
        B, C, D, H, W = x.shape 
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        features = self.features_2d(x) 
        _, C_feat, H_feat, W_feat = features.shape
        features = features.view(B, D, C_feat, H_feat, W_feat).permute(0, 2, 1, 3, 4)
        out = self.adapter_3d(features) 
        return out

# --- 2. GRAPH GENERATOR (Giữ nguyên bản ổn định 7x7) ---
class SuperRegionGraphGenerator(nn.Module):
    def __init__(self, input_channels, n_clusters=128, n_neighbors=8, num_features=128):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.node_features_encoder = nn.Linear(input_channels + 3, num_features)
        self.saved_labels = [] 

    def run_kmeans_gpu(self, x, n_clusters, n_iter=10):
        N, D = x.shape
        device = x.device
        random_idx = torch.randperm(N, device=device)[:n_clusters]
        centroids = x[random_idx]
        labels = None
        for i in range(n_iter):
            dists = torch.cdist(x, centroids)
            labels = torch.argmin(dists, dim=1)
            mask = F.one_hot(labels, n_clusters).float()
            cluster_sums = torch.matmul(mask.t(), x)
            cluster_counts = mask.sum(dim=0).unsqueeze(1).clamp(min=1.0)
            centroids = cluster_sums / cluster_counts
        return labels, centroids

    def forward(self, feature_map):
        feature_map = feature_map.squeeze(2) 
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        z_coords = torch.tensor([0.5], device=device)
        
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_z = z_coords.expand(H, W)
        coords = torch.stack([grid_z, grid_y, grid_x], dim=0) 
        
        fm_flat = feature_map.flatten(2) 
        coords_flat = coords.flatten(1).unsqueeze(0).expand(B, -1, -1) * 5.0
        combined_flat = torch.cat([fm_flat, coords_flat], dim=1).transpose(1, 2) 
        
        node_features_batch = []
        adj_batch = []
        self.saved_labels = [] 
        
        for i in range(B):
            item_data = combined_flat[i]
            labels, centroids = self.run_kmeans_gpu(item_data, self.n_clusters, n_iter=10)
            self.saved_labels.append(labels.detach().cpu().numpy()) 
            feats = self.node_features_encoder(centroids)
            node_features_batch.append(feats)
            centroids_coords = centroids[:, -3:] 
            dist = torch.cdist(centroids_coords.unsqueeze(0), centroids_coords.unsqueeze(0)).squeeze(0)
            _, indices = torch.topk(-dist, k=self.n_neighbors, dim=1)
            adj = torch.zeros((self.n_clusters, self.n_clusters), device=device)
            adj.scatter_(1, indices, 1.0)
            adj = (adj + adj.t() > 0).float()
            adj_batch.append(adj)

        return torch.stack(node_features_batch), torch.stack(adj_batch)

# --- 3. CÁC MODULE GNN (Giữ nguyên) ---
class DenseSAGPooling(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(DenseSAGPooling, self).__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(in_channels, 1)
    def forward(self, x, adj):
        B, N, C = x.size()
        score = self.score_layer(x).squeeze(-1)
        k = int(self.ratio * N)
        if k < 1: k = 1
        top_k_val, top_k_idx = torch.topk(score, k, dim=1)
        idx_expand = top_k_idx.unsqueeze(-1).expand(-1, -1, C)
        new_x = torch.gather(x, 1, idx_expand)
        idx_rows = top_k_idx.unsqueeze(-1).expand(-1, -1, N)
        temp_adj = torch.gather(adj, 1, idx_rows)
        idx_cols = top_k_idx.unsqueeze(1).expand(-1, k, -1)
        new_adj = torch.gather(temp_adj, 2, idx_cols)
        return new_x, new_adj, score

class HierarchicalViewGNN(nn.Module):
    def __init__(self, in_features=128, hidden_features=128, out_features=128, dropout=0.5):
        super().__init__()
        self.conv1 = DenseGATConv(in_features, hidden_features, heads=4, dropout=dropout)
        self.pool1 = DenseSAGPooling(hidden_features * 4, ratio=0.5)
        self.conv2 = DenseGATConv(hidden_features * 4, hidden_features, heads=4, dropout=dropout)
        self.pool2 = DenseSAGPooling(hidden_features * 4, ratio=0.5)
        self.conv3 = DenseGATConv(hidden_features * 4, out_features, heads=1, dropout=dropout)
        self.att_gate = nn.Sequential(nn.Linear(out_features, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x, adj):
        x = self.conv1(x, adj).relu()
        x, adj, node_importance = self.pool1(x, adj)
        x = self.conv2(x, adj).relu()
        x, adj, _ = self.pool2(x, adj)
        x = self.conv3(x, adj)
        scores = self.att_gate(x)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1), node_importance

class LatentGraphLearner(nn.Module):
    def __init__(self, embed_dim, num_latents=32, num_heads=4, dropout=0.5):
        super().__init__()
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, embed_dim))
        self.attn_read = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_read = nn.LayerNorm(embed_dim)
        self.attn_process = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_process = nn.LayerNorm(embed_dim)
        self.attn_write = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_write = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim*2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim*2, embed_dim))
        self.norm_ffn = nn.LayerNorm(embed_dim)
    def forward(self, all_views_features):
        B = all_views_features.shape[0]
        latents = self.latent_tokens.expand(B, -1, -1)
        attn_out, _ = self.attn_read(latents, all_views_features, all_views_features)
        latents = self.norm_read(latents + attn_out)
        attn_out, _ = self.attn_process(latents, latents, latents)
        latents = self.norm_process(latents + attn_out)
        attn_out, _ = self.attn_write(all_views_features, latents, latents)
        enriched = self.norm_write(all_views_features + attn_out)
        enriched = enriched + self.ffn(enriched)
        return self.norm_ffn(enriched), latents

class MultiViewSRRNet_V9(nn.Module):
    def __init__(self, pretrained_path=None, n_clusters=128, n_neighbors=8, node_features=128, 
                 gnn_out_features=128, num_heads=8, dropout=0.5):
        super().__init__()
        # Quay về bản Stable (Không Dilated)
        self.backbone_sag = HybridBackbone(pretrained_path)
        self.backbone_cor = HybridBackbone(pretrained_path)
        self.backbone_axi = HybridBackbone(pretrained_path)
        
        self.n_clusters = n_clusters
        self.graph_gen_sag = SuperRegionGraphGenerator(512, n_clusters, n_neighbors, node_features)
        self.graph_gen_cor = SuperRegionGraphGenerator(512, n_clusters, n_neighbors, node_features)
        self.graph_gen_axi = SuperRegionGraphGenerator(512, n_clusters, n_neighbors, node_features)
        
        latent_dim = max(32, n_clusters // 4)
        self.latent_bridge = LatentGraphLearner(node_features, num_latents=latent_dim, num_heads=4, dropout=dropout)
        self.gnn_sag = HierarchicalViewGNN(node_features, gnn_out_features, gnn_out_features, dropout)
        self.gnn_cor = HierarchicalViewGNN(node_features, gnn_out_features, gnn_out_features, dropout)
        self.gnn_axi = HierarchicalViewGNN(node_features, gnn_out_features, gnn_out_features, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=gnn_out_features, nhead=num_heads, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.fuser = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, gnn_out_features))
        
        self.classifier = nn.Sequential(nn.LayerNorm(gnn_out_features), nn.Linear(gnn_out_features, 3))
        self.aux_classifier = nn.Sequential(nn.LayerNorm(gnn_out_features), nn.Linear(gnn_out_features, 3))
        
    def forward(self, sag, cor, axi):
        f_sag = self.backbone_sag(sag)
        f_cor = self.backbone_cor(cor)
        f_axi = self.backbone_axi(axi)
        
        x_sag, adj_sag = self.graph_gen_sag(f_sag)
        x_cor, adj_cor = self.graph_gen_cor(f_cor)
        x_axi, adj_axi = self.graph_gen_axi(f_axi)
        
        all_feats = torch.cat([x_sag, x_cor, x_axi], dim=1) 
        enriched, latents = self.latent_bridge(all_feats)
        
        latent_summary = latents.mean(dim=1)
        aux_logits = self.aux_classifier(latent_summary)
        
        N = self.n_clusters
        x_sag_new = enriched[:, 0:N, :]
        x_cor_new = enriched[:, N:2*N, :]
        x_axi_new = enriched[:, 2*N:3*N, :]
        
        h_sag, att_sag = self.gnn_sag(x_sag_new, adj_sag)
        h_cor, att_cor = self.gnn_cor(x_cor_new, adj_cor)
        h_axi, att_axi = self.gnn_axi(x_axi_new, adj_axi)
        
        h_sag_emb, h_cor_emb, h_axi_emb = h_sag.unsqueeze(1), h_cor.unsqueeze(1), h_axi.unsqueeze(1)
        x = torch.cat([self.cls_token.expand(sag.size(0), -1, -1), h_sag_emb, h_cor_emb, h_axi_emb], dim=1) 
        fused = self.fuser(x) 
        final_emb = fused[:, 0]
        logits = self.classifier(final_emb)
        
        if self.training: return logits, aux_logits, (h_sag, h_cor, h_axi)
        else: return logits, aux_logits, (att_sag, att_cor, att_axi)