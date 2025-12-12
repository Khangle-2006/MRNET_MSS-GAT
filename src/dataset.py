import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

class MRNetDataset(Dataset):
    def __init__(self, root_dir, split_type='train', transform=None, cache_to_ram=True):
        self.root_dir = root_dir
        self.split_type = split_type
        self.transform = transform 
        self.cache_to_ram = cache_to_ram
        self.data_path = os.path.join(root_dir, split_type)
        
        try:
            df_abn = pd.read_csv(os.path.join(root_dir, f'{split_type}-abnormal.csv'), header=None, names=['Case', 'abnormal'])
            df_acl = pd.read_csv(os.path.join(root_dir, f'{split_type}-acl.csv'), header=None, names=['Case', 'acl'])
            df_men = pd.read_csv(os.path.join(root_dir, f'{split_type}-meniscus.csv'), header=None, names=['Case', 'meniscus'])
            self.labels_df = df_abn.merge(df_acl, on='Case').merge(df_men, on='Case')
            self.patient_list = self.labels_df['Case'].astype(str).str.zfill(4).tolist()
        except:
            self.patient_list = []

        self.cache = {}
        if self.cache_to_ram and len(self.patient_list) > 0:
            print(f"--> [RAM CACHE] Loading {split_type} set (RAW VARIABLE DEPTH)...")
            for patient_id in tqdm(self.patient_list):
                self.cache[patient_id] = self._load_case(patient_id)

    def _load_case(self, patient_id):
        views_data = {}
        for view in ['sagittal', 'coronal', 'axial']:
            file_path = os.path.join(self.data_path, view, f'{patient_id}.npy')
            try:
                scan = np.load(file_path).astype(np.float32)
            except:
                scan = np.zeros((32, 256, 256), dtype=np.float32)

            mx, mn = np.max(scan), np.min(scan)
            scan = (scan - mn) / (mx - mn + 1e-8) if (mx - mn) > 0 else scan - mn
            
            # TRẢ VỀ RAW (D, H, W)
            views_data[view] = torch.tensor(scan) 
            
        return views_data

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        if self.cache_to_ram and patient_id in self.cache:
            views = self.cache[patient_id]
            views = {k: v.clone() for k, v in views.items()} 
        else:
            views = self._load_case(patient_id)

        labels = self.labels_df.iloc[idx][['abnormal', 'acl', 'meniscus']].values
        return views, torch.tensor(labels, dtype=torch.float32)