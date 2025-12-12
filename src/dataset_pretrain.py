import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class RadImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, save_map_path='class_mapping.json'):
        """
        root_dir: Đường dẫn đến thư mục gốc (ví dụ: 'radiology_ai')
        Cấu trúc mong đợi: root_dir/MODALITY/ANATOMY/PATHOLOGY/image.png
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 1. Tự động quét các Modality (Cấp 1)
        # Tìm các folder con trong root (CT, MR, US...)
        self.modalities = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.modality_to_idx = {cls_name: i for i, cls_name in enumerate(self.modalities)}
        
        # 2. Tự động quét các Anatomy (Cấp 2)
        # Chúng ta sẽ duyệt qua TẤT CẢ modality để gom đủ các loại anatomy
        anatomy_names = set()
        for mod in self.modalities:
            mod_path = os.path.join(root_dir, mod)
            # Lấy tên các folder con trong mỗi modality (đây chính là Anatomy: abd, knee...)
            anats = [d.name for d in os.scandir(mod_path) if d.is_dir()]
            anatomy_names.update(anats)
            
        self.anatomies = sorted(list(anatomy_names))
        self.anatomy_to_idx = {anat: i for i, anat in enumerate(self.anatomies)}
        
        # In ra để bạn biết nó tìm thấy gì
        print(f"--> Tìm thấy {len(self.modalities)} Modalities: {self.modalities}")
        print(f"--> Tìm thấy {len(self.anatomies)} Anatomies: {self.anatomies}")
        
        # Lưu map ra file để sau này check
        with open(save_map_path, 'w') as f:
            json.dump({
                'modality_map': self.modality_to_idx,
                'anatomy_map': self.anatomy_to_idx
            }, f, indent=4)
        print(f"--> Đã lưu bảng mapping vào '{save_map_path}'")

        # 3. Quét file ảnh (Crawl Data)
        print("Đang index toàn bộ file ảnh (có thể mất vài giây)...")
        self._crawl_data()
        print(f"--> Tổng cộng: {len(self.samples)} ảnh sẵn sàng training.")

    def _crawl_data(self):
        # Duyệt Modality -> Anatomy -> Pathology -> Image
        for mod_name in self.modalities:
            mod_idx = self.modality_to_idx[mod_name]
            mod_path = os.path.join(self.root_dir, mod_name)
            
            # Duyệt các folder Anatomy bên trong Modality này
            for anat_name in os.listdir(mod_path):
                anat_path = os.path.join(mod_path, anat_name)
                if not os.path.isdir(anat_path): continue
                
                # Lấy ID từ map đã tạo tự động
                if anat_name not in self.anatomy_to_idx: continue # (Phòng hờ)
                anat_idx = self.anatomy_to_idx[anat_name]
                
                # Duyệt folder bệnh lý (Pathology) - Chúng ta ko cần label bệnh, chỉ cần ảnh
                for pathol_name in os.listdir(anat_path):
                    pathol_path = os.path.join(anat_path, pathol_name)
                    if not os.path.isdir(pathol_path): continue
                    
                    # Lấy file ảnh
                    with os.scandir(pathol_path) as entries:
                        for entry in entries:
                            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                                # Lưu: (Đường dẫn, Nhãn Modality, Nhãn Anatomy)
                                self.samples.append((entry.path, mod_idx, anat_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mod_label, anat_label = self.samples[idx]
        
        try:
            # Convert RGB ngay để khớp ResNet
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, mod_label, anat_label
            
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path}: {e}")
            # Trả về ảnh đen nếu lỗi
            return torch.zeros((3, 224, 224)), mod_label, anat_label
        