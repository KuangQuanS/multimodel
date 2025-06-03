import numpy as np
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities, indices=None, include_ct=False):
        data = np.load(npz_path)
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        self.include_ct = include_ct
        
        if include_ct and 'CT' in data:
            self.ct_data = data['CT']
        else:
            self.ct_data = None
            
        if indices is not None:
            self.Xs = [x[indices] for x in self.Xs]
            self.y = self.y[indices]
            if self.ct_data is not None:
                self.ct_data = self.ct_data[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {}
        
        # 添加常规模态数据
        for i, mod in enumerate(self.Xs):
            sample[f'X{i}'] = torch.FloatTensor(mod[idx])
        
        # 添加CT数据（如果有）
        if self.include_ct and self.ct_data is not None:
            sample['CT'] = torch.FloatTensor(self.ct_data[idx])
        
        sample['y'] = torch.tensor(self.y[idx], dtype=torch.long)
        
        return sample