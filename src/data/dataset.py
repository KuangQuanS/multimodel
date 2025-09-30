"""
数据加载和处理模块
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """多模态数据集类，支持cfDNA和CT数据"""
    def __init__(self, npz_path, modalities, indices=None, include_ct=False):
        data = np.load(npz_path, allow_pickle=True)  # ensure support for string arrays
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        self.include_ct = include_ct
        self.ct_data = data['CT'] if include_ct and 'CT' in data else None
        self.id = data['id'] if 'id' in data else None

        if indices is not None:
            self.Xs = [x[indices] for x in self.Xs]
            self.y = self.y[indices]
            if self.ct_data is not None:
                self.ct_data = self.ct_data[indices]
            if self.id is not None:
                self.id = self.id[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {f'X{i}': torch.FloatTensor(mod[idx]) for i, mod in enumerate(self.Xs)}
        
        if self.include_ct and self.ct_data is not None:
            sample['CT'] = torch.FloatTensor(self.ct_data[idx].transpose(2, 0, 1))
        
        sample['y'] = torch.tensor(self.y[idx], dtype=torch.long)

        if self.id is not None:
            sample['id'] = str(self.id[idx])
        
        return sample


def load_data(data_path, modalities, include_ct=False):
    """
    加载数据文件并返回相关信息
    
    Args:
        data_path: 数据文件路径
        modalities: 模态列表
        include_ct: 是否包含CT数据
    
    Returns:
        tuple: (dataset, label_distribution)
    """
    dataset = MultiModalDataset(data_path, modalities, include_ct=include_ct)
    
    # 计算标签分布
    unique, counts = np.unique(dataset.y, return_counts=True)
    label_distribution = (unique, counts)
    
    return dataset, label_distribution


def get_modality_dimensions():
    """返回各个模态的特征维度"""
    return {
        'Frag': 888,
        'CNV': 21870,
        'PFE': 19415,
        'NDR': 19434,
        'NDR2K': 19434
    }