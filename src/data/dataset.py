"""
数据加载和处理模块
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class MultiModalDataset(Dataset):
    """多模态数据集类，支持cfDNA和CT数据"""
    def __init__(self, npz_path, modalities, indices=None, include_ct=False, feature_selectors=None):
        data = np.load(npz_path, allow_pickle=True)  # ensure support for string arrays
        self.modalities = modalities
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        self.include_ct = include_ct
        self.ct_data = data['CT'] if include_ct and 'CT' in data else None
        self.id = data['id'] if 'id' in data else None
        
        # 应用特征选择
        self.feature_selectors = feature_selectors or {}
        if self.feature_selectors:
            for i, mod in enumerate(modalities):
                if mod in self.feature_selectors:
                    original_shape = self.Xs[i].shape
                    self.Xs[i] = self.feature_selectors[mod].transform(self.Xs[i])
                    new_shape = self.Xs[i].shape
                    print(f"{mod} 特征选择: {original_shape[1]} -> {new_shape[1]} 特征")

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


def create_feature_selectors(train_data_path, modalities, method='combined', k_features=50):
    """
    为cfDNA模态创建特征选择器
    
    Args:
        train_data_path: 训练数据路径
        modalities: cfDNA模态列表
        method: 特征选择方法 ('variance', 'kbest', 'rfe', 'combined')
        k_features: 每个模态保留的特征数量
    
    Returns:
        feature_selectors: 特征选择器字典
    """
    data = np.load(train_data_path)
    y_train = data['y']
    
    feature_selectors = {}
    
    for mod in modalities:
        X_mod = data[mod]
        print(f"为{mod}创建特征选择器: {X_mod.shape[1]} 个原始特征")
        
        if method == 'variance':
            # 方差阈值选择
            selector = VarianceThreshold(threshold=0.0)
            
        elif method == 'kbest':
            # 单变量特征选择
            selector = SelectKBest(f_classif, k=min(k_features, X_mod.shape[1]))
            
        elif method == 'rfe':
            # 递归特征消除
            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k_features, X_mod.shape[1]))
            
        elif method == 'combined':
            # 组合方法：先方差筛选，再K-best选择
            variance_selector = VarianceThreshold(threshold=0.0)
            k_selector = SelectKBest(f_classif, k=min(k_features, X_mod.shape[1]))
            selector = Pipeline([
                ('variance', variance_selector),
                ('k_best', k_selector)
            ])
        
        selector.fit(X_mod, y_train)
        feature_selectors[mod] = selector
        
        # 显示选择后的特征数量
        X_selected = selector.transform(X_mod)
        print(f"{mod}特征选择完成: {X_mod.shape[1]} -> {X_selected.shape[1]} 特征")
    
    return feature_selectors


def load_data(data_path, modalities, include_ct=False, feature_selectors=None):
    """
    加载数据文件并返回相关信息
    
    Args:
        data_path: 数据文件路径
        modalities: 模态列表
        include_ct: 是否包含CT数据
        feature_selectors: cfDNA特征选择器字典
    
    Returns:
        tuple: (dataset, label_distribution)
    """
    dataset = MultiModalDataset(data_path, modalities, include_ct=include_ct, 
                               feature_selectors=feature_selectors)
    
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