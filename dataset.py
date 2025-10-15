"""
数据加载和处理模块
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from scipy import ndimage
import random


class ElasticNetVotingSelector(BaseEstimator, TransformerMixin):
    """ElasticNet投票特征选择器
    
    流程：
    1. 方差过滤：移除低方差特征
    2. 多次ElasticNet：运行多次ElasticNet回归，每次记录选择的特征
    3. 投票统计：统计每个特征被选中的次数
    4. 阈值过滤：保留投票比例 >= 阈值的特征
    5. Top特征：从通过阈值的特征中选择Top K个
    
    ElasticNet = α * L1_ratio * ||w||_1 + 0.5 * α * (1 - L1_ratio) * ||w||_2^2
    L1_ratio=1.0 相当于LASSO, L1_ratio=0.0 相当于Ridge
    """
    
    def __init__(self, variance_threshold=0.01, n_runs=10, voting_threshold=0.6, 
                 k_features=200, l1_ratio=0.7, random_state=42):
        self.variance_threshold = variance_threshold
        self.n_runs = n_runs
        self.voting_threshold = voting_threshold
        self.k_features = k_features
        self.l1_ratio = l1_ratio  # ElasticNet的L1/L2混合比例
        self.random_state = random_state
        
    def fit(self, X, y):
        """训练特征选择器"""
        np.random.seed(self.random_state)
        
        # Step 1: 方差过滤
        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        X_variance_filtered = self.variance_selector_.fit_transform(X)
        
        print(f"方差过滤: {X.shape[1]} -> {X_variance_filtered.shape[1]} 特征")
        
        # Step 2-4: 多次ElasticNet投票
        n_features = X_variance_filtered.shape[1]
        vote_counts = np.zeros(n_features)
        
        # 标准化数据用于ElasticNet
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_variance_filtered)
        
        print(f"🔍 ElasticNet投票特征选择 ({self.n_runs}次运行)...")
        successful_runs = 0
        
        for i in range(self.n_runs):
            # 每次使用不同的随机状态和L1_ratio
            try:
                # 在l1_ratio周围添加一些随机性，增加多样性
                current_l1_ratio = max(0.1, min(0.9, self.l1_ratio + np.random.normal(0, 0.1)))
                
                elastic_net = ElasticNetCV(
                    l1_ratio=current_l1_ratio,  # L1/L2混合比例
                    cv=5, 
                    random_state=self.random_state + i, 
                    max_iter=10000,
                    tol=1e-3,  # 更宽松的收敛容忍度
                    selection='random',  # 随机特征选择，提高收敛性
                    alphas=np.logspace(-4, -1, 50)
                )
                
                # 忽略收敛警告和其他警告
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")  # 忽略所有警告
                    elastic_net.fit(X_scaled, y)
                
                # 统计非零系数的特征
                selected_features = np.abs(elastic_net.coef_) > 1e-6
                n_selected = np.sum(selected_features)
                vote_counts += selected_features
                successful_runs += 1
                
                # 简化输出：只在关键点显示结果
                if i == 0 or (i + 1) % max(1, self.n_runs // 3) == 0 or i == self.n_runs - 1:
                    print(f"  运行 {i+1:2d}/{self.n_runs}: 选择了 {n_selected:4d} 特征")
                
            except Exception as e:
                # 如果ElasticNet失败，使用随机选择作为fallback
                n_random = min(self.k_features, n_features // 2)
                random_indices = np.random.choice(n_features, n_random, replace=False)
                random_features = np.zeros(n_features, dtype=bool)
                random_features[random_indices] = True
                vote_counts += random_features
                if i == 0 or (i + 1) % max(1, self.n_runs // 3) == 0:
                    print(f"  运行 {i+1:2d}/{self.n_runs}: 收敛失败，使用随机选择")
            
        # Step 4: 计算投票比例并应用阈值
        vote_ratios = vote_counts / self.n_runs
        threshold_mask = vote_ratios >= self.voting_threshold
        
        # 简化投票统计信息
        print(f"📊 ElasticNet投票完成: 成功{successful_runs}/{self.n_runs}次, 最高投票率{np.max(vote_ratios):.3f}")
        print(f"🎯 阈值过滤(>={self.voting_threshold}): {np.sum(threshold_mask):,} 特征通过")
        
        # Step 5: 从通过阈值的特征中选择Top K个
        if np.sum(threshold_mask) > self.k_features:
            # 按投票比例排序，选择Top K个
            threshold_indices = np.where(threshold_mask)[0]
            threshold_ratios = vote_ratios[threshold_mask]
            top_k_indices = threshold_indices[np.argsort(threshold_ratios)[::-1][:self.k_features]]
            
            final_mask = np.zeros(n_features, dtype=bool)
            final_mask[top_k_indices] = True
            self.selected_features_ = final_mask
            
            print(f"✂️ Top-K选择: 保留前{self.k_features:,}个特征")
        else:
            # 如果通过阈值的特征不足K个，全部保留
            self.selected_features_ = threshold_mask
            print(f"✂️ 保留所有通过阈值的{np.sum(threshold_mask):,}个特征")
        
        print(f"✅ 最终选择: {np.sum(self.selected_features_):,}个特征")
        
        # 保存投票统计信息用于分析
        self.vote_ratios_ = vote_ratios
        self.scaler_ = scaler
        
        return self
    
    def transform(self, X):
        """应用特征选择"""
        # 先应用方差过滤
        X_variance_filtered = self.variance_selector_.transform(X)
        # 再应用ElasticNet投票选择
        return X_variance_filtered[:, self.selected_features_]
    
    def get_support(self, indices=False):
        """返回被选择的特征索引或mask"""
        # 需要将两步选择的结果合并
        variance_support = self.variance_selector_.get_support()
        final_support = np.zeros(len(variance_support), dtype=bool)
        final_support[variance_support] = self.selected_features_
        
        if indices:
            return np.where(final_support)[0]
        return final_support


def apply_ct_augmentation(ct_data, training=True, strong_augment=False):
    """
    为CT数据应用数据增强
    
    Args:
        ct_data: CT数据 numpy array, shape (H, W, C) 或 (C, H, W)
        training: 是否为训练模式
        strong_augment: 是否使用强增强
    
    Returns:
        augmented_ct_data: 增强后的CT数据
    """
    if not training:
        return ct_data.copy()  # 确保返回连续数组
    
    # 确保数据是连续的且是 (H, W, C) 格式
    if ct_data.shape[0] == 3:  # 如果是 (3, 64, 64)
        ct_data = np.ascontiguousarray(ct_data.transpose(1, 2, 0))  # 转换为 (64, 64, 3)
    else:
        ct_data = np.ascontiguousarray(ct_data)
    
    H, W, C = ct_data.shape
    augmented = ct_data.copy()
    
    if strong_augment:
        return apply_strong_ct_augmentation(augmented)
    
    # 原始的轻度增强
    # 1. 随机旋转 (-10° to +10°)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        for c in range(C):
            rotated = ndimage.rotate(augmented[:, :, c], angle, 
                                   reshape=False, mode='nearest')
            augmented[:, :, c] = np.ascontiguousarray(rotated)
    
    # 2. 随机平移 (±5 pixels)
    if random.random() < 0.5:
        shift_x = random.randint(-5, 5)
        shift_y = random.randint(-5, 5)
        for c in range(C):
            shifted = ndimage.shift(augmented[:, :, c], (shift_y, shift_x), 
                                  mode='nearest')
            augmented[:, :, c] = np.ascontiguousarray(shifted)
    
    # 3. 随机亮度调整 (±10%)
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.9, 1.1)
        augmented = np.ascontiguousarray(np.clip(augmented * brightness_factor, 0, 1))
    
    # 4. 随机对比度调整 (±15%)
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.85, 1.15)
        mean_val = augmented.mean()
        augmented = np.ascontiguousarray(np.clip((augmented - mean_val) * contrast_factor + mean_val, 0, 1))
    
    # 5. 随机噪声添加 (微弱高斯噪声)
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, augmented.shape)
        augmented = np.ascontiguousarray(np.clip(augmented + noise, 0, 1))
    
    # 6. 随机水平翻转
    if random.random() < 0.5:
        augmented = np.ascontiguousarray(np.flip(augmented, axis=1))
    
    # 7. 随机缩放 (0.95-1.05倍) - 简化实现避免复杂操作
    if random.random() < 0.2:  # 降低概率避免过度增强
        scale_factor = random.uniform(0.98, 1.02)  # 更小的缩放范围
        for c in range(C):
            original_slice = augmented[:, :, c]
            zoomed_slice = ndimage.zoom(original_slice, scale_factor, 
                                       mode='nearest', order=1)
            zoomed_slice = np.ascontiguousarray(zoomed_slice)
            
            # 简单的中心裁剪或填充
            current_h, current_w = zoomed_slice.shape
            if current_h == H and current_w == W:
                augmented[:, :, c] = zoomed_slice
            elif current_h >= H and current_w >= W:
                # 中心裁剪
                start_h = (current_h - H) // 2
                start_w = (current_w - W) // 2
                augmented[:, :, c] = zoomed_slice[start_h:start_h+H, start_w:start_w+W]
            else:
                # 中心填充
                padded = np.zeros((H, W), dtype=augmented.dtype)
                start_h = max(0, (H - current_h) // 2)
                start_w = max(0, (W - current_w) // 2)
                end_h = min(H, start_h + current_h)
                end_w = min(W, start_w + current_w)
                
                src_end_h = min(current_h, end_h - start_h)
                src_end_w = min(current_w, end_w - start_w)
                
                padded[start_h:end_h, start_w:end_w] = zoomed_slice[:src_end_h, :src_end_w]
                augmented[:, :, c] = padded
    
    # 确保最终结果是连续的
    return np.ascontiguousarray(augmented)


def apply_strong_ct_augmentation(ct_data):
    """
    应用强CT数据增强
    
    包括更激进的变换：
    1. 更大范围的旋转和平移
    2. 更强的亮度和对比度变化
    3. 弹性变形
    4. Cutout/擦除
    5. 多种噪声
    6. 混合增强（MixUp风格的变换）
    """
    H, W, C = ct_data.shape
    augmented = ct_data.copy()
    
    # 1. 强旋转 (-30° to +30°)
    if random.random() < 0.7:  # 更高概率
        angle = random.uniform(-30, 30)  # 更大角度范围
        for c in range(C):
            rotated = ndimage.rotate(augmented[:, :, c], angle, 
                                   reshape=False, mode='nearest')
            augmented[:, :, c] = np.ascontiguousarray(rotated)
    
    # 2. 强平移 (±15 pixels)
    if random.random() < 0.7:
        shift_x = random.randint(-15, 15)
        shift_y = random.randint(-15, 15)
        for c in range(C):
            shifted = ndimage.shift(augmented[:, :, c], (shift_y, shift_x), 
                                  mode='nearest')
            augmented[:, :, c] = np.ascontiguousarray(shifted)
    
    # 3. 强亮度调整 (±30%)
    if random.random() < 0.6:
        brightness_factor = random.uniform(0.7, 1.3)
        augmented = np.ascontiguousarray(np.clip(augmented * brightness_factor, 0, 1))
    
    # 4. 强对比度调整 (±40%)
    if random.random() < 0.6:
        contrast_factor = random.uniform(0.6, 1.4)
        mean_val = np.nanmean(augmented)  # 使用nanmean避免NaN
        if np.isfinite(mean_val):  # 只有mean_val是有限值时才应用对比度调整
            augmented = np.ascontiguousarray(np.clip((augmented - mean_val) * contrast_factor + mean_val, 0, 1))
    
    # 5. 多种噪声
    if random.random() < 0.5:
        noise_type = random.choice(['gaussian', 'uniform', 'salt_pepper'])
        
        if noise_type == 'gaussian':
            # 高斯噪声
            noise_std = random.uniform(0.02, 0.08)
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented = np.ascontiguousarray(np.clip(augmented + noise, 0, 1))
            
        elif noise_type == 'uniform':
            # 均匀噪声
            noise_range = random.uniform(0.02, 0.06)
            noise = np.random.uniform(-noise_range, noise_range, augmented.shape)
            augmented = np.ascontiguousarray(np.clip(augmented + noise, 0, 1))
            
        elif noise_type == 'salt_pepper':
            # 椒盐噪声
            noise_prob = random.uniform(0.01, 0.05)
            mask = np.random.random(augmented.shape) < noise_prob
            augmented[mask] = np.random.choice([0, 1], size=np.sum(mask))
            augmented = np.ascontiguousarray(augmented)
    
    # 6. 弹性变形 (简化版)
    if random.random() < 0.3:
        alpha = random.uniform(10, 30)  # 变形强度
        sigma = random.uniform(3, 6)    # 平滑参数
        
        for c in range(C):
            # 生成随机位移场
            dx = ndimage.gaussian_filter((np.random.rand(H, W) - 0.5), sigma) * alpha
            dy = ndimage.gaussian_filter((np.random.rand(H, W) - 0.5), sigma) * alpha
            
            # 生成网格
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # 应用变形
            deformed = ndimage.map_coordinates(augmented[:, :, c], indices, 
                                             order=1, mode='nearest').reshape(H, W)
            augmented[:, :, c] = np.ascontiguousarray(deformed)
    
    # 7. Cutout/Random Erasing
    if random.random() < 0.4:
        n_holes = random.randint(1, 3)  # 1-3个擦除区域
        
        for _ in range(n_holes):
            hole_size = random.randint(8, 16)  # 擦除区域大小
            y1 = random.randint(0, H - hole_size)
            x1 = random.randint(0, W - hole_size)
            y2 = y1 + hole_size
            x2 = x1 + hole_size
            
            # 随机填充值
            fill_value = random.uniform(0, 1)
            augmented[y1:y2, x1:x2, :] = fill_value
    
    # 8. 随机翻转（包括垂直翻转）
    if random.random() < 0.5:
        augmented = np.ascontiguousarray(np.flip(augmented, axis=1))  # 水平翻转
    
    if random.random() < 0.3:
        augmented = np.ascontiguousarray(np.flip(augmented, axis=0))  # 垂直翻转
    
    # 9. 强缩放 (0.8-1.3倍)
    if random.random() < 0.5:
        scale_factor = random.uniform(0.8, 1.3)
        for c in range(C):
            original_slice = augmented[:, :, c]
            zoomed_slice = ndimage.zoom(original_slice, scale_factor, 
                                       mode='nearest', order=1)
            zoomed_slice = np.ascontiguousarray(zoomed_slice)
            
            current_h, current_w = zoomed_slice.shape
            if current_h >= H and current_w >= W:
                # 中心裁剪
                start_h = (current_h - H) // 2
                start_w = (current_w - W) // 2
                augmented[:, :, c] = zoomed_slice[start_h:start_h+H, start_w:start_w+W]
            else:
                # 中心填充
                padded = np.zeros((H, W), dtype=augmented.dtype)
                start_h = (H - current_h) // 2
                start_w = (W - current_w) // 2
                end_h = start_h + current_h
                end_w = start_w + current_w
                
                # 确保边界不越界
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h = min(H, end_h)
                end_w = min(W, end_w)
                
                padded[start_h:end_h, start_w:end_w] = zoomed_slice[:end_h-start_h, :end_w-start_w]
                augmented[:, :, c] = padded
    
    # 10. Gamma变换（模拟不同的成像条件）
    if random.random() < 0.3:
        gamma = random.uniform(0.5, 2.0)
        # 确保数据严格非负，避免0^负数的情况
        augmented = np.clip(augmented, 1e-8, 1.0)  # 避免完全的0值
        
        # 安全的幂运算
        try:
            augmented = np.power(augmented, gamma)
            # 检查结果是否有效
            if np.isnan(augmented).any() or np.isinf(augmented).any():
                # 如果有问题，跳过gamma变换
                augmented = np.clip(ct_data.copy(), 0, 1)
        except (RuntimeWarning, FloatingPointError):
            # 如果出现数值问题，跳过gamma变换
            augmented = np.clip(ct_data.copy(), 0, 1)
        
        augmented = np.ascontiguousarray(augmented)
    
    # 11. 色彩抖动（对于多通道CT）
    if random.random() < 0.3 and C > 1:
        for c in range(C):
            # 通道特定的亮度调整
            channel_factor = random.uniform(0.8, 1.2)
            augmented[:, :, c] = np.clip(augmented[:, :, c] * channel_factor, 0, 1)
    
    # 12. 局部对比度调整（CLAHE风格）
    if random.random() < 0.2:
        for c in range(C):
            # 简化版的自适应直方图均衡
            slice_data = augmented[:, :, c]
            # 分块处理
            block_size = 16
            for i in range(0, H, block_size):
                for j in range(0, W, block_size):
                    block = slice_data[i:min(i+block_size, H), j:min(j+block_size, W)]
                    if block.size > 0:
                        # 局部对比度增强
                        block_mean = block.mean()
                        block_std = block.std()
                        if block_std > 0:
                            enhanced_block = (block - block_mean) * random.uniform(1.0, 1.5) + block_mean
                            slice_data[i:min(i+block_size, H), j:min(j+block_size, W)] = np.clip(enhanced_block, 0, 1)
    
    # 最终检查：确保没有NaN、inf或超出范围的值
    augmented = np.nan_to_num(augmented, nan=0.0, posinf=1.0, neginf=0.0)
    augmented = np.clip(augmented, 0, 1)
    
    return np.ascontiguousarray(augmented)



class MultiModalDataset(Dataset):
    """多模态数据集类，支持cfDNA和CT数据"""
    def __init__(self, npz_path, modalities, indices=None, include_ct=False, feature_selectors=None, training=True, strong_augment=False):
        data = np.load(npz_path, allow_pickle=True)  # ensure support for string arrays
        self.modalities = modalities
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        self.include_ct = include_ct
        self.ct_data = data['CT'] if include_ct and 'CT' in data else None
        self.id = data['id'] if 'id' in data else None
        self.training = training  # 训练模式标志，用于控制数据增强
        self.strong_augment = strong_augment  # 强增强标志
        
        # 应用特征选择
        self.feature_selectors = feature_selectors or {}
        if self.feature_selectors:
            for i, mod in enumerate(modalities):
                if mod in self.feature_selectors:
                    self.Xs[i] = self.feature_selectors[mod].transform(self.Xs[i])

        if indices is not None:
            self.Xs = [x[indices] for x in self.Xs]
            self.y = self.y[indices]
            if self.ct_data is not None:
                self.ct_data = self.ct_data[indices]
            if self.id is not None:
                self.id = self.id[indices]

    def __len__(self):
        return len(self.y)
    
    def set_training_mode(self, training):
        """设置数据集的训练模式"""
        self.training = training

    def __getitem__(self, idx):
        # 获取cfDNA数据（不进行额外增强）
        cfdna_data = {}
        for i, mod in enumerate(self.Xs):
            data = mod[idx].copy()
            cfdna_data[f'X{i}'] = torch.FloatTensor(data)
        
        sample = cfdna_data
        
        if self.include_ct and self.ct_data is not None:
            ct_slice = self.ct_data[idx].copy()  # 复制数据避免修改原始数据
            
            # 应用CT数据增强（仅在训练模式下）
            if self.training:
                ct_slice = apply_ct_augmentation(ct_slice, training=True, strong_augment=self.strong_augment)
            
            # 转换为torch tensor格式 (C, H, W)
            if ct_slice.shape[-1] == 3:  # 如果是 (H, W, C) 格式
                ct_slice = np.ascontiguousarray(ct_slice.transpose(2, 0, 1))  # 转换为 (C, H, W) 并确保连续
            else:
                ct_slice = np.ascontiguousarray(ct_slice)  # 确保数据是连续的
                
            sample['CT'] = torch.FloatTensor(ct_slice)
        
        sample['y'] = torch.tensor(self.y[idx], dtype=torch.long)

        if self.id is not None:
            sample['id'] = str(self.id[idx])
        
        return sample


def create_feature_selectors(train_data_path, modalities, method='combined', k_features=50, 
                           variance_threshold=0.01, lasso_voting_threshold=0.6, lasso_n_runs=10, elastic_l1_ratio=0.7):
    """
    为cfDNA模态创建特征选择器
    
    Args:
        train_data_path: 训练数据路径
        modalities: cfDNA模态列表
        method: 特征选择方法 ('variance', 'kbest', 'rfe', 'combined', 'lasso_voting', 'elastic_voting')
        k_features: 每个模态保留的特征数量
        variance_threshold: 方差阈值（用于lasso_voting/elastic_voting方法）
        lasso_voting_threshold: 投票阈值
        lasso_n_runs: 运行次数
        elastic_l1_ratio: ElasticNet的L1比例 (0.0=纯L2, 1.0=纯L1)
    
    Returns:
        feature_selectors: 特征选择器字典
    """
    data = np.load(train_data_path)
    y_train = data['y']
    
    feature_selectors = {}
    
    for mod in modalities:
        X_mod = data[mod]
        # 简化输出：只显示模态名和原始特征数
        if len(modalities) <= 4:  # 只在模态数量少时显示详情
            print(f"🔧 {mod}: {X_mod.shape[1]}特征")
        
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
            
        elif method == 'lasso_voting':
            # ElasticNet投票方法：方差过滤 -> 多次ElasticNet + 投票 -> Top特征
            selector = ElasticNetVotingSelector(
                variance_threshold=variance_threshold,
                n_runs=lasso_n_runs, 
                voting_threshold=lasso_voting_threshold,
                k_features=min(k_features, X_mod.shape[1]),
                l1_ratio=elastic_l1_ratio  # 使用传入的L1/L2混合比例
            )
            
        elif method == 'elastic_voting':
            # ElasticNet投票方法（新方法名）：方差过滤 -> 多次ElasticNet + 投票 -> Top特征
            selector = ElasticNetVotingSelector(
                variance_threshold=variance_threshold,
                n_runs=lasso_n_runs, 
                voting_threshold=lasso_voting_threshold,
                k_features=min(k_features, X_mod.shape[1]),
                l1_ratio=elastic_l1_ratio  # 使用传入的L1/L2混合比例
            )
            
        elif method == 'pca':
            # PCA降维方法：将特征压缩到指定维度
            n_components = min(k_features, X_mod.shape[1], X_mod.shape[0])  # 不能超过样本数
            selector = PCA(n_components=n_components, random_state=42)
            
            # 详细说明维度限制
            if X_mod.shape[0] < k_features:
                print(f"⚠️  {mod}PCA维度受样本数限制: 期望{k_features}维 → 实际{n_components}维 (样本数={X_mod.shape[0]})")
            else:
                print(f"🔄 {mod}使用PCA降维: {X_mod.shape[1]:,} -> {n_components} 维")
        
        # 根据方法类型决定fit调用方式
        if method == 'pca':
            # PCA是无监督方法，不需要标签
            selector.fit(X_mod)
        else:
            # 有监督方法需要标签
            selector.fit(X_mod, y_train)
        feature_selectors[mod] = selector
        
        # 显示选择后的特征数量
        X_selected = selector.transform(X_mod)
        # 简化输出
        if len(modalities) <= 4:
            print(f"✅ {mod}: {X_mod.shape[1]:,} -> {X_selected.shape[1]:,}")
    
    # 打印总体特征选择摘要
    total_original = sum(np.load(train_data_path)[mod].shape[1] for mod in modalities)
    total_selected = sum(selector.transform(np.load(train_data_path)[mod]).shape[1] 
                        for mod, selector in feature_selectors.items())
    reduction_ratio = (1 - total_selected / total_original) * 100
    
    print(f"\n🎯 特征选择总结 ({method}):")
    print(f"  📊 总特征数: {total_original:,} -> {total_selected:,}")
    print(f"  📉 压缩率: {reduction_ratio:.1f}%")
    print(f"  🔢 每模态平均: {total_selected//len(modalities):,} 特征\n")
    
    return feature_selectors


def load_data(data_path, modalities, include_ct=False, feature_selectors=None, training=True):
    """
    加载数据文件并返回相关信息
    
    Args:
        data_path: 数据文件路径
        modalities: 模态列表
        include_ct: 是否包含CT数据
        feature_selectors: cfDNA特征选择器字典
        training: 是否为训练模式（影响CT数据增强）
    
    Returns:
        tuple: (dataset, label_distribution)
    """
    dataset = MultiModalDataset(data_path, modalities, include_ct=include_ct, 
                               feature_selectors=feature_selectors, training=training)
    
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