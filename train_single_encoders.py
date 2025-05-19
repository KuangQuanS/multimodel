import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from datetime import datetime
import glob
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./log/training_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'),
        logging.StreamHandler()
    ]
)

class SingleModalityEncoder(nn.Module):
    """单模态分类编码器"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        
        # 编码器部分
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        self.encoder = nn.Sequential(*layers)
        
        # 分类头
        self.classifier = nn.Linear(dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回特征和分类logits"""
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits

class SingleModalityDataset(Dataset):
    """单模态数据集"""
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

def load_data_from_folders(normal_dir: str, cancer_dir: str, modality: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """加载特定模态的数据，返回训练数据、测试数据、训练标签、测试标签、样本名和特征名"""
    # 加载正常样本
    normal_files = glob.glob(os.path.join(normal_dir, f"*{modality}*.csv"))
    if not normal_files:
        raise FileNotFoundError(f"No {modality} files found in {normal_dir}")
    
    normal_data = []
    normal_labels = []
    sample_names = []
    feature_names = None
    
    for file in normal_files:
        df = pd.read_csv(file, index_col=0)
        if feature_names is None:
            feature_names = df.columns.tolist()
        sample_names.extend(df.index.tolist())
        normal_data.append(df.values)
        normal_labels.extend([0] * len(df))
    
    # 加载癌症样本
    cancer_files = glob.glob(os.path.join(cancer_dir, f"*{modality}*.csv"))
    if not cancer_files:
        raise FileNotFoundError(f"No {modality} files found in {cancer_dir}")
    
    cancer_data = []
    cancer_labels = []
    
    for file in cancer_files:
        df = pd.read_csv(file, index_col=0)
        sample_names.extend(df.index.tolist())
        cancer_data.append(df.values)
        cancer_labels.extend([1] * len(df))
    
    # 合并并划分数据集
    def split_data(data, labels, test_size=0.2):
        indices = np.random.permutation(len(data))
        split = int(len(data) * (1 - test_size))
        return data[indices[:split]], data[indices[split:]], labels[indices[:split]], labels[indices[split:]]
    
    # 分别划分正常和癌症数据
    normal_train, normal_test, normal_train_labels, normal_test_labels = split_data(
        np.vstack(normal_data), np.array(normal_labels))
    cancer_train, cancer_test, cancer_train_labels, cancer_test_labels = split_data(
        np.vstack(cancer_data), np.array(cancer_labels))
    
    # 合并训练集和测试集
    train_data = np.vstack([normal_train, cancer_train])
    test_data = np.vstack([normal_test, cancer_test])
    train_labels = np.concatenate([normal_train_labels, cancer_train_labels])
    test_labels = np.concatenate([normal_test_labels, cancer_test_labels])
    
    return train_data, test_data, train_labels, test_labels, sample_names, feature_names

def train_single_encoder(modality: str, 
                        normal_dir: str, 
                        cancer_dir: str,
                        hidden_dims: List[int] = [256, 128],
                        encoder_output_dim: int = 64,
                        num_epochs: int = 100,
                        batch_size: int = 32,
                        learning_rate: float = 0.001,
                        dropout_rate: float = 0.3) -> None:
    """训练单个模态的编码器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"------------------------------------------------------------------------")
    logging.info(f"Training {modality} encoder on {device}")
    
    # 加载并划分数据
    train_data, test_data, train_labels, test_labels, _, feature_names = load_data_from_folders(
        normal_dir, cancer_dir, modality)
    
    # 数据预处理
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    # 创建数据集和数据加载器
    train_dataset = SingleModalityDataset(train_data, train_labels)
    test_dataset = SingleModalityDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_dim = train_data.shape[1]  # 使用train_data的shape而不是未定义的data
    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))  # 合并标签获取类别数
    num_classes = len(unique_labels)
    
    model = SingleModalityEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        dropout_rate=dropout_rate
    ).to(device)
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            _, logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)  # 使用torch.max替代直接.max()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        
        # 验证 (使用测试集作为验证集)
        test_loss, test_acc = validate(model, test_loader, device, criterion)
        
        # 学习率调整
        scheduler.step()
        
        # 记录日志
        if epoch % 25 == 0:
            logging.info(f'Epoch {epoch+1}/{num_epochs} - {modality}:')
            logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logging.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # 保存最佳模型 (包括测试数据)
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'scaler': scaler,
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'output_dim': num_classes,
                'feature_names': feature_names
                # 'test_data': test_data,  # 保存测试数据用于后续评估
                # 'test_labels': test_labels  # 保存测试标签
            }, f'./checkpoint/best_{modality}_encoder.pth')
            logging.info(f'Saved new best {modality} model with test accuracy: {best_val_acc:.2f}%')

def validate(model: nn.Module, 
            val_loader: DataLoader, 
            device: torch.device,
            criterion: nn.Module) -> Tuple[float, float]:
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            _, logits = model(inputs)
            loss = criterion(logits, targets)
            
            val_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def evaluate_single_encoder(modality: str, normal_dir: str, cancer_dir: str) -> None:
    """评估单个模态的编码器（从原始文件夹重新加载测试数据）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    checkpoint = torch.load(f'./checkpoint/best_{modality}_encoder.pth', map_location=device)
    model = SingleModalityEncoder(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        output_dim=checkpoint['output_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 重新从文件夹加载测试数据
    _, test_data, _, test_labels, _, feature_names = load_data_from_folders(
        normal_dir, cancer_dir, modality)
    
    # 使用保存的scaler预处理数据
    scaler = checkpoint['scaler']
    test_data = scaler.transform(test_data)
    
    # 创建测试数据集
    test_dataset = SingleModalityDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 评估
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, device, criterion)
    
    # 计算分类报告和混淆矩阵
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, logits = model(inputs)
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # 打印结果
    logging.info(f"\n{modality} Encoder Evaluation:")
    logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    logging.info("\nClassification Report:")
    logging.info(classification_report(all_labels, all_preds, target_names=['Normal', 'Cancer']))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Cancer'], 
                yticklabels=['Normal', 'Cancer'])
    plt.title(f'{modality} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{modality}_confusion_matrix.png')
    plt.close()

    return feature_names

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据路径 - 请替换为您的实际路径
    normal_dir = "../database/cfDNA/normal"  # 正常样本目录
    cancer_dir = "../database/cfDNA/cancer"  # 癌症样本目录
    h_dims=[2048, 1024, 512]
    # 训练所有模态的编码器
    modalities = ['Frag', 'CNV', 'PFE', 'NDR', 'NDR2K']
    logging.info(f"---------------------encoder dim:{h_dims}---------------------")
    for modality in modalities:
        train_single_encoder(
            modality=modality,
            normal_dir=normal_dir,
            cancer_dir=cancer_dir,
            hidden_dims=h_dims,
            encoder_output_dim=256,
            num_epochs=100,
            batch_size=16,
            learning_rate=0.001,
            dropout_rate=0.3
        )
        #evaluate_single_encoder(modality)

if __name__ == '__main__':
    main()