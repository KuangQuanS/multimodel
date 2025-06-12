import math
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from collections import Counter
import torch_optimizer as optim
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2

def generate_attention_visualization(model, image_tensor, original_image=None, save_path=None, alpha=0.5):
    """
    生成注意力可视化图并与原始图像叠加
    
    参数:
        model: CTModel实例
        image_tensor: 输入张量 [1, 3, H, W]
        original_image: 原始图像numpy数组 (H, W, 3)，如果为None则使用tensor
        save_path: 保存路径，如果为None则不保存
        alpha: 注意力图与原始图像的混合比例
    
    返回:
        overlaid_image: 叠加了注意力图的原始图像
        attention_maps: 所有注意力图
    """
    # 确保模型处于评估模式
    model.eval()
    # 生成注意力图
    with torch.no_grad():
        attention_maps = model.visualize_attention(image_tensor.unsqueeze(0))
    # 创建叠加图像
    overlaid_images = []
    for i, attention_map in enumerate(attention_maps):
        # 转换注意力图为热力图
        attention_map = attention_map.squeeze().cpu().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        
        # 转换为RGB（如果需要）
        if len(heatmap.shape) == 2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        elif heatmap.shape[2] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 归一化热力图
        heatmap = heatmap.astype(float) / 255
        
        # 确保原始图像是float类型且在0-1范围内
        if original_image.dtype != np.float32:
            original_image = original_image.astype(float) / 255
        
        # 叠加图像
        overlaid = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
        overlaid_images.append(overlaid)
    
    # 如果需要保存图像
    if save_path is not None:
        # 创建图像网格
        n_maps = len(overlaid_images)
        n_cols = min(4, n_maps)  # 每行最多4张图
        n_rows = (n_maps + n_cols - 1) // n_cols
        
        plt.figure(figsize=(4*n_cols, 4*n_rows))
        for i, img in enumerate(overlaid_images):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(img)
            plt.title(f'Attention Layer {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    return overlaid_images, attention_maps

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """
        alpha:         None 或者 shape=[num_classes] 的 Tensor 或 float，
                       二分类时 float 会被转换为 [alpha, 1-alpha] Tensor
        gamma:         聚焦系数 γ，典型值 2.0
        reduction:     'none' | 'mean' | 'sum'
        ignore_index:  忽略标签
        """
        super().__init__()
        # 如果是单一 float（用于二分类），转换为 [α, 1−α] Tensor
        if isinstance(alpha, float):
            self.alpha = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32)
        else:
            self.alpha = alpha  # None 或 已经是 Tensor
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs:  [N, C] logits 或者 [N] logits（二分类）
        targets: [N] LongTensor
        """
        # 二分类分支
        if inputs.dim() == 1 or inputs.size(1) == 1:
            logits = inputs.view(-1)
            targets = targets.view(-1).float()
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none')
            pt = torch.exp(-bce_loss)  # pt = sigmoid(logit) 或 1−pt
            if self.alpha is not None:
                # 对二分类 alpha 张量广播
                alpha_factor = (targets * self.alpha[0] +
                                (1 - targets) * self.alpha[1]).to(bce_loss.device)
                loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
            else:
                loss = (1 - pt) ** self.gamma * bce_loss

        # 多分类分支
        else:
            logp = F.log_softmax(inputs, dim=1)     # [N, C] :contentReference[oaicite:0]{index=0}
            p = torch.exp(logp)                     # [N, C]
            # 取对应类别的 log p_t 和 p_t
            targets = targets.view(-1, 1)
            logpt = logp.gather(1, targets).view(-1)
            pt = p.gather(1, targets).view(-1)
            # 准备 α_t
            if self.alpha is not None:
                # 确保 α 是 Tensor 并在同 device
                if not isinstance(self.alpha, torch.Tensor):
                    raise ValueError("alpha must be Tensor for multiclass")
                at = self.alpha.to(inputs.device).gather(0, targets.view(-1))
            else:
                at = 1.0
            loss = -at * (1 - pt) ** self.gamma * logpt

        # 忽略特定标签
        if self.ignore_index >= 0:
            valid = targets.view(-1) != self.ignore_index
            loss = loss[valid]

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class ChannelAttention2D(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super().__init__()
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction, 1),
            nn.LayerNorm([inplanes // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1),
            nn.LayerNorm([inplanes, 1, 1])
        )
        self.sigmoid = nn.Sigmoid()

    def spatial_pool(self, x):
        B, C, H, W = x.size()
        input_x = x.view(B, C, -1).unsqueeze(1)              # B × 1 × C × (H×W)
        context_mask = self.conv_mask(x).view(B, 1, -1)      # B × 1 × (H×W)
        context_mask = self.softmax(context_mask).unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)        # B × 1 × C × 1
        return context.view(B, C, 1, 1)

    def forward(self, x):
        context = self.spatial_pool(x)
        channel_mul_term = self.sigmoid(self.channel_mul_conv(context))
        return x * channel_mul_term

class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([max_out, avg_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))

class GCSAM2D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention2D(in_channels, reduction)
        self.spatial_attention = SpatialAttention2D()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Bottle2neck2D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, baseWidth=26, scale=4, stype='normal'):
        super().__init__()
        width = int(math.floor(inplanes / 4 * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.relu = nn.ReLU(inplace=True)

        self.nums = scale - 1 if scale != 1 else 1
        self.stype = stype
        self.scale = scale
        self.width = width

        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(width)
            for _ in range(self.nums)
        ])

        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        if inplanes != planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

        self.GCS = GCSAM2D(planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
            sp = spx[i] if i == 0 or self.stype == 'stage' else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        if self.scale != 1:
            if self.stype == 'normal':
                out = torch.cat((out, spx[self.nums]), 1)
            elif self.stype == 'stage':
                out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(self.conv3(out))
        residual = self.shortcut(residual)
        out = self.GCS(out)
        return self.relu(out + residual)

class Encoder2D(nn.Module):
    def __init__(self, channels, dropout_prob=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                Bottle2neck2D(channels[i], channels[i+1]),
                Bottle2neck2D(channels[i+1], channels[i+1]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout_prob)
            ) for i in range(len(channels)-1)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class CTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.preBlock = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder = Encoder2D([128, 256, 256])

        self.gap = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=1),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.LayerNorm(256*8*8)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(256*8*8, 128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_attention=False):
        """
        前向传播，支持注意力可视化
        
        参数:
            x: 输入张量，可以是:
               - 标准图像张量 [B, 3, H, W]
               - 单通道CT图像 [B, 1, H, W]
            return_attention: 是否返回注意力图
        """
        # 处理单通道输入
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # 将单通道扩展为3通道
            
        x = self.preBlock(x)
        
        # 存储注意力图
        attention_maps = []
        
        # 在encoder中获取注意力图
        for block in self.encoder.blocks:
            # 获取Bottle2neck2D的GCSAM注意力
            bottle_block = block[0]  # 第一个Bottle2neck2D
            
            # 应用第一个卷积层
            feat = bottle_block.relu(bottle_block.bn1(bottle_block.conv1(x)))
            spx = torch.split(feat, bottle_block.width, 1)
            
            # 处理split特征
            out = None
            for i in range(bottle_block.nums):
                sp = spx[i] if i == 0 or bottle_block.stype == 'stage' else sp + spx[i]
                sp = bottle_block.relu(bottle_block.bns[i](bottle_block.convs[i](sp)))
                out = sp if i == 0 else torch.cat((out, sp), 1)
            
            if bottle_block.scale != 1:
                if bottle_block.stype == 'normal':
                    out = torch.cat((out, spx[bottle_block.nums]), 1)
                elif bottle_block.stype == 'stage':
                    out = torch.cat((out, bottle_block.pool(spx[bottle_block.nums])), 1)
            
            # 获取注意力图
            attention_feat = bottle_block.bn3(bottle_block.conv3(out))
            channel_attention = bottle_block.GCS.channel_attention(attention_feat)
            spatial_attention = bottle_block.GCS.spatial_attention(channel_attention)
            
            # 存储注意力图
            attention_maps.append(spatial_attention)
            
            # 正常前向传播
            x = block(x)
        
        x = self.gap(x)
        x = self.mlp(x)
        
        if return_attention:
            return x, attention_maps
        return x

    def visualize_attention(self, x, original_size=None):
        """
        生成注意力可视化图
        
        参数:
            x: 输入张量
            original_size: 原始图片大小，用于上采样 (H, W)
        
        返回:
            attention_visualizations: 注意力可视化图列表
        """
        _, attention_maps = self.forward(x, return_attention=True)
        attention_visualizations = []
        
        for attention_map in attention_maps:
            # 提取空间注意力权重
            spatial_weights = attention_map.mean(1, keepdim=True)  # [B, 1, H, W]
            
            # 归一化到[0, 1]范围
            spatial_weights = (spatial_weights - spatial_weights.min()) / (spatial_weights.max() - spatial_weights.min() + 1e-8)
            
            # 如果需要，调整大小到原始图片尺寸
            if original_size is not None:
                spatial_weights = F.interpolate(spatial_weights, size=original_size, mode='bilinear', align_corners=False)
            
            attention_visualizations.append(spatial_weights)
        
        return attention_visualizations
# ---- Training & Evaluation ----
def cross_validation(dataset, model_class, num_folds=10, epochs=100, batch_size=64, 
                    criterion=None, optimizer_fn=None, scheduler_fn=None, 
                    device=None, save_dir='./cv_checkpoints', 
                    verbose=True, stratified=True, visualize_attention=False):
    """
    执行K折交叉验证
    
    参数:
        dataset: 数据集
        model_class: 模型类，用于创建新的模型实例
        num_folds: 折数，默认为10
        epochs: 每折训练的轮数
        batch_size: 批量大小
        criterion: 损失函数
        optimizer_fn: 优化器函数，接受model.parameters()作为参数
        scheduler_fn: 学习率调度器函数，接受optimizer作为参数
        device: 训练设备
        save_dir: 保存模型的目录
        verbose: 是否打印详细信息
        stratified: 是否使用分层抽样
    
    返回:
        结果字典，包含每折的评估指标和平均指标
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备交叉验证
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取所有标签用于分层抽样
    all_labels = [label for _, label in dataset]
    
    # 选择K折交叉验证方法
    if stratified:
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        splits = kfold.split(np.arange(len(dataset)), all_labels)
    else:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        splits = kfold.split(np.arange(len(dataset)))
    
    # 存储每折的结果
    fold_results = []
    best_models = []
    
    # 开始K折交叉验证
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*20} Fold {fold+1}/{num_folds} {'='*20}")
        
        # 创建数据加载器
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, 
                                 sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=batch_size,
                               sampler=val_sampler, num_workers=4)
        
        # 创建新的模型实例
        model = model_class().to(device)
        
        # 创建优化器和调度器
        optimizer = optimizer_fn(model.parameters()) if optimizer_fn else \
                   optim.Lookahead(optim.RAdam(model.parameters(), lr=1e-4))
        scheduler = scheduler_fn(optimizer) if scheduler_fn else \
                   torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练跟踪
        best_f1 = 0.0
        fold_save_path = os.path.join(save_dir, f'fold_{fold+1}_best.pth')
        fold_metrics = []
        
        # 创建当前fold的可视化保存目录
        if visualize_attention:
            fold_vis_dir = os.path.join(save_dir, f'fold_{fold+1}_visualizations')
            os.makedirs(fold_vis_dir, exist_ok=True)
        else:
            fold_vis_dir = None

        # 训练循环
        progress_bar = tqdm(range(epochs), desc=f"Fold {fold+1} Training", unit="epoch")
        for epoch in progress_bar:
            train_loss, acc, prec, rec, f1 = train(
                model, train_loader, val_loader, 
                criterion, optimizer, scheduler, device,
                visualize_attention=visualize_attention,
                vis_save_dir=fold_vis_dir,
                current_fold=fold+1,
                current_epoch=epoch+1
            )
            
            # 记录指标
            metrics = {
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_acc': acc,
                'val_prec': prec,
                'val_rec': rec,
                'val_f1': f1
            }
            fold_metrics.append(metrics)
            
            # 更新进度条
            progress_bar.set_description(
                f"Fold {fold+1} | Epoch {epoch+1} | Loss: {train_loss:.4f} | "
                f"Acc: {acc:.4f} | F1: {f1:.4f}"
            )
            
            # 保存最佳模型
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), fold_save_path)
                if verbose:
                    print(f"  🎉 New best F1: {best_f1:.4f}, saved to {fold_save_path}")
        
        # 加载最佳模型进行最终评估
        model.load_state_dict(torch.load(fold_save_path))
        final_acc, final_prec, final_rec, final_f1 = evaluate(model, val_loader, device)
        
        # 记录该折的最终结果
        fold_result = {
            'fold': fold + 1,
            'acc': final_acc,
            'prec': final_prec,
            'rec': final_rec,
            'f1': final_f1
        }
        fold_results.append(fold_result)
        best_models.append((model, fold_save_path, final_f1))
        
        print(f"Fold {fold+1} Final Results: Acc={final_acc:.4f}, Prec={final_prec:.4f}, "
              f"Rec={final_rec:.4f}, F1={final_f1:.4f}")
    
    # 计算平均指标
    avg_acc = sum(r['acc'] for r in fold_results) / num_folds
    avg_prec = sum(r['prec'] for r in fold_results) / num_folds
    avg_rec = sum(r['rec'] for r in fold_results) / num_folds
    avg_f1 = sum(r['f1'] for r in fold_results) / num_folds
    
    # 找出最佳模型
    best_model_idx = max(range(len(best_models)), key=lambda i: best_models[i][2])
    best_model, best_path, best_f1_score = best_models[best_model_idx]
    
    # 保存最佳模型到总体最佳路径
    overall_best_path = os.path.join(save_dir, 'overall_best.pth')
    torch.save(best_model.state_dict(), overall_best_path)
    
    # 打印总结
    print("\n" + "="*50)
    print(f"Cross-Validation Complete - {num_folds} Folds")
    print(f"Average Metrics: Acc={avg_acc:.4f}, Prec={avg_prec:.4f}, "
          f"Rec={avg_rec:.4f}, F1={avg_f1:.4f}")
    print(f"Best Model from Fold {best_model_idx+1} with F1={best_f1_score:.4f}")
    print(f"Best Model saved to {overall_best_path}")
    print("="*50)
    
    # 返回结果
    return {
        'fold_results': fold_results,
        'avg_acc': avg_acc,
        'avg_prec': avg_prec,
        'avg_rec': avg_rec,
        'avg_f1': avg_f1,
        'best_model_fold': best_model_idx + 1,
        'best_model_path': overall_best_path,
        'best_f1': best_f1_score
    }

def evaluate(model, dataloader, device, visualize_attention=False, vis_save_dir=None):
    """
    评估模型性能，可选择是否可视化注意力权重
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        visualize_attention: 是否可视化注意力权重
        vis_save_dir: 可视化结果保存目录
    
    返回:
        acc, prec, rec, f1: 评估指标
    """
    model.eval()
    all_preds, all_labels = [], []
    
    # 如果需要可视化，确保保存目录存在
    if visualize_attention and vis_save_dir:
        os.makedirs(vis_save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 根据是否需要可视化选择不同的前向传播方式
            if visualize_attention:
                outputs, attention_maps = model(imgs, return_attention=True)
                
                # 只为前几个样本生成可视化，避免生成太多图像
                if i < 5:  # 只处理前5个批次
                    for j, (img, label) in enumerate(zip(imgs[:4], labels[:4])):  # 每批次只处理前4个样本
                        # 获取原始图像
                        original_img = img.cpu().numpy().transpose(1, 2, 0)
                        
                        # 生成注意力可视化
                        save_path = os.path.join(vis_save_dir, f'batch{i}_sample{j}_class{label.item()}.png')
                        overlaid_images, _ = generate_attention_visualization(
                            model=model,
                            image_tensor=img,
                            original_image=original_img,
                            save_path=save_path,
                            alpha=0.5
                        )
            else:
                outputs = model(imgs)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, prec, rec, f1

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
          visualize_attention=False, vis_save_dir=None, current_fold=None, current_epoch=None):
    """
    训练一个epoch并评估
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        visualize_attention: 是否可视化注意力权重
        vis_save_dir: 可视化结果保存目录
        current_fold: 当前折数（用于交叉验证）
        current_epoch: 当前epoch数
    """
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    
    # 在验证时生成可视化
    if visualize_attention and vis_save_dir and current_fold is not None and current_epoch is not None:
        # 为每个fold和epoch创建单独的目录
        epoch_vis_dir = os.path.join(vis_save_dir, f'fold_{current_fold}', f'epoch_{current_epoch}')
        os.makedirs(epoch_vis_dir, exist_ok=True)
    else:
        epoch_vis_dir = None
    
    val_acc, val_prec, val_rec, val_f1 = evaluate(
        model, val_loader, device,
        visualize_attention=visualize_attention,
        vis_save_dir=epoch_vis_dir
    )
    
    return total_loss / len(train_loader), val_acc, val_prec, val_rec, val_f1

# ---- Dataset ----
class NpzPatchDataset(Dataset):
    def __init__(self, root_dirs, labels_map, transform=None):
        self.samples = []
        self.transform = transform
        for label, dir_path in root_dirs.items():
            for npz_path in glob.glob(os.path.join(dir_path, '*.npz')):
                self.samples.append((npz_path, labels_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        data = np.load(npz_path)
        img = data['data']  # (H,W,3)
        img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        #img = torch.from_numpy(img).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, label
#-----Facol loss------
class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """
        alpha:         None 或者 shape=[num_classes] 的 Tensor 或 float，
                       二分类时 float 会被转换为 [alpha, 1-alpha] Tensor
        gamma:         聚焦系数 γ，典型值 2.0
        reduction:     'none' | 'mean' | 'sum'
        ignore_index:  忽略标签
        """
        super().__init__()
        # 如果是单一 float（用于二分类），转换为 [α, 1−α] Tensor
        if isinstance(alpha, float):
            self.alpha = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32)
        else:
            self.alpha = alpha  # None 或 已经是 Tensor
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs:  [N, C] logits 或者 [N] logits（二分类）
        targets: [N] LongTensor
        """
        # 二分类分支
        if inputs.dim() == 1 or inputs.size(1) == 1:
            logits = inputs.view(-1)
            targets = targets.view(-1).float()
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none')
            pt = torch.exp(-bce_loss)  # pt = sigmoid(logit) 或 1−pt
            if self.alpha is not None:
                # 对二分类 alpha 张量广播
                alpha_factor = (targets * self.alpha[0] +
                                (1 - targets) * self.alpha[1]).to(bce_loss.device)
                loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
            else:
                loss = (1 - pt) ** self.gamma * bce_loss

        # 多分类分支
        else:
            logp = F.log_softmax(inputs, dim=1)     # [N, C] :contentReference[oaicite:0]{index=0}
            p = torch.exp(logp)                     # [N, C]
            # 取对应类别的 log p_t 和 p_t
            targets = targets.view(-1, 1)
            logpt = logp.gather(1, targets).view(-1)
            pt = p.gather(1, targets).view(-1)
            # 准备 α_t
            if self.alpha is not None:
                # 确保 α 是 Tensor 并在同 device
                if not isinstance(self.alpha, torch.Tensor):
                    raise ValueError("alpha must be Tensor for multiclass")
                at = self.alpha.to(inputs.device).gather(0, targets.view(-1))
            else:
                at = 1.0
            loss = -at * (1 - pt) ** self.gamma * logpt

        # 忽略特定标签
        if self.ignore_index >= 0:
            valid = targets.view(-1) != self.ignore_index
            loss = loss[valid]

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
                 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ResGSCA Training')
    parser.add_argument('--data_dir', type=str, default='./database/3slice/48', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./resgsca_checkpoint/', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--stratified', action='store_true', help='Use stratified sampling for cross-validation')
    parser.add_argument('--cv_save_dir', type=str, default='./cv_checkpoints', help='Directory to save cross-validation checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize_attention', action='store_true', help='Generate attention visualizations')
    parser.add_argument('--vis_save_dir', type=str, default='./attention_visualizations', help='Directory to save attention visualizations')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    if args.cv:
        os.makedirs(args.cv_save_dir, exist_ok=True)
    
    # 数据目录和标签映射
    root_dirs = {
        'cancer': os.path.join(args.data_dir, 'cancer'),
        'nocancer': os.path.join(args.data_dir, 'nocancer'),
    }
    labels_map = {'cancer': 1, 'nocancer': 0}
    
    # 数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    # 加载数据集
    full_dataset = NpzPatchDataset(root_dirs, labels_map, transform)
    
    # 检查标签分布
    label_list = [label for _, label in full_dataset]
    print("标签分布:", Counter(label_list))
    
    # 决定是否使用交叉验证
    if args.cv:
        print(f"\n{'='*20} 开始 {args.folds} 折交叉验证 {'='*20}")
        start_time = time.time()
        
        # 定义优化器和调度器创建函数
        def create_optimizer(params):
            return optim.Lookahead(optim.RAdam(params, lr=args.lr))
        
        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # 执行交叉验证
        # 创建可视化保存目录
        if args.visualize_attention:
            vis_save_dir = os.path.join(args.cv_save_dir, 'attention_maps')
            os.makedirs(vis_save_dir, exist_ok=True)
        else:
            vis_save_dir = None

        cv_results = cross_validation(
            dataset=full_dataset,
            model_class=CTModel,
            num_folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            criterion=FocalLoss(alpha=0.25, gamma=2.0),
            optimizer_fn=create_optimizer,
            scheduler_fn=create_scheduler,
            device=device,
            save_dir=args.cv_save_dir,
            stratified=args.stratified,
            visualize_attention=args.visualize_attention
        )
        
        # 保存交叉验证结果到CSV
        results_df = pd.DataFrame(cv_results['fold_results'])
        results_csv_path = os.path.join(args.cv_save_dir, 'cv_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"交叉验证结果已保存到: {results_csv_path}")
        
        # 打印总运行时间
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n交叉验证总时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        
    else:
        # 常规训练模式（不使用交叉验证）
        print("\n使用常规训练模式（80%训练/20%验证）")
        
        # 分割数据集
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        model = CTModel(num_classes=2).to(device)
        
        # 创建损失函数、优化器和调度器
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.Lookahead(optim.RAdam(model.parameters(), lr=args.lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        print(f"\n{'='*20} 开始常规训练 {'='*20}")
        start_time = time.time()
        
        # 训练模型
        best_f1 = 0.0
        save_path = os.path.join(args.save_dir, 'res2gcsa_best.pth')
        
        progress_bar = tqdm(range(args.epochs), desc="训练进度", unit="epoch")
        for epoch in progress_bar:
            train_loss, acc, prec, rec, f1 = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
            
            # 动态更新进度条描述
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
            )
            
            # 保存最佳模型
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), save_path)
                print(f"  🎉 新的最佳F1: {best_f1:.4f}, 已保存到 {save_path}")
        
        # 保存最终模型
        final_save_path = os.path.join(args.save_dir, 'res2gcsa_final.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f"最终模型已保存到: {final_save_path}")
        
        # 打印总运行时间
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n训练总时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")