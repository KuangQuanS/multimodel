"""
训练和评估工具函数
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from fusion_models import ModalityEncoder, CTModel
from dataset import get_modality_dimensions

# 设置matplotlib为非交互模式，图片保存到文件而不显示
plt.ioff()
plt.switch_backend('Agg')


class FocalLoss(nn.Module):
    """
    论文: Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        if isinstance(self.alpha, (list, np.ndarray, torch.Tensor)):
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha
            
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_encoder(mod, checkpoint_dir=None, latent_dim=256, feature_dims=None):
    if feature_dims and mod in feature_dims:
        dim_in = feature_dims[mod]  # 使用特征选择后的维度
    else:
        modality_dims = get_modality_dimensions()
        dim_in = modality_dims.get(mod, 1000)  # 默认维度
    
    encoder = ModalityEncoder(dim_in=dim_in, dim_latent=latent_dim)

    return encoder


def train_epoch(model, encoders, loader, optimizer, criterion, device, l1_lambda=0.0):
    """训练一个epoch"""
    model.train()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        yb = batch['y'].to(device)
        
        modality_features = [encoders[mod](batch[f'X{i}'].to(device)) 
                           for i, mod in enumerate(encoders.keys()) if f'X{i}' in batch]
        
        zs = torch.stack(modality_features, dim=1)
        ct_data = batch.get('CT', None)
        ct_data = ct_data.to(device) if ct_data is not None else None
        
        logits = model(zs, ct_data)
        loss = criterion(logits, yb)
        
        # L1
        if l1_lambda > 0:
            l1_regularization = 0.0
            for param in model.parameters():
                l1_regularization += torch.sum(torch.abs(param))
            loss += l1_lambda * l1_regularization
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        _, predicted = torch.max(logits.data, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100.0 * correct / total
    
    return accuracy, avg_loss


def evaluate(model, encoders, loader, device, criterion=None):
    """评估模型性能"""
    model.eval()
    # 确保所有编码器也设置为eval模式
    for encoder in encoders.values():
        encoder.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    error_cases = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            yb = batch['y'].to(device)
            batch_size = yb.size(0)
            
            # 处理批次大小为1的情况，避免BatchNorm错误
            if batch_size == 1:
                # 对于单样本批次，使用eval模式避免BatchNorm问题
                model.eval()
                for encoder in encoders.values():
                    encoder.eval()
            
            ids = batch.get('id', [f'sample_{i}' for i in range(len(yb))])
            modality_features = [encoders[mod](batch[f'X{i}'].to(device)) 
                               for i, mod in enumerate(encoders.keys()) if f'X{i}' in batch]
            
            zs = torch.stack(modality_features, dim=1)
            ct_data = batch.get('CT', None)
            ct_data = ct_data.to(device) if ct_data is not None else None
            
            try:
                logits = model(zs, ct_data)
            except RuntimeError as e:
                if "Expected more than 1 value per channel" in str(e):
                    print(f"⚠️  BatchNorm错误 (batch_size={batch_size})，跳过该批次")
                    continue
                else:
                    raise e
            
            # 计算损失
            if criterion:
                loss = criterion(logits, yb)
                total_loss += loss.item()
                num_batches += 1
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)

            # 检查并处理NaN/inf值
            probs_np = probs.cpu().numpy()
            if np.isnan(probs_np).any() or np.isinf(probs_np).any():
                print(f"⚠️  检测到NaN/inf概率值，跳过该批次")
                continue
            
            all_preds.extend(preds.cpu().numpy())
            # 确保取正类概率（如果是二分类）
            if probs.shape[1] >= 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs[:, 0].cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

            for i in range(len(yb)):
                if preds[i] != yb[i]:
                    error_cases.append({
                        'id': ids[i] if hasattr(ids, '__getitem__') else ids,
                        'true_label': yb[i].item(),
                        'pred_label': preds[i].item(),
                        'prob': probs[i, 1].item()
                    })
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # 最终检查：确保没有NaN值传入AUC计算
    all_probs_clean = np.array(all_probs)
    all_labels_clean = np.array(all_labels)
    
    # 如果有NaN，用0.5替换（中性概率）
    nan_mask = np.isnan(all_probs_clean) | np.isinf(all_probs_clean)
    if nan_mask.any():
        print(f"⚠️  发现并修复了 {nan_mask.sum()} 个NaN/inf概率值")
        all_probs_clean[nan_mask] = 0.5
    
    try:
        auc_score = roc_auc_score(all_labels_clean, all_probs_clean)
    except ValueError as e:
        print(f"⚠️  AUC计算失败: {e}，使用默认值0.5")
        auc_score = 0.5
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'f1': f1_score(all_labels, all_preds),
        'auc': auc_score,
        'loss': avg_loss,
        'preds': all_preds,
        'probs': all_probs_clean.tolist(),
        'labels': all_labels,
        'errors': error_cases
    }


def run_single_fold(model, encoders, train_loader, val_loader, args, fold=None, cv_dir=None):
    """运行单折训练"""
    cv_dir = cv_dir or os.path.join(args.output_dir, "cv_results")
    
    # Setup optimizer
    params = []
    params.extend(model.parameters())
    if args.finetune:
        params.extend([p for encoder in encoders.values() for p in encoder.parameters()])
    
    optimizer = optim.AdamW(params, lr=args.lr)

    T_max = max(1, args.epochs//3)  # 确保T_max至少为1
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr/10)
    
    # Automatic class weight calculation (默认启用)
    auto_class_weights = getattr(args, 'auto_class_weights', True)  
    
    if auto_class_weights:
        # Calculate class weights based on training data distribution
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['y'].cpu().numpy().tolist())  # Labels are in 'y' key
        
        # Count class frequencies
        from collections import Counter
        class_counts = Counter(train_labels)
        n_classes = len(class_counts)
        n_samples = len(train_labels)
        
        # Calculate inverse frequency weights: weight = n_samples / (n_classes * count)
        class_weights = []
        for i in range(n_classes):
            weight = n_samples / (n_classes * class_counts[i])
            class_weights.append(weight)
        
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Auto-calculated class weights: {[f'{w:.3f}' for w in class_weights]}")
    
    # Create criterion - Focal Loss or CrossEntropyLoss
    if hasattr(args, 'use_focal_loss') and args.use_focal_loss:
        # Use Focal Loss for imbalanced data
        if auto_class_weights:
            alpha = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
        elif hasattr(args, 'use_class_weights') and args.use_class_weights:
            alpha = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
        else:
            alpha = args.focal_alpha if hasattr(args, 'focal_alpha') else 1.0
            
        gamma = args.focal_gamma if hasattr(args, 'focal_gamma') else 2.0
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        print(f"Using Focal Loss: alpha={alpha}, gamma={gamma}")
        
    elif auto_class_weights or (hasattr(args, 'use_class_weights') and args.use_class_weights):
        # Use CrossEntropyLoss with class weights (auto or manual)
        if auto_class_weights:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
        else:
            weights_tensor = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"Using weighted CrossEntropyLoss with weights: {[f'{w:.3f}' for w in weights_tensor.cpu().tolist()]}")
    else:
        # Standard CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (no class weights)")
    best_auc, best_epoch, best_results = 0, 0, None
    
    # Early stopping variables
    early_stopping_counter = 0
    use_early_stopping = hasattr(args, 'early_stopping') and args.early_stopping
    patience = getattr(args, 'patience', 10)
    min_delta = getattr(args, 'min_delta', 0.001)
    
    if use_early_stopping:
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
    
    # Track training history
    train_losses, train_accs, val_losses, val_accs, val_f1s, val_aucs = [], [], [], [], [], []
    
    # 添加tqdm进度条
    fold_str = f" Fold {fold}" if fold is not None else ""
    epoch_range = tqdm(range(1, args.epochs + 1), 
                       desc=f"Training{fold_str}", 
                       unit="epoch",
                       leave=False)
    
    for epoch in epoch_range:
        train_acc, train_loss = train_epoch(model, encoders, train_loader, optimizer, criterion, args.device, args.l1_lambda)
        scheduler.step()
        
        # 每个epoch都评估，获得更平滑的训练曲线
        val_results = evaluate(model, encoders, val_loader, args.device, criterion)
        
        # 更新tqdm描述信息
        epoch_range.set_postfix({
            'Train_Loss': f'{train_loss:.3f}',
            'Acc': f'{train_acc:.1f}%', 
            'Val_Loss': f'{val_results["loss"]:.3f}',
            'Val_AUC': f'{val_results["auc"]:.3f}'
        })
        
        # 每个epoch都记录训练历史，确保平滑曲线
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_results['loss'])  # 添加验证损失记录
        val_accs.append(val_results['accuracy'])
        val_f1s.append(val_results['f1'])
        val_aucs.append(val_results['auc'])
        
        # Check for improvement and save best model
        if val_results['auc'] > best_auc + min_delta:
            best_auc = val_results['auc']
            best_epoch = epoch
            best_results = val_results
            early_stopping_counter = 0  # Reset counter on improvement
            
            model_path = os.path.join(cv_dir if fold else args.output_dir, 
                                    f"best_model{'_fold_'+str(fold) if fold else ''}.pth")
            torch.save(model.state_dict(), model_path)
            
                
        elif use_early_stopping:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                epoch_range.set_description(f"Early Stop{fold_str}")
                break
    
    # Add training history to results
    if best_results:
        best_results['training_history'] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,  # 添加验证损失历史
            'val_accs': val_accs,
            'val_f1s': val_f1s,
            'val_aucs': val_aucs,
            'eval_epochs': list(range(1, len(train_losses) + 1))  # 每个epoch都有数据
        }
    
    return best_results, best_epoch


def plot_training_curves(results, output_dir=None, fold=None, show=False):
    """
    绘制训练过程中的loss和指标曲线
    
    Args:
        results: 包含training_history的结果字典
        output_dir: 保存图片的目录，为None则不保存
        fold: 折数，用于文件命名
        show: 是否显示图片 (默认False，只保存到文件)
    """
    if 'training_history' not in results:
        print("Warning: No training history found in results")
        return
    
    history = results['training_history']
    epochs = history['eval_epochs']
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training Curves{"" if fold is None else f" - Fold {fold}"}', fontsize=16)
    
    # Loss曲线
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    if 'val_losses' in history:
        axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Accuracy曲线
    axes[0, 1].plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # F1分数曲线
    axes[1, 0].plot(epochs, history['val_f1s'], 'g-', label='Val F1', linewidth=2)
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # AUC曲线
    axes[1, 1].plot(epochs, history['val_aucs'], 'm-', label='Val AUC', linewidth=2)
    axes[1, 1].set_title('AUC Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存图片
    if output_dir:
        results_dir = os.path.join(output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        filename = f"training_curves{'_fold_'+str(fold) if fold else ''}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"📊 训练曲线已保存: {filepath}")
    
    # 显示图片 (默认不显示，只保存)
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_folds_curves(all_results, output_dir=None):
    """
    绘制所有折的训练曲线对比图
    
    Args:
        all_results: 包含所有折结果的列表
        output_dir: 保存目录
    """
    if not all_results or 'training_history' not in all_results[0]:
        print("Warning: No training history found in results")
        return
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Validation Training Curves', fontsize=16)
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']
    
    for fold, results in enumerate(all_results):
        if 'training_history' not in results:
            continue
            
        history = results['training_history']
        epochs = history['eval_epochs']
        color = colors[fold % len(colors)]
        
        # Loss曲线
        axes[0, 0].plot(epochs, history['train_losses'], color=color, 
                       alpha=0.7, label=f'Fold {fold+1}', linewidth=1.5)
        
        # Accuracy曲线
        axes[0, 1].plot(epochs, history['val_accs'], color=color, 
                       alpha=0.7, label=f'Fold {fold+1}', linewidth=1.5)
        
        # F1曲线
        axes[1, 0].plot(epochs, history['val_f1s'], color=color, 
                       alpha=0.7, label=f'Fold {fold+1}', linewidth=1.5)
        
        # AUC曲线
        axes[1, 1].plot(epochs, history['val_aucs'], color=color, 
                       alpha=0.7, label=f'Fold {fold+1}', linewidth=1.5)
    
    # 设置子图标题和标签
    titles = ['Training Loss', 'Validation Accuracy', 'Validation F1', 'Validation AUC']
    ylabels = ['Loss', 'Accuracy', 'F1 Score', 'AUC']
    
    for i, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存图片
    if output_dir:
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        filepath = os.path.join(output_dir, 'results', 'cross_validation_curves.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"📊 交叉验证曲线已保存: {filepath}")
    
    plt.show()


def plot_roc_curve(labels, probs, output_dir, suffix=""):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, f"roc_curve_{suffix}.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 ROC曲线已保存: {roc_path}")


class CTFeatureExtractor(nn.Module):
    """CT特征提取器包装器，返回特征而不是分类结果"""
    def __init__(self, ct_model):
        super().__init__()
        self.ct_model = ct_model
        
    def forward(self, x):
        # 提取特征，在MLP分类器之前停止
        x = self.ct_model.preBlock(x)
        x = self.ct_model.encoder(x)  
        x = self.ct_model.gap(x)  # 输出 [batch, 256*8*8]
        return x


def create_ct_model(ct_model_path, device, num_classes=2):
    """创建并加载CT模型"""
    # 直接使用本地的CTModel
    ct_model = CTModel(num_classes=num_classes)
    
    if ct_model_path and os.path.exists(ct_model_path):
        try:
            # 加载预训练权重
            checkpoint = torch.load(ct_model_path, map_location=device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                ct_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                ct_model.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接加载权重字典
                ct_model.load_state_dict(checkpoint)
        except Exception as e:
            print("no pretrain CTmodel")
    
    # 包装为特征提取器
    ct_feature_extractor = CTFeatureExtractor(ct_model)
    return ct_feature_extractor.to(device)