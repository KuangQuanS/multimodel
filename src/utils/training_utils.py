"""
训练和评估工具函数
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

from ..models.fusion_models import ModalityEncoder, CTModel
from ..data.dataset import get_modality_dimensions


def load_encoder(mod, checkpoint_dir=None, latent_dim=256, feature_dims=None):
    """
    创建编码器（不加载预训练权重版本）
    
    Args:
        mod: 模态名称
        checkpoint_dir: 检查点目录（现在忽略）
        latent_dim: 潜在空间维度
        feature_dims: 实际特征维度字典（考虑特征选择后）
    
    Returns:
        ModalityEncoder: 编码器模型
    """
    if feature_dims and mod in feature_dims:
        dim_in = feature_dims[mod]  # 使用特征选择后的维度
    else:
        modality_dims = get_modality_dimensions()
        dim_in = modality_dims.get(mod, 1000)  # 默认维度
    
    encoder = ModalityEncoder(dim_in=dim_in, dim_latent=latent_dim)
    print(f"创建新的 {mod} 编码器，输入维度: {dim_in}")
    return encoder


def train_epoch(model, encoders, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        yb = batch['y'].to(device)
        
        # Extract modality features
        modality_features = [encoders[mod](batch[f'X{i}'].to(device)) 
                           for i, mod in enumerate(encoders.keys()) if f'X{i}' in batch]
        
        zs = torch.stack(modality_features, dim=1)
        ct_data = batch.get('CT', None)
        ct_data = ct_data.to(device) if ct_data is not None else None
        
        logits = model(zs, ct_data)
        loss = criterion(logits, yb)
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


def evaluate(model, encoders, loader, device):
    """评估模型性能"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    error_cases = []

    with torch.no_grad():
        for batch in loader:
            yb = batch['y'].to(device)
            ids = batch.get('id', [f'sample_{i}' for i in range(len(yb))])
            modality_features = [encoders[mod](batch[f'X{i}'].to(device)) 
                               for i, mod in enumerate(encoders.keys()) if f'X{i}' in batch]
            
            zs = torch.stack(modality_features, dim=1)
            ct_data = batch.get('CT', None)
            ct_data = ct_data.to(device) if ct_data is not None else None
            
            logits = model(zs, ct_data)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
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
    
    return {
        'accuracy': accuracy,
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'preds': all_preds,
        'probs': all_probs,
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
    from torch.optim.lr_scheduler import CosineAnnealingLR
    T_max = max(1, args.epochs//3)  # 确保T_max至少为1
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr/10)
    
    # Create criterion with class weights if specified
    if hasattr(args, 'use_class_weights') and args.use_class_weights:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {args.class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    best_auc, best_epoch, best_results = 0, 0, None
    
    # Track training history
    train_losses, train_accs, val_losses, val_accs, val_f1s, val_aucs = [], [], [], [], [], []
    
    for epoch in range(1, args.epochs + 1):
        train_acc, train_loss = train_epoch(model, encoders, train_loader, optimizer, criterion, args.device)
        scheduler.step()
        
        if epoch % args.eval_interval == 0:
            val_results = evaluate(model, encoders, val_loader, args.device)
            fold_str = f" Fold {fold}" if fold is not None else ""
            print(f"Epoch {epoch:3d}{fold_str}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, Val Acc={val_results['accuracy']:.3f}, Val F1={val_results['f1']:.3f}, Val AUC={val_results['auc']:.3f}")
            
            # Record training history
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_accs.append(val_results['accuracy'])
            val_f1s.append(val_results['f1'])
            val_aucs.append(val_results['auc'])
            
            if val_results['auc'] > best_auc:
                best_auc = val_results['auc']
                best_epoch = epoch
                best_results = val_results
                model_path = os.path.join(cv_dir if fold else args.output_dir, 
                                        f"best_model{'_fold_'+str(fold) if fold else ''}.pth")
                torch.save(model.state_dict(), model_path)
    
    # Add training history to results
    if best_results:
        best_results['training_history'] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'val_f1s': val_f1s,
            'val_aucs': val_aucs,
            'eval_epochs': list(range(args.eval_interval, args.epochs + 1, args.eval_interval))
        }
    
    return best_results, best_epoch


def plot_training_curves(results, output_dir=None, fold=None, show=True):
    """
    绘制训练过程中的loss和指标曲线
    
    Args:
        results: 包含training_history的结果字典
        output_dir: 保存图片的目录，为None则不保存
        fold: 折数，用于文件命名
        show: 是否显示图片
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
    axes[0, 0].set_title('Training Loss')
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
        print(f"Training curves saved to: {filepath}")
    
    # 显示图片
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
        print(f"Cross-validation curves saved to: {filepath}")
    
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
    print(f"ROC curve saved to {roc_path}")


def create_ct_model(ct_model_path, device, num_classes=2):
    """创建并加载CT模型"""
    # 暂时使用简化的CT模型，不加载预训练权重
    ct_model = CTModel(num_classes=num_classes)
    print("使用随机初始化的CT模型（预训练权重加载已跳过）")
    return ct_model