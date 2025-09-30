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
    total = correct = 0
    
    for batch in loader:
        yb = batch['y'].to(device)
        modality_features = [encoders[mod](batch[f'X{i}'].to(device)) 
                           for i, mod in enumerate(encoders.keys()) if f'X{i}' in batch]
        
        zs = torch.stack(modality_features, dim=1)
        ct_data = batch.get('CT', None)
        ct_data = ct_data.to(device) if ct_data is not None else None
        
        optimizer.zero_grad()
        outputs = model(zs, ct_data)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        total += yb.size(0)
        correct += (outputs.argmax(1) == yb).sum().item()
    
    return 100.0 * correct / total


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
    
    return {
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
    criterion = nn.CrossEntropyLoss()
    best_auc, best_epoch, best_results = 0, 0, None
    
    for epoch in range(1, args.epochs + 1):
        train_acc = train_epoch(model, encoders, train_loader, optimizer, criterion, args.device)
        scheduler.step()
        
        if epoch % args.eval_interval == 0:
            val_results = evaluate(model, encoders, val_loader, args.device)
            fold_str = f" Fold {fold}" if fold is not None else ""
            print(f"Epoch {epoch}{fold_str}: Train Acc={train_acc:.1f}%, Val F1={val_results['f1']:.3f}, Val AUC={val_results['auc']:.3f}")
            
            if val_results['auc'] > best_auc:
                best_auc = val_results['auc']
                best_epoch = epoch
                best_results = val_results
                model_path = os.path.join(cv_dir if fold else args.output_dir, 
                                        f"best_model{'_fold_'+str(fold) if fold else ''}.pth")
                torch.save(model.state_dict(), model_path)
    
    # 调试信息
    print(f"Fold完成，best_results是否为None: {best_results is None}, best_auc: {best_auc}")
    return best_results, best_epoch


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