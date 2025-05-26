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
from model import CTModel
# ---- Training & Evaluation ----
def cross_validation(dataset, model_class, num_folds=10, epochs=100, batch_size=64, 
                    criterion=None, optimizer_fn=None, scheduler_fn=None, 
                    device=None, save_dir='./cv_checkpoints', 
                    verbose=True, stratified=True):
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
        
        # 训练循环
        progress_bar = tqdm(range(epochs), desc=f"Fold {fold+1} Training", unit="epoch")
        for epoch in progress_bar:
            train_loss, acc, prec, rec, f1 = train(model, train_loader, val_loader, 
                                                  criterion, optimizer, scheduler, device)
            
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

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, prec, rec, f1

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
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
    val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
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
    parser.add_argument('--save_dir', type=str, default='./resgsca_checkpoint/32', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--stratified', action='store_true', help='Use stratified sampling for cross-validation')
    parser.add_argument('--cv_save_dir', type=str, default='./cv_checkpoints', help='Directory to save cross-validation checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
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
            stratified=args.stratified
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