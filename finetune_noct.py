import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR

class ModalityMAE(nn.Module):
    """
    单模态的 Masked Autoencoder
    输入: x ∈ R^(B×D_in)
    输出: 重建 x_hat ∈ R^(B×D_in)
    Encoder 输出 latent z ∈ R^(B×D_latent)
    """
    def __init__(self, dim_in, dim_latent=256, encoder_layers=4, decoder_layers=2,
                 n_heads=8, mlp_ratio=4., mask_ratio=0.3):
        super().__init__()
        self.mask_ratio = mask_ratio
        # 投影到 latent 维度
        self.encoder_embed = nn.Linear(dim_in, dim_latent)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_latent, nhead=n_heads, dim_feedforward=int(dim_latent*mlp_ratio), batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        # Decoder: 从 latent 恢复到原始维度
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim_latent, nhead=n_heads, dim_feedforward=int(dim_latent*mlp_ratio), batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_pred = nn.Linear(dim_latent, dim_in)


    def random_mask(self, x):
        B, D = x.shape
        mask = torch.rand(B, D, device=x.device) > self.mask_ratio
        return mask

    def forward(self, x):
        # x: B×D_in
        mask = self.random_mask(x)        # B×D_in，True 表示保留
        x_masked = x * mask.float()
        
        # Encoder
        z = self.encoder_embed(x_masked)  # B×D_latent
        z = z.unsqueeze(1)                # B×1×D_latent
        z = self.encoder(z)               # B×1×D_latent
        z = z.squeeze(1)                  # B×D_latent

        # Decoder 重建（注意：现在是在 latent 空间做 transformer）
        z = z.unsqueeze(1)                # B×1×D_latent
        z = self.decoder(z)               # B×1×D_latent
        z = z.squeeze(1)                  # B×D_latent
        x_rec = self.decoder_pred(z)      # B×D_in

        return x_rec, mask
# -----------------------------
# 1) 辅助函数
# -----------------------------
def load_encoder(mod, args):
    """加载预训练的编码器"""
    ckpt = torch.load(os.path.join(args.checkpoint_dir, f"{mod}_best.pth"),
                      map_location="cpu", weights_only=True)
    dim_in = ckpt['encoder_embed.weight'].shape[1]
    mae = ModalityMAE(
        dim_in=dim_in,
        dim_latent=args.latent_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        n_heads=args.nhead,
        mlp_ratio=args.mlp_ratio,
        mask_ratio=args.mask_ratio
    )
    mae.encoder_embed.load_state_dict({
        'weight': ckpt['encoder_embed.weight'],
        'bias':   ckpt['encoder_embed.bias']
    })
    mae.encoder.load_state_dict({k:v for k,v in ckpt.items() if k.startswith('layers.')})
    return mae

def prepare_data(args, is_single_modal=False):
    """准备数据加载器"""
    data = np.load(args.data_file)
    N = len(data['y'])
    idx = np.random.permutation(N)
    split = int(N * args.val_ratio)
    train_idx, val_idx = idx[split:], idx[:split]
    
    if is_single_modal:
        X = data[args.single_mod]
        y = data['y']
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).long()
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
    else:
        train_ds = MultiModalDataset(args.data_file, args.modalities, train_idx)
        val_ds = MultiModalDataset(args.data_file, args.modalities, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_epoch(fusion, encoders, train_loader, optimizer, criterion, args, scheduler=None):
    """训练一个epoch"""
    fusion.train()
    total = correct = 0
    
    for batch in train_loader:
        if args.single_mod is not None:
            # 单模态模式
            Xb, yb = batch
            Xb, yb = Xb.to(args.device), yb.to(args.device)
            z = encoders[args.single_mod].encoder(
                encoders[args.single_mod].encoder_embed(Xb).unsqueeze(1)
            ).squeeze(1)
            zs = z.unsqueeze(1)
        else:
            # 多模态模式
            Xlist, yb = batch
            yb = yb.to(args.device)
            zs = torch.stack([
                encoders[mod].encoder(
                    encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                ).squeeze(1)
                for mod, x in zip(args.modalities, Xlist)
            ], dim=1)
        
        logits = fusion(zs)
        loss = criterion(logits, yb)
        # L1正则化
        l1_reg = torch.tensor(0., device=args.device)
        for p in fusion.parameters():
            l1_reg += torch.norm(p, 1)
        loss = loss + args.l1_lambda * l1_reg
        # 优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
    
    if scheduler is not None:
        scheduler.step()
    
    return correct / total * 100

def evaluate(fusion, encoders, val_loader, args):
    """评估模型"""
    fusion.eval()
    val_true, val_pred, val_probs = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            if args.single_mod is not None:
                # 单模态模式
                Xb, yb = batch
                z = encoders[args.single_mod].encoder(
                    encoders[args.single_mod].encoder_embed(Xb.to(args.device)).unsqueeze(1)
                ).squeeze(1)
                zs = z.unsqueeze(1)
            else:
                # 多模态模式
                Xlist, yb = batch
                zs = torch.stack([
                    encoders[mod].encoder(
                        encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                    ).squeeze(1)
                    for mod, x in zip(args.modalities, Xlist)
                ], dim=1)
            
            logits = fusion(zs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            
            if args.single_mod is not None:
                val_true.extend(yb.numpy())
            else:
                val_true.extend(yb.cpu().numpy())
            val_pred.extend(preds)
            val_probs.extend(probs)
    
    val_true = np.array(val_true)
    val_pred = np.array(val_pred)
    val_probs = np.array(val_probs)
    
    val_acc = (val_true == val_pred).mean() * 100
    f1 = f1_score(val_true, val_pred, average="macro")
    auc = roc_auc_score(val_true, val_probs)
    
    return val_acc, f1, auc

# -----------------------------
# 2) 多模态融合 + 分类模块
# -----------------------------
class Fusion(nn.Module):
    def __init__(self, dim_latent, n_modalities, nhead, num_layers, num_classes, dropout=0.3):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim_latent, nhead=nhead,
            dim_feedforward=dim_latent*4, batch_first=True, dropout=dropout
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(dim_latent * n_modalities)
        self.dropout  = nn.Dropout(p=0.5)  
        # 分类头，输入维度 dim_latent * n_modalities
        self.cls_head = nn.Linear(dim_latent * n_modalities, num_classes)

    def forward(self, zs):
        # zs: [B, M, D]
        B, M, D = zs.shape
        fused = self.fusion(zs)         # [B, M, D]
        agg = fused.reshape(B, M*D)     # [B, M*D]
        agg = self.bn(agg)
        agg = self.dropout(agg)
        logits = self.cls_head(agg)     # [B, num_classes]
        return logits

# -----------------------------
# 2) Dataset
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities, indices=None):
        npz = np.load(npz_path)
        self.Xs_all = [npz[mod] for mod in modalities]
        self.y_all  = npz["y"]
        if indices is None:
            indices = np.arange(len(self.y_all))
        self.indices = indices
        self.Xs = [x[indices] for x in self.Xs_all]
        self.y  = self.y_all[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = [torch.from_numpy(x[idx]).float() for x in self.Xs]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y

# -----------------------------
# 3) 训练逻辑
# -----------------------------
def train_one_fold(fold_idx, train_idx, val_idx, encoders, n_modalities, args):
    """训练单个折"""
    # 准备数据
    if args.single_mod is not None:
        data = np.load(args.data_file)
        X = data[args.single_mod]
        y = data['y']
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).long()
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
    else:
        train_ds = MultiModalDataset(args.data_file, args.modalities, train_idx)
        val_ds = MultiModalDataset(args.data_file, args.modalities, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 创建融合模型
    fusion = Fusion(
        dim_latent=args.latent_dim,
        n_modalities=n_modalities,
        nhead=args.nhead,
        num_layers=args.fusion_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(args.device)
    
    # 设置优化器和损失函数
    params = list(fusion.parameters())
    if args.finetune and args.single_mod is None:
        for m in encoders.values():
            for p in m.encoder.parameters(): p.requires_grad=False
            last = m.encoder.layers[-1]
            for p in last.parameters(): p.requires_grad=True
            params += list(last.parameters())
    
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = None if args.single_mod is not None else CosineAnnealingLR(optimizer, T_max=args.epochs//3, eta_min=args.lr/10)
    
    # 训练循环
    best_f1 = best_auc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs+1):
        # 训练
        train_acc = train_epoch(fusion, encoders, train_loader, optimizer, criterion, args, scheduler)
        
        # 评估
        val_acc, f1, auc = evaluate(fusion, encoders, val_loader, args)
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_auc = auc
            best_epoch = epoch
            os.makedirs(args.output_dir, exist_ok=True)
            
            if args.single_mod is not None:
                save_path = os.path.join(args.output_dir, f"{args.single_mod}_fusion_fold{fold_idx}_best.pth")
            else:
                save_path = os.path.join(args.output_dir, f"fold{fold_idx}_best.pth")
            
            torch.save(fusion.state_dict(), save_path)
        
        # 打印进度
        if epoch % 50 == 0 or epoch == args.epochs:
            prefix = f"[{args.single_mod}] " if args.single_mod is not None else ""
            print(f"{prefix}Fold {fold_idx}, Epoch {epoch}: "
                  f"TrAcc={train_acc:.2f}% VaAcc={val_acc:.2f}% "
                  f"F1={f1:.4f} AUC={auc:.4f}")
    
    return best_f1, best_auc, best_epoch

def train(args):
    """统一的训练函数，支持单模态和多模态，使用十折交叉验证"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 确定模式和加载编码器
    encoders = {}
    if args.single_mod is not None:
        # 单模态模式
        mae = load_encoder(args.single_mod, args)
        mae.eval()
        mae.to(args.device)
        encoders[args.single_mod] = mae
        n_modalities = 1
        print(f"单模态模式: {args.single_mod}")
    else:
        # 多模态模式
        for mod in args.modalities:
            mae = load_encoder(mod, args)
            if not args.finetune:
                mae.eval()
                for p in mae.parameters(): p.requires_grad=False
            mae.to(args.device)
            encoders[mod] = mae
        n_modalities = len(args.modalities)
        print(f"多模态模式: {args.modalities}")
    
    # 准备十折交叉验证
    data = np.load(args.data_file)
    y = data['y']
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # 存储每折的结果
    fold_results = []
    
    # 十折交叉验证
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"\n===== 开始第 {fold_idx}/10 折 =====")
        print(f"训练集样本比例: {np.bincount(y[train_idx])}")
        print(f"验证集样本比例: {np.bincount(y[val_idx])}")
        
        # 训练当前折
        best_f1, best_auc, best_epoch = train_one_fold(
            fold_idx, train_idx, val_idx, encoders, n_modalities, args
        )
        
        fold_results.append({
            'fold': fold_idx,
            'f1': best_f1,
            'auc': best_auc,
            'epoch': best_epoch
        })
        print(f"第 {fold_idx} 折最佳结果: F1={best_f1:.4f}, AUC={best_auc:.4f}, Epoch={best_epoch}")
    
    # 计算并打印总体结果
    f1_scores = [res['f1'] for res in fold_results]
    auc_scores = [res['auc'] for res in fold_results]
    
    mode_str = f"单模态 `{args.single_mod}`" if args.single_mod is not None else "多模态"
    print(f"\n===== {mode_str}十折交叉验证结果 =====")
    print(f"平均 F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"平均 AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    
    # 保存详细结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, 
                              f"{'single_'+args.single_mod if args.single_mod else 'multi'}_cv_results.txt")
    
    with open(result_file, 'w') as f:
        f.write(f"{mode_str}十折交叉验证结果:\n")
        f.write(f"平均 F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
        f.write(f"平均 AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n\n")
        f.write("各折详细结果:\n")
        for res in fold_results:
            f.write(f"Fold {res['fold']}: F1={res['f1']:.4f}, AUC={res['auc']:.4f}, "
                   f"Best Epoch={res['epoch']}\n")
    
    print(f"\n✅ 详细结果已保存至: {result_file}")

def evaluate_test_set(fusion, encoders, test_loader, args):
    """在测试集上评估模型"""
    fusion.eval()
    test_true, test_pred, test_probs = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            if args.single_mod is not None:
                # 单模态模式
                Xb, yb = batch
                z = encoders[args.single_mod].encoder(
                    encoders[args.single_mod].encoder_embed(Xb.to(args.device)).unsqueeze(1)
                ).squeeze(1)
                zs = z.unsqueeze(1)
            else:
                # 多模态模式
                Xlist, yb = batch
                zs = torch.stack([
                    encoders[mod].encoder(
                        encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                    ).squeeze(1)
                    for mod, x in zip(args.modalities, Xlist)
                ], dim=1)
            
            logits = fusion(zs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            
            test_true.extend(yb.numpy() if args.single_mod is not None else yb.cpu().numpy())
            test_pred.extend(preds)
            test_probs.extend(probs)
    
    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    test_probs = np.array(test_probs)
    
    acc = (test_true == test_pred).mean() * 100
    f1 = f1_score(test_true, test_pred, average="macro")
    auc = roc_auc_score(test_true, test_probs)
    
    return acc, f1, auc

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--modalities",   nargs="+")
    p.add_argument("--data_file",     type=str, required=True, help="训练数据文件路径")
    p.add_argument("--test_file",     type=str, required=True, help="测试数据文件路径")
    p.add_argument("--checkpoint_dir",type=str, default="./pretrained")
    p.add_argument("--output_dir",    type=str, default="./fusion_out")
    p.add_argument("--nhead",         type=int, default=8)
    p.add_argument("--mlp_ratio",     type=float, default=4.0)
    p.add_argument("--latent_dim",    type=int, default=64)
    p.add_argument("--fusion_layers", type=int, default=6)
    p.add_argument("--num_classes",   type=int, default=2)
    p.add_argument("--single_mod",    type=str,   default=None,
                   help="指定单模态名称（如 Frag），启用单模态模式）")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--l1_lambda",  type=float, default=1e-4)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--finetune",     action="store_true",
                   help="只微调每个 encoder 最后一层（否则全冻结）")
    p.add_argument("--encoder_layers",type=int, default=2)
    p.add_argument("--decoder_layers",type=int, default=2)
    p.add_argument("--mask_ratio",    type=float, default=0.25)
    p.add_argument("--val_ratio",     type=float, default=0.2, help="验证集比例")
    p.add_argument("--device",        type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    
    # 训练模型
    train(args)
    
    # 在测试集上评估最佳模型
    print("\n===== 在测试集上评估最佳模型 =====")
    
    # 加载最佳模型（这里使用最后一折的最佳模型）
    best_fold = 10
    if args.single_mod is not None:
        model_path = os.path.join(args.output_dir, f"{args.single_mod}_fusion_fold{best_fold}_best.pth")
    else:
        model_path = os.path.join(args.output_dir, f"fold{best_fold}_best.pth")
    
    fusion = Fusion(
        dim_latent=args.latent_dim,
        n_modalities=1 if args.single_mod is not None else len(args.modalities),
        nhead=args.nhead,
        num_layers=args.fusion_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(args.device)
    fusion.load_state_dict(torch.load(model_path, weights_only=True))
    
    # 准备测试数据
    if args.single_mod is not None:
        test_data = np.load(args.test_file)
        X_test = torch.from_numpy(test_data[args.single_mod]).float()
        y_test = torch.from_numpy(test_data['y']).long()
        test_ds = torch.utils.data.TensorDataset(X_test, y_test)
    else:
        test_ds = MultiModalDataset(args.test_file, args.modalities)
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # 加载编码器
    encoders = {}
    if args.single_mod is not None:
        mae = load_encoder(args.single_mod, args)
        mae.eval()
        mae.to(args.device)
        encoders[args.single_mod] = mae
    else:
        for mod in args.modalities:
            mae = load_encoder(mod, args)
            mae.eval()
            mae.to(args.device)
            encoders[mod] = mae
    
    # 评估测试集
    test_acc, test_f1, test_auc = evaluate_test_set(fusion, encoders, test_loader, args)
    print(f"测试集结果 - Acc: {test_acc:.2f}%, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")