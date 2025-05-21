import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from model import ModalityMAE
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# -----------------------------
# 1) 辅助函数
# -----------------------------
def load_encoder(mod, args):
    """加载预训练的编码器"""
    ckpt = torch.load(os.path.join(args.checkpoint_dir, f"{mod}_best.pth"),
                      map_location="cpu")
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
        # 分类头，输入维度 dim_latent * n_modalities
        self.cls_head = nn.Linear(dim_latent * n_modalities, num_classes)

    def forward(self, zs):
        # zs: [B, M, D]
        B, M, D = zs.shape
        fused = self.fusion(zs)         # [B, M, D]
        agg = fused.reshape(B, M*D)     # [B, M*D]
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
def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    # —— 单模态模式 —— 
    if args.single_mod is not None:
        # 1) 加载 MAE encoder
        ckpt = torch.load(os.path.join(args.checkpoint_dir, f"{args.single_mod}_best.pth"),
                          map_location="cpu")
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
        mae.eval(); mae.to(args.device)

        # 2) 构建单模态 DataLoader
        data = np.load(args.data_file)
        X = data[args.single_mod]
        y = data['y']
        N = len(y)
        idx = np.random.permutation(N)
        split = int(N*args.val_ratio)
        train_idx, val_idx = idx[split:], idx[:split]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
        )
        val_ds   = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),   torch.from_numpy(y_val).long()
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

        # 3) 构建 Fusion（M=1）并训练
        fusion = Fusion(
            dim_latent=args.latent_dim,
            n_modalities=1,
            nhead=args.nhead,
            num_layers=args.fusion_layers,
            num_classes=args.num_classes,
            dropout=args.dropout
        ).to(args.device)

        optimizer = optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        crit = nn.CrossEntropyLoss()

        best_f1 = 0.0
        for epoch in range(1, args.epochs+1):
            fusion.train()
            total=correct=0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(args.device), yb.to(args.device)
                z = mae.encoder(mae.encoder_embed(Xb).unsqueeze(1)).squeeze(1)  # [B,D]
                zs = z.unsqueeze(1)  # [B,1,D]
                logits = fusion(zs)
                loss = crit(logits, yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                preds = logits.argmax(1)
                total   += yb.size(0)
                correct += (preds==yb).sum().item()
            train_acc = correct/total*100

            fusion.eval()
            v_tot=v_corr=0
            val_true,val_pred = [],[]
            with torch.no_grad():
                for Xb,yb in val_loader:
                    zb = mae.encoder(mae.encoder_embed(Xb.to(args.device)).unsqueeze(1)).squeeze(1)
                    logits = fusion(zb.unsqueeze(1))
                    preds  = logits.argmax(1)
                    v_tot  += yb.size(0)
                    v_corr += (preds==yb.to(args.device)).sum().item()
                    val_true.append(yb.numpy()); val_pred.append(preds.cpu().numpy())
            val_acc = v_corr/v_tot*100
            y_true = np.concatenate(val_true); y_pred = np.concatenate(val_pred)
            f1 = f1_score(y_true, y_pred, average="macro")

            if f1>best_f1:
                best_f1 = f1
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(fusion.state_dict(),
                           os.path.join(args.output_dir, f"{args.single_mod}_fusion_best.pth"))
            if epoch%20==0:
                print(f"[{args.single_mod}] Epo {epoch}: TrAcc={train_acc:.2f}% VaAcc={val_acc:.2f}% F1={f1:.4f}")

        print(f"✅ 单模态 `{args.single_mod}` 最佳 F1: {best_f1:.4f}")
        return

    # —— 多模态模式 —— 
    # 1) 加载所有 MAE encoder
    encoders={}
    for mod in args.modalities:
        ckpt = torch.load(os.path.join(args.checkpoint_dir, f"{mod}_best.pth"),
                          map_location="cpu")
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
        if not args.finetune:
            mae.eval()
            for p in mae.parameters(): p.requires_grad=False
        mae.to(args.device)
        encoders[mod]=mae

    # 2) 划分数据 & DataLoader
    arr = np.load(args.data_file)
    N   = len(arr['y'])
    idx = np.random.permutation(N)
    split = int(N*args.val_ratio)
    val_idx,train_idx = idx[:split], idx[split:]
    train_ds = MultiModalDataset(args.data_file, args.modalities, train_idx)
    val_ds   = MultiModalDataset(args.data_file, args.modalities, val_idx)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
    val_loader  =DataLoader(val_ds,  batch_size=args.batch_size,shuffle=False)

    # 3) 构建 & 训练 Fusion
    fusion = Fusion(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        nhead=args.nhead,
        num_layers=args.fusion_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(args.device)
    crit = nn.CrossEntropyLoss()
    params = list(fusion.parameters())
    # 如启用 --finetune，则只解冻每个 encoder 最后一层
    if args.finetune:
        for m in encoders.values():
            for p in m.encoder.parameters(): p.requires_grad=False
            last = m.encoder.layers[-1]
            for p in last.parameters(): p.requires_grad=True
            params += list(last.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs//3, eta_min=args.lr/10)

    best_f1=0.0
    for epoch in range(1, args.epochs+1):
        fusion.train()
        total=correct=0
        for Xlist, y in train_loader:
            y = y.to(args.device)
            # 5 路编码堆叠
            zs = torch.stack([
                encoders[mod].encoder(
                    encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                ).squeeze(1)
                for mod,x in zip(args.modalities,Xlist)
            ], dim=1)  # [B,M,D]
            logits = fusion(zs)
            loss = crit(logits,y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(1)
            total   += y.size(0)
            correct += (preds==y).sum().item()
        train_acc = correct/total*100
        scheduler.step()

        fusion.eval()
        v_tot=v_corr=0
        val_true,val_pred = [],[]
        with torch.no_grad():
            for Xlist,y in val_loader:
                y=y.to(args.device)
                zs = torch.stack([
                    encoders[mod].encoder(
                        encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                    ).squeeze(1)
                    for mod,x in zip(args.modalities,Xlist)
                ], dim=1)
                logits = fusion(zs)
                preds = logits.argmax(1)
                v_tot  += y.size(0)
                v_corr += (preds==y).sum().item()
                val_true.append(y.cpu().numpy())
                val_pred.append(preds.cpu().numpy())
        val_acc = v_corr/v_tot*100
        y_true=np.concatenate(val_true); y_pred=np.concatenate(val_pred)
        f1 = f1_score(y_true,y_pred,average="macro")
        if f1>best_f1:
            best_f1=f1
            torch.save(fusion.state_dict(), os.path.join(args.output_dir, "best.pth"))
        if epoch%20==0:
            print(f"Epo{epoch}: TrAcc={train_acc:.2f}% VaAcc={val_acc:.2f}% F1={f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(fusion.state_dict(), os.path.join(args.output_dir, "end.pth"))
    print(f"✅ 多模态最佳 F1: {best_f1:.4f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--modalities",   nargs="+")
    p.add_argument("--data_file",     type=str, required=True)
    p.add_argument("--checkpoint_dir",type=str, default="./pretrained")
    p.add_argument("--output_dir",    type=str, default="./fusion_out")
    p.add_argument("--nhead",         type=int, default=8)
    p.add_argument("--mlp_ratio",     type=float, default=4.0)
    p.add_argument("--latent_dim",    type=int, default=256)
    p.add_argument("--fusion_layers", type=int, default=6)
    p.add_argument("--num_classes",   type=int, default=2)
    p.add_argument("--single_mod",    type=str,   default=None,
                   help="指定单模态名称（如 Frag），启用单模态模式）")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--finetune",     action="store_true",
                   help="只微调每个 encoder 最后一层（否则全冻结）")
    p.add_argument("--encoder_layers",type=int, default=3)
    p.add_argument("--decoder_layers",type=int, default=2)
    p.add_argument("--mask_ratio",    type=float, default=0.25)
    p.add_argument("--val_ratio",     type=float, default=0.3)
    p.add_argument("--device",        type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    train(args)