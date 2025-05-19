import argparse, os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from model import FusionCrossAttn

def finetune_crossattn(args):
    # 加载所有 modality 的 encoder
    encoders = []
    for mod in args.modalities:
        encoder = ModalityMAE(dim_in=0).encoder_embed  # 只占位，后面加载权重
        state = torch.load(os.path.join(args.pretrain_dir, f"{mod}_encoder.pth"), map_location=args.device)
        encoder = nn.Linear( args.latent_dim, args.latent_dim )  # placeholder
        encoder.load_state_dict(state)
        encoders.append(encoder.to(args.device))
        encoder.eval()

    # 读取所有 modality 的特征矩阵
    Xs = [ np.load(os.path.join(args.data_root, f"{mod}.npy")) for mod in args.modalities ]
    y  = np.load(os.path.join(args.data_root, "labels.npy"))
    Xs = [ torch.from_numpy(x).float().to(args.device) for x in Xs ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(Xs[0], y)):
        # 构造融合特征
        Zs = []
        for x, encoder in zip(Xs, encoders):
            z = encoder(x)                  # N×D_latent
            Zs.append(z)
        Z = torch.stack(Zs, dim=1)         # N×M×D_latent

        # 划分 fold
        Z_train = Z[train_idx]; y_train = y[train_idx]
        Z_val   = Z[val_idx];   y_val   = y[val_idx]

        # DataLoader
        train_ds = TensorDataset(Z_train, torch.from_numpy(y_train))
        val_ds   = TensorDataset(Z_val,   torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

        # 定义融合模型
        model = FusionCrossAttn(dim_latent=args.latent_dim,
                                num_modalities=len(args.modalities),
                                fusion_layers=args.fusion_layers,
                                n_heads=args.n_heads,
                                mlp_ratio=args.mlp_ratio,
                                num_classes=args.num_classes,
                                dropout=args.dropout).to(args.device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.class_weight], device=args.device))

        # 训练
        best_auc = 0
        for epoch in range(args.epochs):
            model.train()
            for Zb, yb in train_loader:
                logits = model(Zb)
                loss = criterion(logits, yb.to(args.device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # 验证
            model.eval()
            all_pred = []
            all_y    = []
            with torch.no_grad():
                for Zb, yb in val_loader:
                    logits = model(Zb)
                    preds  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                    all_pred.extend(preds)
                    all_y.extend(yb.numpy())
            auc = roc_auc_score(all_y, all_pred)
            f1  = f1_score(all_y, (np.array(all_pred)>0.5).astype(int))
            print(f"[Fold {fold+1}] Epoch {epoch+1}/{args.epochs}  AUC: {auc:.4f}  F1: {f1:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_fold{fold+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities",   nargs="+", required=True)
    parser.add_argument("--data_root",    type=str, required=True)
    parser.add_argument("--pretrain_dir", type=str, default="./pretrained")
    parser.add_argument("--save_dir",     type=str, default="./finetuned")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--latent_dim",   type=int, default=256)
    parser.add_argument("--fusion_layers",type=int, default=2)
    parser.add_argument("--n_heads",      type=int, default=8)
    parser.add_argument("--mlp_ratio",    type=float, default=4.0)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--num_classes",  type=int, default=2)
    parser.add_argument("--class_weight", type=float, default= (500-110)/110 )  # 平衡正负样本
    parser.add_argument("--device",       type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    finetune_crossattn(args)
