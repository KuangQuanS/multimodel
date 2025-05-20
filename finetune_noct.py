import argparse, os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from model import ModalityMAE  # ← 直接导入你原来的 MAE 定义
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. GRL
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# 2. 融合 + 分类 + 域判别模块
class FusionWithDomain(nn.Module):
    def __init__(self, dim_latent, n_modalities, nhead, num_layers, num_classes, dropout=0.3):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim_latent, nhead=nhead,
            dim_feedforward=dim_latent*4, batch_first=True, dropout=dropout
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(dim_latent*n_modalities, num_classes)
        self.dom_head = nn.Sequential(
            nn.Linear(dim_latent, dim_latent),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(dim_latent, dim_latent//2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(dim_latent//2, n_modalities)
        )

    def forward(self, zs, domain_mat, lambd=1.0):
        B, M, D = zs.shape
        fused = self.fusion(zs)                  # [B, M, D]
        # 拼接所有模态用于分类
        agg = fused.reshape(B, -1)    
           # [B, M*D]
        logits_cls = self.cls_head(agg)
        # 每个模态分别判别（基于原始 [B, M, D]）
        z_flat = grad_reverse(fused.reshape(B*M, D), lambd)  # [B*M, D]
        dom_labels = domain_mat.reshape(B*M)                 # [B*M]
        logits_dom = self.dom_head(z_flat)

        return logits_cls, logits_dom, dom_labels


# 3. 多模态数据集 + 支持子集选择
class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities, indices=None):
        npz = np.load(npz_path)
        self.Xs_all = [npz[mod] for mod in modalities]
        self.y_all  = npz["y"]
        self.dom_all= npz["domain"]

        if indices is None:
            indices = np.arange(len(self.y_all))
        self.indices = indices
        self.Xs = [x[indices] for x in self.Xs_all]
        self.y  = self.y_all[indices]
        self.dom= self.dom_all[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = [torch.from_numpy(x[idx]).float() for x in self.Xs]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        dom = torch.tensor(self.dom[idx], dtype=torch.long)
        return X, y, dom

# 4. 训练主函数
def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    def get_current_lambda(epoch, max_epoch, args):
        progress = epoch / max_epoch
        if args.lambda_schedule == "linear":
            return args.lambda_max - (args.lambda_max - args.lambda_min) * progress
        elif args.lambda_schedule == "cosine":
            return args.lambda_min + 0.5 * (args.lambda_max - args.lambda_min) * (
                1 + math.cos(progress * math.pi))
        else:  # step
            return args.lambda_max if epoch < max_epoch//2 else args.lambda_min
    
    if args.single_mod is not None:
        # 加载该模态的 MAE encoder
        ckpt = torch.load(
            os.path.join(args.checkpoint_dir, f"{args.single_mod}_best.pth"),
            map_location="cpu"
        )
        dim_in = ckpt['encoder_embed.weight'].shape[1]
        mae = ModalityMAE(
            dim_in=dim_in,
            dim_latent=256,
            encoder_layers=3,
            decoder_layers=2,
            n_heads=8,
            mlp_ratio=4,
            mask_ratio=0.25
        )
        # 只加载 encoder_embed + encoder
        mae.encoder_embed.load_state_dict({
            'weight': ckpt['encoder_embed.weight'],
            'bias':   ckpt['encoder_embed.bias']
        })
        mae.encoder.load_state_dict({ k:v for k,v in ckpt.items() if k.startswith('layers.') })
        mae.eval()  # 不微调
        mae.to(args.device)
        # 构建单模态数据集
        data = np.load(args.data_file)
        X = data[args.single_mod]          # e.g. data['Frag']
        y = data['y']
        # train/val split
        N = len(y)
        idx = np.random.permutation(N)
        split = int(N * args.val_ratio)
        train_idx, val_idx = idx[split:], idx[:split]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        # DataLoader
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
        )
        val_ds   = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),   torch.from_numpy(y_val).long()
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
        # 线性分类头
        clf = nn.Linear(args.latent_dim, args.num_classes).to(args.device)
        optimizer = optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        crit = nn.CrossEntropyLoss()
        # 训练循环
        best_f1 = 0
        for epoch in range(1, args.epochs+1):
            # ——训练——
            clf.train()
            total = correct = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(args.device), yb.to(args.device)
                # 编码
                z = mae.encoder(mae.encoder_embed(Xb).unsqueeze(1)).squeeze(1)
                logits = clf(z)
                loss = crit(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                preds = logits.argmax(1)
                total   += yb.size(0)
                correct += (preds==yb).sum().item()
            acc = correct/total*100

            # ——验证——
            clf.eval()
            v_tot = v_corr = 0
            val_true, val_pred = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    zb = mae.encoder(mae.encoder_embed(Xb.to(args.device)).unsqueeze(1)).squeeze(1)
                    preds = clf(zb).argmax(1)
                    v_tot  += yb.size(0)
                    v_corr += (preds==yb.to(args.device)).sum().item()
                    val_true.append(yb.cpu().numpy()) 
                    val_pred.append(preds.cpu().numpy()) 
            v_acc = v_corr/v_tot*100

            val_y_true = np.concatenate(val_true) 
            val_y_pred = np.concatenate(val_pred)
            f1 = f1_score(val_y_true, val_y_pred, average="macro")
            f1 = f1_score(val_y_true, val_y_pred, average="macro")
            if f1 > best_f1:
                best_f1 = f1
            if epoch % 50 == 0:
                print(f"[{args.single_mod}] Epoch {epoch}: TrainAcc={acc:.2f}% ValAcc={v_acc:.2f}% | F1={f1:.4f}")
        print(f"✅ 单模态 {args.single_mod} 最佳 F1: {best_f1:.2f}%")
        return
    
    # ——多模态，加载每一路 encoder —— 
    encoders = {}
    for mod in args.modalities:
        checkpoint = torch.load(
            os.path.join(args.checkpoint_dir, f"{mod}_best.pth"),
            map_location="cpu",
            weights_only=True
        )
        dim_in = checkpoint['encoder_embed.weight'].shape[1]
        mae = ModalityMAE(
            dim_in=dim_in,
            dim_latent=256,
            encoder_layers=3,
            decoder_layers=2,
            n_heads=8,
            mlp_ratio=4,
            mask_ratio=0.25
        )
        mae.encoder_embed.load_state_dict({
            'weight': checkpoint['encoder_embed.weight'],
            'bias':   checkpoint['encoder_embed.bias']
        })
        mae.encoder.load_state_dict({ k:v for k,v in checkpoint.items() if k.startswith('layers.') })
        if not args.finetune:
            mae.eval()
            for p in mae.parameters(): p.requires_grad = False
        mae.to(args.device)
        encoders[mod] = mae

    # —— 划分训练 / 验证 —— 
    full_npz = np.load(args.data_file)
    N = len(full_npz['y'])
    indices = np.random.permutation(N)
    split = int(N * args.val_ratio)
    val_idx, train_idx = indices[:split], indices[split:]

    train_ds = MultiModalDataset(args.data_file, args.modalities, train_idx)
    val_ds   = MultiModalDataset(args.data_file, args.modalities, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # —— 融合模型 & 优化器 —— 
    fusion = FusionWithDomain(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        nhead=args.nhead,
        num_layers=args.fusion_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(args.device)

    ce = nn.CrossEntropyLoss()
    params = list(fusion.parameters())

    if args.finetune:
        # 只微调每个 encoder 的最后一个 Transformer 层
        for m in encoders.values():
            # 冻结 encoder 里其他所有参数
            for p in m.encoder.parameters():
                p.requires_grad = False
            # 解冻最后一层的参数
            last = m.encoder.layers[-1]
            for p in last.parameters():
                p.requires_grad = True
            # 可选：如果你也想微调输入投影（encoder_embed），也解冻它
            # for p in m.encoder_embed.parameters():
            #    p.requires_grad = True
            params += list(last.parameters())

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs//3, eta_min=args.lr/10)
    best_f1 = 0
    # —— 训练循环 —— 
    for epoch in range(1, args.epochs+1):
        fusion.train()
        sum_cls = sum_dom = total = correct = 0
        for Xlist, y, dom in train_loader:
            y   = y.to(args.device)
            dom = dom.to(args.device)

            zs = torch.stack([
                encoders[mod].encoder(
                    encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                ).squeeze(1)
                for mod, x in zip(args.modalities, Xlist)
            ], dim=1)

            current_lambda = get_current_lambda(epoch, args.epochs, args)
            logits_cls, logits_dom, dom_labels = fusion(zs, dom, lambd=current_lambda)

            loss_cls = ce(logits_cls, y)
            loss_dom = ce(logits_dom, dom_labels)
            loss = loss_cls + args.domain_weight * loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_cls += loss_cls.item() * y.size(0)
            sum_dom += loss_dom.item() * dom_labels.size(0)
            preds = logits_cls.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        scheduler.step()
        # —— 验证 —— 
        fusion.eval()
        val_correct = val_total = 0
        val_y_true = []
        val_y_pred = []
        with torch.no_grad():
            for Xlist, y, dom in val_loader:
                y   = y.to(args.device)
                dom = dom.to(args.device)

                zs = torch.stack([
                    encoders[mod].encoder(
                        encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                    ).squeeze(1)
                    for mod, x in zip(args.modalities, Xlist)
                ], dim=1)

                logits_cls, _, _ = fusion(zs, dom, lambd=0.0)
                preds = logits_cls.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)

                val_y_true.append(y.cpu())
                val_y_pred.append(preds.cpu())

        # 汇总 F1 分数
        val_y_true = torch.cat(val_y_true).numpy()
        val_y_pred = torch.cat(val_y_pred).numpy()
        f1 = f1_score(val_y_true, val_y_pred, average="macro")  # or "binary" 视任务类别而定
        val_acc = val_correct / val_total * 100
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Cls={sum_cls/total:.4f} | Dom={sum_dom/(total*len(args.modalities)):.4f} "
                f"| Acc={correct/total*100:.2f}% | ValAcc={val_acc:.2f}% | F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                print(f"🎉find best f1 {best_f1:.4f},epoch: {epoch}")
                torch.save(fusion.state_dict(), os.path.join(args.output_dir, "best.pth"))
    # —— 保存融合模型 —— 
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(fusion.state_dict(), os.path.join(args.output_dir, "end.pth"))
    print("✅ Saved fusion_with_domain.pth")

# 5. 参数解析
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modalities",   nargs="+")
    p.add_argument("--data_file",     type=str, required=True)
    p.add_argument("--checkpoint_dir",type=str, default="./pretrained")
    p.add_argument("--output_dir",    type=str, default="./fusion_out")
    p.add_argument("--nhead",         type=int, default=8)
    p.add_argument("--mlp_ratio",     type=float, default=4.0)
    p.add_argument("--latent_dim",    type=int, default=256)
    p.add_argument("--fusion_layers", type=int, default=6)
    p.add_argument("--num_classes",   type=int, default=2)
    p.add_argument("--domain_weight", type=float, default=0.25)
    p.add_argument("--single_mod",    type=str, default=None)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--finetune",     action="store_true")
    p.add_argument("--mask_ratio",    type=float, default=0.25)
    p.add_argument("--val_ratio",     type=float, default=0.2)
    p.add_argument("--device",        type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lambda_max", type=float, default=2.0, help="初始最大lambda值（推荐1.0~3.0）")
    p.add_argument("--lambda_min", type=float, default=0.1, help="最终最小lambda值（推荐0~0.5）")
    p.add_argument("--lambda_schedule", type=str, default="linear", choices=["linear", "cosine", "step"],help="lambda衰减策略")
    args = p.parse_args()
    train(args)
