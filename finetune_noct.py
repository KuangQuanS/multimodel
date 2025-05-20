import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from model import ModalityMAE  # ← 直接导入你原来的 MAE 定义

# -----------------------------
# 1) GRL
# -----------------------------
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

# -----------------------------
# 2) 融合 + 分类 + 域判别
# -----------------------------
class FusionWithDomain(nn.Module):
    def __init__(self, dim_latent, n_modalities, nhead, num_layers, num_classes, dropout=0.3):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim_latent, nhead=nhead,
            dim_feedforward=dim_latent*4, batch_first=True, dropout=dropout
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(dim_latent, num_classes)
        self.dom_head = nn.Sequential(
            nn.Linear(dim_latent, dim_latent//2),
            nn.ReLU(),
            nn.Linear(dim_latent//2, n_modalities)
        )

    def forward(self, zs, domain_mat, lambd=1.0):
        B, M, D = zs.shape
        fused = self.fusion(zs)          # [B, M, D]
        agg = fused.mean(dim=1)          # [B, D]
        logits_cls = self.cls_head(agg)  # [B, num_classes]

        # 域判别
        z_flat = agg.unsqueeze(1).expand(-1, M, -1).reshape(B*M, D)
        dom_labels = domain_mat.reshape(B*M)
        z_rev = grad_reverse(z_flat, lambd)
        logits_dom = self.dom_head(z_rev)  # [B*M, M]

        return logits_cls, logits_dom, dom_labels

# -----------------------------
# 3) Dataset
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities):
        npz = np.load(npz_path)
        self.Xs = [npz[mod] for mod in modalities]
        self.y  = npz["y"]
        self.dom= npz["domain"]
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        X = [torch.from_numpy(x[idx]).float() for x in self.Xs]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        dom = torch.tensor(self.dom[idx], dtype=torch.long)
        return X, y, dom

# -----------------------------
# 4) 训练流程
# -----------------------------
def train(args):
    # —— 加载 5 路 MAE encoder —— 
    encoders = {}
    for mod in args.modalities:
        # 和预训练时完全一致的超参

        # 从 checkpoint 加载 state_dict（你之前保存的 encoder.state_dict()）
        checkpoint = torch.load(
            os.path.join(args.checkpoint_dir, f"{mod}_best.pth"),
            map_location="cpu",
            weights_only=True
        )
        dim_in = checkpoint['encoder_embed.weight'].shape[1]
        mae = ModalityMAE(
            dim_in=dim_in,
            dim_latent=args.latent_dim,
            encoder_layers=args.enc_layers,
            decoder_layers=args.dec_layers,
            n_heads=args.nhead,
            mlp_ratio=args.mlp_ratio,
            mask_ratio=args.mask_ratio
        )
                   # TransformerEncoder 部分
        mae.encoder_embed.load_state_dict({
            'weight': checkpoint['encoder_embed.weight'],
            'bias':   checkpoint['encoder_embed.bias']
        })
        mae.encoder.load_state_dict({ k:v for k,v in checkpoint.items() if k.startswith('layers.') })
        # 冻结或微调
        if not args.fine_tune:
            mae.eval()
            for p in mae.parameters(): p.requires_grad = False
        mae.to(args.device)
        encoders[mod] = mae

    # —— DataLoader —— 
    ds = MultiModalDataset(args.data_file, args.modalities)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

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
    if args.fine_tune:
        for m in encoders.values():
            params += list(m.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # —— 训练循环 —— 
    for epoch in range(1, args.epochs+1):
        fusion.train()
        sum_cls = sum_dom = total = correct = 0
        for Xlist, y, dom in loader:
            y   = y.to(args.device)
            dom = dom.to(args.device)

            # 1) 五路编码
            zs = torch.stack([
                encoders[mod].encoder(
                    encoders[mod].encoder_embed(x.to(args.device)).unsqueeze(1)
                ).squeeze(1)
                for mod, x in zip(args.modalities, Xlist)
            ], dim=1)  # [B, M, D_latent]

            # 2) 前向融合 + 对抗
            logits_cls, logits_dom, dom_labels = fusion(zs, dom, lambd=args.domain_lambda)

            # 3) 损失
            loss_cls = ce(logits_cls, y)
            loss_dom = ce(logits_dom, dom_labels)
            loss = loss_cls + args.domain_weight * loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4) 统计
            sum_cls += loss_cls.item() * y.size(0)
            sum_dom += loss_dom.item() * dom_labels.size(0)
            preds = logits_cls.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        print(f"Epoch {epoch}: Cls={sum_cls/total:.4f} | Dom={sum_dom/(total*len(args.modalities)):.4f} "
              f"| Acc={correct/total*100:.2f}%")

    # 保存融合模型
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(fusion.state_dict(), os.path.join(args.output_dir, "fusion_with_domain.pth"))
    print("Saved fusion_with_domain.pth")

# -----------------------------
# 5) 参数解析
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modalities",   nargs="+", required=True)
    p.add_argument("--data_file",     type=str, required=True)
    p.add_argument("--checkpoint_dir",type=str, default="./pretrained")
    p.add_argument("--output_dir",    type=str, default="./fusion_out")
    p.add_argument("--enc_layers",    type=int, default=3)
    p.add_argument("--dec_layers",    type=int, default=2)
    p.add_argument("--nhead",         type=int, default=8)
    p.add_argument("--mlp_ratio",     type=float, default=4.0)
    p.add_argument("--latent_dim",    type=int, default=256)
    p.add_argument("--fusion_layers", type=int, default=2)
    p.add_argument("--num_classes",   type=int, default=2)
    p.add_argument("--domain_lambda", type=float, default=1.0)
    p.add_argument("--domain_weight", type=float, default=0.5)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--fine_tune",     action="store_true")
    p.add_argument("--mask_ratio",    type=float, default=0.25)
    p.add_argument("--device",        type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)
