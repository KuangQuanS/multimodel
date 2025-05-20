import argparse, os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # 新增
from torch.utils.data import TensorDataset, DataLoader
from model import ModalityMAE

def train_mae(modality, data_tensor, args):
    device = args.device
    dataset = TensorDataset(data_tensor)

    # 1. train/val 划分
    N = len(dataset)
    val_size = int(N * args.val_ratio)
    train_size = N - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = ModalityMAE(dim_in=data_tensor.size(1),
                        dim_latent=args.latent_dim,
                        encoder_layers=args.enc_layers,
                        decoder_layers=args.dec_layers,
                        n_heads=args.n_heads,
                        mlp_ratio=args.mlp_ratio,
                        mask_ratio=args.mask_ratio).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_history, val_history = [], []
    best_loss = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(args.save_dir, f"{modality}_best_encoder.pth")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        # —— 训练阶段 —— 
        model.train()
        train_loss = 0.
        for (x,) in train_loader:
            x = x.to(device)
            x_rec, mask = model(x)
            loss = criterion(x_rec[~mask], x[~mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # —— 验证阶段 —— 
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_rec, mask = model(x)
                loss = criterion(x_rec[~mask], x[~mask])
                val_loss += loss.item()
        val_loss /= len(val_loader)

        train_history.append(train_loss)
        val_history.append(val_loss)

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.encoder.state_dict(), best_model_path)

        # 打印进度
        if epoch % 20 == 0 or epoch==1:
            print(f"[{modality}] Epoch {epoch}/{args.epochs}  "
                  f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    print(f"[{modality}] Best @ epoch {best_epoch}, Val Loss={best_loss:.4f}")

    # —— 画图并保存 —— 
    epochs = list(range(1, args.epochs+1))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_history, label="Train Loss")
    plt.plot(epochs, val_history,   label="Val   Loss")
    # 标注最低点
    plt.scatter([best_epoch], [best_loss], color='red')
    plt.text(best_epoch, best_loss,
             f"  min={best_loss:.4f}\n  epoch={best_epoch}",
             verticalalignment='bottom', fontsize=9)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"{modality} MAE Loss Curves")
    plt.grid(True); plt.legend()
    save_path = os.path.join(
        args.save_dir,
        f"{modality}_{args.latent_dim}_epoch{args.epochs}_loss_curve.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{modality}] Loss curves saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--data_root", type=str, required=True,
                        help="每个模态的 .npz 数据路径，形如 data/{modality}.npz")
    parser.add_argument("--save_dir",  type=str, default="./pretrained")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--enc_layers", type=int, default=3)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--n_heads",    type=int, default=8)
    parser.add_argument("--mlp_ratio",  type=float, default=4.0)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                    help="验证集比例，0~1 之间")
    args = parser.parse_args()

    for mod in args.modalities:
        npz_path = os.path.join(args.data_root, f"{mod}.npz")
        npzfile = np.load(npz_path)
        X = npzfile["X"]
        data = torch.from_numpy(X).float().to(args.device)
        train_mae(mod, data, args)
