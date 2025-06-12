import argparse, os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # 新增
from torch.utils.data import TensorDataset, DataLoader

class ModalityEncoder(nn.Module):
    """
    单模态的编码器-解码器网络
    输入: x ∈ R^(B×D_in)
    输出: 重建 x_hat ∈ R^(B×D_in)
    Encoder 输出 latent z ∈ R^(B×D_latent)
    """
    def __init__(self, dim_in, dim_latent=256, hidden_dim=512):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_mu = nn.Linear(hidden_dim, dim_latent)
        self.fc_logvar = nn.Linear(hidden_dim, dim_latent)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim_in)
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 取 exp(logvar / 2)
        eps = torch.randn_like(std)   # 与 std 相同形状的标准正态噪声
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        h = self.encoder(x)  # B×D_latent
        mu = self.fc_mu(h)               # B×latent
        logvar = self.fc_logvar(h)       # B×latent
        z = self.reparameterize(mu, logvar)  # B×latent
        # 解码
        x_rec = self.decoder(z)  # B×D_in
        return x_rec, z
    
def train_encoder(modality, data_tensor, args):
    device = args.device
    dataset = TensorDataset(data_tensor)

    # train/val 划分
    N = len(dataset)
    val_size = int(N * args.val_ratio)
    train_size = N - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 创建模型
    model = ModalityEncoder(
        dim_in=data_tensor.size(1),
        dim_latent=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_history, val_history = [], []
    best_loss = float('inf')
    best_epoch = -1
    early_stop_counter = 0
    early_stop_patience = 50  # 提前终止容忍度

    encoder_path = os.path.join(args.save_dir, f"{modality}_encoder_best.pth")
    full_model_path = os.path.join(args.save_dir, f"{modality}_full_model_best.pth")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0.
        for (x,) in train_loader:
            x = x.to(device)
            x_rec, z = model(x)
            loss = criterion(x_rec, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_rec, z = model(x)
                loss = criterion(x_rec, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            # 保存最优 encoder 和整个 model
            torch.save(model.encoder.state_dict(), encoder_path)
            torch.save(model.state_dict(), full_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"[{modality}] Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"[{modality}] Epoch {epoch}/{args.epochs}  "
                  f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    print(f"[{modality}] Best @ epoch {best_epoch}, Val Loss={best_loss:.4f}")
    print(f"[{modality}] Saved best encoder to {encoder_path}")
    print(f"[{modality}] Saved best full model to {full_model_path}")

    # —— 可视化 Loss 曲线 —— 
    epochs = list(range(1, len(train_history)+1))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_history, label="Train Loss")
    plt.plot(epochs, val_history,   label="Val   Loss")
    plt.scatter([best_epoch], [best_loss], color='red')
    plt.text(best_epoch, best_loss,
             f"  min={best_loss:.4f}\n  epoch={best_epoch}",
             verticalalignment='bottom', fontsize=9)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"{modality} Autoencoder Loss Curves")
    plt.grid(True); plt.legend()
    save_path = os.path.join(
        args.save_dir,
        f"{modality}_{args.latent_dim}_epoch{len(epochs)}_loss_curve.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{modality}] Loss curves saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--data_file", type=str, required=True,
                        help="包含所有模态和标签的NPZ文件路径")
    parser.add_argument("--save_dir",  type=str, default="./pretrained")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                    help="验证集比例，0~1 之间")
    args = parser.parse_args()

    # 加载数据
    data = np.load(args.data_file)
    
    for mod in args.modalities:
        print(f"Training Autoencoder for {mod}...")
        X = torch.from_numpy(data[mod]).float()
        train_encoder(mod, X, args)