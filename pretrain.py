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
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim_latent),
            nn.BatchNorm1d(dim_latent)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim_in)
        )

    def forward(self, x):
        # 编码
        z = self.encoder(x)  # B×D_latent
        # 解码
        x_rec = self.decoder(z)  # B×D_in
        return x_rec, z
    
def train_encoder(modality, data_tensor, args):
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

    # 创建模型
    model = ModalityEncoder(
        dim_in=data_tensor.size(1),
        dim_latent=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用MSE损失进行重建任务
    criterion = nn.MSELoss()

    train_history, val_history = [], []
    best_loss = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(args.save_dir, f"{modality}_best.pth")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        # —— 训练阶段 —— 
        model.train()
        train_loss = 0.
        for (x,) in train_loader:
            x = x.to(device)
            x_rec, z = model(x)
            # 使用重建损失
            loss = criterion(x_rec, x)
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
                x_rec, z = model(x)
                loss = criterion(x_rec, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        train_history.append(train_loss)
        val_history.append(val_loss)

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            # 保存编码器部分
            torch.save(model.encoder.state_dict(), best_model_path)

        # 打印进度
        if epoch % 20 == 0 or epoch==1:
            print(f"[{modality}] Epoch {epoch}/{args.epochs}  "
                  f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    print(f"[{modality}] Best @ epoch {best_epoch}, Val Loss={best_loss:.4f}")

    # # —— 画图并保存 —— 
    # epochs = list(range(1, args.epochs+1))
    # plt.figure(figsize=(6,4))
    # plt.plot(epochs, train_history, label="Train Loss")
    # plt.plot(epochs, val_history,   label="Val   Loss")
    # # 标注最低点
    # plt.scatter([best_epoch], [best_loss], color='red')
    # plt.text(best_epoch, best_loss,
    #          f"  min={best_loss:.4f}\n  epoch={best_epoch}",
    #          verticalalignment='bottom', fontsize=9)
    # plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    # plt.title(f"{modality} Autoencoder Loss Curves")
    # plt.grid(True); plt.legend()
    # save_path = os.path.join(
    #     args.save_dir,
    #     f"{modality}_{args.latent_dim}_epoch{args.epochs}_loss_curve.png"
    # )
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"[{modality}] Loss curves saved to {save_path}")

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