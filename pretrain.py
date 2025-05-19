import argparse, os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import ModalityMAE

def train_mae(modality, data_tensor, args):
    """
    data_tensor: Tensor of shape [N, D_in]
    """
    device = args.device
    dataset = TensorDataset(data_tensor)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ModalityMAE(dim_in=data_tensor.size(1),
                        dim_latent=args.latent_dim,
                        encoder_layers=args.enc_layers,
                        decoder_layers=args.dec_layers,
                        n_heads=args.n_heads,
                        mlp_ratio=args.mlp_ratio,
                        mask_ratio=args.mask_ratio).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for (x,) in loader:
            x = x.to(device)
            x_rec, mask = model(x)
            # 只计算 mask 掩盖位置
            loss = criterion(x_rec[~mask], x[~mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{modality}] Epoch {epoch+1}/{args.epochs}  Loss: {total_loss/len(loader):.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), os.path.join(args.save_dir, f"{modality}_encoder.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--data_root", type=str, required=True, help="每个模态的 .npy 数据路径，形如 data/{modality}.npy")
    parser.add_argument("--save_dir",  type=str, default="./pretrained")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--n_heads",    type=int, default=8)
    parser.add_argument("--mlp_ratio",  type=float, default=4.0)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for mod in args.modalities:
        # ─────────── 这里改为加载 .npz ───────────
        npz_path = os.path.join(args.data_root, f"{mod}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"{npz_path} 不存在")
        npzfile = np.load(npz_path)
        if "X" not in npzfile:
            raise KeyError(f"{mod}.npz 中没有键 'X'，可用 npzfile.files 查看")
        X = npzfile["X"]  
        data = torch.from_numpy(X).float().to(args.device)
        # ────────────────────────────────────────────

        train_mae(mod, data, args)