import torch
import torch.nn as nn

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
