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
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_latent, nhead=n_heads, dim_feedforward=int(dim_latent*mlp_ratio))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        # Decoder: 从 latent 恢复到原始维度
        self.decoder_embed = nn.Linear(dim_latent, dim_in)
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=n_heads, dim_feedforward=int(dim_in*mlp_ratio))
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)

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
        z = z.unsqueeze(1)                # B×1×D_latent（补充 seq len 维度）
        z = self.encoder(z)               # B×1×D_latent
        z = z.squeeze(1)                  # B×D_latent
        # Decoder 重建
        x_rec = self.decoder_embed(z)     # B×D_in
        x_rec = x_rec.unsqueeze(1)        # B×1×D_in
        x_rec = self.decoder(x_rec)       # B×1×D_in
        x_rec = x_rec.squeeze(1)          # B×D_in
        # 只计算 mask 掩盖位置的重建误差
        return x_rec, mask

class FusionCrossAttn(nn.Module):
    """
    多模态融合 + 分类头
    输入: list of latent z_i ∈ R^(B×D_latent) for i=1..M
    输出: logits ∈ R^(B×num_classes)
    """
    def __init__(self, dim_latent=256, num_modalities=5,
                 fusion_layers=2, n_heads=8, mlp_ratio=4., num_classes=2, dropout=0.2):
        super().__init__()
        self.num_modalities = num_modalities
        self.dim = dim_latent

        # 一个 learnable token 作为融合 queries（可选）
        self.fuse_token = nn.Parameter(torch.randn(1, 1, dim_latent))

        # Cross-Attention Layers
        layers = []
        for _ in range(fusion_layers):
            # MultiheadAttention: query=fuse_token, key/value=all z
            layers.append(nn.MultiheadAttention(embed_dim=dim_latent, num_heads=n_heads, dropout=dropout))
            # FeedForward
            layers.append(nn.Sequential(
                nn.LayerNorm(dim_latent),
                nn.Linear(dim_latent, int(dim_latent*mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(dim_latent*mlp_ratio), dim_latent),
                nn.Dropout(dropout),
            ))
        self.fusion_layers = nn.ModuleList(layers)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim_latent),
            nn.Linear(dim_latent, num_classes)
        )

    def forward(self, zs: torch.Tensor):
        """
        zs: B×M×D_latent （M=modalities 数量）
        """
        B, M, D = zs.shape
        # 准备 fuse_token
        fuse = self.fuse_token.expand(B, -1, -1)  # B×1×D
        # flatten zs 作为 key/value
        kv = zs.transpose(0,1)  # M×B×D
        fuse = fuse.transpose(0,1)  # 1×B×D

        # 依次做交叉注意力 + FFN
        for i in range(0, len(self.fusion_layers), 2):
            attn = self.fusion_layers[i]
            ffn  = self.fusion_layers[i+1]
            # Cross-Attn: query=fuse, kv=kv
            fuse2, _ = attn(query=fuse, key=kv, value=kv)
            fuse = fuse + fuse2
            # FFN
            fuse = fuse + ffn(fuse)

        # 分类
        # fuse: 1×B×D →  B×D
        fuse = fuse.transpose(0,1).squeeze(1)
        logits = self.cls_head(fuse)  # B×num_classes
        return logits
