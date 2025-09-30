"""
模型定义文件
包含所有神经网络模型的定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CTModel(nn.Module):
    """CT图像特征提取模型"""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.preBlock = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 这里需要添加更多层，但为了简化先保留核心结构
        self.features = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.classifier = nn.Linear(256*8*8, num_classes)

    def forward(self, x):
        x = self.preBlock(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten for classification
        return x  # 返回特征而不是分类结果，用于融合


class ModalityEncoder(nn.Module):
    """单模态编码器，使用变分自编码器结构"""
    def __init__(self, dim_in, dim_latent=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim, dim_latent)
        self.fc_logvar = nn.Linear(hidden_dim, dim_latent)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模型，融合cfDNA和CT特征"""
    def __init__(self, dim_latent, n_modalities, num_classes=2, 
                 conv_channels=512, d_model=768, n_heads=4, dropout=0.3, train=True, 
                 ct_feature_extractor=None, finetune_ct=True):
        super().__init__()
        self.training = train
        
        # cfDNA feature processing
        self.cfdna_conv = nn.Sequential(
            nn.Conv1d(dim_latent, conv_channels, kernel_size=min(3, n_modalities), bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SEBlock(conv_channels),
        )
        self.token_proj = nn.Linear(conv_channels, d_model)
        
        # CT model components
        assert ct_feature_extractor is not None, "ct_feature_extractor must be provided"
        self.ct_feature_extractor = ct_feature_extractor

        if not finetune_ct:
            for p in self.ct_feature_extractor.parameters():
                p.requires_grad = False
            self.ct_feature_extractor.eval()  # 不启用 BN/Dropout 的训练行为

        self.ct_proj_k = nn.Linear(256*8*8, d_model)
        self.ct_proj_v = nn.Linear(256*8*8, d_model)
        
        # Attention mechanisms
        self.multihead_attn_cf_to_ct = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.multihead_attn_ct_to_cf = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        
        # Fusion and classification
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, zs, ct_images=None):
        if self.training:
            zs = zs + torch.randn_like(zs) * 0.01

        # cfDNA feature extraction
        x = self.cfdna_conv(zs.permute(0, 2, 1)).permute(0, 2, 1)
        cfdna_tokens = self.token_proj(x)

        if ct_images is None:
            pooled = cfdna_tokens.mean(dim=1)
            fused = self.fusion_layer(torch.cat([pooled, pooled], dim=1))
            return self.classifier(fused)

        # CT feature extraction
        ct_feat = self.ct_feature_extractor(ct_images)
        ct_feat_proj = self.ct_proj_v(ct_feat)
        
        # Cross-attention mechanisms
        k_ct = self.ct_proj_k(ct_feat).unsqueeze(1)
        v_ct = self.ct_proj_v(ct_feat).unsqueeze(1)
        
        attn_cf_to_ct, _ = self.multihead_attn_cf_to_ct(query=cfdna_tokens, key=k_ct, value=v_ct)
        attn_cf_to_ct_pooled = attn_cf_to_ct.mean(dim=1)
        
        q_ct = ct_feat_proj.unsqueeze(1)
        attn_ct_to_cf, _ = self.multihead_attn_ct_to_cf(query=q_ct, key=cfdna_tokens, value=cfdna_tokens)
        attn_ct_to_cf_pooled = attn_ct_to_cf.squeeze(1)
        
        # Fusion and classification
        fused = self.fusion_layer(torch.cat([attn_cf_to_ct_pooled, attn_ct_to_cf_pooled], dim=1))
        return self.classifier(fused)