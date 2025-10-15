"""
模型定义文件
包含所有神经网络模型的定义
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super().__init__()
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction, 1),
            nn.LayerNorm([inplanes // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1),
            nn.LayerNorm([inplanes, 1, 1])
        )
        self.sigmoid = nn.Sigmoid()

    def spatial_pool(self, x):
        B, C, H, W = x.size()
        input_x = x.view(B, C, -1).unsqueeze(1)              # B × 1 × C × (H×W)
        context_mask = self.conv_mask(x).view(B, 1, -1)      # B × 1 × (H×W)
        context_mask = self.softmax(context_mask).unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)        # B × 1 × C × 1
        return context.view(B, C, 1, 1)

    def forward(self, x):
        context = self.spatial_pool(x)
        channel_mul_term = self.sigmoid(self.channel_mul_conv(context))
        return x * channel_mul_term

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([max_out, avg_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, channels, length]
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Res2Block(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, baseWidth=26, scale=4, stype='normal'):
        super().__init__()
        width = int(math.floor(inplanes / 4 * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.relu = nn.ReLU(inplace=True)

        self.nums = scale - 1 if scale != 1 else 1
        self.stype = stype
        self.scale = scale
        self.width = width

        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(width)
            for _ in range(self.nums)
        ])

        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        if inplanes != planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

        self.GCS = CBAM(planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
            sp = spx[i] if i == 0 or self.stype == 'stage' else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        if self.scale != 1:
            if self.stype == 'normal':
                out = torch.cat((out, spx[self.nums]), 1)
            elif self.stype == 'stage':
                out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(self.conv3(out))
        residual = self.shortcut(residual)
        out = self.GCS(out)
        return self.relu(out + residual)

class Encoder(nn.Module):
    def __init__(self, channels, dropout_prob=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                Res2Block(channels[i], channels[i+1]),
                Res2Block(channels[i+1], channels[i+1]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout_prob)
            ) for i in range(len(channels)-1)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class CTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.preBlock = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.encoder = Encoder([128, 256, 256])

        self.gap = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=1),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.LayerNorm(256*8*8)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(256*8*8, 128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.preBlock(x)
        x = self.encoder(x)
        x = self.gap(x)
        x = self.mlp(x)
        return x
    
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
                 ct_feature_extractor=None, finetune_ct=True, use_ct=True, use_se_block=True):
        super().__init__()
        self.training = train
        self.use_ct = use_ct
        
        # cfDNA feature processing
        cfdna_layers = [
            nn.Conv1d(dim_latent, conv_channels, kernel_size=min(2, n_modalities), bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ]
        
        # 可选地添加SEBlock
        if use_se_block:
            cfdna_layers.append(SEBlock(conv_channels))
            
        self.cfdna_conv = nn.Sequential(*cfdna_layers)

        self.token_proj = nn.Linear(conv_channels, d_model)
        
        # CT model components (only if using CT)
        if self.use_ct:
            assert ct_feature_extractor is not None, "ct_feature_extractor must be provided when use_ct=True"
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
            
            # Fusion layer for CT+cfDNA
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model * 2, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # cfDNA-only fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Classification head
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

        # cfDNA-only mode
        if not self.use_ct or ct_images is None:
            pooled = cfdna_tokens.mean(dim=1)  # Global average pooling
            fused = self.fusion_layer(pooled)
            return self.classifier(fused)

        # CT+cfDNA mode with cross-attention
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