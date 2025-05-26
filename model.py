import math
import torch
import torch.nn as nn

# ----------------CfDNA模型-----------------
class ModalityEncoder(nn.Module):
    """Simplified encoder for single modality"""
    def __init__(self, dim_in, dim_latent=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim_latent),
            nn.BatchNorm1d(dim_latent)
        )

    def forward(self, x):
        return self.encoder(x)

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, C, L] -> [B, C, 1]
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        y = self.pool(x).view(b, c)         # [B, C]
        y = self.fc(y).view(b, c, 1)        # [B, C, 1]  
        return x * y.expand_as(x)           # [B, C, L]

class Conv1dFusion(nn.Module):
    def __init__(self, dim_latent, n_modalities, num_classes=2,
                 conv_channels=512, dropout=0.3, train=True):
        super().__init__()

        self.training = train
    
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=dim_latent,
                      out_channels=conv_channels,
                      kernel_size=min(3,n_modalities),
                      bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SEBlock1D(conv_channels),
            # 池化到长度 1
            #nn.AdaptiveAvgPool1d(1),
            #nn.Flatten(), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channels,
                      out_channels=conv_channels,
                      kernel_size=min(3,n_modalities),
                      bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SEBlock1D(conv_channels),
            # 池化到长度 1
            #nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), 
        )
        self.classifier = nn.Sequential(
             # [B, conv_channels]
            nn.Linear(conv_channels, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, zs):
        # zs: [B, M, D]
        if self.training:
            noise = torch.randn_like(zs) * 0.01
            zs = zs + noise
        x = zs.permute(0, 2, 1)  # -> [B, D, M]
        x = self.conv1(x)         # -> [B, conv_channels, 1]
        x = self.conv2(x)

        return self.classifier(x)  # -> [B, num_classes]

# ----------------CT模型-----------------    
class ChannelAttention2D(nn.Module):
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

class SpatialAttention2D(nn.Module):
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

class GCSAM2D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention2D(in_channels, reduction)
        self.spatial_attention = SpatialAttention2D()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Bottle2neck2D(nn.Module):
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

        self.GCS = GCSAM2D(planes)

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

class Encoder2D(nn.Module):
    def __init__(self, channels, dropout_prob=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                Bottle2neck2D(channels[i], channels[i+1]),
                Bottle2neck2D(channels[i+1], channels[i+1]),
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
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder = Encoder2D([128, 256, 512])

        self.gap = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=1),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.LayerNorm(512*2*2)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(512*2*2, 128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.preBlock(x)
        x = self.encoder(x)
        x = self.gap(x)
        x = self.mlp(x)
        return x