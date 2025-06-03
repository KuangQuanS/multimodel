import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tabulate import tabulate
import json
from datetime import datetime

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

class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities, indices=None, include_ct=False):
        data = np.load(npz_path)
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        self.include_ct = include_ct
        
        if include_ct and 'CT' in data:
            self.ct_data = data['CT']
        else:
            self.ct_data = None
            
        if indices is not None:
            self.Xs = [x[indices] for x in self.Xs]
            self.y = self.y[indices]
            if self.ct_data is not None:
                self.ct_data = self.ct_data[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {}
        
        # 添加常规模态数据
        for i, mod in enumerate(self.Xs):
            sample[f'X{i}'] = torch.FloatTensor(mod[idx])
        
        # 添加CT数据（如果有）
        if self.include_ct and self.ct_data is not None:
            # 将CT数据从[64, 64, 3]转换为[3, 64, 64]以适应PyTorch的卷积层
            ct_data = self.ct_data[idx].transpose(2, 0, 1)  # [3, 64, 64]
            sample['CT'] = torch.FloatTensor(ct_data)
        
        sample['y'] = torch.tensor(self.y[idx], dtype=torch.long)
        
        return sample

class ModalityEncoder(nn.Module):
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
            nn.ReLU(inplace=True),
        )
        
        self.encoder = Encoder2D([128, 256, 256])

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

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_latent, n_modalities, num_classes=2, 
                 conv_channels=512, d_model=768, n_heads=4, dropout=0.3, train=True):
        super().__init__()
        self.training = train
        
        # CfDNA特征处理，conv_channels可调，保留原设计
        self.cfdna_conv = nn.Sequential(
            nn.Conv1d(in_channels=dim_latent, out_channels=conv_channels,
                      kernel_size=min(3, n_modalities), bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SEBlock1D(conv_channels),
        )
        
        # 把conv输出做通道降维，准备做attention token表示
        self.token_proj = nn.Linear(conv_channels, d_model)
        
        # CT模型
        self.ct_model = CTModel(in_channels=3, num_classes=num_classes)
        self.ct_feature_extractor = nn.Sequential(
            self.ct_model.preBlock,
            self.ct_model.encoder,
            self.ct_model.gap
        )
        
        # 将CT特征映射到d_model维度，作为Key和Value输入
        self.ct_proj_k = nn.Linear(256*8*8, d_model)
        self.ct_proj_v = nn.Linear(256*8*8, d_model)
        
        # Multi-head Attention，query来自cfdna token，key/value来自ct特征
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.multihead_attn_cf_to_ct = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.multihead_attn_ct_to_cf = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        # 融合层，输入维度 d_model * 2，因为concat了attention输出和cfdna token的平均
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, zs, ct_images=None):
        # zs shape [B, D, M]，M=模态数量

        if self.training:
            noise = torch.randn_like(zs) * 0.01
            zs = zs + noise

        # --- cfDNA 特征提取 ---
        x = zs  # [B, D, M]
        x = self.cfdna_conv(x.permute(0, 2, 1))  # -> [B, conv_channels, M']
        x = x.permute(0, 2, 1)  # -> [B, M', conv_channels]

        cfdna_tokens = self.token_proj(x)  # -> [B, M', d_model]

        if ct_images is None:
            pooled = cfdna_tokens.mean(dim=1)
            fused = self.fusion_layer(torch.cat([pooled, pooled], dim=1))
            logits = self.classifier(fused)
            return logits

        # --- CT 特征提取 ---
        ct_feat = self.ct_feature_extractor(ct_images)  # [B, 512*2*2]
        ct_feat_proj = self.ct_proj_v(ct_feat)  # [B, d_model]

        # reshape为 [B, 1, d_model] 用作 cross-attn 的输入
        k_ct = self.ct_proj_k(ct_feat).unsqueeze(1)  # [B, 1, d_model]
        v_ct = self.ct_proj_v(ct_feat).unsqueeze(1)  # [B, 1, d_model]

        # === 第一次 Cross-Attention（cfDNA -> CT）===
        attn_cf_to_ct, _ = self.multihead_attn_cf_to_ct(query=cfdna_tokens, key=k_ct, value=v_ct)
        attn_cf_to_ct_pooled = attn_cf_to_ct.mean(dim=1)  # [B, d_model]

        # === 第二次 Cross-Attention（CT -> cfDNA）===
        # 将 ct_feat 作为 query，cfDNA tokens 为 key/value
        q_ct = ct_feat_proj.unsqueeze(1)  # [B, 1, d_model]
        attn_ct_to_cf, _ = self.multihead_attn_ct_to_cf(query=q_ct, key=cfdna_tokens, value=cfdna_tokens)
        attn_ct_to_cf_pooled = attn_ct_to_cf.squeeze(1)  # [B, d_model]

        # --- 融合 ---
        fused = self.fusion_layer(torch.cat([attn_cf_to_ct_pooled, attn_ct_to_cf_pooled], dim=1))  # [B, d_model*2]
        logits = self.classifier(fused)

        return logits

def load_encoder(mod, checkpoint_dir, latent_dim):
    ckpt = torch.load(os.path.join(checkpoint_dir, f"{mod}_encoder_best.pth"),
                     map_location="cpu", weights_only=True)
    dim_in = ckpt['0.weight'].shape[1]
    encoder = ModalityEncoder(dim_in=dim_in, dim_latent=latent_dim)
    encoder.encoder.load_state_dict(ckpt)
    return encoder

def train_epoch(model, encoders, loader, optimizer, criterion, device):
    model.train()
    total = correct = 0
    for batch in loader:
        yb = batch['y'].to(device)
        
        # 处理常规模态数据
        modality_features = []
        for i, mod in enumerate(encoders.keys()):
            if f'X{i}' in batch:
                x = batch[f'X{i}'].to(device)
                modality_features.append(encoders[mod](x))
        
        # 将所有模态特征堆叠在一起
        zs = torch.stack(modality_features, dim=1)  # [B, M, D]
        # 处理CT数据
        ct_data = batch.get('CT')
        if ct_data is not None:
            ct_data = ct_data.to(device)
            logits = model(zs, ct_data)
        else:
            logits = model(zs)
            
        loss = criterion(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
    
    return correct / total * 100

def evaluate(model, encoders, loader, device):
    """在测试集上评估模型，计算F1分数和AUC"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            yb = batch['y'].to(device)
            
            # 处理常规模态数据
            modality_features = []
            for i, mod in enumerate(encoders.keys()):
                if f'X{i}' in batch:
                    x = batch[f'X{i}'].to(device)
                    modality_features.append(encoders[mod](x))
            
            # 将所有模态特征堆叠在一起
            zs = torch.stack(modality_features, dim=1)  # [B, M, D]
            
            # 处理CT数据
            ct_data = batch.get('CT')
            if ct_data is not None:
                ct_data = ct_data.to(device)
                logits = model(zs, ct_data)
            else:
                logits = model(zs)
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 取正类的概率
            all_labels.extend(yb.cpu().numpy())
    
    # 计算F1分数
    f1 = f1_score(all_labels, all_preds)
    
    # 计算AUC
    auc_score = roc_auc_score(all_labels, all_probs)
    
    return {
        'f1': f1,
        'auc': auc_score,
        'preds': all_preds,
        'probs': all_probs,
        'labels': all_labels
    }

def plot_roc_curve(labels, probs, output_dir, name):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, f'roc_curve_{name}.png'))
    plt.close()

def run_single_fold(model, encoders, train_loader, val_loader, args, fold=None, cv_dir=None):
    """运行单次训练和验证"""
    if cv_dir is None:
        cv_dir = os.path.join(args.output_dir, "cv_results")

    # 根据finetune参数决定优化器参数
    if args.finetune:
        # 如果是微调模式，将所有可训练参数加入优化
        params = []
        # 添加融合模型参数
        params.extend(model.parameters())
        # 添加编码器参数
        for encoder in encoders.values():
            params.extend(encoder.parameters())
        optimizer = optim.AdamW(params, lr=args.lr)
        print("优化器包含融合模型和编码器参数")
    else:
        # 否则只优化融合模型参数
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        print("优化器仅包含融合模型参数")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs//3, eta_min=args.lr/10)
    criterion = nn.CrossEntropyLoss()
    best_auc = 0
    best_epoch = 0
    best_results = None
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_acc = train_epoch(model, encoders, train_loader, optimizer, criterion, args.device)
        scheduler.step()
        
        # 验证
        if epoch % args.eval_interval == 0:
            val_results = evaluate(model, encoders, val_loader, args.device)
            fold_str = f" Fold {fold}" if fold is not None else ""
            print(f"Epoch {epoch}{fold_str}:")
            print(f"Train Acc={train_acc:.1f}%")
            print(f"Val F1={val_results['f1']:.3f}")
            print(f"Val AUC={val_results['auc']:.3f}")
            
            # 保存最佳模型
            if val_results['auc'] > best_auc:
                best_auc = val_results['auc']
                best_epoch = epoch
                best_results = val_results
                if fold is not None:
                    model_path = os.path.join(cv_dir , f"best_model_fold_{fold}.pth")
                else:
                    model_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
    
    return best_results, best_epoch

def run_cross_validation(args, dataset):
    """运行k折交叉验证"""
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    all_results = []
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = os.path.join(args.output_dir, f"cv_results_{timestamp}")
    os.makedirs(cv_dir, exist_ok=True)
    
    # 加载预训练编码器
    encoders = {mod: load_encoder(mod, args.checkpoint_dir, args.latent_dim).to(args.device) 
                for mod in args.modalities}
    
    # 设置编码器参数是否可更新
    for encoder in encoders.values():
        if args.finetune:
            encoder.train()
        else:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        print(f"\n{'='*20} Fold {fold}/{args.k_folds} {'='*20}")
        
        # 创建数据加载器
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
        
        # 初始化模型
        model = CrossAttentionFusion(
            dim_latent=args.latent_dim,
            n_modalities=len(args.modalities),
            num_classes=2,
            dropout=0.3,
            train=True
        ).to(args.device)
        
        # 训练和验证
        fold_results, best_epoch = run_single_fold(
            model, encoders, train_loader, val_loader, args, fold, cv_dir
        )
        
        # 保存结果
        fold_results['fold'] = fold
        fold_results['best_epoch'] = best_epoch
        all_results.append(fold_results)
        
        # 绘制ROC曲线
        plot_roc_curve(
            fold_results['labels'],
            fold_results['probs'],
            cv_dir,
            f"fold_{fold}"
        )
    
    # 计算平均结果
    f1_scores = [r['f1'] for r in all_results]
    auc_scores = [r['auc'] for r in all_results]
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    # 创建结果表格
    results_table = [
        ["Fold", "F1 Score", "AUC Score", "Best Epoch"]
    ]
    for r in all_results:
        results_table.append([
            r['fold'],
            f"{r['f1']:.3f}",
            f"{r['auc']:.3f}",
            r['best_epoch']
        ])
    results_table.append([
        "Mean ± Std",
        f"{mean_f1:.3f} ± {std_f1:.3f}",
        f"{mean_auc:.3f} ± {std_auc:.3f}",
        "-"
    ])
    
    # 保存详细结果
    # 确保所有NumPy类型都转换为Python原生类型
    processed_results = []
    for r in all_results:
        processed_result = {
            'fold': int(r['fold']),
            'f1': float(r['f1']),
            'auc': float(r['auc']),
            'best_epoch': int(r['best_epoch']),
            'preds': [int(p) for p in r['preds']],
            'probs': [float(p) for p in r['probs']],
            'labels': [int(l) for l in r['labels']]
        }
        processed_results.append(processed_result)
    
    results_dict = {
        'timestamp': timestamp,
        'args': vars(args),
        'fold_results': processed_results,
        'summary': {
            'mean_f1': float(mean_f1),
            'std_f1': float(std_f1),
            'mean_auc': float(mean_auc),
            'std_auc': float(std_auc)
        }
    }
    with open(os.path.join(cv_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # 打印结果表格
    print("\n交叉验证结果:")
    print(tabulate(results_table, headers="firstrow", tablefmt="grid"))
    
    return results_dict

def main():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, default=None, help="独立测试集文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default="./pretrained")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_only", action="store_true", help="仅进行评估，不训练模型")
    parser.add_argument("--model_path", type=str, default=None, help="用于评估的模型路径")
    parser.add_argument("--ct_model_path", type=str, default=None, help="CT模型预训练参数路径")
    parser.add_argument("--cross_val", action="store_true", help="执行交叉验证")
    parser.add_argument("--k_folds", type=int, default=10, help="交叉验证的折数")
    parser.add_argument("--finetune", action="store_true", help="允许更新encoder和CT模型参数")
    parser.add_argument("--device", type=str, 
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载训练数据
    train_dataset = MultiModalDataset(args.data_file, args.modalities, include_ct=True)
    print(f"加载训练数据: {args.data_file}, 样本数: {len(train_dataset)}")
    
    # 交叉验证模式
    if args.cross_val:
        print(f"\n执行{args.k_folds}折交叉验证...")
        cv_results = run_cross_validation(args, train_dataset)
        return
    
    # 加载独立测试集或从训练集划分验证集
    if args.test_file:
        test_dataset = MultiModalDataset(args.test_file, args.modalities, include_ct=True)
        print(f"加载独立测试集: {args.test_file}, 样本数: {len(test_dataset)}")
        
        # 如果不是仅评估模式，则从训练集划分验证集
        if not args.eval_only:
            dataset_size = len(train_dataset)
            val_size = int(dataset_size * args.val_size)
            train_size = dataset_size - val_size
            train_subset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            print(f"训练集: {train_size}样本, 验证集: {val_size}样本")
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        # 从训练集划分验证集作为测试集
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * args.val_size)
        train_size = dataset_size - val_size
        train_subset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_dataset = val_dataset
        print(f"训练集: {train_size}样本, 验证/测试集: {val_size}样本")
    
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载预训练编码器
    encoders = {mod: load_encoder(mod, args.checkpoint_dir, args.latent_dim).to(args.device) 
                for mod in args.modalities}
    
    # 初始化模型
    model = CrossAttentionFusion(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        num_classes=2,
        dropout=0.3,
        train=not args.eval_only
    ).to(args.device)

    # 仅评估模式
    if args.eval_only:
        if args.model_path:
            print(f"加载模型: {args.model_path}")
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        else:
            print("警告: 未指定模型路径，使用随机初始化的模型进行评估")
        
        model.eval()
        test_results = evaluate(model, encoders, test_loader, args.device)
        
        # 创建结果表格
        results_table = [
            ["指标", "值"],
            ["F1 Score", f"{test_results['f1']:.3f}"],
            ["AUC Score", f"{test_results['auc']:.3f}"]
        ]
        
        print("\n独立测试集评估结果:")
        print(tabulate(results_table, headers="firstrow", tablefmt="grid"))
        
        # 绘制ROC曲线
        plot_roc_curve(
            test_results['labels'],
            test_results['probs'],
            args.output_dir,
            "test_eval"
        )
        
        # 保存结果
        processed_results = {
            'f1': float(test_results['f1']),
            'auc': float(test_results['auc']),
            'preds': [int(p) for p in test_results['preds']],
            'probs': [float(p) for p in test_results['probs']],
            'labels': [int(l) for l in test_results['labels']]
        }
        
        results_dict = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'args': vars(args),
            'metrics': processed_results
        }
        with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        return

    # 设置编码器参数是否可更新
    for encoder in encoders.values():
        if args.finetune:
            encoder.train()
            print(f"设置编码器为微调模式，参数可更新")
        else:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
    
    # 加载CT模型预训练参数（如果提供）
    if 'ct' in args.modalities and args.ct_model_path:
        print(f"加载CT模型预训练参数: {args.ct_model_path}")
        # 获取CT模型
        ct_model = None
        for mod, encoder in encoders.items():
            if mod == 'ct':
                ct_model = encoder
                break
        
        if ct_model:
            # 加载预训练参数
            pretrained_state_dict = torch.load(args.ct_model_path, map_location=args.device)
            
            # 过滤掉不匹配的参数
            model_state_dict = ct_model.state_dict()
            filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() 
                                  if k in model_state_dict and v.shape == model_state_dict[k].shape}
            
            # 加载匹配的参数
            ct_model.load_state_dict(filtered_state_dict, strict=False)
            
            # 设置CT模型参数是否可更新
            if args.finetune:
                ct_model.train()
                print(f"设置CT模型为微调模式，参数可更新")
            else:
                ct_model.eval()
                for p in ct_model.parameters():
                    p.requires_grad = False
    
    # 如果有独立测试集，使用验证集进行训练
    if args.test_file:
        print("\n使用验证集进行模型选择...")
        best_results, best_epoch = run_single_fold(
            model, encoders, train_loader, val_loader, args
        )
        
        # 加载最佳模型进行测试集评估
        print(f"\n加载最佳模型 (epoch {best_epoch}) 进行测试集评估")
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
        test_results = evaluate(model, encoders, test_loader, args.device)
        
        # 创建结果表格
        results_table = [
            ["数据集", "F1 Score", "AUC Score"],
            ["验证集 (最佳)", f"{best_results['f1']:.3f}", f"{best_results['auc']:.3f}"],
            ["测试集", f"{test_results['f1']:.3f}", f"{test_results['auc']:.3f}"]
        ]
        
        print("\n最终评估结果:")
        print(tabulate(results_table, headers="firstrow", tablefmt="grid"))
        
        # 绘制ROC曲线
        plot_roc_curve(
            test_results['labels'],
            test_results['probs'],
            args.output_dir,
            "final_test"
        )
        
        # 保存结果
        processed_val_results = {
            'f1': float(best_results['f1']),
            'auc': float(best_results['auc']),
            'preds': [int(p) for p in best_results['preds']],
            'probs': [float(p) for p in best_results['probs']],
            'labels': [int(l) for l in best_results['labels']]
        }
        
        processed_test_results = {
            'f1': float(test_results['f1']),
            'auc': float(test_results['auc']),
            'preds': [int(p) for p in test_results['preds']],
            'probs': [float(p) for p in test_results['probs']],
            'labels': [int(l) for l in test_results['labels']]
        }
        
        results_dict = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'args': vars(args),
            'best_epoch': int(best_epoch),
            'validation': processed_val_results,
            'test': processed_test_results
        }
        with open(os.path.join(args.output_dir, 'train_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
    else:
        # 如果没有独立测试集，使用划分的测试集进行训练和评估
        print("\n使用划分的测试集进行模型选择和评估...")
        best_results, best_epoch = run_single_fold(
            model, encoders, train_loader, test_loader, args
        )
        
        # 创建结果表格
        results_table = [
            ["指标", "值"],
            ["最佳Epoch", str(best_epoch)],
            ["F1 Score", f"{best_results['f1']:.3f}"],
            ["AUC Score", f"{best_results['auc']:.3f}"]
        ]
        
        print("\n最终评估结果:")
        print(tabulate(results_table, headers="firstrow", tablefmt="grid"))
        
        # 绘制ROC曲线
        plot_roc_curve(
            best_results['labels'],
            best_results['probs'],
            args.output_dir,
            "final"
        )
        
        # 保存结果
        processed_results = {
            'f1': float(best_results['f1']),
            'auc': float(best_results['auc']),
            'preds': [int(p) for p in best_results['preds']],
            'probs': [float(p) for p in best_results['probs']],
            'labels': [int(l) for l in best_results['labels']]
        }
        
        results_dict = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'args': vars(args),
            'best_epoch': int(best_epoch),
            'metrics': processed_results
        }
        with open(os.path.join(args.output_dir, 'train_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    main()