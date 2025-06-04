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

# ==================== Model Components ====================

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
        input_x = x.view(B, C, -1).unsqueeze(1)
        context_mask = self.conv_mask(x).view(B, 1, -1)
        context_mask = self.softmax(context_mask).unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)
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

        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        ) if inplanes != planes or stride != 1 else nn.Identity()

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

# ==================== Core Models ====================

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
        return self.mlp(x)

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

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_latent, n_modalities, num_classes=2, 
                 conv_channels=512, d_model=768, n_heads=4, dropout=0.3, train=True):
        super().__init__()
        self.training = train
        
        # cfDNA feature processing
        self.cfdna_conv = nn.Sequential(
            nn.Conv1d(dim_latent, conv_channels, kernel_size=min(3, n_modalities), bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SEBlock1D(conv_channels),
        )
        self.token_proj = nn.Linear(conv_channels, d_model)
        
        # CT model components
        self.ct_model = CTModel(in_channels=3, num_classes=num_classes)
        self.ct_feature_extractor = nn.Sequential(
            self.ct_model.preBlock,
            self.ct_model.encoder,
            self.ct_model.gap
        )
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

# ==================== Data Handling ====================

class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities, indices=None, include_ct=False):
        data = np.load(npz_path)
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        self.include_ct = include_ct
        self.ct_data = data['CT'] if include_ct and 'CT' in data else None
            
        if indices is not None:
            self.Xs = [x[indices] for x in self.Xs]
            self.y = self.y[indices]
            if self.ct_data is not None:
                self.ct_data = self.ct_data[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {f'X{i}': torch.FloatTensor(mod[idx]) for i, mod in enumerate(self.Xs)}
        
        if self.include_ct and self.ct_data is not None:
            sample['CT'] = torch.FloatTensor(self.ct_data[idx].transpose(2, 0, 1))
        
        sample['y'] = torch.tensor(self.y[idx], dtype=torch.long)
        return sample

# ==================== Training Utilities ====================

def load_encoder(mod, checkpoint_dir, latent_dim):
    ckpt = torch.load(os.path.join(checkpoint_dir, f"{mod}_encoder_best.pth"),
                     map_location="cpu", weights_only=True)
    encoder = ModalityEncoder(dim_in=ckpt['0.weight'].shape[1], dim_latent=latent_dim)
    encoder.encoder.load_state_dict(ckpt)
    return encoder

def train_epoch(model, encoders, loader, optimizer, criterion, device):
    model.train()
    total = correct = 0
    
    for batch in loader:
        yb = batch['y'].to(device)
        modality_features = [encoders[mod](batch[f'X{i}'].to(device)) 
                           for i, mod in enumerate(encoders.keys()) if f'X{i}' in batch]
        
        zs = torch.stack(modality_features, dim=1)
        ct_data = batch.get('CT', None)
        ct_data = ct_data.to(device) if ct_data is not None else None
        
        logits = model(zs, ct_data)
        loss = criterion(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
    
    return correct / total * 100

def evaluate(model, encoders, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            yb = batch['y'].to(device)
            modality_features = [encoders[mod](batch[f'X{i}'].to(device)) 
                               for i, mod in enumerate(encoders.keys()) if f'X{i}' in batch]
            
            zs = torch.stack(modality_features, dim=1)
            ct_data = batch.get('CT', None)
            ct_data = ct_data.to(device) if ct_data is not None else None
            
            logits = model(zs, ct_data)
            probs = torch.softmax(logits, dim=1)
            
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    return {
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'preds': all_preds,
        'probs': all_probs,
        'labels': all_labels
    }

def plot_roc_curve(labels, probs, output_dir, name):
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
    plt.savefig(os.path.join(output_dir, f'roc_curve_{name}.png'))
    plt.close()

# ==================== Core Training Functions ====================

def run_single_fold(model, encoders, train_loader, val_loader, args, fold=None, cv_dir=None):
    cv_dir = cv_dir or os.path.join(args.output_dir, "cv_results")
    
    # Setup optimizer
    params = []
    params.extend(model.parameters())
    if args.finetune:
        params.extend([p for encoder in encoders.values() for p in encoder.parameters()])
    
    optimizer = optim.AdamW(params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs//3, eta_min=args.lr/10)
    criterion = nn.CrossEntropyLoss()
    
    best_auc, best_epoch, best_results = 0, 0, None
    
    for epoch in range(1, args.epochs + 1):
        train_acc = train_epoch(model, encoders, train_loader, optimizer, criterion, args.device)
        scheduler.step()
        
        if epoch % args.eval_interval == 0:
            val_results = evaluate(model, encoders, val_loader, args.device)
            fold_str = f" Fold {fold}" if fold is not None else ""
            print(f"Epoch {epoch}{fold_str}: Train Acc={train_acc:.1f}%, Val F1={val_results['f1']:.3f}, Val AUC={val_results['auc']:.3f}")
            
            if val_results['auc'] > best_auc:
                best_auc = val_results['auc']
                best_epoch = epoch
                best_results = val_results
                model_path = os.path.join(cv_dir if fold else args.output_dir, 
                                        f"best_model{'_fold_'+str(fold) if fold else ''}.pth")
                torch.save(model.state_dict(), model_path)
    
    return best_results, best_epoch

def run_cross_validation(args, dataset):
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = os.path.join(args.output_dir, f"cv_results_{timestamp}")
    os.makedirs(cv_dir, exist_ok=True)
    
    # Load and configure encoders
    encoders = {mod: load_encoder(mod, args.checkpoint_dir, args.latent_dim).to(args.device) 
               for mod in args.modalities}
    
    for encoder in encoders.values():
        if args.finetune:
            encoder.train()
        else:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
    
    all_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        print(f"\n{'='*20} Fold {fold}/{args.k_folds} {'='*20}")
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                                 sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(dataset, batch_size=args.batch_size, 
                               sampler=SubsetRandomSampler(val_idx))
        
        model = CrossAttentionFusion(
            dim_latent=args.latent_dim,
            n_modalities=len(args.modalities),
            num_classes=2,
            dropout=0.3,
            train=True
        ).to(args.device)
        
        fold_results, best_epoch = run_single_fold(
            model, encoders, train_loader, val_loader, args, fold, cv_dir
        )
        
        fold_results.update({'fold': fold, 'best_epoch': best_epoch})
        all_results.append(fold_results)
        plot_roc_curve(fold_results['labels'], fold_results['probs'], cv_dir, f"fold_{fold}")
    
    # Calculate and save results
    f1_scores = [r['f1'] for r in all_results]
    auc_scores = [r['auc'] for r in all_results]
    
    results_dict = {
        'timestamp': timestamp,
        'args': vars(args),
        'fold_results': [{
            'fold': int(r['fold']),
            'f1': float(r['f1']),
            'auc': float(r['auc']),
            'best_epoch': int(r['best_epoch']),
            'preds': [int(p) for p in r['preds']],
            'probs': [float(p) for p in r['probs']],
            'labels': [int(l) for l in r['labels']]
        } for r in all_results],
        'summary': {
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
            'mean_auc': float(np.mean(auc_scores)),
            'std_auc': float(np.std(auc_scores))
        }
    }
    
    with open(os.path.join(cv_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Print results table
    results_table = [["Fold", "F1 Score", "AUC Score", "Best Epoch"]]
    results_table.extend([
        [r['fold'], f"{r['f1']:.3f}", f"{r['auc']:.3f}", r['best_epoch']] 
        for r in all_results
    ])
    results_table.append([
        "Mean ± Std",
        f"{np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}",
        f"{np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}",
        "-"
    ])
    
    print("\n交叉验证结果:")
    print(tabulate(results_table, headers="firstrow", tablefmt="grid"))
    
    return results_dict

# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./pretrained")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ct_model_path", type=str, default=None)
    parser.add_argument("--cross_val", action="store_true")
    parser.add_argument("--k_folds", type=int, default=10)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--device", type=str, 
                      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    train_dataset = MultiModalDataset(args.data_file, args.modalities, include_ct=True)
    print(f"Loaded training data: {args.data_file}, samples: {len(train_dataset)}")
    
    if args.cross_val:
        print(f"\nRunning {args.k_folds}-fold cross-validation...")
        run_cross_validation(args, train_dataset)
        return
    
    # Handle test/validation split
    if args.test_file:
        test_dataset = MultiModalDataset(args.test_file, args.modalities, include_ct=True)
        print(f"Loaded test set: {args.test_file}, samples: {len(test_dataset)}")
        
        if not args.eval_only:
            dataset_size = len(train_dataset)
            val_size = int(dataset_size * args.val_size)
            train_subset, val_dataset = torch.utils.data.random_split(
                train_dataset, [dataset_size - val_size, val_size]
            )
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * args.val_size)
        train_subset, test_dataset = torch.utils.data.random_split(
            train_dataset, [dataset_size - val_size, val_size]
        )
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load encoders
    encoders = {mod: load_encoder(mod, args.checkpoint_dir, args.latent_dim).to(args.device) 
               for mod in args.modalities}
    
    # Initialize model
    model = CrossAttentionFusion(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        num_classes=2,
        dropout=0.3,
        train=not args.eval_only
    ).to(args.device)

    # Evaluation only mode
    if args.eval_only:
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        
        test_results = evaluate(model, encoders, test_loader, args.device)
        
        # Save and print results
        results_dict = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'args': vars(args),
            'metrics': {
                'f1': float(test_results['f1']),
                'auc': float(test_results['auc']),
                'preds': [int(p) for p in test_results['preds']],
                'probs': [float(p) for p in test_results['probs']],
                'labels': [int(l) for l in test_results['labels']]
            }
        }
        
        with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print("\nTest Results:")
        print(tabulate([
            ["Metric", "Value"],
            ["F1 Score", f"{test_results['f1']:.3f}"],
            ["AUC Score", f"{test_results['auc']:.3f}"]
        ], headers="firstrow", tablefmt="grid"))
        
        plot_roc_curve(test_results['labels'], test_results['probs'], args.output_dir, "test_eval")
        return

    # Configure model training
    for encoder in encoders.values():
        if args.finetune:
            encoder.train()
        else:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
    
    # Train and evaluate
    if args.test_file:
        print("\nUsing validation set for model selection...")
        best_results, best_epoch = run_single_fold(model, encoders, train_loader, val_loader, args)
        
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
        test_results = evaluate(model, encoders, test_loader, args.device)
        
        results_dict = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'args': vars(args),
            'best_epoch': int(best_epoch),
            'validation': {
                'f1': float(best_results['f1']),
                'auc': float(best_results['auc']),
                'preds': [int(p) for p in best_results['preds']],
                'probs': [float(p) for p in best_results['probs']],
                'labels': [int(l) for l in best_results['labels']]
            },
            'test': {
                'f1': float(test_results['f1']),
                'auc': float(test_results['auc']),
                'preds': [int(p) for p in test_results['preds']],
                'probs': [float(p) for p in test_results['probs']],
                'labels': [int(l) for l in test_results['labels']]
            }
        }
    else:
        print("\nUsing split test set for evaluation...")
        best_results, best_epoch = run_single_fold(model, encoders, train_loader, test_loader, args)
        
        results_dict = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'args': vars(args),
            'best_epoch': int(best_epoch),
            'metrics': {
                'f1': float(best_results['f1']),
                'auc': float(best_results['auc']),
                'preds': [int(p) for p in best_results['preds']],
                'probs': [float(p) for p in best_results['probs']],
                'labels': [int(l) for l in best_results['labels']]
            }
        }
    
    # Save final results
    with open(os.path.join(args.output_dir, 'train_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\nFinal Results:")
    if args.test_file:
        print(tabulate([
            ["Dataset", "F1 Score", "AUC Score"],
            ["Validation", f"{best_results['f1']:.3f}", f"{best_results['auc']:.3f}"],
            ["Test", f"{test_results['f1']:.3f}", f"{test_results['auc']:.3f}"]
        ], headers="firstrow", tablefmt="grid"))
        plot_roc_curve(test_results['labels'], test_results['probs'], args.output_dir, "final_test")
    else:
        print(tabulate([
            ["Metric", "Value"],
            ["Best Epoch", str(best_epoch)],
            ["F1 Score", f"{best_results['f1']:.3f}"],
            ["AUC Score", f"{best_results['auc']:.3f}"]
        ], headers="firstrow", tablefmt="grid"))
        plot_roc_curve(best_results['labels'], best_results['probs'], args.output_dir, "final")

if __name__ == "__main__":
    main()