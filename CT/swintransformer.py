import math
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch_optimizer as optim
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from collections import Counter
import pandas as pd
import time
from torchvision import transforms

class PatchEmbed(nn.Module):
    """图像到Patch嵌入"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    """窗口多头自注意力模块"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # 获取窗口内每对位置的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 移动到从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = int(np.sqrt(L)), int(np.sqrt(L))
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 周期性移动
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # 分区窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 反向移动
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinTransformer(nn.Module):
    """Swin Transformer主模型"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # 分割嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        # 位置嵌入
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

# 辅助函数和类
def window_partition(x, window_size):
    """将特征图划分为不重叠的窗口"""
    B, H, W, C = x.shape
    # 确保H和W能被window_size整除
    assert H % window_size == 0, f"Height {H} must be divisible by window_size {window_size}"
    assert W % window_size == 0, f"Width {W} must be divisible by window_size {window_size}"
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """将窗口重新组合为特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """随机深度"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class PatchMerging(nn.Module):
    """Patch合并层"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """基本Swin Transformer层"""
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # 计算注意力掩码
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        
        # 计算注意力掩码
        attn_mask = self.create_mask(x, H, W)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x

# 从resgsca.py复制必要的训练相关代码
from CT.resgsca import (cross_validation, evaluate, train, NpzPatchDataset, FocalLoss)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Swin Transformer Training')
    parser.add_argument('--data_dir', type=str, default='./database/3slice/48', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./swin_checkpoint', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--stratified', action='store_true', help='Use stratified sampling for cross-validation')
    parser.add_argument('--cv_save_dir', type=str, default='./cv_checkpoints', help='Directory to save cross-validation checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    if args.cv:
        os.makedirs(args.cv_save_dir, exist_ok=True)
    
    # 数据目录和标签映射
    root_dirs = {
        'cancer': os.path.join(args.data_dir, 'cancer'),
        'nocancer': os.path.join(args.data_dir, 'nocancer'),
    }
    labels_map = {'cancer': 1, 'nocancer': 0}
    
    # 数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    # 加载数据集
    full_dataset = NpzPatchDataset(root_dirs, labels_map, transform)
    
    # 检查标签分布
    from collections import Counter
    label_list = [label for _, label in full_dataset]
    print("标签分布:", Counter(label_list))
    
    # 决定是否使用交叉验证
    if args.cv:
        print(f"\n{'='*20} 开始 {args.folds} 折交叉验证 {'='*20}")
        import time
        start_time = time.time()
        
        # 定义优化器和调度器创建函数
        def create_optimizer(params):
            return optim.Lookahead(optim.RAdam(params, lr=args.lr))
        
        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # 执行交叉验证
        cv_results = cross_validation(
            dataset=full_dataset,
            model_class=lambda: SwinTransformer(img_size=48, patch_size=2, in_chans=3, num_classes=2),
            num_folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            criterion=FocalLoss(alpha=0.25, gamma=2.0),
            optimizer_fn=create_optimizer,
            scheduler_fn=create_scheduler,
            device=device,
            save_dir=args.cv_save_dir,
            stratified=args.stratified
        )
        
        # 保存交叉验证结果到CSV
        import pandas as pd
        results_df = pd.DataFrame(cv_results['fold_results'])
        results_csv_path = os.path.join(args.cv_save_dir, 'cv_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"交叉验证结果已保存到: {results_csv_path}")
        
        # 打印总运行时间
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n交叉验证总时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        
    else:
        # 常规训练模式（不使用交叉验证）
        print("\n使用常规训练模式（80%训练/20%验证）")
        
        # 分割数据集
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        model = SwinTransformer(img_size=64, patch_size=2, in_chans=3, num_classes=2,
                              embed_dim=48, depths=[2, 2, 2, 2], num_heads=[2, 4, 8, 16],
                              window_size=4, mlp_ratio=4., qkv_bias=True,
                              drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1).to(device)
        
        # 创建损失函数、优化器和调度器
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.Lookahead(optim.RAdam(model.parameters(), lr=args.lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        print(f"\n{'='*20} 开始常规训练 {'='*20}")
        start_time = time.time()
        
        # 训练模型
        best_f1 = 0.0
        save_path = os.path.join(args.save_dir, 'swin_best.pth')
        
        from tqdm import tqdm
        progress_bar = tqdm(range(args.epochs), desc="训练进度", unit="epoch")
        for epoch in progress_bar:
            train_loss, acc, prec, rec, f1 = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
            
            # 动态更新进度条描述
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
            )
            
            # 保存最佳模型
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), save_path)
                print(f"  🎉 新的最佳F1: {best_f1:.4f}, 已保存到 {save_path}")
        
        # 保存最终模型
        final_save_path = os.path.join(args.save_dir, 'swin_final.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f"最终模型已保存到: {final_save_path}")
        
        # 打印总运行时间
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n训练总时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")