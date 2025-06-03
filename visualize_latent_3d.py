import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from finetune import MultiModalDataset, CrossAttentionFusion, load_encoder
import json

# 尝试导入UMAP，如果安装了的话
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

class FeatureExtractor(torch.nn.Module):
    """用于从模型中提取潜在特征的类，支持双向 cross-attention"""
    def __init__(self, model, extract_layer='fusion'):
        """
        初始化特征提取器

        参数:
            model: 训练好的模型
            extract_layer: 提取特征的层，可选值:
                - 'fusion': 从融合层之前提取 (pooled + attn_pooled_cf + attn_pooled_ct)
                - 'classifier': classifier 前一层的输入特征
                - 'raw_modalities': 原始输入 cfDNA zs
        """
        super().__init__()
        self.model = model
        self.features = None
        self.raw_features = None
        self.extract_layer = extract_layer

        if extract_layer == 'fusion':
            def hook_fusion(module, input, output):
                # 正确访问模型中保存的中间特征
                attn_pooled_cf = self.model._attn_pooled_cf  # cfDNA <- CT
                attn_pooled_ct = self.model._attn_pooled_ct  # CT <- cfDNA
                if attn_pooled_cf is not None and attn_pooled_ct is not None:
                    self.features = torch.cat([attn_pooled_cf, attn_pooled_ct], dim=1).detach()

            original_forward = self.model.forward  # 保存旧 forward 以便需要时恢复

            def new_forward(zs, ct_images=None):
                # === cfDNA token 特征 ===
                x = zs  # [B, D, M]
                x = self.model.cfdna_conv(x.permute(0, 2, 1))  # [B, C, M]
                x = x.permute(0, 2, 1)  # [B, M, C]
                cfdna_tokens = self.model.token_proj(x)  # [B, M, d_model]

                # === CT 特征 ===
                ct_feat = self.model.ct_feature_extractor(ct_images)  # [B, N]
                k_ct = self.model.ct_proj_k(ct_feat).unsqueeze(1)     # [B, 1, d_model]
                v_ct = self.model.ct_proj_v(ct_feat).unsqueeze(1)     # [B, 1, d_model]

                # === 第一次 Cross-Attention（cfDNA → CT）===
                attn_cf_to_ct, _ = self.model.multihead_attn_cf_to_ct(
                    query=cfdna_tokens, key=k_ct, value=v_ct
                )
                attn_pooled_cf = attn_cf_to_ct.mean(dim=1)  # [B, d_model]

                # === 第二次 Cross-Attention（CT → cfDNA）===
                attn_ct_to_cf, _ = self.model.multihead_attn_ct_to_cf(
                    query=k_ct, key=cfdna_tokens, value=cfdna_tokens
                )
                attn_pooled_ct = attn_ct_to_cf.squeeze(1)  # [B, d_model]

                # === 保存中间变量以供 hook 提取 ===
                pooled = cfdna_tokens.mean(dim=1)
                self.model._pooled = pooled
                self.model._attn_pooled_cf = attn_pooled_cf  # cfDNA 融合 CT
                self.model._attn_pooled_ct = attn_pooled_ct  # CT 融合 cfDNA

                # === 融合后送入分类器 ===
                fused = self.model.fusion_layer(torch.cat([attn_pooled_cf, attn_pooled_ct], dim=1))
                return self.model.classifier(fused)

            self.model.forward = new_forward
            self.model.fusion_layer.register_forward_hook(hook_fusion)

        elif extract_layer == 'classifier':
            def hook_classifier(module, input, output):
                self.features = input[0].detach()
            self.model.classifier.register_forward_hook(hook_classifier)

        elif extract_layer == 'raw_modalities':
            def hook_raw(zs, ct_images=None):
                self.raw_features = zs.detach()

            original_forward = self.model.forward

            def new_forward(zs, ct_images=None):
                hook_raw(zs)
                return original_forward(zs, ct_images)

            self.model.forward = new_forward
    
    def forward(self, *args, **kwargs):
        """前向传播，返回提取的特征"""
        self.model(*args, **kwargs)
        
        if self.extract_layer == 'raw_modalities':
            return self.raw_features
        else:
            return self.features

def extract_features(model, encoders, loader, device, extract_layer='fusion', sample_ids=None):
    """
    提取所有样本的潜在特征
    
    参数:
        model: 特征提取器模型
        encoders: 编码器字典
        loader: 数据加载器
        device: 设备
        extract_layer: 提取特征的层
        sample_ids: 如果提供，将保存样本ID
        
    返回:
        features: 提取的特征
        labels: 对应的标签
        ids: 样本ID（如果提供）
    """
    features = []
    labels = []
    ids = [] if sample_ids is not None else None
    
    model.eval()
    for encoder in encoders.values():
        encoder.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # 处理常规模态数据
            modality_features = []
            for i, mod in enumerate(encoders.keys()):
                if f'X{i}' in batch:
                    x = batch[f'X{i}'].to(device)
                    modality_features.append(encoders[mod](x))
                else:
                    print(f"警告: 模态 {mod} (X{i}) 在批次中不存在")
            
            # 确保有模态数据
            if not modality_features:
                print(f"警告: 批次 {i} 没有有效的模态数据，跳过")
                continue
                
            # 将所有模态特征堆叠在一起
            zs = torch.stack(modality_features, dim=1)  # [B, M, D]
            
            # 处理CT数据
            ct_data = batch.get('CT')
            if ct_data is not None:
                ct_data = ct_data.to(device)
                feature = model(zs, ct_data)
                # 打印调试信息
                if i == 0:  # 只打印第一个批次的信息
                    print(f"使用CT模态，CT数据形状: {ct_data.shape}")
            else:
                feature = model(zs)
                if i == 0:  # 只打印第一个批次的信息
                    print("未使用CT模态")
            
            # 如果是原始模态特征，需要特殊处理
            if extract_layer == 'raw_modalities':
                # 对每个样本，将所有模态特征展平
                batch_size, n_modalities, dim = feature.shape
                feature = feature.reshape(batch_size, -1)  # [B, M*D]
            
            features.append(feature.cpu().numpy())
            labels.append(batch['y'].numpy())
            
            # 如果提供了样本ID，保存它们
            if sample_ids is not None:
                batch_ids = np.arange(i * loader.batch_size, 
                                     min((i + 1) * loader.batch_size, len(loader.dataset)))
                ids.extend(batch_ids)
                
            # 打印第一个批次的特征形状
            if i == 0:
                print(f"提取的特征形状: {feature.shape}")
    
    if ids is not None:
        return np.vstack(features), np.concatenate(labels), np.array(ids)
    else:
        return np.vstack(features), np.concatenate(labels)

def reduce_dimensions(features, method='tsne', n_components=3, **kwargs):
    """
    使用指定方法降维
    
    参数:
        features: 高维特征
        method: 降维方法，可选 'tsne', 'pca', 'umap'
        n_components: 降维后的维度
        **kwargs: 传递给降维方法的额外参数
        
    返回:
        降维后的特征
    """
    print(f"正在使用{method}进行{n_components}D降维...")
    
    if method.lower() == 'tsne':
        # 默认参数
        params = {
            'perplexity': 30,
            'random_state': 42,
            'n_iter': 1000
        }
        params.update(kwargs)
        reducer = TSNE(n_components=n_components, **params)
        
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            print("警告: UMAP未安装，将使用t-SNE替代")
            return reduce_dimensions(features, method='tsne', n_components=n_components)
        
        # 默认参数
        params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'random_state': 42
        }
        params.update(kwargs)
        reducer = UMAP(n_components=n_components, **params)
        
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    return reducer.fit_transform(features)

def visualize_3d(features_3d, labels, output_dir, method='tsne', 
                 color_map='viridis', point_size=5, opacity=0.7,
                 class_names=None, title=None):
    """
    创建3D可视化
    
    参数:
        features_3d: 3D特征
        labels: 标签
        output_dir: 输出目录
        method: 降维方法名称（用于文件名）
        color_map: 颜色映射
        point_size: 点大小
        opacity: 点透明度
        class_names: 类别名称字典 {0: 'Class A', 1: 'Class B', ...}
        title: 图表标题
    """
    # 创建DataFrame
    df = pd.DataFrame({
        'x': features_3d[:, 0],
        'y': features_3d[:, 1],
        'z': features_3d[:, 2],
        'label': labels
    })
    
    # 如果提供了类别名称，创建新列
    if class_names is not None:
        df['class'] = df['label'].map(class_names)
        color_column = 'class'
    else:
        color_column = 'label'
    
    # 设置标题
    if title is None:
        title = f'3D {method.upper()} Visualization of Latent Features'
    
    # 使用plotly创建交互式3D散点图
    fig = px.scatter_3d(
        df, 
        x='x', 
        y='y', 
        z='z',
        color=color_column,
        title=title,
        labels={'color': 'Class'},
        color_continuous_scale=color_map if class_names is None else None,
        size_max=point_size,
        opacity=opacity
    )
    
    # 调整布局
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # 保存为HTML文件（可交互）
    html_path = os.path.join(output_dir, f'{method}_3d.html')
    fig.write_html(html_path)
    print(f"交互式3D可视化已保存到: {html_path}")
    
    # 保存为静态图片
    png_path = os.path.join(output_dir, f'{method}_3d.png')
    fig.write_image(png_path, width=1200, height=800)
    print(f"静态图片已保存到: {png_path}")
    
    return fig

def save_features(features, labels, output_dir, method='raw', sample_ids=None):
    """保存特征和标签到文件"""
    output_file = os.path.join(output_dir, f'{method}_features.npz')
    
    save_dict = {
        'features': features,
        'labels': labels
    }
    
    if sample_ids is not None:
        save_dict['sample_ids'] = sample_ids
    
    np.savez(output_file, **save_dict)
    print(f"特征和标签已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='3D visualization of latent features')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--data_file', type=str, required=True, help='NPZ格式的数据文件路径')
    parser.add_argument('--modalities', nargs='+', required=True, help='模态列表')
    parser.add_argument('--checkpoint_dir', type=str, default='./pretrained', help='编码器检查点目录')
    parser.add_argument('--output_dir', type=str, default='./vis_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--latent_dim', type=int, default=256, help='潜在空间维度')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca', 'umap'], 
                        help='降维方法')
    parser.add_argument('--perplexity', type=float, default=30, help='t-SNE困惑度参数')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP邻居数量')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP最小距离')
    parser.add_argument('--extract_layer', type=str, default='fusion', 
                        choices=['fusion', 'classifier', 'raw_modalities'],
                        help='从哪一层提取特征')
    parser.add_argument('--color_map', type=str, default='viridis', help='颜色映射')
    parser.add_argument('--point_size', type=int, default=5, help='点大小')
    parser.add_argument('--opacity', type=float, default=0.7, help='点透明度')
    parser.add_argument('--save_features', action='store_true', help='保存原始特征')
    parser.add_argument('--class_names', type=str, default=None, 
                        help='类别名称JSON文件，格式: {"0": "Class A", "1": "Class B", ...}')
    parser.add_argument('--title', type=str, default=None, help='图表标题')
    parser.add_argument('--include_ct', action='store_true', help='是否包含CT模态数据')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载类别名称（如果提供）
    class_names = None
    if args.class_names:
        if os.path.exists(args.class_names):
            with open(args.class_names, 'r') as f:
                class_names = json.load(f)
        else:
            try:
                class_names = json.loads(args.class_names)
            except:
                print(f"警告: 无法解析类别名称: {args.class_names}")
    
    # 加载数据
    dataset = MultiModalDataset(args.data_file, args.modalities, include_ct=args.include_ct)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"加载数据: {args.data_file}")
    print(f"模态: {args.modalities}")
    print(f"包含CT模态: {args.include_ct}")
    print(f"样本数: {len(dataset)}")
    
    # 加载编码器
    encoders = {mod: load_encoder(mod, args.checkpoint_dir, args.latent_dim).to(args.device) 
                for mod in args.modalities}
    
    # 加载模型
    model = CrossAttentionFusion(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        num_classes=2,
        train=False
    ).to(args.device)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()  # 确保模型处于评估模式
    
    # 添加属性用于存储中间特征
    model._pooled = None
    model._attn_pooled = None
    
    print("模型加载完成，参数数量:", sum(p.numel() for p in model.parameters()))
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(model, extract_layer=args.extract_layer)
    
    # 提取特征
    print(f"正在从{args.extract_layer}层提取特征...")
    if args.save_features:
        features, labels, sample_ids = extract_features(
            feature_extractor, encoders, loader, args.device, 
            extract_layer=args.extract_layer, sample_ids=True
        )
    else:
        features, labels = extract_features(
            feature_extractor, encoders, loader, args.device,
            extract_layer=args.extract_layer
        )
        sample_ids = None
    
    print(f"提取的特征形状: {features.shape}")
    
    # 保存原始特征（如果需要）
    if args.save_features:
        save_features(features, labels, args.output_dir, 'raw', sample_ids)
    
    # 降维参数
    dim_reduction_params = {}
    if args.method == 'tsne':
        dim_reduction_params['perplexity'] = args.perplexity
    elif args.method == 'umap':
        dim_reduction_params['n_neighbors'] = args.n_neighbors
        dim_reduction_params['min_dist'] = args.min_dist
    
    # 降维
    features_3d = reduce_dimensions(
        features, method=args.method, n_components=3, **dim_reduction_params
    )
    
    # 保存降维后的特征（如果需要）
    if args.save_features:
        save_features(features_3d, labels, args.output_dir, args.method, sample_ids)
    
    # 可视化
    visualize_3d(
        features_3d, labels, args.output_dir, args.method,
        color_map=args.color_map, point_size=args.point_size,
        opacity=args.opacity, class_names=class_names,
        title=args.title
    )
    
    print("可视化完成!")

if __name__ == '__main__':
    main()