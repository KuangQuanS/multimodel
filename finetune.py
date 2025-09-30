#!/usr/bin/env python3
"""
重构后的训练脚本
使用模块化设计，代码更清晰
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fusion_models import CrossAttentionFusion
from src.data.dataset import load_data, create_feature_selectors
from src.utils.training_utils import (
    load_encoder, create_ct_model, run_single_fold, 
    evaluate, plot_roc_curve
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multi-modal Classification Training')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True, help='Path to training data NPZ file')
    parser.add_argument('--test_file', type=str, help='Path to test data NPZ file')
    parser.add_argument('--modalities', nargs='+', default=['Frag', 'PFE', 'NDR', 'NDR2K'], 
                       help='List of modalities to use')
    
    # Feature selection parameters
    parser.add_argument('--use_feature_selection', action='store_true', 
                       help='Enable cfDNA feature selection')
    parser.add_argument('--feature_selection_method', type=str, default='combined',
                       choices=['variance', 'kbest', 'rfe', 'combined'],
                       help='Feature selection method')
    parser.add_argument('--k_features', type=int, default=50,
                       help='Number of features to select per cfDNA modality')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension for encoders')
    parser.add_argument('--ct_model_path', type=str, help='Path to pretrained CT model')
    parser.add_argument('--finetune', action='store_true', help='Finetune encoders')
    parser.add_argument('--finetune_ct', action='store_true', help='Finetune CT model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluation interval')
    
    # Cross-validation parameters
    parser.add_argument('--cross_val', action='store_true', help='Run cross-validation')
    parser.add_argument('--k_folds', type=int, default=10, help='Number of folds for cross-validation')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, no training')
    parser.add_argument('--model_path', type=str, help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args


def run_cross_validation(args, dataset, feature_selectors=None):
    """运行交叉验证"""
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = os.path.join(args.output_dir, f"cv_results_{timestamp}")
    os.makedirs(cv_dir, exist_ok=True)

    # 获取实际特征维度
    feature_dims = {}
    if feature_selectors:
        # 通过检查数据集来获取特征选择后的维度
        sample = dataset[0]
        for i, mod in enumerate(args.modalities):
            feature_dims[mod] = sample[f'X{i}'].shape[0]
    
    # Load and configure encoders
    encoders = {mod: load_encoder(mod, None, args.latent_dim, feature_dims).to(args.device) 
               for mod in args.modalities}

    for encoder in encoders.values():
        if args.finetune:
            encoder.train()
        else:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False

    # Load CT model
    ct_model = create_ct_model(args.ct_model_path, args.device)
    
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), dataset.y), 1):
        print(f"\n{'='*20} Fold {fold}/{args.k_folds} {'='*20}")
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
        
        # Create model
        model = CrossAttentionFusion(
            dim_latent=args.latent_dim,  # Features per modality (256)
            n_modalities=len(args.modalities),  # Number of modalities
            num_classes=2,
            ct_feature_extractor=ct_model,
            finetune_ct=args.finetune_ct
        ).to(args.device)
        
        fold_results, best_epoch = run_single_fold(
            model, encoders, train_loader, val_loader, args, fold, cv_dir
        )
        
        if fold_results is None:
            print(f"[Fold {fold}] skipped due to invalid results.")
            continue
            
        fold_results.update({'fold': fold, 'best_epoch': best_epoch})
        all_results.append(fold_results)
        plot_roc_curve(fold_results['labels'], fold_results['probs'], cv_dir, f"fold_{fold}")

    # Calculate cross-validation statistics
    if all_results:
        f1_scores = [r['f1'] for r in all_results]
        auc_scores = [r['auc'] for r in all_results]
        
        results_dict = {
            'timestamp': timestamp,
            'args': vars(args),
            'fold_results': all_results,
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_auc': np.mean(auc_scores),
            'std_auc': np.std(auc_scores)
        }
        
        # Save results
        results_path = os.path.join(cv_dir, 'cv_results.npz')
        np.savez(results_path, **results_dict)
        
        # Print summary table
        table_data = []
        for result in all_results:
            table_data.append([
                f"Fold {result['fold']}", 
                f"{result['f1']:.3f}", 
                f"{result['auc']:.3f}", 
                result['best_epoch']
            ])
        
        table_data.append([
            "Mean ± Std", 
            f"{np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}", 
            f"{np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}", 
            "-"
        ])
        
        print(f"\n交叉验证结果:")
        print(tabulate(table_data, headers=["Fold", "F1 Score", "AUC Score", "Best Epoch"], tablefmt="grid"))
        
        print(f"\n结果已保存到: {cv_dir}")
    else:
        print("所有折都失败了，无法生成交叉验证结果")


def main():
    """主函数"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create feature selectors if enabled
    feature_selectors = None
    if args.use_feature_selection:
        print("🔍 创建cfDNA特征选择器...")
        feature_selectors = create_feature_selectors(
            args.data_file, 
            args.modalities, 
            method=args.feature_selection_method,
            k_features=args.k_features
        )
        print(f"特征选择方法: {args.feature_selection_method}, 每个模态保留: {args.k_features} 个特征")
    
    # Load training data
    train_dataset, label_dist = load_data(args.data_file, args.modalities, 
                                        include_ct=True, feature_selectors=feature_selectors)
    print(f"Loaded training data: {args.data_file}, samples: {len(train_dataset)}")
    print(f"Label distribution: {label_dist}")
    
    # Run cross-validation or single training
    if args.cross_val:
        print(f"\nRunning {args.k_folds}-fold cross-validation...")
        run_cross_validation(args, train_dataset, feature_selectors)
        return
    
    # Single training (not implemented in this simplified version)
    print("Single training mode not implemented in this version. Use --cross_val flag.")


if __name__ == "__main__":
    main()