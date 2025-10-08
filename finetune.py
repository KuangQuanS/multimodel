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
    evaluate, plot_roc_curve, plot_training_curves, plot_all_folds_curves
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
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval (print every N epochs)')
    parser.add_argument('--use_test_set', action='store_true', help='Evaluate on independent test set after training')
    parser.add_argument('--use_class_weights', action='store_true', 
                       help='Use class weights to handle imbalanced data')
    parser.add_argument('--class_weights', nargs=2, type=float, default=[1.0, 1.0],
                       help='Class weights [benign_weight, malignant_weight]')
    
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


def evaluate_test_set(args, feature_selectors, cv_results):
    """在独立测试集上评估最佳模型"""
    print(f"加载测试数据: {args.test_file}")
    
    try:
        # Load test data with same feature selectors as training
        # load_data返回(dataset, label_distribution)元组，需要解包
        test_dataset, test_label_dist = load_data(
            args.test_file, 
            args.modalities, 
            feature_selectors=feature_selectors
        )
        print(f"✅ 测试集加载成功，样本数: {len(test_dataset)}")
        
        # Check if dataset is empty
        if len(test_dataset) == 0:
            print("❌ 测试集为空，跳过评估")
            return
            
    except Exception as e:
        print(f"❌ 测试集加载失败: {e}")
        print("🔄 尝试不使用特征选择器加载测试集...")
        try:
            # Try loading without feature selectors
            test_dataset, test_label_dist = load_data(
                args.test_file, 
                args.modalities, 
                feature_selectors=None
            )
            print(f"✅ 测试集加载成功 (无特征选择)，样本数: {len(test_dataset)}")
        except Exception as e2:
            print(f"❌ 测试集加载完全失败: {e2}")
            print("跳过独立测试集评估")
            return
    
    # Get best fold model (highest AUC)
    best_fold_idx = np.argmax([r['auc'] for r in cv_results['fold_results']])
    best_fold = cv_results['fold_results'][best_fold_idx]['fold']
    
    print(f"使用最佳模型: Fold {best_fold} (AUC: {cv_results['fold_results'][best_fold_idx]['auc']:.3f})")
    
    # Load encoders and model
    encoders = {}
    feature_dims = {}
    
    try:
        # Get feature dimensions from test dataset
        sample = test_dataset[0]
        print(f"测试样本类型: {type(sample)}")
        
        # Ensure sample is a dictionary
        if not isinstance(sample, dict):
            print(f"❌ 测试样本不是字典格式，而是: {type(sample)}")
            return
            
        print(f"测试样本keys: {list(sample.keys())}")
        
        for i, mod in enumerate(args.modalities):
            key = f'X{i}'
            if key in sample:
                feature_dims[mod] = sample[key].shape[0]
                print(f"{mod} 特征维度: {feature_dims[mod]}")
            else:
                print(f"❌ 缺少模态 {mod} (key: {key})")
                print(f"可用的keys: {list(sample.keys())}")
                return
                
        for mod in args.modalities:
            encoders[mod] = load_encoder(mod, None, args.latent_dim, feature_dims).to(args.device)
            
    except Exception as e:
        print(f"❌ 特征维度获取失败: {e}")
        print(f"sample类型: {type(test_dataset[0]) if len(test_dataset) > 0 else 'Empty dataset'}")
        return
    
    # Create model
    ct_model = create_ct_model(args.ct_model_path, args.finetune_ct)
    model = CrossAttentionFusion(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        num_classes=2,
        ct_feature_extractor=ct_model,
        finetune_ct=args.finetune_ct
    ).to(args.device)
    
    # Load best model weights
    timestamp = cv_results['timestamp']
    cv_dir = os.path.join(args.output_dir, f"cv_results_{timestamp}")
    model_path = os.path.join(cv_dir, f"best_model_fold_{best_fold}.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"✅ 已加载模型: {model_path}")
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        return
    
    # Evaluate on test set
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_results = evaluate(model, encoders, test_loader, args.device)
    
    # Print results
    print(f"\n🎯 独立测试集结果:")
    print(f"  样本数量: {len(test_dataset)}")
    print(f"  准确率: {test_results['accuracy']:.4f}")
    print(f"  F1分数: {test_results['f1']:.4f}")
    print(f"  AUC分数: {test_results['auc']:.4f}")
    
    # Compare with CV results
    print(f"\n📊 与交叉验证对比:")
    print(f"  CV准确率: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"  CV F1分数: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")  
    print(f"  CV AUC分数: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    
    # Save test results
    test_results_path = os.path.join(cv_dir, 'test_results.npz')
    np.savez(test_results_path, 
             test_accuracy=test_results['accuracy'],
             test_f1=test_results['f1'],
             test_auc=test_results['auc'],
             test_labels=test_results['labels'],
             test_probs=test_results['probs'])
    
    print(f"  测试结果已保存: {test_results_path}")


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
        
        # 绘制单个折的训练曲线
        plot_training_curves(fold_results, cv_dir, fold, show=False)

    # Calculate cross-validation statistics
    if all_results:
        accuracy_scores = [r['accuracy'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]
        auc_scores = [r['auc'] for r in all_results]
        
        results_dict = {
            'timestamp': timestamp,
            'args': vars(args),
            'fold_results': all_results,
            'mean_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_auc': np.mean(auc_scores),
            'std_auc': np.std(auc_scores)
        }
        
        # Save results
        results_path = os.path.join(cv_dir, 'cv_results.npz')
        np.savez(results_path, **results_dict)
        
        # Print summary table with accuracy
        accuracy_scores = [result['accuracy'] for result in all_results]
        
        table_data = []
        for result in all_results:
            table_data.append([
                f"Fold {result['fold']}", 
                f"{result['accuracy']:.3f}",
                f"{result['f1']:.3f}", 
                f"{result['auc']:.3f}", 
                result['best_epoch']
            ])
        
        table_data.append([
            "Mean ± Std", 
            f"{np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}",
            f"{np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}", 
            f"{np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}", 
            "-"
        ])
        
        print(f"\n📊 交叉验证结果:")
        print(tabulate(table_data, headers=["Fold", "Accuracy", "F1 Score", "AUC Score", "Best Epoch"], tablefmt="grid"))
        
        print(f"\n✅ 交叉验证完成！结果已保存到: {cv_dir}")
        
        # 绘制所有折的训练曲线对比图
        plot_all_folds_curves(all_results, cv_dir)
        
        # Independent test set evaluation if provided
        if args.test_file and args.use_test_set:
            print(f"\n🧪 独立测试集评估...")
            evaluate_test_set(args, feature_selectors, results_dict)
            
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