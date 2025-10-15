"""
简化的多模态分类训练脚本
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabulate import tabulate
from tqdm import tqdm

from fusion_models import CrossAttentionFusion
from dataset import load_data, create_feature_selectors, MultiModalDataset
from training_utils import (
    load_encoder, create_ct_model, run_single_fold, 
    evaluate, plot_roc_curve, plot_training_curves, plot_all_folds_curves
)
from evaluation import evaluate_on_test_set, evaluate_single_training

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置cudnn以确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子设置为: {seed}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multi-modal Classification Training')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True, help='Path to training data NPZ file')
    parser.add_argument('--test_file', type=str, help='Path to test data NPZ file')
    parser.add_argument('--modalities', nargs='+', default=['Frag', 'PFE', 'NDR', 'NDR2K'], 
                       help='List of modalities to use')
    
    # Feature selection parameters
    parser.add_argument('--use_feature_selection', action='store_true', help='Enable cfDNA feature selection')
    parser.add_argument('--feature_selection_method', type=str, default='combined',
                       choices=['variance', 'kbest', 'rfe', 'combined', 'lasso_voting', 'elastic_voting', 'pca'],
                       help='Feature selection method')
    parser.add_argument('--k_features', type=int, default=50, help='Number of features to select per cfDNA modality')
    parser.add_argument('--lasso_voting_threshold', type=float, default=0.6, help='Voting threshold for LASSO feature selection')
    parser.add_argument('--lasso_n_runs', type=int, default=10, help='Number of LASSO runs for voting')
    parser.add_argument('--elastic_l1_ratio', type=float, default=0.7, help='ElasticNet L1 ratio')
    parser.add_argument('--variance_threshold', type=float, default=0.0001, help='Variance threshold for initial filtering')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension for encoders')
    parser.add_argument('--no_ct', action='store_true', help='Disable CT modality')
    parser.add_argument('--ct_model_path', type=str, default='CT/resgsca_checkpoint/best_model.pth', 
                       help='Path to pre-trained CT model')
    parser.add_argument('--finetune_ct', action='store_true', help='Fine-tune CT feature extractor')
    parser.add_argument('--no_se_block', action='store_true', help='Disable SE block in model')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune encoders')
    parser.add_argument('--l1_lambda', type=float, default=0.0, help='L1 regularization strength')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=4, help='Early stopping patience')
    parser.add_argument('--strong_augment', action='store_true', help='Use strong data augmentation')
    parser.add_argument('--aug_prob', type=float, default=0.8, help='Probability of applying augmentation')
    
    # Cross-validation parameters
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/validation split ratio for single training')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--use_test_set', action='store_true', help='Evaluate on independent test set after training')
    parser.add_argument('--test_eval_strategy', type=str, default='both', 
                       choices=['best', 'ensemble', 'both'], help='Test evaluation strategy')
    
    # Reproducibility parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Class imbalance handling
    parser.add_argument('--auto_class_weights', action='store_true', default=True, 
                       help='Automatically calculate class weights based on class distribution (default: True)')
    parser.add_argument('--no_auto_class_weights', dest='auto_class_weights', action='store_false',
                       help='Disable automatic class weight calculation')
    
    # Cross-fold error analysis
    parser.add_argument('--analyze_errors', action='store_true', 
                       help='Analyze samples that are consistently misclassified across folds')
    parser.add_argument('--min_error_folds', type=int, default=2, 
                       help='Minimum number of folds where sample must be wrong to be flagged')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args


def cross_validate(args, dataset, feature_selectors):
    """进行K折交叉验证训练"""
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv_dir = os.path.join(args.output_dir, f'cv_results_{timestamp}')
    os.makedirs(cv_dir, exist_ok=True)
    
    logger.info(f"开始{args.k_folds}折交叉验证")
    logger.info(f"结果保存到: {cv_dir}")
    
    # 获取特征维度
    feature_dims = {}
    sample = dataset[0]
    for i, mod in enumerate(args.modalities):
        key = f'X{i}'
        if key in sample:
            feature_dims[mod] = sample[key].shape[0]
            logger.info(f"{mod}特征维度: {feature_dims[mod]}")
    
    # 交叉验证设置
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), dataset.y), 1):
        logger.info(f"\n{'='*15} Fold {fold}/{args.k_folds} {'='*15}")
        
        # 重新初始化编码器参数，避免fold间参数泄露
        logger.debug("重新初始化编码器参数")
        encoders = {mod: load_encoder(mod, None, args.latent_dim, feature_dims).to(args.device) 
                   for mod in args.modalities}

        for encoder in encoders.values():
            if args.finetune:
                encoder.train()
            else:
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad = False

        # 每个fold都重新创建CT模型
        ct_model = None if args.no_ct else create_ct_model(args.ct_model_path, args.device)
        
        # 创建训练和验证数据集
        train_dataset_fold = MultiModalDataset(
            args.data_file, args.modalities,
            indices=train_idx,
            include_ct=not args.no_ct,
            feature_selectors=feature_selectors,
            training=True,  # 启用数据增强
            strong_augment=args.strong_augment  # 强增强选项
        )
        
        val_dataset_fold = MultiModalDataset(
            args.data_file, args.modalities,
            indices=val_idx,
            include_ct=not args.no_ct,
            feature_selectors=feature_selectors,
            training=False,  # 禁用数据增强
            strong_augment=False  # 验证集不使用增强
        )
        
        train_loader = DataLoader(train_dataset_fold, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=args.batch_size, shuffle=False)
        
        # 每个fold都重新创建主模型
        model = CrossAttentionFusion(
            dim_latent=args.latent_dim,
            n_modalities=len(args.modalities),
            num_classes=2,
            ct_feature_extractor=ct_model,
            finetune_ct=args.finetune_ct,
            use_ct=not args.no_ct,
            use_se_block=not args.no_se_block
        ).to(args.device)
        
        fold_results, best_epoch = run_single_fold(
            model, encoders, train_loader, val_loader, args, fold, cv_dir
        )
        
        if fold_results is None:
            logger.warning(f"Fold {fold} 训练失败")
            continue
            
        fold_results['fold'] = fold
        fold_results['best_epoch'] = best_epoch
        all_results.append(fold_results)
        
        # 获取最终的训练和验证损失（从training_history的最后一个值）
        if 'training_history' in fold_results:
            final_train_loss = fold_results['training_history']['train_losses'][-1] if fold_results['training_history']['train_losses'] else 0.0
            # 获取最佳epoch的验证损失，而不是最后一个epoch的
            final_val_loss = fold_results.get('loss', 0.0)  # 这是最佳epoch的验证损失
        else:
            final_train_loss = 0.0
            final_val_loss = fold_results.get('loss', 0.0)
        
        logger.info(f"Fold {fold} 完成 - Acc: {fold_results['accuracy']:.4f}, AUC: {fold_results['auc']:.4f}, F1: {fold_results['f1']:.4f}, Train_Loss: {final_train_loss:.4f}, Val_Loss: {final_val_loss:.4f}")
        
        # 保存fold详细结果（包含错误样本信息）
        fold_detail_file = os.path.join(cv_dir, f'fold_{fold}_results.json')
        fold_detail = {
            'fold_number': fold,
            'accuracy': fold_results['accuracy'],
            'f1': fold_results['f1'],
            'auc': fold_results['auc'],
            'loss': fold_results.get('loss', 0.0),
            'errors': fold_results.get('errors', []),
            'num_errors': len(fold_results.get('errors', [])),
            'total_samples': len(fold_results.get('labels', [])),
            'error_rate': len(fold_results.get('errors', [])) / len(fold_results.get('labels', [])) if fold_results.get('labels') else 0.0
        }
        
        try:
            with open(fold_detail_file, 'w', encoding='utf-8') as f:
                json.dump(fold_detail, f, indent=2, ensure_ascii=False)
            logger.debug(f"Fold {fold} 详细结果已保存: {len(fold_detail['errors'])} 个错误样本")
        except Exception as e:
            logger.warning(f"保存fold {fold} 详细结果失败: {e}")
        
        # 绘制单fold结果
        plot_roc_curve(fold_results['labels'], fold_results['probs'], cv_dir, f"fold_{fold}")
        plot_training_curves(fold_results, cv_dir, fold, show=False)
    
    # 处理交叉验证结果
    if not all_results:
        logger.error("所有fold都失败了")
        return None
    
    # 计算统计信息
    auc_scores = [r['auc'] for r in all_results]
    acc_scores = [r['accuracy'] for r in all_results] 
    f1_scores = [r['f1'] for r in all_results]
    
    results_dict = {
        'results': all_results,
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores),
        'mean_accuracy': np.mean(acc_scores),
        'std_accuracy': np.std(acc_scores),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'best_fold': all_results[np.argmax(auc_scores)]['fold'],
        'cv_dir': cv_dir
    }
    
    # 保存交叉验证结果
    np.savez(
        os.path.join(cv_dir, 'cv_results.npz'),
        **{k: v for k, v in results_dict.items() if k != 'cv_dir'}
    )
    
    # 输出结果表格
    table_data = []
    for result in all_results:
        table_data.append([
            result['fold'],
            f"{result['accuracy']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['auc']:.4f}",
            result['best_epoch']
        ])
    
    table_data.append([
        "平均值",
        f"{results_dict['mean_accuracy']:.4f} ± {results_dict['std_accuracy']:.4f}",
        f"{results_dict['mean_f1']:.4f} ± {results_dict['std_f1']:.4f}", 
        f"{results_dict['mean_auc']:.4f} ± {results_dict['std_auc']:.4f}", 
        "-"
    ])
    
    logger.info(f"\n交叉验证结果:")
    print(tabulate(table_data, headers=["Fold", "Accuracy", "F1 Score", "AUC Score", "Best Epoch"], tablefmt="grid"))
    
    logger.info(f"交叉验证完成！结果已保存到: {cv_dir}")
    
    # 绘制所有折的训练曲线对比图
    plot_all_folds_curves(all_results, cv_dir)
    
    # 跨fold错误样本分析
    if args.analyze_errors:
        logger.info("开始跨fold错误样本分析...")
        analyze_cross_fold_errors(args, dataset, feature_selectors, cv_dir, args.min_error_folds)
    
    # 独立测试集评估
    if args.test_file and args.use_test_set:
        logger.info("开始独立测试集评估...")
        evaluate_on_test_set(args, feature_selectors, results_dict)
        
    return results_dict


def run_single_training(args, dataset, feature_selectors=None):
    """运行单次训练（不使用交叉验证）"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'single_training_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"开始单次训练")
    logger.info(f"结果保存到: {output_dir}")
    
    # 获取特征维度
    feature_dims = {}
    sample = dataset[0]
    for i, mod in enumerate(args.modalities):
        key = f'X{i}'
        if key in sample:
            feature_dims[mod] = sample[key].shape[0]
    
    # 划分训练验证集
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)), 
        test_size=1-args.split_ratio, 
        stratify=dataset.y, 
        random_state=42
    )
    
    # 创建数据集
    train_dataset = MultiModalDataset(
        args.data_file, args.modalities,
        indices=train_indices,
        include_ct=not args.no_ct,
        feature_selectors=feature_selectors,
        training=True,
        strong_augment=args.strong_augment
    )
    
    val_dataset = MultiModalDataset(
        args.data_file, args.modalities,
        indices=val_indices,
        include_ct=not args.no_ct,
        feature_selectors=feature_selectors,
        training=False,
        strong_augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    encoders = {mod: load_encoder(mod, None, args.latent_dim, feature_dims).to(args.device) 
               for mod in args.modalities}
    
    for encoder in encoders.values():
        if args.finetune:
            encoder.train()
        else:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
    
    ct_model = None if args.no_ct else create_ct_model(args.ct_model_path, args.device)
    
    model = CrossAttentionFusion(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        num_classes=2,
        ct_feature_extractor=ct_model,
        finetune_ct=args.finetune_ct,
        use_ct=not args.no_ct,
        use_se_block=not args.no_se_block
    ).to(args.device)
    
    # 训练
    results, best_epoch = run_single_fold(
        model, encoders, train_loader, val_loader, args, 1, output_dir
    )
    
    if results:
        logger.info(f"训练完成 - AUC: {results['auc']:.4f}, F1: {results['f1']:.4f}")
        plot_roc_curve(results['labels'], results['probs'], output_dir)
        plot_training_curves(results, output_dir, show=False)
        
        # 独立测试集评估
        if args.test_file:
            logger.info("开始独立测试集评估...")
            evaluate_single_training(args, results, output_dir, feature_selectors)
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子确保可重复性
    set_seed(args.seed)
    
    logger.info(f"使用设备: {args.device}")
    logger.info(f"使用模态: {args.modalities}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建特征选择器
    feature_selectors = None
    if args.use_feature_selection:
        logger.info("创建cfDNA特征选择器...")
        feature_selectors = create_feature_selectors(
            args.data_file, 
            args.modalities, 
            method=args.feature_selection_method,
            k_features=args.k_features,
            variance_threshold=args.variance_threshold,
            lasso_voting_threshold=args.lasso_voting_threshold,
            lasso_n_runs=args.lasso_n_runs,
            elastic_l1_ratio=args.elastic_l1_ratio
        )
        logger.info(f"特征选择方法: {args.feature_selection_method}, 每个模态保留: {args.k_features} 个特征")
    
    # 加载训练数据
    train_dataset, label_dist = load_data(
        args.data_file, 
        args.modalities, 
        include_ct=not args.no_ct, 
        feature_selectors=feature_selectors,
        training=True
    )
    
    logger.info(f"加载训练数据: {args.data_file}, 样本数: {len(train_dataset)}")
    logger.info(f"标签分布: {label_dist}")
    
    # 运行训练
    if args.k_folds > 1:
        logger.info("运行交叉验证")
        results = cross_validate(args, train_dataset, feature_selectors)
    else:
        logger.info("运行单次训练")
        results = run_single_training(args, train_dataset, feature_selectors)
    
    if results:
        logger.info("训练完成!")
    else:
        logger.error("训练失败!")


def analyze_cross_fold_errors(args, dataset, feature_selectors, cv_dir, min_error_folds=2):
    """
    跨fold错误分析：用每个fold的模型对全量数据预测，找出持续预测错误的样本
    
    Args:
        args: 参数对象
        dataset: 完整数据集
        feature_selectors: 特征选择器
        cv_dir: 交叉验证结果目录
        min_error_folds: 最少错误fold数量
    """
    import glob
    from collections import defaultdict, Counter
    import matplotlib.pyplot as plt
    
    logger.info("=" * 60)
    logger.info("🔍 跨Fold样本错误分析")
    logger.info("=" * 60)
    
    # 1. 查找所有fold模型
    model_files = glob.glob(os.path.join(cv_dir, "best_model_fold_*.pth"))
    if not model_files:
        logger.error("未找到fold模型文件！")
        return
    
    fold_models = {}
    for model_file in model_files:
        try:
            fold_num = int(os.path.basename(model_file).split('_fold_')[1].split('.pth')[0])
            fold_models[fold_num] = model_file
        except (IndexError, ValueError):
            logger.warning(f"无法解析模型文件: {model_file}")
    
    logger.info(f"找到 {len(fold_models)} 个fold模型: {sorted(fold_models.keys())}")
    
    # 2. 创建数据加载器
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 3. 创建CT模型
    ct_model = None if args.no_ct else create_ct_model(args.ct_model_path, args.device)
    
    # 4. 获取特征维度并创建编码器
    sample = dataset[0]
    feature_dims = {}
    for i, mod in enumerate(args.modalities):
        key = f'X{i}'
        if key in sample:
            feature_dims[mod] = sample[key].shape[0]
    
    encoders = {mod: load_encoder(mod, None, args.latent_dim, feature_dims).to(args.device) 
               for mod in args.modalities if mod in feature_dims}
    
    # 5. 创建主模型
    model = CrossAttentionFusion(
        dim_latent=args.latent_dim,
        n_modalities=len(args.modalities),
        num_classes=2,  # 假设二分类
        use_ct=not args.no_ct,
        use_se_block=not args.no_se_block,
        ct_feature_extractor=ct_model
    ).to(args.device)
    
    # 6. 存储预测结果
    sample_predictions = {}  # sample_idx -> {fold: {'pred': label, 'prob': prob, 'correct': bool}}
    sample_true_labels = {}  # sample_idx -> true_label
    sample_ids = {}  # sample_idx -> original_id
    
    # 7. 用每个fold模型预测全量数据
    for fold_num in sorted(fold_models.keys()):
        model_path = fold_models[fold_num]
        logger.info(f"使用 Fold {fold_num} 模型进行预测...")
        
        # 加载模型权重
        try:
            model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
            model.eval()
        except Exception as e:
            logger.error(f"加载 Fold {fold_num} 模型失败: {e}")
            continue
        
        # 预测
        sample_idx = 0
        correct_count = 0
        
        with torch.no_grad():
            for batch in full_loader:
                yb = batch['y'].to(args.device)
                batch_size = yb.size(0)
                
                try:
                    # 特征提取
                    modality_features = []
                    for i, mod in enumerate(args.modalities):
                        if f'X{i}' in batch and mod in encoders:
                            features = encoders[mod](batch[f'X{i}'].to(args.device))
                            modality_features.append(features)
                    
                    if not modality_features:
                        sample_idx += batch_size
                        continue
                    
                    # 融合和预测
                    zs = torch.stack(modality_features, dim=1)
                    ct_data = batch.get('CT', None)
                    if ct_data is not None:
                        ct_data = ct_data.to(args.device)
                    
                    logits = model(zs, ct_data)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(1)
                    
                    # 记录结果
                    for i in range(batch_size):
                        idx = sample_idx + i
                        true_label = yb[i].item()
                        pred_label = preds[i].item()
                        pred_prob = probs[i, 1].item()  # 正类概率
                        is_correct = pred_label == true_label
                        
                        if is_correct:
                            correct_count += 1
                        
                        # 初始化记录
                        if idx not in sample_predictions:
                            sample_predictions[idx] = {}
                            sample_true_labels[idx] = true_label
                            # 保存原始样本ID
                            if 'id' in batch:
                                sample_ids[idx] = batch['id'][i]
                            else:
                                sample_ids[idx] = f"sample_{idx}"
                        
                        # 保存预测
                        sample_predictions[idx][fold_num] = {
                            'pred': pred_label,
                            'prob': pred_prob,
                            'correct': is_correct
                        }
                    
                    sample_idx += batch_size
                    
                except Exception as e:
                    logger.error(f"Fold {fold_num} 批次预测失败: {e}")
                    sample_idx += batch_size
                    continue
        
        fold_accuracy = correct_count / sample_idx if sample_idx > 0 else 0
        logger.info(f"Fold {fold_num} 准确率: {fold_accuracy:.3f} ({correct_count}/{sample_idx})")
    
    # 7. 分析持续错误样本
    logger.info(f"\n分析在≥{min_error_folds}个fold中预测错误的样本...")
    
    consistent_errors = {}
    total_folds = len(fold_models)
    
    for sample_idx, fold_predictions in sample_predictions.items():
        available_folds = len(fold_predictions)
        error_count = sum(1 for pred in fold_predictions.values() if not pred['correct'])
        
        if error_count >= min_error_folds and available_folds >= min_error_folds:
            true_label = sample_true_labels[sample_idx]
            error_rate = error_count / available_folds
            
            pred_labels = [p['pred'] for p in fold_predictions.values()]
            pred_probs = [p['prob'] for p in fold_predictions.values()]
            
            consistent_errors[sample_idx] = {
                'true_label': true_label,
                'error_count': error_count,
                'total_predictions': available_folds,
                'error_rate': error_rate,
                'predictions': pred_labels,
                'avg_prob': np.mean(pred_probs),
                'prediction_consistency': len(set(pred_labels)) == 1,
                'consistent_pred_label': pred_labels[0] if len(set(pred_labels)) == 1 else None
            }
    
    # 8. 生成报告
    if not consistent_errors:
        logger.info("✅ 未发现持续错误的样本，数据质量良好！")
        return
    
    logger.info(f"🚨 找到 {len(consistent_errors)} 个持续错误样本")
    
    # 分类统计
    very_high_error = [idx for idx, info in consistent_errors.items() if info['error_rate'] >= 0.8]
    high_error = [idx for idx, info in consistent_errors.items() if info['error_rate'] >= 0.6]
    consistent_wrong = [idx for idx, info in consistent_errors.items() 
                      if info['prediction_consistency'] and info['error_rate'] >= 0.6]
    
    logger.info(f"📊 分析结果:")
    logger.info(f"   🔴 极高错误率(≥80%): {len(very_high_error)} 个")
    logger.info(f"   🟠 高错误率(≥60%): {len(high_error)} 个")
    logger.info(f"   🏷️  可能标签错误: {len(consistent_wrong)} 个 (预测一致但与标签不符)")
    
    # 详细报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("跨Fold样本错误分析详细报告")
    report_lines.append("=" * 80)
    report_lines.append(f"总样本数: {len(sample_predictions)}")
    report_lines.append(f"分析fold数: {total_folds}")
    report_lines.append(f"持续错误样本: {len(consistent_errors)}")
    report_lines.append(f"极高错误率(≥80%): {len(very_high_error)} 个")
    report_lines.append(f"可能标签错误: {len(consistent_wrong)} 个")
    report_lines.append("")
    
    # 最严重的样本
    sorted_errors = sorted(consistent_errors.items(), 
                         key=lambda x: (x[1]['error_rate'], -x[1]['avg_prob']), 
                         reverse=True)
    
    report_lines.append("最严重的错误样本 (Top 20):")
    report_lines.append("-" * 80)
    for i, (sample_idx, info) in enumerate(sorted_errors[:20]):
        status = "⚠️极高" if info['error_rate'] >= 0.8 else "🔴高"
        consistency = "一致" if info['prediction_consistency'] else "不一致"
        pred_info = f"→{info['consistent_pred_label']}" if info['prediction_consistency'] else "混合"
        
        report_lines.append(f"{i+1:2d}. 样本#{sample_idx:<6} | 标签:{info['true_label']} {pred_info} | "
                           f"错误率:{info['error_rate']:.1%} ({info['error_count']}/{info['total_predictions']}) | "
                           f"概率:{info['avg_prob']:.3f} | {status}错误 | 预测{consistency}")
    
    report_lines.append("")
    report_lines.append("🔧 数据清洗建议:")
    report_lines.append("-" * 40)
    
    if very_high_error:
        report_lines.append(f"1. 【高优先级】检查以下 {len(very_high_error)} 个极高错误率样本:")
        for idx in very_high_error[:10]:
            info = consistent_errors[idx]
            pred_label = info['consistent_pred_label'] if info['prediction_consistency'] else '混合'
            original_id = sample_ids.get(idx, f"sample_{idx}")
            report_lines.append(f"   样本ID {original_id}: 标签{info['true_label']} vs 预测{pred_label} (错误率{info['error_rate']:.1%})")
        if len(very_high_error) > 10:
            report_lines.append(f"   ... 还有{len(very_high_error)-10}个")
    
    if consistent_wrong:
        report_lines.append(f"\n2. 【标签检查】以下 {len(consistent_wrong)} 个样本可能标签错误:")
        for idx in consistent_wrong[:5]:
            info = consistent_errors[idx]
            original_id = sample_ids.get(idx, f"sample_{idx}")
            report_lines.append(f"   样本ID {original_id}: 标签{info['true_label']} → 所有fold都预测为{info['consistent_pred_label']}")
    
    report_lines.append(f"\n3. 【预期改进】:")
    improvement = len(very_high_error) / len(sample_predictions) * 100
    report_lines.append(f"   移除/修正极高错误率样本后，预期准确率提升: ~{improvement:.1f}%")
    
    # 保存报告
    report_file = os.path.join(cv_dir, 'cross_fold_error_analysis.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"📄 详细报告已保存: {report_file}")
    
    # 简化版控制台输出
    if very_high_error:
        logger.info(f"\n⚠️  建议立即检查以下极高错误率样本:")
        for i, idx in enumerate(very_high_error[:5]):
            info = consistent_errors[idx]
            pred_label = info['consistent_pred_label'] if info['prediction_consistency'] else '不一致'
            original_id = sample_ids.get(idx, f"sample_{idx}")
            logger.info(f"   {i+1}. 样本ID {original_id}: 标签{info['true_label']} vs 预测{pred_label} (错误{info['error_count']}/{info['total_predictions']}次)")
        if len(very_high_error) > 5:
            logger.info(f"   ... 还有{len(very_high_error)-5}个，详见报告")
    
    logger.info(f"✅ 跨fold错误分析完成！")


if __name__ == "__main__":
    main()