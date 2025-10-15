"""
评估模块 - 处理模型评估相关的功能
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from fusion_models import CrossAttentionFusion
from dataset import load_data
from training_utils import load_encoder, create_ct_model, evaluate


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, args, feature_dims):
        self.args = args
        self.feature_dims = feature_dims
        
    def _create_model_components(self):
        """创建模型组件"""
        # 创建编码器
        encoders = {}
        for mod in self.args.modalities:
            encoders[mod] = load_encoder(
                mod, None, self.args.latent_dim, self.feature_dims
            ).to(self.args.device)
        
        # 创建CT模型
        ct_model = None if self.args.no_ct else create_ct_model(
            self.args.ct_model_path, self.args.device
        )
        
        # 创建主模型
        model = CrossAttentionFusion(
            dim_latent=self.args.latent_dim,
            n_modalities=len(self.args.modalities),
            num_classes=2,
            ct_feature_extractor=ct_model,
            finetune_ct=self.args.finetune_ct,
            use_ct=not self.args.no_ct,
            use_se_block=not self.args.no_se_block
        ).to(self.args.device)
        
        return model, encoders
    
    def evaluate_single_fold(self, test_loader, cv_dir, fold_num):
        """评估单个fold的模型"""
        model, encoders = self._create_model_components()
        
        # 加载模型权重
        model_path = os.path.join(cv_dir, f"best_model_fold_{fold_num}.pth")
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return None
            
        model.load_state_dict(torch.load(model_path, map_location=self.args.device, weights_only=True))
        
        # 评估
        results = evaluate(model, encoders, test_loader, self.args.device)
        
        # 添加调试信息
        logger.info(f"单模型 Fold {fold_num} 测试集结果详情:")
        logger.info(f"  样本数: {len(results['labels'])}")
        unique_preds, pred_counts = np.unique(results['preds'], return_counts=True)
        unique_labels, label_counts = np.unique(results['labels'], return_counts=True)
        logger.info(f"  预测分布: {dict(zip(unique_preds, pred_counts))}")
        logger.info(f"  标签分布: {dict(zip(unique_labels, label_counts))}")
        probs_array = np.array(results['probs'])
        logger.info(f"  概率范围: [{probs_array.min():.4f}, {probs_array.max():.4f}]")
        logger.info(f"  概率均值: {probs_array.mean():.4f} ± {probs_array.std():.4f}")
        
        return results
    
    def evaluate_ensemble(self, test_loader, cv_dir, fold_results):
        """集成评估所有fold的模型"""
        all_probs = []
        all_labels = None
        successful_folds = 0
        
        logger.info(f"加载 {len(fold_results)} 个fold的模型进行集成评估...")
        
        for fold_result in fold_results:
            fold_num = fold_result['fold']
            
            model, encoders = self._create_model_components()
            
            # 加载模型权重
            model_path = os.path.join(cv_dir, f"best_model_fold_{fold_num}.pth")
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.args.device, weights_only=True))
                    
                    # 获取预测
                    fold_results_dict = evaluate(model, encoders, test_loader, self.args.device)
                    all_probs.append(fold_results_dict['probs'])
                    
                    # 只在第一次时保存标签
                    if all_labels is None:
                        all_labels = fold_results_dict['labels']
                    
                    successful_folds += 1
                    logger.debug(f"成功加载 Fold {fold_num}")
                    
                except Exception as e:
                    logger.error(f"Fold {fold_num} 加载失败: {e}")
            else:
                logger.warning(f"Fold {fold_num} 模型文件不存在")
        
        if successful_folds == 0:
            logger.error("没有成功加载任何fold的模型")
            return None
        
        # 计算集成结果
        logger.info(f"成功加载 {successful_folds}/{len(fold_results)} 个模型")
        
        # 确保数据类型正确
        all_probs = np.array(all_probs)  # shape: [n_folds, n_samples]
        all_labels = np.array(all_labels)  # shape: [n_samples]
        
        logger.info(f"集成概率形状: {all_probs.shape}, 标签形状: {all_labels.shape}")
        
        # 计算平均概率
        ensemble_probs_pos = np.mean(all_probs, axis=0)
        ensemble_probs_neg = 1 - ensemble_probs_pos
        ensemble_probs = np.column_stack([ensemble_probs_neg, ensemble_probs_pos])
        ensemble_preds = (ensemble_probs_pos > 0.5).astype(int)
        
        # 计算集成指标
        ensemble_accuracy = np.mean(ensemble_preds == all_labels)
        
        # 详细调试信息
        unique_preds, pred_counts = np.unique(ensemble_preds, return_counts=True)
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        logger.info(f"集成预测分布: {dict(zip(unique_preds, pred_counts))}")
        logger.info(f"真实标签分布: {dict(zip(unique_labels, label_counts))}")
        logger.info(f"集成概率范围: [{ensemble_probs_pos.min():.4f}, {ensemble_probs_pos.max():.4f}]")
        logger.info(f"集成概率均值: {ensemble_probs_pos.mean():.4f} ± {ensemble_probs_pos.std():.4f}")
        
        # 检查是否所有样本都被预测为同一类
        if len(unique_preds) == 1:
            logger.warning(f"⚠️ 集成模型预测所有样本为类别 {unique_preds[0]}！这可能表明有问题")
            
        # 检查概率分布是否异常
        extreme_probs = np.sum((ensemble_probs_pos < 0.01) | (ensemble_probs_pos > 0.99))
        if extreme_probs > len(ensemble_probs_pos) * 0.8:
            logger.warning(f"⚠️ {extreme_probs}/{len(ensemble_probs_pos)} 样本的概率过于极端 (<0.01 或 >0.99)")
            
        # 显示单个fold的表现对比
        logger.info(f"参与集成的fold数: {successful_folds}")
        for i, probs in enumerate(all_probs):
            fold_preds = (probs > 0.5).astype(int)
            fold_acc = np.mean(fold_preds == all_labels)
            fold_f1 = f1_score(all_labels, fold_preds, average='binary', zero_division=0)
            logger.info(f"  Fold {i+1} 在测试集: Acc={fold_acc:.4f}, F1={fold_f1:.4f}")
        
        # 使用sklearn计算F1分数（更可靠）
        try:
            ensemble_f1 = f1_score(all_labels, ensemble_preds, average='binary', zero_division=0)
            logger.info(f"sklearn计算的F1: {ensemble_f1:.4f}")
        except Exception as e:
            logger.warning(f"F1计算失败: {e}")
            # 手动计算作为备选
            tp = np.sum((ensemble_preds == 1) & (all_labels == 1))
            fp = np.sum((ensemble_preds == 1) & (all_labels == 0))
            fn = np.sum((ensemble_preds == 0) & (all_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ensemble_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            logger.info(f"手动计算 - TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {ensemble_f1:.4f}")
        
        # 使用sklearn计算AUC（更可靠）
        try:
            ensemble_auc = roc_auc_score(all_labels, ensemble_probs_pos)
        except Exception as e:
            logger.warning(f"AUC计算失败: {e}")
            # 手动计算作为备选
            pos_indices = np.where(all_labels == 1)[0]
            neg_indices = np.where(all_labels == 0)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                ensemble_auc = 0.5
            else:
                total_comparisons = 0
                correct_comparisons = 0
                
                for pos_idx in pos_indices:
                    for neg_idx in neg_indices:
                        total_comparisons += 1
                        if ensemble_probs_pos[pos_idx] > ensemble_probs_pos[neg_idx]:
                            correct_comparisons += 1
                        elif ensemble_probs_pos[pos_idx] == ensemble_probs_pos[neg_idx]:
                            correct_comparisons += 0.5
                
                ensemble_auc = correct_comparisons / total_comparisons
        
        return {
            'accuracy': ensemble_accuracy,
            'f1': ensemble_f1, 
            'auc': ensemble_auc,
            'probs': ensemble_probs_pos,
            'preds': ensemble_preds,
            'labels': all_labels,
            'folds_used': successful_folds
        }


def evaluate_on_test_set(args, feature_selectors, cv_results):
    """在独立测试集上评估模型"""
    if not args.test_file or not args.use_test_set:
        logger.info("跳过测试集评估 - 未指定测试文件或未启用测试集评估")
        return
    
    logger.info(f"开始独立测试集评估: {args.test_file}")
    
    # 加载测试数据
    try:
        test_dataset, _ = load_data(
            args.test_file, 
            args.modalities, 
            include_ct=not args.no_ct,
            feature_selectors=feature_selectors,
            training=False
        )
        
        if len(test_dataset) == 0:
            logger.warning("测试集为空，跳过评估")
            return
            
        logger.info(f"测试集加载成功，样本数: {len(test_dataset)}")
        
    except Exception as e:
        logger.error(f"测试集加载失败: {e}")
        try:
            # 尝试不使用特征选择器加载
            test_dataset, _ = load_data(
                args.test_file, 
                args.modalities, 
                include_ct=not args.no_ct,
                feature_selectors=None,
                training=False
            )
            logger.info(f"测试集加载成功 (无特征选择)，样本数: {len(test_dataset)}")
        except Exception as e2:
            logger.error(f"测试集加载完全失败: {e2}")
            return
    
    # 获取特征维度
    feature_dims = {}
    sample = test_dataset[0]
    for i, mod in enumerate(args.modalities):
        key = f'X{i}'
        if key in sample:
            feature_dims[mod] = sample[key].shape[0]
        else:
            logger.error(f"测试样本中缺少 {key} 键")
            return
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    evaluator = ModelEvaluator(args, feature_dims)
    
    # 获取最佳fold和CV目录
    best_fold = cv_results['best_fold']
    cv_dir = cv_results['cv_dir']
    fold_results = cv_results['results']
    
    # 在测试集上评估所有fold模型，计算平均性能
    logger.info("在独立测试集上评估所有fold模型...")
    all_test_results = []
    
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        logger.info(f"  评估 Fold {fold_num}...")
        test_result = evaluator.evaluate_single_fold(test_loader, cv_dir, fold_num)
        if test_result:
            test_result['fold'] = fold_num
            test_result['val_auc'] = fold_result['auc']  # 记录验证集AUC
            all_test_results.append(test_result)
    
    if not all_test_results:
        logger.error("没有成功评估任何fold在测试集上的表现")
        return
    
    # 计算跨fold的平均性能和标准差
    test_accuracies = [r['accuracy'] for r in all_test_results]
    test_f1s = [r['f1'] for r in all_test_results]
    test_aucs = [r['auc'] for r in all_test_results]
    
    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    mean_f1 = np.mean(test_f1s)
    std_f1 = np.std(test_f1s)
    mean_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs)
    
    # 显示详细结果
    logger.info("="*80)
    logger.info("独立测试集性能评估 - 跨fold平均结果:")
    logger.info(f"准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"F1分数:  {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"AUC:    {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info("")
    logger.info("各fold详细结果:")
    for i, result in enumerate(all_test_results):
        logger.info(f"  Fold {result['fold']}: Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, AUC={result['auc']:.4f}")
    logger.info("="*80)
    
    # 保存统计结果（使用测试集的结果，不是验证集的结果）
    test_results = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'all_accuracies': test_accuracies,  # 使用测试集结果
        'all_f1s': test_f1s,
        'all_aucs': test_aucs,
        'num_folds': len(all_test_results),
        'fold_results': all_test_results,  # 使用测试集结果
        'test_samples': len(test_dataset)
    }
    
    # 保存结果文件
    test_results_file = os.path.join(cv_dir, 'test_results.npz')
    np.savez(test_results_file, **test_results)
    
    # 输出结果摘要
    logger.info(f"\n最终测试集评估结果 (样本数: {len(test_dataset)})")
    logger.info(f"平均AUC: {mean_auc:.4f} ± {std_auc:.4f} (跨{len(all_test_results)}个fold)")
    logger.info(f"平均F1: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return test_results


def evaluate_single_training(args, results, output_dir, feature_selectors):
    """评估单次训练的模型"""
    if not args.test_file:
        logger.info("跳过测试集评估 - 未指定测试文件")
        return
    
    logger.info(f"单次训练模型测试集评估: {args.test_file}")
    
    # 加载测试数据 (与上面类似的逻辑，但简化)
    try:
        test_dataset, _ = load_data(
            args.test_file, 
            args.modalities, 
            include_ct=not args.no_ct,
            feature_selectors=feature_selectors,
            training=False
        )
        
        if len(test_dataset) == 0:
            logger.warning("测试集为空")
            return
            
        logger.info(f"测试集样本数: {len(test_dataset)}")
        
    except Exception as e:
        logger.error(f"测试集加载失败: {e}")
        return
    
    # 获取特征维度并评估
    feature_dims = {}
    sample = test_dataset[0]
    for i, mod in enumerate(args.modalities):
        key = f'X{i}'
        if key in sample:
            feature_dims[mod] = sample[key].shape[0]
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 加载训练好的模型
    model_path = os.path.join(output_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    evaluator = ModelEvaluator(args, feature_dims)
    model, encoders = evaluator._create_model_components()
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    
    test_results = evaluate(model, encoders, test_loader, args.device)
    
    logger.info(f"测试结果 - 准确率: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}, AUC: {test_results['auc']:.4f}")
    
    return test_results