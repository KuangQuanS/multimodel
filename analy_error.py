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