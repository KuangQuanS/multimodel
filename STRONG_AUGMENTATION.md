# 强数据增强功能

本项目实现了CT数据的强增强功能，以提高模型的泛化能力和鲁棒性。

## 功能概述

强数据增强通过 `--strong_augment` 参数启用，专门针对CT图像数据：
- **CT图像强增强**: 12种不同的增强技术
- **cfDNA数据**: 保持原始特征，不进行额外增强

## 使用方法

在训练命令中添加 `--strong_augment` 参数：

```bash
python finetune.py \
    --modalities Frag PFE NDR NDR2K \
    --data_file /path/to/train.npz \
    --test_file /path/to/test.npz \
    --strong_augment \
    --other_parameters...
```

## CT强增强功能

### 原始增强 vs 强增强对比

| 增强类型 | 原始增强 | 强增强 |
|---------|---------|--------|
| 旋转 | ±10° | ±30° |
| 平移 | ±5px | ±15px |
| 亮度调整 | ±10% | ±30% |
| 对比度调整 | ±15% | ±40% |
| 缩放 | 0.98-1.02x | 0.8-1.3x |

### 强增强新增功能

1. **多种噪声类型**
   - 高斯噪声：标准差0.02-0.08
   - 均匀噪声：范围±0.02-0.06
   - 椒盐噪声：概率1-5%

2. **弹性变形**
   - 变形强度：10-30
   - 平滑参数：3-6
   - 模拟组织形变

3. **Cutout/Random Erasing**
   - 1-3个擦除区域
   - 擦除大小：8x8到16x16像素
   - 随机填充值

4. **垂直翻转**
   - 30%概率的垂直翻转
   - 50%概率的水平翻转

5. **Gamma变换**
   - Gamma值：0.5-2.0
   - 模拟不同成像条件

6. **色彩抖动**（多通道CT）
   - 通道特定的亮度调整
   - 每通道独立变换

7. **局部对比度调整**
   - 16x16块的自适应增强
   - 模拟CLAHE效果

## cfDNA数据处理

为了保持cfDNA特征的完整性和生物学意义，cfDNA数据不进行额外的增强处理，保持原始特征用于训练。这样可以确保：
- 保留cfDNA特征的生物学解释性
- 避免过度增强导致的噪声
- 专注于CT图像的增强效果

## 参数说明

- `--strong_augment`: 启用强增强功能
- `--aug_prob`: 增强应用概率（默认0.8，暂未完全实现）

## 使用建议

1. **训练时间**: 强增强会增加训练时间，建议适当增加epochs
2. **学习率**: 可能需要降低学习率以适应更强的噪声
3. **批大小**: 强增强可能需要更大的批大小来稳定训练
4. **验证**: 强增强只在训练集上应用，验证集和测试集保持原始数据

## 示例配置

### 癌症正常分类（强增强）
```bash
python finetune.py \
    --modalities Frag PFE NDR NDR2K \
    --data_file /path/to/cancer_normal/train.npz \
    --test_file /path/to/cancer_normal/test.npz \
    --ct_model_path /path/to/ct_model.pth \
    --use_feature_selection \
    --feature_selection_method pca \
    --k_features 100 \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --k_folds 10 \
    --use_test_set \
    --finetune_ct \
    --latent_dim 512 \
    --strong_augment \
    --output_dir ./results/cancer_normal_strong_aug/
```

## 注意事项

1. **数据质量**: 确保原始数据质量良好，强增强不能修复损坏的数据
2. **过拟合vs欠拟合**: 强增强有助于防止过拟合，但可能导致欠拟合
3. **计算资源**: 强增强需要更多的计算资源，特别是弹性变形
4. **实验对比**: 建议同时运行普通增强和强增强进行对比

## 技术细节

- 所有增强操作都保证数据类型和形状一致
- 使用`np.ascontiguousarray()`确保内存连续性
- 支持多通道CT数据的独立处理
- cfDNA特征保持原始数据范围和分布特性