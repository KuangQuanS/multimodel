# 多模态表格数据编码器

这个项目实现了一个用于多模态表格数据的编码器训练框架，结合了对比学习和分类任务。主要用于处理五种cfDNA相关的表格数据：片段特征、CNV、PFE、NDR和NDR2K。

## 特点

- 使用对比学习和分类任务的联合训练
- 为每种模态数据提供独立的编码器
- 包含特征可视化和分析工具
- 支持模型预测和评估

## 文件结构

```
.
├── train_tabular_encoders.py  # 训练脚本
├── predict.py                 # 预测脚本
├── visualize_features.py      # 特征可视化脚本
└── README.md                 # 说明文档
```

## 环境要求

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn umap-learn
```

## 数据格式

每个模态的数据应该是一个CSV文件，包含以下格式：
- 每行代表一个样本
- 除最后一列外的所有列都是特征
- 最后一列是标签（0表示正常，1表示癌症）

数据目录结构示例：
```
data/
├── fragment.csv
├── cnv.csv
├── pfe.csv
├── ndr.csv
└── ndr2k.csv
```

## 使用方法

### 1. 训练编码器

```bash
python train_tabular_encoders.py \
    --data_path /path/to/your/data \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001
```

训练过程会自动保存最佳模型到 `best_model.pth`。

### 2. 进行预测

```bash
python predict.py \
    --model best_model.pth \
    --data /path/to/test/data \
    --output results
```

这将生成：
- 分类报告
- 混淆矩阵
- 提取的特征

### 3. 可视化特征

```bash
python visualize_features.py \
    --features_dir results \
    --output_dir visualizations
```

这将生成多种可视化结果：
- t-SNE可视化
- UMAP可视化
- PCA可视化
- 模态间相关性热图
- 特征分布图
- 特征重要性分析

## 主要参数说明

### 训练参数

- `batch_size`: 批次大小（默认：32）
- `num_epochs`: 训练轮数（默认：100）
- `learning_rate`: 学习率（默认：0.001）
- `lambda_contrast`: 对比学习损失权重（默认：0.5）

### 模型架构参数

- `hidden_dims`: 编码器隐藏层维度（默认：[256, 128]）
- `encoder_output_dim`: 编码器输出维度（默认：64）
- `projection_dim`: 投影头输出维度（默认：128）

## 训练策略

该实现采用了以下策略来提高模型性能：

1. **联合训练**
   - 结合对比学习和分类任务
   - 使用加权损失函数平衡两个任务

2. **数据增强**
   - 添加高斯噪声
   - 随机特征遮蔽

3. **正则化**
   - Dropout
   - 批量归一化
   - L2正则化

4. **学习率调度**
   - 使用余弦退火学习率
   - 支持早停

## 特征分析

可视化脚本提供了多种分析工具：

1. **降维可视化**
   - t-SNE
   - UMAP
   - PCA

2. **相关性分析**
   - 模态间相关性热图
   - 特征分布分析

3. **特征重要性**
   - 基于效应量的特征重要性分析
   - 模态贡献度分析

## 注意事项

1. 确保数据已经正确预处理（如处理缺失值、异常值等）
2. 根据实际数据规模调整批次大小和学习率
3. 如果出现过拟合，可以：
   - 增加dropout率
   - 减小模型复杂度
   - 增加正则化强度
   - 调整对比学习权重

## 结果解释

1. **分类性能指标**
   - 准确率（Accuracy）
   - 精确率（Precision）
   - 召回率（Recall）
   - F1分数

2. **特征分析**
   - 特征空间分布
   - 模态间关系
   - 特征重要性排序

## 常见问题

1. **内存不足**
   - 减小批次大小
   - 减少特征维度
   - 使用数据生成器

2. **训练不稳定**
   - 调整学习率
   - 检查数据归一化
   - 调整损失权重

3. **过拟合**
   - 增加正则化
   - 减小模型复杂度
   - 增加数据增强

## 后续改进方向

1. 添加更多数据增强方法
2. 实现更多的特征可视化方法
3. 支持模型集成
4. 添加交叉验证支持
5. 实现特征选择功能

## 引用

如果您使用了这个项目，请引用以下论文：
- [对比学习相关论文]
- [多模态学习相关论文]
- [生物信息学相关论文]