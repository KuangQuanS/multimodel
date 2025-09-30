# 多模态癌症分类模型

## 📁 项目结构

```
├── src/                          # 模块化代码
│   ├── models/
│   │   └── fusion_models.py      # 模型定义（ModalityEncoder, CrossAttentionFusion, CTModel）
│   ├── data/
│   │   └── dataset.py            # 数据加载和处理
│   └── utils/
│       └── training_utils.py     # 训练工具函数
├── finetune_v2.py               # 主训练脚本（重构版）
├── finetune.py                  # 原版训练脚本（保留作参考）
├── finetune.ipynb              # Jupyter notebook训练界面
├── CT/                         # CT模型相关文件
├── cfDNA/                      # cfDNA预训练模型（已禁用）
└── results/                    # 训练结果输出
```

## 🚀 使用方法

### 命令行训练
```bash
python finetune_v2.py \
    --modalities Frag PFE NDR NDR2K \
    --data_file /path/to/train.npz \
    --epochs 10 \
    --batch_size 64 \
    --lr 3e-4 \
    --cross_val \
    --output_dir ./results/
```

### Notebook训练
使用 `finetune.ipynb` 进行交互式训练

## ✨ 主要特性

- ✅ **无预训练依赖**: cfDNA编码器使用随机初始化，避免预训练权重加载问题
- 🧩 **模块化设计**: 代码按功能分离，易于维护和扩展  
- 📊 **交叉验证**: 支持K折交叉验证
- 🎯 **多任务支持**: 亚型分类、良性vs恶性、癌症vs正常分类
- 📈 **可视化**: 自动生成ROC曲线和训练结果

## 🔧 更新说明

- **重构**: 将单文件代码拆分为模块化结构
- **简化**: 移除cfDNA预训练权重依赖  
- **清理**: 删除过时文件和调试代码