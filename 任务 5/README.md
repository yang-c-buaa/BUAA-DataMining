# 任务5：多模态统一异常检测框架

本方案实现了一个“模态专属编码器 + 共享潜在空间 + 统一异常评分器”的原型，用于同时处理图像与结构化数据（以任务2的 MVTec 图像数据和任务4的甲状腺数值数据为例）。核心目标：

- **统一潜在空间**：不同模态经各自编码器映射到同一维度的 latent 表示。
- **共享异常评分器**：在潜在空间内使用高斯密度估计/马氏距离给出异常分数，有标签时可叠加监督头。
- **灵活扩展**：新增模态只需实现对应编码器与数据加载器。

## 目录结构

```
task5_multi_modal/
├── multi_modal_anomaly.py    # 框架主脚本
├── README.md                 # 说明文档
└── requirements.txt          # 依赖列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python multi_modal_anomaly.py \
  --image_root "../任务 2：图像异常检测任务/Image_Anomaly_Detection" \
  --tabular_root "../任务 4：无监督疾病判断任务/thyroid" \
  --category hazelnut \
  --latent_dim 256 \
  --mode train_and_eval
```

- **train_and_eval**：读取正常训练样本，拟合统一高斯模型，并在测试集上给出异常分数。若 tabular 集含标签会输出指标。
- **train_only / eval_only**：分别只训练或只评估（需先保存模型工件）。

## 方法概要

1. **数据加载**  
   - 图像：沿用任务2数据划分，并支持 heatmap 生成。  
   - 表格：读取任务4的 `train-set.csv`、`test-set.csv`，仅在训练阶段使用正常样本。

2. **模态编码器**  
   - 图像编码器：使用预训练 `wide_resnet50_2`，提取 Layer2+Layer3 特征，融合后经投影层得到指定维度 latent。  
   - 表格编码器：两层 MLP + LayerNorm，将数值特征映射到相同维度。

3. **潜在对齐与评分**  
   - 对所有正常样本 latent 拼接，估计均值与协方差，形成多元高斯。  
   - 测试时通过马氏距离计算异常分数。  
   - 可选监督头：若存在标签，可对 latent 接一个小型分类器，使用 BCE/Focal Loss 微调。

4. **输出**  
   - `artifacts/gaussian_stats.pt`：存储均值、协方差、逆协方差。  
   - `artifacts/image_scores.csv`、`artifacts/tabular_scores.csv`：保存各模态分数。  
   - `visualizations/heatmaps/`：图像异常热力图。  
   - `reports/metrics.json`：若有标签则输出精确率、召回率、ROC-AUC、PR-AUC。

## 关键配置

| 参数 | 说明 |
| ---- | ---- |
| `--latent_dim` | 统一潜在空间维度，默认 256 |
| `--image_size` | 输入图像尺寸（默认 256） |
| `--batch_size` | 训练/评估批大小 |
| `--use_supervised_head` | 是否启用监督头（需提供标签） |

