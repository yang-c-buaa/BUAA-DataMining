## 北航数据挖掘课程作业总览（BUAA-DataMining）

本仓库汇总了北航数据挖掘相关的 5 个作业任务，每个任务在各自子目录中实现并附带独立的说明文档和依赖文件。  
整体上涵盖了 **聚类分析、图像异常检测、时间序列预测、无监督疾病判断、多模态异常检测** 等典型数据挖掘场景。

- **任务 1：图像聚类任务**（无监督聚类 + 可视化 + Word 报告）
- **任务 2：图像异常检测任务**（基于深度特征 + 高斯建模的无监督检测）
- **任务 3：时间序列预测任务**（多模型回归预测 ΔOT）
- **任务 4：无监督疾病判断任务**（基于 IsolationForest 的异常检测）
- **任务 5：多模态统一异常检测框架**（图像 + 表格数据统一潜在空间）

---

## 仓库结构

```text
BUAA-DataMining/
├── 任务 1：聚类任务/
│   └── Cluster/              # 图像聚类代码与数据
├── 任务 2：图像异常检测任务/
│   └── Image_Anomaly_Detection/
├── 任务 3：时间序列预测任务/
├── 任务 4：无监督疾病判断任务/
│   └── thyroid/
└── 任务 5/
    └── task5_multi_modal/    # 多模态异常检测框架
```

各子目录下均包含独立的 `README.md` 与 `requirements.txt`，推荐按任务单独创建虚拟环境、安装依赖与运行。

---

## 环境与依赖

整体建议使用 **Python 3.9+**。  
每个任务目录中都提供了对应的 `requirements.txt`，可在进入该任务目录后执行：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 注：部分任务（尤其是任务 2 和任务 5）建议在具备 **GPU（CUDA）** 的环境下运行，以加速深度模型推理。

---

## 各任务简要说明与入口

### 任务 1：图像聚类任务

- **路径**：`任务 1：聚类任务/Cluster/`
- **主要目标**：对 `dataset/` 中 600 张 PNG 图像进行无监督聚类，并与 `cluster_labels.json` 中的真实类别对照评估；可生成聚类可视化图与 Word 报告。
- **核心脚本**：
  - `image_clustering.py`：特征提取（直方图/像素/组合）、PCA 降维、K-means / 层次聚类、指标评估、结果可视化。
  - `generate_word_report.py`：基于相同流程生成 `聚类分析报告.docx`。
- **快速运行**（在 `Cluster/` 目录）：
  - 执行聚类分析：`python image_clustering.py`
  - 生成 Word 报告：`python generate_word_report.py`
- **详细说明**：见 `任务 1：聚类任务/Cluster/README.md`。

### 任务 2：图像异常检测任务

- **路径**：`任务 2：图像异常检测任务/Image_Anomaly_Detection/`
- **主要目标**：在 MVTec AD 风格数据集（目前包含 `hazelnut` 与 `zipper` 两类）上进行图像级与像素级的无监督异常检测，并生成热力图与性能可视化。
- **核心脚本**：
  - `main.py`：从正常样本中提取深度特征（Wide ResNet50-2，Layer2+Layer3 融合），基于多元高斯 + 马氏距离进行异常评分，输出 AUROC 等指标与可视化结果。
- **快速运行**（在 `Image_Anomaly_Detection/` 目录）：
  1. 在 `main.py` 中按需修改 `CONFIG`（数据路径、类别、图像尺寸等）。
  2. 运行：`python main.py`
- **输出**：
  - `visualizations/{category}/heatmaps/`：每张测试图像的异常热力图及叠加结果。
  - `visualizations/{category}/summary/performance_summary.png`：包含 PCA 分布、ROC 曲线、混淆矩阵等。
- **详细说明**：见 `任务 2：图像异常检测任务/Image_Anomaly_Detection/README.md`。

### 任务 3：时间序列预测任务（ΔOT）

- **路径**：`任务 3：时间序列预测任务/`
- **主要目标**：根据过去 2 小时（12 个时间点）的 21 维气象特征，预测下一时间点室外温度变化量 ΔOT。
- **核心脚本**：
  - `train.py`：读取 `weather.csv`，构造滑动窗口特征，训练 Ridge 与 Random Forest 等模型，并输出评估指标与预测曲线图。
- **快速运行**（在该目录）：
  - `python train.py`
- **输出**：
  - `artifacts/ridge.joblib`、`artifacts/rf.joblib`：训练好的模型。
  - `artifacts/scaler.joblib`：标准化器。
  - `artifacts/prediction_plot.png`：真实 vs 预测 ΔOT 可视化。
  - `artifacts/metrics.txt`：MAE、RMSE、R2 等指标。
- **详细说明**：见 `任务 3：时间序列预测任务/README.md`。

### 任务 4：无监督疾病判断任务（甲状腺）

- **路径**：`任务 4：无监督疾病判断任务/thyroid/`
- **主要目标**：在仅使用正常样本训练的前提下，对甲状腺疾病进行无监督异常检测，并在测试集上评估。
- **核心脚本**：
  - `analyze_thyroid.py`：对 `train-set.csv` 与 `test-set.csv` 进行预处理、标准化，训练 IsolationForest，计算异常分数并生成评估指标与图像。
- **快速运行**（在 `thyroid/` 目录）：
  - `python analyze_thyroid.py`
- **输出**（默认到 `outputs/`）：
  - `artifact_model.joblib`：训练好的模型。
  - `results.json`、`eda_summary.json`：指标与数据分析结果。
  - `roc_curve.png`、`pr_curve.png`、`confusion_matrix.png` 等可视化。
- **详细说明**：见 `任务 4：无监督疾病判断任务/thyroid/README.md`。

### 任务 5：多模态统一异常检测框架

- **路径**：`任务 5/task5_multi_modal/`
- **主要目标**：构建“模态专属编码器 + 共享潜在空间 + 统一异常评分器”的多模态异常检测框架，示例模态为任务 2 的图像数据与任务 4 的表格数据。
- **核心脚本**：
  - `multi_modal_anomaly.py`：整合图像编码器（Wide ResNet50-2）与表格编码器（MLP），在统一潜在空间中基于高斯分布/马氏距离进行异常打分，可选叠加监督头。
- **快速运行示例**（在 `task5_multi_modal/` 目录）：

  ```bash
  python multi_modal_anomaly.py ^
    --image_root "../任务 2：图像异常检测任务/Image_Anomaly_Detection" ^
    --tabular_root "../任务 4：无监督疾病判断任务/thyroid" ^
    --category hazelnut ^
    --latent_dim 256 ^
    --mode train_and_eval
  ```

- **输出**：
  - `artifacts/gaussian_stats.pt`：潜在空间高斯模型参数。
  - `artifacts/image_scores.csv`、`artifacts/tabular_scores.csv`：各模态异常分数。
  - `visualizations/heatmaps/`：图像异常热力图。
  - `reports/metrics.json`：若有标签则保存多模态指标。
- **详细说明**：见 `任务 5/README.md` 及 `任务 5/task5_multi_modal/README.md`（如存在）。

---

## 建议阅读顺序与学习路径

若将本仓库视作一个循序渐进的数据挖掘实践，可以按以下顺序阅读与运行：

1. **任务 1：图像聚类**  
   - 侧重传统特征工程 + 聚类算法 + 指标与可视化。
2. **任务 2：图像异常检测**  
   - 体验深度特征提取 + 统计建模的无监督异常检测。
3. **任务 3：时间序列预测**  
   - 了解回归建模、滑动窗口构造与时间序列评估指标。
4. **任务 4：无监督疾病判断**  
   - 练习基于 IsolationForest 的异常检测与阈值设定。
5. **任务 5：多模态异常检测框架**  
   - 综合前面任务，理解如何在统一潜在空间下融合多模态数据。



