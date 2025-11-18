# 图像聚类任务说明文档

## 1. 项目概述
- 目标：对 `dataset/` 中 600 张 PNG 图像进行无监督聚类分析，并与 `cluster_labels.json` 中的真实类别对照评估表现。
- 核心产物：聚类脚本 `image_clustering.py`、可视化结果 `clustering_results.png`、可选 Word 报告 `聚类分析报告.docx`。
- 类别覆盖：`cable`、`tile`、`bottle`、`pill`、`leather`、`transistor` 六大类型。

## 2. 目录结构
- `dataset/`：原始图像数据（600 张 PNG，已按文件名排序读取）。
- `cluster_labels.json`：文件名到真实类别的映射，供评估与报告使用。
- `image_clustering.py`：主分析脚本，涵盖特征提取、PCA 降维、K-means/层次聚类、指标评估与可视化。
- `generate_word_report.py`：读取同一套数据，复现直方图特征 + PCA + K-means 流程，输出 Word 报告。
- `clustering_results.png`：`image_clustering.py` 运行后生成的 2×2 子图（真实标签、聚类结果、类别对比、混淆矩阵）。
- `requirements.txt`：项目依赖清单。

## 3. 环境准备
1. 建议使用 Python 3.9+。
2. 创建虚拟环境（可选）：
   - Windows PowerShell：`python -m venv .venv && .\.venv\Scripts\activate`
3. 安装依赖：`pip install -r requirements.txt`

## 4. 核心流程说明
### 4.1 `image_clustering.py`
- `ImageClustering.load_data()`：读取图像与真实标签并缓存，输出类别分布。
- `extract_features(method)`：提供 `histogram`（默认）、`pixel`、`combined` 三种特征方案；同时在 `run_complete_analysis()` 中循环比较，基于轮廓系数挑选最佳方案。
- `apply_pca(n_components=50)`：对特征做 `StandardScaler` 标准化后执行 PCA，输出累积解释方差。
- `cluster_images(method, n_clusters=6)`：支持 `kmeans` 和 `hierarchical`，默认簇数与真实类别一致。
- `evaluate_clustering()`：计算轮廓系数、调整兰德指数（ARI）、聚类纯度、标准化互信息（NMI）。
- `visualize_results()`：将 PCA 投影至二维，生成 4 子图并保存为 `clustering_results.png`。
- `run_complete_analysis()`：自动串联上述步骤，比较不同特征 & 聚类算法，打印综合结论。

### 4.2 `generate_word_report.py`
- 再次执行颜色直方图特征 + PCA + K-means 流程，统计样本数、聚类簇规模及指标（Silhouette、ARI、Purity）。
- 通过 `python-docx` 生成 `聚类分析报告.docx`，包含概述、数据分布表、方法说明、指标表格、可视化插图与文字结论。
- 若 `clustering_results.png` 存在，会自动嵌入 Word；否则插入提示文本。

## 5. 使用指南
1. **准备数据**：确认 `dataset/`、`cluster_labels.json` 均位于项目根目录；可根据需要替换为其他同构数据集。
2. **运行聚类分析**：
   - `python image_clustering.py`
   - 控制台会输出每种特征方法的指标、最终算法对比以及可视化保存位置。
3. **生成 Word 报告（可选）**：
   - `python generate_word_report.py`
   - 默认输出 `聚类分析报告.docx`，可在调用 `generate_report()` 时自定义文件名。

## 6. 结果解读
- **指标含义**：
  - 轮廓系数（-1~1）：越大越证明簇分离度越好。
  - 调整兰德指数（-1~1）：越大越接近真实标签。
  - 聚类纯度（0~1）：评估簇内主导类别的占比。
  - 标准化互信息（0~1）：衡量聚类结果与真实标签的互信息强度。
- **可视化**：通过颜色映射直观对比真实标签与聚类标签，同时提供类别柱状对比与混淆矩阵帮助定位误分布。
- **最佳配置**：`run_complete_analysis()` 会自动报告得分最高的特征方案，你也可以根据任务需求手动指定方法与参数（例如修改 PCA 维度、聚类簇数或更换算法）。

## 7. 常见问题
- **运行缓慢**：可调低特征维度（如缩小直方图 bins 或减小 `_extract_pixel_features` 的分辨率），或减少 PCA 主成分数量。
- **缺少字体/库**：若 `matplotlib` 或 `python-docx` 报缺失，确认已从 `requirements.txt` 安装；中文字体可改为系统已安装的字体名称。
- **自定义评估**：可在 `ImageClustering.evaluate_clustering()` 中添加新的指标（如 Calinski-Harabasz、Davies-Bouldin），并在 `visualize_results()` 中扩展更多图表。

## 8. 后续扩展建议
- 引入深度特征：利用预训练 CNN（如 ResNet 特征）替代手工直方图，以提高聚类区分度。
- 试验更多聚类器：如 DBSCAN、Spectral 或 Gaussian Mixture，并在脚本中统一封装接口进行网格比较。
- 自动化报告：将 `image_clustering.py` 的结果直接传递给 `WordReportGenerator`，避免重复计算并保持指标一致。

