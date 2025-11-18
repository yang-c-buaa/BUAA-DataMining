# 甲状腺异常检测（无监督）

本仓库包含一份可复现脚本，用于在提供的甲状腺数据集上训练无监督异常检测器（训练集仅包含正常样本），并在测试集上进行评估。

文件说明：
- `train-set.csv`, `test-set.csv` — 数据文件（6 个特征）。`test-set.csv` 包含 `label` 列（1 = 异常/疾病，0 = 正常）。
- `analyze_thyroid.py` — 主脚本：完成 EDA、在训练数据上训练 IsolationForest、在测试数据上评估，并将图表与结果保存到 `outputs/`。
- `requirements.txt` — Python 依赖清单。

工作流程（概要）：
- 使用训练集对特征进行标准化。
- 训练 IsolationForest，若测试集中含有标签，则脚本会用测试集中正样本比例估计 contamination，从而设定与预期患病率相符的判决阈值。
- 计算异常分数（取反的 decision_function），并基于训练分数分布的阈值生成二分类预测。
- 若有标签，则使用混淆矩阵、精确率、召回率、F1、ROC-AUC 和 PR-AUC 等指标评估，并保存 ROC/PR 曲线与混淆矩阵图像。

运行方法：
1. 安装依赖：`pip install -r requirements.txt`
2. 运行脚本：`python analyze_thyroid.py`

脚本会将输出写入 `outputs/` 目录。

说明与设计选择：
- 方法：IsolationForest — 之所以选择它，是因为训练集中只包含正常样本，IsolationForest 能学习“正常”样本的分布并将偏离该分布的样本孤立出来。此方法计算效率高、对多维数据鲁棒，适合该问题场景。
- 阈值设定：脚本在可能的情况下会从测试集标签估计 contamination，以得到一个合理的判定阈值；同时脚本也会保存原始异常分数，便于你事后按业务需求调整阈值（例如更关注召回时降低阈值）。
