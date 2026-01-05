# Weather 时间序列预测（ΔOT）


目标：根据过去 2 小时（12 个时间点，每 10 分钟记录一次）的 21 维气象特征，预测下一时间点室外温度（OT）的变化 ΔOT = OT_{t+1} - OT_t。

包含文件：
- `weather.csv`：原始数据（请放在同目录下）
- `train.py`：训练与评估脚本，生成 `artifacts/` 目录包含模型、评估结果与图像
- `requirements.txt`：所需 Python 包

使用方法（Windows PowerShell）：

1. 建议创建并激活虚拟环境，然后安装依赖：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 运行训练脚本：

```powershell
python train.py
```

输出：
- `artifacts/scaler.joblib`：标准化器
- `artifacts/ridge.joblib`：Ridge 基线模型
- `artifacts/rf.joblib`：随机森林模型
- `artifacts/prediction_plot.png`：部分测试集真实 ΔOT 与预测对比图
- `artifacts/metrics.txt`：模型评估指标（MAE、RMSE、R2）

评价指标：脚本中计算 MAE、RMSE 与 R2，是常用回归指标；其中 RMSE 更强调大误差，MAE 易解释，R2 用于衡量拟合程度。

可扩展方向（建议）：
- 尝试 LSTM/Transformer 等序列模型（需 reshape 为三维并使用 tensorflow/torch）
- 使用滑动窗口时保留时间信息（hour of day、sin/cos 周期特征）
- 做交叉验证或基于时间的验证集（例如向前滚动验证）
