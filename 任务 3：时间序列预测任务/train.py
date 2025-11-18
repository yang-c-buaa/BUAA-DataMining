import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt


def load_data(csv_path='weather.csv'):
    # 读取 CSV，清理列名空白
    # 指定编码以处理 CSV 中可能的非 UTF-8 字符（数据文件包含特殊符号）
    df = pd.read_csv(csv_path, sep=',', skipinitialspace=True, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    # 尝试解析时间列，列名为 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

    # 把 OT 转为数值
    if 'OT' not in df.columns:
        raise ValueError('目标列 OT 未找到，请检查 csv 列名')
    df['OT'] = pd.to_numeric(df['OT'], errors='coerce')

    return df


def create_sliding_windows(df, past_steps=12, feature_cols=None):
    # 预测目标：下一时刻 OT 的变化 (delta)
    ot = df['OT'].values
    n = len(df)
    if feature_cols is None:
        # 除了日期和 OT 外其它列作为特征
        feature_cols = [c for c in df.columns if c not in ['date', 'OT']]

    X_list = []
    y_list = []
    idx_list = []
    feats = df[feature_cols].values

    # 生成窗口：输入为过去 past_steps 的特征（按时间顺序），输出为 OT_{t+1}-OT_t
    for t in range(past_steps - 1, n - 1):
        start = t - (past_steps - 1)
        end = t + 1  # inclusive t
        X_win = feats[start:end]  # shape (past_steps, n_features)
        y = ot[t + 1] - ot[t]
        if np.isnan(X_win).any() or np.isnan(y):
            continue
        X_list.append(X_win.flatten())
        y_list.append(y)
        idx_list.append(t)

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, feature_cols


def time_split(X, y, train_ratio=0.8):
    n = len(X)
    split = int(n * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    # Some sklearn versions don't accept the `squared` keyword; compute RMSE via sqrt(MSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}, y_pred


def main():
    csv_path = 'weather.csv'
    print('加载数据...')
    df = load_data(csv_path)
    print('样本数:', len(df), '列数:', len(df.columns))

    past_steps = 12  # 过去2小时（12个时间点，每10min）
    print('构建滑动窗口...')
    X, y, feature_cols = create_sliding_windows(df, past_steps=past_steps)
    print('生成样本数:', X.shape[0], '每样本特征维度:', X.shape[1])

    # 时间序列划分（按顺序）
    X_train, X_test, y_train, y_test = time_split(X, y, train_ratio=0.8)
    print('训练集样本:', len(X_train), '测试集样本:', len(X_test))

    # 标准化（在训练集上拟合）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 模型1: 岭回归（线性基线）
    print('训练 Ridge 基线模型...')
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    metrics_ridge, y_pred_ridge = evaluate_model(ridge, X_test_s, y_test)
    print('Ridge 评估:', metrics_ridge)

    # 模型2: 随机森林
    print('训练 RandomForest 回归模型...（可能需要一点时间，使用 50 棵树以加快速度）')
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42, verbose=0)
    rf.fit(X_train_s, y_train)
    print('RandomForest 训练完成')
    metrics_rf, y_pred_rf = evaluate_model(rf, X_test_s, y_test)
    print('RandomForest 评估:', metrics_rf)

    # 保存模型与标准化器
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    joblib.dump(ridge, 'artifacts/ridge.joblib')
    joblib.dump(rf, 'artifacts/rf.joblib')

    # 绘制一小段真实值 vs 预测值
    n_plot = min(200, len(y_test))
    plt.figure(figsize=(10, 4))
    plt.plot(y_test[:n_plot], label='真实 ΔOT')
    plt.plot(y_pred_rf[:n_plot], label='RF 预测 ΔOT')
    plt.plot(y_pred_ridge[:n_plot], label='Ridge 预测 ΔOT', alpha=0.7)
    plt.legend()
    plt.title('测试集：真实 ΔOT 与 预测 ΔOT（前 %d 样本）' % n_plot)
    plt.tight_layout()
    plt.savefig('artifacts/prediction_plot.png', dpi=150)
    print('图像已保存到 artifacts/prediction_plot.png')

    # 输出评估表
    with open('artifacts/metrics.txt', 'w', encoding='utf-8') as f:
        f.write('Ridge:\n')
        f.write(str(metrics_ridge) + '\n')
        f.write('RandomForest:\n')
        f.write(str(metrics_rf) + '\n')

    print('所有工作完成，结果保存在 artifacts/ 目录下')


if __name__ == '__main__':
    main()
