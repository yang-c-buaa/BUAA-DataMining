import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data(base_path):
    train_path = os.path.join(base_path, "train-set.csv")
    test_path = os.path.join(base_path, "test-set.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def basic_eda(train, test, outdir):
    summary = {}
    summary['train_shape'] = train.shape
    summary['test_shape'] = test.shape
    summary['train_columns'] = train.columns.tolist()
    summary['test_columns'] = test.columns.tolist()
    summary['train_missing'] = train.isna().sum().to_dict()
    summary['test_missing'] = test.isna().sum().to_dict()
    # numeric summary
    summary['train_describe'] = train.describe().to_dict()
    summary['test_describe'] = test.describe().to_dict()

    # label distribution if present
    if 'label' in test.columns:
        summary['test_label_counts'] = test['label'].value_counts().to_dict()

    with open(os.path.join(outdir, 'eda_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("EDA 摘要已写入：", os.path.join(outdir, 'eda_summary.json'))


def train_and_evaluate(train, test, outdir, random_state=42):
    features = [c for c in train.columns if c != 'label']
    X_train = train[features].values
    X_test = test[features].values

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # estimate contamination from test label distribution if available
    if 'label' in test.columns:
        # label==1 considered disease/anomaly
        pos_frac = float((test['label'] == 1).mean())
        contamination = max(pos_frac, 1e-3)
    else:
        contamination = 0.01

    print(f"使用 contamination={contamination:.6f} 训练 IsolationForest（根据测试集估计）")

    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    iso.fit(X_train_s)

    # decision_function: higher -> more normal; we invert to get anomaly score
    test_scores = -iso.decision_function(X_test_s)

    # threshold based on contamination on train (percentile)
    train_scores = -iso.decision_function(X_train_s)
    thr = np.percentile(train_scores, 100 * (1 - contamination))

    y_pred = (test_scores >= thr).astype(int)

    results = {}
    if 'label' in test.columns:
        y_true = test['label'].astype(int).values
        cm = confusion_matrix(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, test_scores)
        except Exception:
            roc_auc = None
        pr_auc = average_precision_score(y_true, test_scores)

        results.update({
            'confusion_matrix': cm.tolist(),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': None if roc_auc is None else float(roc_auc),
            'pr_auc': float(pr_auc),
        })

        # save confusion matrix heatmap
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'confusion_matrix.png'))
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, test_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC = {results["roc_auc"]:.4f}' if results['roc_auc'] else 'ROC')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'roc_curve.png'))
        plt.close()

        # PR curve
        precs, recs, _ = precision_recall_curve(y_true, test_scores)
        plt.figure()
        plt.plot(recs, precs, label=f'PR AUC = {results["pr_auc"]:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'pr_curve.png'))
        plt.close()

        print("评估结果：")
        print(json.dumps(results, indent=2))

    else:
        # no labels: just output anomaly scores and top anomalies
        results['scores_summary'] = {
            'min': float(test_scores.min()),
            'max': float(test_scores.max()),
            'median': float(np.median(test_scores)),
        }
        topk_idx = np.argsort(-test_scores)[:20]
        results['top20_anomalies_index'] = topk_idx.tolist()
        print("测试集中没有标签。已保存异常分数摘要。")

    # persist artifacts
    pd.DataFrame({'score': test_scores, 'pred': y_pred}).to_csv(os.path.join(outdir, 'test_scores_and_pred.csv'), index=False)
    joblib.dump({'model': iso, 'scaler': scaler}, os.path.join(outdir, 'artifact_model.joblib'))
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"已将工件写入：{outdir}")


def main():
    base_path = os.path.dirname(__file__)
    outdir = os.path.join(base_path, 'outputs')
    ensure_dir(outdir)

    train, test = load_data(base_path)
    basic_eda(train, test, outdir)
    train_and_evaluate(train, test, outdir)


if __name__ == '__main__':
    main()
