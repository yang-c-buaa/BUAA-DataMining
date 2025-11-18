import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
# 新增的库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import cv2

# --- 1. 配置参数 (与之前相同) ---
CONFIG = {
    "dataset_path": "./",  # !<-- 修改为你的数据集根目录
    "category": "zipper",  # !<-- 修改为你需要处理的类别 ('hazelnut' 或 'zipper')
    "image_size": 256,
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "visualization_dir": "visualizations"
}

print(f"使用的设备: {CONFIG['device']}")
# 为两种可视化结果创建文件夹
os.makedirs(os.path.join(CONFIG['visualization_dir'], CONFIG['category'], 'heatmaps'), exist_ok=True)
os.makedirs(os.path.join(CONFIG['visualization_dir'], CONFIG['category'], 'summary'), exist_ok=True)

# --- 新增: 设置matplotlib支持中文的字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# --- 2. 数据集加载器 (与之前相同, 但getitem现在返回5个值) ---
class MVTecDataset(Dataset):
    def __init__(self, root_path, category, is_train=True, transform=None, label_file=None):
        self.root_path = root_path
        self.category = category
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.image_dir = os.path.join(root_path, category, 'train', 'good')
            self.image_files = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)])
            self.labels = [0] * len(self.image_files)
        else:
            if label_file is None:
                raise ValueError("测试模式下必须提供 label_file (JSON文件路径)")
            with open(label_file, 'r') as f:
                label_data = json.load(f)
            self.image_files = []
            self.labels = []
            test_image_dir = os.path.join(root_path, category, 'test')
            for filename in sorted(os.listdir(test_image_dir)):
                json_key = f"{category}/test/{filename}"
                if json_key in label_data:
                    label_str = label_data[json_key]["label"]
                    label = 0 if label_str == "good" else 1
                    self.image_files.append(os.path.join(test_image_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(transforms.functional.resize(image, (CONFIG['image_size'], CONFIG['image_size'])))

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label = self.labels[idx]
        return image_tensor, label, original_image, os.path.basename(image_path)


# --- 3. 特征提取器 (与之前相同) ---
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        self.model.eval()
        self.features = []
        # 使用钩子捕获中间层输出
        self.model.layer2.register_forward_hook(self.get_features('layer2'))
        self.model.layer3.register_forward_hook(self.get_features('layer3'))

    def get_features(self, name):
        def hook(model, input, output):
            self.features.append(output)

        return hook

    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        # 组合特征图
        layer3_upsampled = F.interpolate(self.features[1], size=self.features[0].shape[-2:], mode='bilinear',
                                         align_corners=False)
        return torch.cat((self.features[0], layer3_upsampled), dim=1)


# --- 4. “训练”/拟合 过程 (与之前相同) ---
def fit_normal_distribution(model, train_loader, device):
    print("开始从正常训练数据中提取区域特征...")
    all_patch_features = []
    with torch.no_grad():
        for images, _, _, _ in tqdm(train_loader):
            images = images.to(device)
            feature_maps = model(images)
            patches = feature_maps.permute(0, 2, 3, 1).reshape(-1, feature_maps.size(1))
            all_patch_features.append(patches.cpu().numpy())
    all_patch_features = np.concatenate(all_patch_features, axis=0)
    print(f"计算 {all_patch_features.shape[0]} 个区域特征的均值和协方差...")
    mean = np.mean(all_patch_features, axis=0)
    covariance = np.cov(all_patch_features, rowvar=False) + 1e-6 * np.identity(all_patch_features.shape[1])
    inv_covariance = np.linalg.inv(covariance)
    print("区域正态分布拟合完成。")
    return mean, inv_covariance


# --- 5. 评估过程 (修改后返回聚合的特征) ---
def evaluate_and_visualize(model, test_loader, mean, inv_covariance, device):
    print("开始在测试集上评估并生成可视化...")
    all_labels = []
    all_scores = []
    # **** 新增：收集池化后的特征用于PCA分析 ****
    aggregated_features = []

    with torch.no_grad():
        for images, labels, original_images, filenames in tqdm(test_loader):
            images = images.to(device)
            feature_maps = model(images)
            b, c, h, w = feature_maps.shape

            # **** 新增：为PCA计算聚合特征 ****
            pooled_features = F.adaptive_avg_pool2d(feature_maps, 1).reshape(b, c)
            aggregated_features.append(pooled_features.cpu().numpy())

            patches = feature_maps.permute(0, 2, 3, 1).reshape(-1, c)
            patch_scores = [mahalanobis(patch, mean, inv_covariance) for patch in patches.cpu().numpy()]
            anomaly_maps = np.array(patch_scores).reshape(b, h, w)
            image_scores = np.max(anomaly_maps, axis=(1, 2))

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(image_scores)

            for i in range(b):
                generate_heatmap_visualization(
                    original_images[i], anomaly_maps[i], image_scores[i], labels[i].item(), filenames[i]
                )

    aggregated_features = np.concatenate(aggregated_features, axis=0)
    return all_labels, all_scores, aggregated_features


# --- 6. 热力图可视化函数 (名称变更) ---
def generate_heatmap_visualization(original_image, anomaly_map, score, label, filename):
    # 如果 original_image 是 PyTorch Tensor，则转换为 numpy 数组
    if torch.is_tensor(original_image):
        original_image = original_image.numpy()

    # 确保 original_image 是 uint8 类型
    if original_image.dtype != np.uint8:
        # 如果是浮点型且范围在[0,1]之间，则转换到[0,255]
        if original_image.dtype in [np.float32, np.float64] and original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)

    map_smoothed = gaussian_filter(anomaly_map, sigma=4)
    map_resized = cv2.resize(map_smoothed, (original_image.shape[1], original_image.shape[0]))
    map_normalized = (map_resized - np.min(map_resized)) / (np.max(map_resized) - np.min(map_resized) + 1e-8)
    map_uint8 = (map_normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, 0.4, original_image, 0.6, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title(f'原始图像\n标签: {"异常" if label == 1 else "正常"}')
    axes[0].axis('off')
    axes[1].imshow(heatmap)
    axes[1].set_title('异常热力图')
    axes[1].axis('off')
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'叠加效果\n分数: {score:.2f}')
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(CONFIG['visualization_dir'], CONFIG['category'], 'heatmaps', f"heatmap_{filename}")
    plt.savefig(save_path)
    plt.close(fig)


# --- 7. 新增: 生成综合性能报告的函数 ---
def generate_summary_visualizations(true_labels, anomaly_scores, features):
    print("正在生成综合性能可视化报告...")
    true_labels = np.array(true_labels)
    anomaly_scores = np.array(anomaly_scores)

    # 1. PCA降维
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'类别 "{CONFIG["category"]}" 的异常检测性能分析', fontsize=20)

    # 图1: 真实标签的特征分布
    scatter1 = axes[0, 0].scatter(features_pca[:, 0], features_pca[:, 1], c=true_labels, cmap='coolwarm', alpha=0.7)
    axes[0, 0].set_title('真实标签特征分布 (PCA)', fontsize=14)
    axes[0, 0].set_xlabel('PC 1')
    axes[0, 0].set_ylabel('PC 2')
    axes[0, 0].legend(handles=scatter1.legend_elements()[0], labels=['正常', '异常'])
    axes[0, 0].grid(True)

    # 图2: 模型异常分数的特征分布
    scatter2 = axes[0, 1].scatter(features_pca[:, 0], features_pca[:, 1], c=anomaly_scores, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('模型异常分数分布 (PCA)', fontsize=14)
    axes[0, 1].set_xlabel('PC 1')
    axes[0, 1].set_ylabel('PC 2')
    fig.colorbar(scatter2, ax=axes[0, 1], label='异常分数')
    axes[0, 1].grid(True)

    # 3. ROC曲线
    fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
    auroc = roc_auc_score(true_labels, anomaly_scores)
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {auroc:.4f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('假阳性率 (FPR)')
    axes[1, 0].set_ylabel('真阳性率 (TPR)')
    axes[1, 0].set_title('ROC 曲线', fontsize=14)
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True)

    # 4. 混淆矩阵
    # 找到最佳阈值 (最大化 TPR - FPR)
    best_threshold_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_idx]
    predictions = (anomaly_scores >= best_threshold).astype(int)
    cm = confusion_matrix(true_labels, predictions)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['预测正常', '预测异常'], yticklabels=['实际正常', '实际异常'])
    axes[1, 1].set_title(f'混淆矩阵 (阈值={best_threshold:.2f})', fontsize=14)
    axes[1, 1].set_ylabel('实际标签')
    axes[1, 1].set_xlabel('预测标签')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(CONFIG['visualization_dir'], CONFIG['category'], 'summary', 'performance_summary.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"综合报告已保存至: {save_path}")


# --- 主程序 ---
def main():
    data_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MVTecDataset(CONFIG["dataset_path"], CONFIG["category"], is_train=True, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    label_file_path = os.path.join(CONFIG["dataset_path"], "image_anomaly_labels.json")
    test_dataset = MVTecDataset(CONFIG["dataset_path"], CONFIG["category"], is_train=False, transform=data_transform,
                                label_file=label_file_path)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = FeatureExtractor().to(CONFIG["device"])

    # 步骤1: 学习正常区域的分布
    mean, inv_covariance = fit_normal_distribution(model, train_loader, CONFIG["device"])

    # 步骤2: 在测试集上计算分数并生成可视化
    labels, scores, features = evaluate_and_visualize(model, test_loader, mean, inv_covariance, CONFIG["device"])

    # 步骤3: 计算AUROC
    auroc = roc_auc_score(labels, scores)

    print("\n--- 评估结果 ---")
    print(f"类别: {CONFIG['category']}")
    print(f"图像级异常检测 AUROC: {auroc:.4f}")
    print(f"热力图可视化已保存至: '{os.path.join(CONFIG['visualization_dir'], CONFIG['category'], 'heatmaps')}' 文件夹")
    print("------------------")

    # 步骤4: 生成综合性能报告
    generate_summary_visualizations(labels, scores, features)


if __name__ == "__main__":
    
    main()