#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像聚类任务解决方案
Task 1: Clustering Task (聚类任务)
- 1.1: 如何处理图像特征 (How to process image features) - 10%
- 1.2: 选择合适的聚类算法 (Select appropriate clustering algorithms) - 10%  
- 1.3: 评估你的聚类效果 (Evaluate your clustering results) - 5%
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
# import seaborn as sns  # 如果seaborn安装有问题，可以注释掉这行
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ImageClustering:
    def __init__(self, dataset_path='dataset', labels_path='cluster_labels.json'):
        """
        初始化图像聚类类
        
        Args:
            dataset_path: 图像数据集路径
            labels_path: 真实标签文件路径
        """
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.images = []
        self.features = []
        self.true_labels = []
        self.image_names = []
        
        # 类别映射
        self.label_to_id = {
            'cable': 0, 'tile': 1, 'bottle': 2, 
            'pill': 3, 'leather': 4, 'transistor': 5
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_data(self):
        """
        加载图像数据和真实标签
        """
        print("正在加载数据...")
        
        # 加载真实标签
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        
        # 加载图像
        for filename in sorted(os.listdir(self.dataset_path)):
            if filename.endswith('.png'):
                image_path = os.path.join(self.dataset_path, filename)
                image = Image.open(image_path)
                self.images.append(image)
                self.image_names.append(filename)
                
                # 获取真实标签
                true_label = labels_dict[filename]
                self.true_labels.append(self.label_to_id[true_label])
        
        print(f"成功加载 {len(self.images)} 张图像")
        print(f"类别分布: {Counter([self.id_to_label[label] for label in self.true_labels])}")
        
    def extract_features(self, method='histogram'):
        """
        任务1.1: 图像特征提取
        
        Args:
            method: 特征提取方法 ('histogram', 'pixel', 'combined')
        """
        print(f"\n=== 任务1.1: 图像特征提取 (方法: {method}) ===")
        
        if method == 'histogram':
            self._extract_histogram_features()
        elif method == 'pixel':
            self._extract_pixel_features()
        elif method == 'combined':
            self._extract_combined_features()
        else:
            raise ValueError("不支持的特征提取方法")
            
        print(f"特征维度: {self.features.shape}")
        print(f"特征范围: [{self.features.min():.3f}, {self.features.max():.3f}]")
        
    def _extract_histogram_features(self):
        """
        提取颜色直方图特征
        """
        features_list = []
        
        for image in self.images:
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 计算RGB直方图
            r_hist = np.histogram(image.getchannel('R'), bins=32, range=(0, 256))[0]
            g_hist = np.histogram(image.getchannel('G'), bins=32, range=(0, 256))[0]
            b_hist = np.histogram(image.getchannel('B'), bins=32, range=(0, 256))[0]
            
            # 归一化直方图
            r_hist = r_hist / np.sum(r_hist)
            g_hist = g_hist / np.sum(g_hist)
            b_hist = b_hist / np.sum(b_hist)
            
            # 合并特征
            feature = np.concatenate([r_hist, g_hist, b_hist])
            features_list.append(feature)
        
        self.features = np.array(features_list)
        
    def _extract_pixel_features(self):
        """
        提取像素级特征
        """
        features_list = []
        
        for image in self.images:
            # 转换为RGB并调整大小
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 调整图像大小以减少特征维度
            image = image.resize((32, 32))
            
            # 展平像素值
            pixels = np.array(image).flatten()
            
            # 归一化
            pixels = pixels / 255.0
            
            features_list.append(pixels)
        
        self.features = np.array(features_list)
        
    def _extract_combined_features(self):
        """
        提取组合特征（直方图 + 统计特征）
        """
        features_list = []
        
        for image in self.images:
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 颜色直方图特征
            r_hist = np.histogram(image.getchannel('R'), bins=16, range=(0, 256))[0]
            g_hist = np.histogram(image.getchannel('G'), bins=16, range=(0, 256))[0]
            b_hist = np.histogram(image.getchannel('B'), bins=16, range=(0, 256))[0]
            
            # 归一化
            r_hist = r_hist / np.sum(r_hist)
            g_hist = g_hist / np.sum(g_hist)
            b_hist = b_hist / np.sum(b_hist)
            
            # 统计特征
            img_array = np.array(image)
            mean_r, mean_g, mean_b = np.mean(img_array, axis=(0, 1))
            std_r, std_g, std_b = np.std(img_array, axis=(0, 1))
            
            # 合并所有特征
            feature = np.concatenate([
                r_hist, g_hist, b_hist,
                [mean_r, mean_g, mean_b],
                [std_r, std_g, std_b]
            ])
            features_list.append(feature)
        
        self.features = np.array(features_list)
        
    def apply_pca(self, n_components=50):
        """
        应用PCA降维
        """
        print(f"\n应用PCA降维到 {n_components} 维...")
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # PCA降维
        pca = PCA(n_components=n_components)
        self.features_pca = pca.fit_transform(features_scaled)
        
        print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"降维后特征维度: {self.features_pca.shape}")
        
        return pca
        
    def cluster_images(self, method='kmeans', n_clusters=6):
        """
        任务1.2: 聚类算法选择
        
        Args:
            method: 聚类方法 ('kmeans', 'hierarchical')
            n_clusters: 聚类数量
        """
        print(f"\n=== 任务1.2: 聚类算法选择 (方法: {method}) ===")
        
        # 使用PCA降维后的特征
        features_to_use = self.features_pca if hasattr(self, 'features_pca') else self.features
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError("不支持的聚类方法")
        
        # 执行聚类
        self.cluster_labels = clusterer.fit_predict(features_to_use)
        
        print(f"聚类完成，聚类数量: {n_clusters}")
        print(f"聚类结果分布: {Counter(self.cluster_labels)}")
        
        return clusterer
        
    def evaluate_clustering(self):
        """
        任务1.3: 聚类效果评估
        """
        print(f"\n=== 任务1.3: 聚类效果评估 ===")
        
        # 计算各种评估指标
        metrics = {}
        
        # 1. 轮廓系数
        features_to_use = self.features_pca if hasattr(self, 'features_pca') else self.features
        silhouette_avg = silhouette_score(features_to_use, self.cluster_labels)
        metrics['轮廓系数'] = silhouette_avg
        
        # 2. 调整兰德指数
        ari = adjusted_rand_score(self.true_labels, self.cluster_labels)
        metrics['调整兰德指数'] = ari
        
        # 3. 聚类纯度
        purity = self._calculate_purity(self.true_labels, self.cluster_labels)
        metrics['聚类纯度'] = purity
        
        # 4. 互信息
        nmi = self._calculate_nmi(self.true_labels, self.cluster_labels)
        metrics['标准化互信息'] = nmi
        
        # 打印结果
        print("评估指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
        
    def _calculate_purity(self, true_labels, cluster_labels):
        """
        计算聚类纯度
        """
        total = len(true_labels)
        correct = 0
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = np.array(true_labels)[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                most_common = Counter(cluster_true_labels).most_common(1)[0][0]
                correct += np.sum(cluster_true_labels == most_common)
        
        return correct / total
    
    def _calculate_nmi(self, true_labels, cluster_labels):
        """
        计算标准化互信息
        """
        def entropy(labels):
            counts = Counter(labels)
            total = len(labels)
            return -sum((count/total) * np.log2(count/total) for count in counts.values())
        
        def mutual_information(true_labels, cluster_labels):
            true_counts = Counter(true_labels)
            cluster_counts = Counter(cluster_labels)
            joint_counts = Counter(zip(true_labels, cluster_labels))
            
            total = len(true_labels)
            
            mi = 0
            for (true_label, cluster_label), joint_count in joint_counts.items():
                p_joint = joint_count / total
                p_true = true_counts[true_label] / total
                p_cluster = cluster_counts[cluster_label] / total
                
                if p_joint > 0:
                    mi += p_joint * np.log2(p_joint / (p_true * p_cluster))
            
            return mi
        
        mi = mutual_information(true_labels, cluster_labels)
        h_true = entropy(true_labels)
        h_cluster = entropy(cluster_labels)
        
        if h_true == 0 or h_cluster == 0:
            return 0
        
        return mi / np.sqrt(h_true * h_cluster)
    
    def visualize_results(self, save_path='clustering_results.png'):
        """
        可视化聚类结果
        """
        print(f"\n生成可视化结果...")
        
        # 使用PCA降维到2D进行可视化
        pca_2d = PCA(n_components=2)
        features_2d = pca_2d.fit_transform(self.features_pca if hasattr(self, 'features_pca') else self.features)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 真实标签分布
        scatter1 = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                     c=self.true_labels, cmap='tab10', alpha=0.7)
        axes[0, 0].set_title('真实标签分布')
        axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
        
        # 2. 聚类结果分布
        scatter2 = axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=self.cluster_labels, cmap='tab10', alpha=0.7)
        axes[0, 1].set_title('聚类结果分布')
        axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
        axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
        
        # 3. 类别分布柱状图
        true_counts = Counter(self.true_labels)
        cluster_counts = Counter(self.cluster_labels)
        
        categories = list(self.id_to_label.values())
        true_values = [true_counts.get(i, 0) for i in range(6)]
        cluster_values = [cluster_counts.get(i, 0) for i in range(6)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, true_values, width, label='真实标签', alpha=0.8)
        axes[1, 0].bar(x + width/2, cluster_values, width, label='聚类结果', alpha=0.8)
        axes[1, 0].set_title('类别分布对比')
        axes[1, 0].set_xlabel('类别')
        axes[1, 0].set_ylabel('样本数量')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories, rotation=45)
        axes[1, 0].legend()
        
        # 4. 混淆矩阵
        confusion_matrix = np.zeros((6, 6))
        for true_label, cluster_label in zip(self.true_labels, self.cluster_labels):
            confusion_matrix[true_label, cluster_label] += 1
        
        # 使用matplotlib绘制热力图（替代seaborn）
        im = axes[1, 1].imshow(confusion_matrix, cmap='Blues')
        axes[1, 1].set_xticks(range(6))
        axes[1, 1].set_yticks(range(6))
        axes[1, 1].set_xticklabels([f'Cluster {i}' for i in range(6)])
        axes[1, 1].set_yticklabels(categories)
        
        # 添加数值标注
        for i in range(6):
            for j in range(6):
                axes[1, 1].text(j, i, f'{int(confusion_matrix[i, j])}', 
                               ha='center', va='center', color='black')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[1, 1])
        axes[1, 1].set_title('混淆矩阵')
        axes[1, 1].set_xlabel('聚类标签')
        axes[1, 1].set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化结果已保存到: {save_path}")
    
    def run_complete_analysis(self):
        """
        运行完整的聚类分析
        """
        print("=" * 60)
        print("图像聚类任务完整分析")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 特征提取（尝试不同方法）
        methods = ['histogram', 'pixel', 'combined']
        best_method = None
        best_score = -1
        
        for method in methods:
            print(f"\n尝试特征提取方法: {method}")
            self.extract_features(method=method)
            
            # PCA降维
            self.apply_pca(n_components=50)
            
            # K-means聚类
            self.cluster_images(method='kmeans', n_clusters=6)
            
            # 评估
            metrics = self.evaluate_clustering()
            
            # 选择最佳方法（基于轮廓系数）
            if metrics['轮廓系数'] > best_score:
                best_score = metrics['轮廓系数']
                best_method = method
        
        print(f"\n最佳特征提取方法: {best_method} (轮廓系数: {best_score:.4f})")
        
        # 3. 使用最佳方法重新运行
        print(f"\n使用最佳方法 {best_method} 重新运行...")
        self.extract_features(method=best_method)
        self.apply_pca(n_components=50)
        
        # 4. 比较不同聚类算法
        print("\n比较不同聚类算法:")
        
        # K-means
        self.cluster_images(method='kmeans', n_clusters=6)
        kmeans_metrics = self.evaluate_clustering()
        
        # 层次聚类
        self.cluster_images(method='hierarchical', n_clusters=6)
        hierarchical_metrics = self.evaluate_clustering()
        
        print("\n算法比较结果:")
        print("K-means:")
        for metric, value in kmeans_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("层次聚类:")
        for metric, value in hierarchical_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 5. 可视化结果
        self.visualize_results()
        
        return {
            'best_method': best_method,
            'kmeans_metrics': kmeans_metrics,
            'hierarchical_metrics': hierarchical_metrics
        }

def main():
    """
    主函数
    """
    # 创建聚类分析器
    cluster_analyzer = ImageClustering()
    
    # 运行完整分析
    results = cluster_analyzer.run_complete_analysis()
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()
