import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ImageAnomalyDataset(Dataset):
    def __init__(
        self,
        root: str,
        category: str,
        split: str,
        image_size: int = 256,
        only_normal: bool = False,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.root = root
        self.category = category
        self.split = split
        self.only_normal = only_normal
        self.labels = labels or {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[str, int]]:
        data_dir = os.path.join(self.root, self.category, self.split)
        samples: List[Tuple[str, int]] = []
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        if self.split == "train":
            # 默认只用 good 作为正常
            good_dir = os.path.join(data_dir, "good")
            if os.path.isdir(good_dir):
                samples.extend(
                    [
                        (os.path.join(good_dir, f), 0)
                        for f in os.listdir(good_dir)
                        if f.lower().endswith(exts)
                    ]
                )
            if not self.only_normal:
                bad_dir = os.path.join(data_dir, "bad")
                if os.path.isdir(bad_dir):
                    samples.extend(
                        [
                            (os.path.join(bad_dir, f), 1)
                            for f in os.listdir(bad_dir)
                            if f.lower().endswith(exts)
                        ]
                    )
        else:
            test_dir = data_dir
            for f in os.listdir(test_dir):
                if not f.lower().endswith(exts):
                    continue
                label = 1 if self.labels.get(f) == "bad" else 0
                if self.only_normal and label == 1:
                    continue
                samples.append((os.path.join(test_dir, f), label))
        return sorted(samples, key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "label": label, "path": path}


class TabularDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str,
        only_normal: bool = False,
        features: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.only_normal = only_normal
        if split == "train" and only_normal and "label" in self.df.columns:
            self.df = self.df[self.df["label"] == 0]
        if features is None:
            features = [c for c in self.df.columns if c != "label"]
        self.features = features
        values = self.df[self.features].values.astype(np.float32)
        if scaler is not None:
            values = scaler.transform(values).astype(np.float32)
        self.x = values
        self.y = self.df["label"].values.astype(np.int64) if "label" in self.df.columns else None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict:
        sample = {"features": torch.from_numpy(self.x[idx])}
        if self.y is not None:
            sample["label"] = int(self.y[idx])
        else:
            sample["label"] = 0
        return sample


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        try:
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        except Exception:
            weights = None
        backbone = wide_resnet50_2(weights=weights)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        self.layer3 = backbone.layer3
        fused_channels = 512 + 1024
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fused_channels, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, return_map: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat2 = self.features(x)  # B, 512, H/8, W/8
        feat3 = self.layer3(feat2)  # B, 1024, H/16, W/16
        feat3_up = F.interpolate(feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([feat2, feat3_up], dim=1)
        latent = self.proj(fused)
        if return_map:
            activation = torch.norm(fused, dim=1, keepdim=True)
            activation = F.interpolate(activation, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return latent, activation
        return latent, None


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnifiedGaussianScorer:
    def __init__(self, latent_dim: int, eps: float = 1e-5):
        self.latent_dim = latent_dim
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.cov_inv: Optional[torch.Tensor] = None

    def fit(self, embeddings: torch.Tensor) -> None:
        mean = embeddings.mean(dim=0, keepdim=True)
        zero_centered = embeddings - mean
        cov = zero_centered.T @ zero_centered / (embeddings.shape[0] - 1)
        cov = cov + self.eps * torch.eye(self.latent_dim, device=embeddings.device)
        cov_inv = torch.linalg.inv(cov)
        self.mean = mean.detach()
        self.cov_inv = cov_inv.detach()

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.cov_inv is None:
            raise ValueError("Gaussian scorer 尚未拟合。")
        diff = embeddings - self.mean
        left = diff @ self.cov_inv
        scores = (left * diff).sum(dim=1)
        return scores

    def save(self, path: str) -> None:
        ensure_dir(os.path.dirname(path))
        torch.save({"mean": self.mean, "cov_inv": self.cov_inv}, path)

    def load(self, path: str, device: torch.device) -> None:
        data = torch.load(path, map_location=device)
        self.mean = data["mean"].to(device)
        self.cov_inv = data["cov_inv"].to(device)


@dataclass
class SystemConfig:
    image_root: str
    tabular_root: str
    category: str = "hazelnut"
    image_size: int = 256
    batch_size: int = 8
    latent_dim: int = 256
    num_workers: int = 0
    artifacts_dir: str = "artifacts"
    visual_dir: str = "visualizations"
    reports_dir: str = "reports"
    use_supervised_head: bool = False


class MultiModalAnomalySystem:
    def __init__(self, config: SystemConfig, device: torch.device):
        self.cfg = config
        self.device = device
        ensure_dir(self.cfg.artifacts_dir)
        ensure_dir(self.cfg.visual_dir)
        ensure_dir(self.cfg.reports_dir)
        labels_path = os.path.join(self.cfg.image_root, "image_anomaly_labels.json")
        self.image_labels = load_json(labels_path) if os.path.exists(labels_path) else {}
        self.image_encoder = ImageEncoder(self.cfg.latent_dim).to(self.device)
        tabular_train_path = os.path.join(self.cfg.tabular_root, "train-set.csv")
        train_df = pd.read_csv(tabular_train_path)
        self.feature_cols = [c for c in train_df.columns if c != "label"]
        self.scaler = StandardScaler().fit(train_df[self.feature_cols])
        self.tabular_encoder = TabularEncoder(len(self.feature_cols), self.cfg.latent_dim).to(self.device)
        self.scorer = UnifiedGaussianScorer(self.cfg.latent_dim)

    def _image_loader(self, split: str, only_normal: bool = False) -> DataLoader:
        dataset = ImageAnomalyDataset(
            self.cfg.image_root,
            self.cfg.category,
            split,
            image_size=self.cfg.image_size,
            only_normal=only_normal,
            labels=self.image_labels,
        )
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=(split == "train"), num_workers=self.cfg.num_workers)

    def _tabular_loader(self, split: str, only_normal: bool = False) -> DataLoader:
        csv_path = os.path.join(self.cfg.tabular_root, f"{split}-set.csv")
        dataset = TabularDataset(
            csv_path,
            split,
            only_normal=only_normal,
            features=self.feature_cols,
            scaler=self.scaler,
        )
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=(split == "train"), num_workers=self.cfg.num_workers)

    @torch.no_grad()
    def _collect_embeddings(self) -> torch.Tensor:
        self.image_encoder.eval()
        self.tabular_encoder.eval()
        image_loader = self._image_loader("train", only_normal=True)
        tab_loader = self._tabular_loader("train", only_normal=True)
        all_embeddings: List[torch.Tensor] = []

        for batch in image_loader:
            imgs = batch["image"].to(self.device)
            latent, _ = self.image_encoder(imgs)
            all_embeddings.append(latent.cpu())

        for batch in tab_loader:
            feats = batch["features"].to(self.device)
            latent = self.tabular_encoder(feats)
            all_embeddings.append(latent.cpu())

        return torch.cat(all_embeddings, dim=0)

    def train(self) -> None:
        embeddings = self._collect_embeddings().to(self.device)
        self.scorer.fit(embeddings)
        stats_path = os.path.join(self.cfg.artifacts_dir, "gaussian_stats.pt")
        self.scorer.save(stats_path)
        print(f"已保存高斯统计量：{stats_path}")

    @torch.no_grad()
    def evaluate(self) -> None:
        stats_path = os.path.join(self.cfg.artifacts_dir, "gaussian_stats.pt")
        if not os.path.exists(stats_path):
            raise FileNotFoundError("请先执行 train 以生成统计量。")
        self.scorer.load(stats_path, self.device)
        self.image_encoder.eval()
        self.tabular_encoder.eval()

        image_loader = self._image_loader("test", only_normal=False)
        tab_loader = self._tabular_loader("test", only_normal=False)

        image_results = self._evaluate_images(image_loader)
        tab_results = self._evaluate_tabular(tab_loader)
        report = {"image": image_results["metrics"], "tabular": tab_results["metrics"]}
        report_path = os.path.join(self.cfg.reports_dir, "metrics.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"评估结果已写入：{report_path}")

    def _evaluate_images(self, loader: DataLoader) -> Dict:
        scores: List[float] = []
        labels: List[int] = []
        paths: List[str] = []
        heatmap_dir = os.path.join(self.cfg.visual_dir, "heatmaps")
        ensure_dir(heatmap_dir)
        for batch in loader:
            imgs = batch["image"].to(self.device)
            latent, act = self.image_encoder(imgs, return_map=True)
            batch_scores = self.scorer.score(latent).cpu().numpy().tolist()
            scores.extend(batch_scores)
            labels.extend(batch["label"].tolist())
            paths.extend(batch["path"])
            if act is not None:
                for activation, img_path in zip(act, batch["path"]):
                    norm_act = activation.squeeze(0).cpu().numpy()
                    norm_act = (norm_act - norm_act.min()) / (norm_act.max() - norm_act.min() + 1e-8)
                    heatmap = cv2.applyColorMap(np.uint8(255 * norm_act), cv2.COLORMAP_JET)
                    original = cv2.imread(img_path)
                    if original is None:
                        continue
                    overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)
                    out_path = os.path.join(heatmap_dir, os.path.basename(img_path))
                    cv2.imwrite(out_path, overlay)
        df = pd.DataFrame({"path": paths, "score": scores, "label": labels})
        df.to_csv(os.path.join(self.cfg.artifacts_dir, "image_scores.csv"), index=False)
        metrics = self._compute_metrics(labels, scores)
        return {"metrics": metrics}

    def _evaluate_tabular(self, loader: DataLoader) -> Dict:
        scores: List[float] = []
        labels: List[int] = []
        for batch in loader:
            feats = batch["features"].to(self.device)
            latent = self.tabular_encoder(feats)
            batch_scores = self.scorer.score(latent).cpu().numpy().tolist()
            scores.extend(batch_scores)
            labels.extend(batch["label"].tolist())
        df = pd.DataFrame({"score": scores, "label": labels})
        df.to_csv(os.path.join(self.cfg.artifacts_dir, "tabular_scores.csv"), index=False)
        metrics = self._compute_metrics(labels, scores)
        return {"metrics": metrics}

    @staticmethod
    def _compute_metrics(labels: List[int], scores: List[float]) -> Dict:
        if len(set(labels)) < 2:
            return {"message": "测试集中无标签或单一类别，无法计算指标"}
        y_true = np.array(labels)
        y_score = np.array(scores)
        threshold = np.percentile(y_score[y_true == 0], 95) if np.any(y_true == 0) else np.median(y_score)
        y_pred = (y_score >= threshold).astype(int)
        metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "threshold": float(threshold),
        }
        return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="任务5：多模态统一异常检测框架")
    parser.add_argument("--image_root", type=str, required=True, help="任务2数据根目录")
    parser.add_argument("--tabular_root", type=str, required=True, help="任务4数据根目录")
    parser.add_argument("--category", type=str, default="hazelnut")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["train_only", "eval_only", "train_and_eval"], default="train_and_eval")
    parser.add_argument("--artifacts_dir", type=str, default="task5_multi_modal/artifacts")
    parser.add_argument("--visual_dir", type=str, default="task5_multi_modal/visualizations")
    parser.add_argument("--reports_dir", type=str, default="task5_multi_modal/reports")
    parser.add_argument("--use_supervised_head", action="store_true", help="占位：需额外实现监督头")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SystemConfig(
        image_root=args.image_root,
        tabular_root=args.tabular_root,
        category=args.category,
        image_size=args.image_size,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        num_workers=args.num_workers,
        artifacts_dir=args.artifacts_dir,
        visual_dir=args.visual_dir,
        reports_dir=args.reports_dir,
        use_supervised_head=args.use_supervised_head,
    )
    system = MultiModalAnomalySystem(config, device)
    if args.mode in ("train_only", "train_and_eval"):
        system.train()
    if args.mode in ("eval_only", "train_and_eval"):
        system.evaluate()


if __name__ == "__main__":
    main()


