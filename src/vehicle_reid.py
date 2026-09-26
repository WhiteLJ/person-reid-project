"""PC-only FastReID VeRi SBS(R50-IBN) feature extraction.

MVP-8.3-PC1 deliberately keeps Vehicle ReID independent from the person
pipeline.  The small inference model below mirrors the inference-relevant
parts of the official FastReID ``configs/VeRi/sbs_R50-ibn.yml``: ResNet-50
with IBN, the configured Non-local blocks, Generalized Mean Pooling, and the
BN neck.  It does not import ``references/fast-reid`` and contains no tracking,
Gallery, SQLite, or Person OSNet logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import VehicleReIDConfig, resolve_device
from .vehicle_reid_common import (
    normalize_vehicle_features,
    prepare_vehicle_batch,
    preprocess_vehicle_crop,
)


LOGGER = getLogger(__name__)


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Validate and L2-normalize a batch of features as float32."""
    return normalize_vehicle_features(features)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        for key in ("features", "embedding", "embeddings"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError("Vehicle ReID model output mapping has no feature value")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


class _FastReIDBatchNorm(nn.BatchNorm2d):
    """State-compatible BN used by the FastReID backbone/head."""

    def __init__(self, channels: int, *, bias_freeze: bool = False) -> None:
        super().__init__(channels, eps=1e-5, momentum=0.1)
        nn.init.constant_(self.weight, 1.0)
        nn.init.constant_(self.bias, 0.0)
        self.bias.requires_grad_(not bias_freeze)


class _IBN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        first = channels // 2
        self.half = first
        self.IN = nn.InstanceNorm2d(first, affine=True)
        self.BN = _FastReIDBatchNorm(channels - first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = torch.split(x, self.half, dim=1)
        return torch.cat(
            (self.IN(first.contiguous()), self.BN(second.contiguous())), dim=1
        )


class _NonLocal(nn.Module):
    """The inference-time Non-local block used by official FastReID 0.1.1."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # This is the value in the official FastReID implementation.
        inter_channels = 1
        self.g = nn.Conv2d(channels, inter_channels, 1, 1, 0)
        self.W = nn.Sequential(
            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            _FastReIDBatchNorm(channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(channels, inter_channels, 1, 1, 0)
        self.phi = nn.Conv2d(channels, inter_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, 1, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, 1, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, 1, -1)
        affinity = torch.matmul(theta_x, phi_x)
        affinity = affinity / affinity.size(-1)
        y = torch.matmul(affinity, g_x)
        y = y.permute(0, 2, 1).contiguous().view(
            batch_size, 1, x.size(2), x.size(3)
        )
        return self.W(y) + x


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        with_ibn: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = _IBN(planes) if with_ibn else _FastReIDBatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = _FastReIDBatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = _FastReIDBatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = nn.Identity()
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    1,
                    stride=stride,
                    bias=False,
                ),
                _FastReIDBatchNorm(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _VehicleResNetIBN(nn.Module):
    """Minimal state-compatible FastReID R50-IBN + Non-local backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = _FastReIDBatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(64, 3, stride=1, with_ibn=True)
        self.layer2 = self._make_layer(128, 4, stride=2, with_ibn=True)
        self.layer3 = self._make_layer(256, 6, stride=2, with_ibn=True)
        self.layer4 = self._make_layer(512, 3, stride=1, with_ibn=False)

        self.NL_1 = nn.ModuleList()
        self.NL_2 = nn.ModuleList([_NonLocal(512) for _ in range(2)])
        self.NL_3 = nn.ModuleList([_NonLocal(1024) for _ in range(3)])
        self.NL_4 = nn.ModuleList()
        self.NL_1_idx = []
        self.NL_2_idx = [2, 3]
        self.NL_3_idx = [3, 4, 5]
        self.NL_4_idx = []

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        *,
        stride: int,
        with_ibn: bool,
    ) -> nn.Sequential:
        layers = [
            _Bottleneck(
                self.inplanes,
                planes,
                stride=stride,
                with_ibn=with_ibn,
            )
        ]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                _Bottleneck(
                    self.inplanes,
                    planes,
                    stride=1,
                    with_ibn=with_ibn,
                )
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _run_stage(
        x: torch.Tensor,
        stage: nn.Sequential,
        non_local: nn.ModuleList,
        indices: list[int],
    ) -> torch.Tensor:
        counter = 0
        for index, block in enumerate(stage):
            x = block(x)
            if counter < len(indices) and index == indices[counter]:
                x = non_local[counter](x)
                counter += 1
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self._run_stage(x, self.layer1, self.NL_1, self.NL_1_idx)
        x = self._run_stage(x, self.layer2, self.NL_2, self.NL_2_idx)
        x = self._run_stage(x, self.layer3, self.NL_3, self.NL_3_idx)
        return self._run_stage(x, self.layer4, self.NL_4, self.NL_4_idx)


class _GeneralizedMeanPoolingP(nn.Module):
    def __init__(self, norm: float = 3.0) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        powered = x.clamp(min=1e-6).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(powered, (1, 1)).pow(
            1.0 / self.p
        )


class _VehicleHeads(nn.Module):
    """State-compatible FastReID EmbeddingHead inference subset."""

    def __init__(self) -> None:
        super().__init__()
        self.pool_layer = _GeneralizedMeanPoolingP()
        self.bottleneck = nn.Sequential(
            _FastReIDBatchNorm(2048, bias_freeze=True)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_layer(features)
        neck_features = self.bottleneck(pooled)
        return neck_features[..., 0, 0]


class _VehicleSBSModel(nn.Module):
    """Evaluation-only Baseline equivalent for the official VeRi config."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _VehicleResNetIBN()
        self.heads = _VehicleHeads()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.heads(features)


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict"):
            if isinstance(checkpoint.get(key), Mapping):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Vehicle checkpoint does not contain a state_dict mapping")
    state = dict(checkpoint)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


class VehicleReIDExtractor:
    """Extract normalized FastReID VeRi embeddings from BGR vehicle crops."""

    def __init__(
        self,
        config: VehicleReIDConfig,
        device: str | None = None,
        *,
        model: Any | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(device or config.device)
        self._embedding_dim = embedding_dim
        if model is None:
            weight_path = Path(config.weight)
            if not weight_path.is_file():
                raise FileNotFoundError(
                    "Vehicle ReID checkpoint not found: "
                    f"{weight_path}. Download the official FastReID VeRi "
                    "checkpoint before constructing VehicleReIDExtractor."
                )
            model = _VehicleSBSModel()
            self._load_checkpoint(model, weight_path)
        self.model = model
        if hasattr(self.model, "to"):
            self.model.to(torch.device(self.device))
        if hasattr(self.model, "eval"):
            self.model.eval()

    @staticmethod
    def _load_checkpoint(model: nn.Module, path: Path) -> None:
        checkpoint = torch.load(str(path), map_location="cpu")
        state = _checkpoint_state_dict(checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            raise RuntimeError(
                "Vehicle ReID checkpoint is incompatible with official VeRi "
                f"SBS(R50-IBN) inference model; missing={missing[:8]}"
            )
        LOGGER.info(
            "VEHICLE_REID_CHECKPOINT_LOADED path=%s unexpected_keys=%d",
            path,
            len(unexpected),
        )

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            preprocess_vehicle_crop(
                crop,
                image_height=self.config.image_height,
                image_width=self.config.image_width,
            )
        )

    def _prepare_batch(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(
            prepare_vehicle_batch(
                crops,
                image_height=self.config.image_height,
                image_width=self.config.image_width,
            )
        )

    def extract(self, crop: np.ndarray) -> np.ndarray:
        return self.extract_batch([crop])[0]

    def extract_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        crop_list = list(crops)
        if not crop_list:
            dimension = self._embedding_dim or 0
            return np.empty((0, dimension), dtype=np.float32)

        batch = self._prepare_batch(crop_list).to(torch.device(self.device))
        with torch.no_grad():
            raw = self.model(batch)
        features = _as_numpy(raw)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[0] != len(crop_list):
            raise ValueError(
                "Vehicle ReID model output must have shape (N, D) matching crops"
            )
        normalized = _normalize_features(features)
        actual_dimension = int(normalized.shape[1])
        if self._embedding_dim is None:
            self._embedding_dim = actual_dimension
            LOGGER.info("VEHICLE_REID_EMBEDDING_DIM dimension=%d", actual_dimension)
        elif actual_dimension != self._embedding_dim:
            raise ValueError(
                f"Vehicle ReID embedding dimension changed from "
                f"{self._embedding_dim} to {actual_dimension}"
            )
        return normalized.copy()
