"""
Internal trace:
- Wrong before: image inference logic mixed true model loading with placeholder branches and did not provide a reliable fallback when weights were missing.
- Fixed now: each image slot attempts its configured open-source model once at startup and falls back to a deterministic forensic scorer only when loading fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import settings
from utils.file_utils import clamp


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def _jpeg_artifact_score(image: Image.Image) -> float:
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    block_rows = np.abs(gray[:, 8:] - gray[:, :-8]).mean() / 255.0
    edge_noise = np.abs(laplacian).mean() / 64.0
    return clamp((0.65 * block_rows) + (0.35 * edge_noise))


def _frequency_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = spectrum.shape
    center = spectrum[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    high_freq = spectrum.sum() - center.sum()
    ratio = high_freq / max(spectrum.sum(), 1e-6)
    return clamp(_sigmoid((ratio - 0.62) * 10.0))


def _color_consistency_score(image: Image.Image) -> float:
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    ycrcb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    channel_gap = np.abs(rgb[..., 0] - rgb[..., 2]).mean()
    chroma_var = float(np.var(ycrcb[..., 1:]))
    return clamp(_sigmoid((channel_gap * 2.4 + chroma_var * 1.8) - 0.9))


def _patch_consistency_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    patches = []
    step = max(gray.shape[0] // 8, 16)
    for y in range(0, gray.shape[0] - step + 1, step):
        for x in range(0, gray.shape[1] - step + 1, step):
            patches.append(float(np.var(gray[y : y + step, x : x + step])))
    if not patches:
        return 0.5
    spread = np.std(patches)
    symmetry = np.abs(gray[:, : gray.shape[1] // 2].mean() - gray[:, gray.shape[1] // 2 :].mean())
    return clamp(_sigmoid((spread * 4.0 + symmetry * 2.0) - 0.8))


def _detect_primary_face(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return image.convert('RGB')
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    margin = int(min(w, h) * 0.25)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(rgb.shape[1], x + w + margin)
    y1 = min(rgb.shape[0], y + h + margin)
    return Image.fromarray(rgb[y0:y1, x0:x1]).convert('RGB')


@dataclass
class ImageModelSlot:
    key: str
    label: str
    weight: float
    repo_id: str
    loader: Callable[[], tuple[Callable[[Image.Image], float], str]]


class ImageModelFactory:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_transformers_classifier(self, repo_id: str) -> tuple[Callable[[Image.Image], float], str]:
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModelForImageClassification.from_pretrained(repo_id)
        model.eval().to(self.device)

        def infer(image: Image.Image) -> float:
            inputs = processor(images=image, return_tensors='pt')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            id2label = {int(key): value.lower() for key, value in model.config.id2label.items()}
            fake_indices = [index for index, label in id2label.items() if 'fake' in label or 'deepfake' in label]
            real_indices = [index for index, label in id2label.items() if 'real' in label or 'auth' in label]
            if fake_indices:
                return clamp(float(sum(probs[index] for index in fake_indices)))
            if real_indices:
                return clamp(1.0 - float(sum(probs[index] for index in real_indices)))
            return clamp(float(probs[np.argmax(probs)]))

        return infer, 'primary'

    def build_timm_classifier(self, repo_id: str, model_name: str) -> tuple[Callable[[Image.Image], float], str]:
        import timm

        model = timm.create_model(f'hf_hub:{repo_id}', pretrained=True)
        model.eval().to(self.device)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)

        def infer(image: Image.Image) -> float:
            tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            if len(probs) >= 2:
                return clamp(float(probs[-1]))
            return clamp(float(_sigmoid(float(probs[0]))))

        return infer, 'primary'


def create_image_slots() -> list[ImageModelSlot]:
    factory = ImageModelFactory()
    return [
        ImageModelSlot(
            key='vit',
            label='ViT',
            weight=0.30,
            repo_id=settings.model_vit_repo,
            loader=lambda: factory.build_transformers_classifier(settings.model_vit_repo),
        ),
        ImageModelSlot(
            key='efficientnet',
            label='EfficientNet-B4',
            weight=0.25,
            repo_id=settings.model_efficientnet_repo,
            loader=lambda: factory.build_timm_classifier(settings.model_efficientnet_repo, 'efficientnet_b4'),
        ),
        ImageModelSlot(
            key='xception',
            label='Xception',
            weight=0.25,
            repo_id=settings.model_xception_repo,
            loader=lambda: factory.build_timm_classifier(settings.model_xception_repo, 'xception'),
        ),
        ImageModelSlot(
            key='convnext',
            label='ConvNeXt',
            weight=0.20,
            repo_id=settings.model_convnext_repo,
            loader=lambda: factory.build_timm_classifier(settings.model_convnext_repo, 'convnext_base'),
        ),
    ]


def create_fallback_scorer(slot_key: str) -> Callable[[Image.Image], float]:
    mapping = {
        'vit': _jpeg_artifact_score,
        'efficientnet': _frequency_score,
        'xception': _color_consistency_score,
        'convnext': _patch_consistency_score,
    }
    return mapping[slot_key]


def prepare_image(image_bytes: bytes) -> Image.Image:
    return _detect_primary_face(Image.open(BytesIO(image_bytes)).convert('RGB'))
