"""Batched alpha prediction from a coarse mask, a bounding box, or click points."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("mask", "bbox", "point")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
INPUT_SIZE = 1024


def transform_images(images: Sequence[np.ndarray], device) -> torch.Tensor:
    tensors = []
    for image in images:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = tensor.float().div_(255.0).unsqueeze(0)
        tensors.append(F.interpolate(tensor, (INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False))
    batch = torch.cat(tensors, dim=0)
    return ((batch - MEAN) / STD).to(device, non_blocking=True)


def transform_masks(masks: Sequence[np.ndarray], device) -> torch.Tensor:
    result = []
    for mask in masks:
        tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]
        result.append(F.interpolate(tensor, (INPUT_SIZE, INPUT_SIZE), mode="nearest"))
    return torch.cat(result, dim=0).to(device, non_blocking=True)


def transform_boxes(boxes, shapes, device) -> dict:
    values = []
    for box, (height, width) in zip(boxes, shapes):
        scaled = np.asarray(box, dtype=np.float32).reshape(2, 2).copy()
        scaled[..., 0] *= INPUT_SIZE / width
        scaled[..., 1] *= INPUT_SIZE / height
        values.append(torch.from_numpy(scaled))
    coords = torch.stack(values).float().to(device)
    labels = torch.tensor([2, 3], dtype=torch.int32, device=device).repeat(len(values), 1)
    return {"point_coords": coords, "point_labels": labels}


def transform_points(points, labels, shapes, device) -> dict:
    coords_list, labels_list = [], []
    for point, label, (height, width) in zip(points, labels, shapes):
        scaled = np.asarray(point, dtype=np.float32).reshape(-1, 2).copy()
        scaled[..., 0] *= INPUT_SIZE / width
        scaled[..., 1] *= INPUT_SIZE / height
        coords_list.append(torch.from_numpy(scaled))
        if label is None:
            label = np.ones(len(scaled), dtype=np.int32)
        labels_list.append(torch.as_tensor(np.asarray(label, dtype=np.int32)))
    if len({len(item) for item in coords_list}) != 1:
        raise ValueError("every image in a point batch needs the same number of points")
    return {"point_coords": torch.stack(coords_list).float().to(device), "point_labels": torch.stack(labels_list).to(device)}


def binarize_mask(mask: np.ndarray, threshold: int = 0) -> np.ndarray:
    """``mask > threshold`` on the uint8 scale; float masks are taken as [0, 1]."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = np.rint(np.clip(mask.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return (mask > int(threshold)).astype(np.uint8)


def tight_box_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("cannot derive a bbox from an empty mask")
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def alpha_to_uint8(alpha: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)


class SAM2AlphaPredictor:
    """Wraps a SAM2AlphaRefiner model.

    mode="mask"  : binary coarse mask (H x W, non-zero = object) goes straight to the head.
    mode="bbox"  : xyxy box in input pixels -> SAM2 mask -> matting.
    mode="point" : click points (label 1 positive / 0 negative) -> SAM2 (best of 3 masks) -> matting.
    """

    def __init__(self, model, device="cuda", amp_dtype: str = "bf16"):
        if amp_dtype not in ("bf16", "fp32"):
            raise ValueError("amp_dtype must be bf16 or fp32")
        self.model = model
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype

    @torch.inference_mode()
    def predict(
        self,
        images: Sequence[np.ndarray],
        mode: str,
        masks: Optional[Sequence[np.ndarray]] = None,
        boxes: Optional[Sequence[np.ndarray]] = None,
        points: Optional[Sequence[np.ndarray]] = None,
        point_labels: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Per-image float32 alpha in [0, 1] and uint8 prompt masks, both at the input size.

        ``images`` are RGB uint8 H x W x 3 arrays.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        images = list(images)
        shapes = [image.shape[:2] for image in images]
        if mode == "mask":
            if masks is None or len(masks) != len(images):
                raise ValueError("mode='mask' needs one binary mask per image")
            for mask, shape in zip(masks, shapes):
                if mask.shape[:2] != shape:
                    raise ValueError(f"mask shape {mask.shape[:2]} != image shape {shape}")
                if not np.any(mask):
                    raise ValueError("coarse mask is empty")
        elif mode == "bbox":
            if boxes is None or len(boxes) != len(images):
                raise ValueError("mode='bbox' needs one xyxy box per image")
        elif points is None or len(points) != len(images):
            raise ValueError("mode='point' needs points for every image")

        use_amp = self.amp_dtype == "bf16" and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            image_batch = transform_images(images, self.device)
            if mode == "mask":
                prompt = transform_masks([(np.asarray(mask) > 0).astype(np.uint8) for mask in masks], self.device)
            features = self.model.image_features(image_batch)
            if mode == "bbox":
                logits = self.model.segment_from_prompts(features, transform_boxes(boxes, shapes, self.device), False)
                prompt = (logits > 0).float()
            elif mode == "point":
                labels = point_labels if point_labels is not None else [None] * len(images)
                logits = self.model.segment_from_prompts(
                    features, transform_points(points, labels, shapes, self.device), True
                )
                prompt = (logits > 0).float()
            alpha = self.model.predict_alpha(image_batch, prompt, features).float()

            alphas, prompts = [], []
            for index, shape in enumerate(shapes):
                resized = F.interpolate(alpha[index : index + 1], shape, mode="bilinear", align_corners=False)[0, 0]
                alphas.append(resized.clamp_(0, 1).cpu().numpy())
                restored = F.interpolate(prompt[index : index + 1].float(), shape, mode="nearest")[0, 0]
                prompts.append((restored > 0.5).cpu().numpy().astype(np.uint8))
        return alphas, prompts
