"""Mask losses for the pixel decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

DECODER_OUTPUTS = ("mask_logits", "mask_logits_aux", "mask_logits_coarse")


def sigmoid_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Per-pixel focal loss averaged over pixels, then over masks. Soft targets allowed."""
    probabilities = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * ce * (1 - p_t) ** gamma).flatten(1).mean(1).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probabilities * targets).sum(1) + 1
    denominator = probabilities.sum(1) + targets.sum(1) + 1
    return (1 - numerator / denominator).mean()


def mask_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    focal_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    """Mean focal + Dice loss over every decoder output.

    ``targets`` holds one full-image mask or alpha matte per query in ``[0, 1]``
    (``[Q, S, S]``); it is area-resized to each output's resolution.
    """
    losses = []
    for key in DECODER_OUTPUTS:
        logits = outputs[key].float()
        target = targets
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target[:, None], size=logits.shape[-2:], mode="area")[:, 0]
        losses.append(
            focal_weight * sigmoid_focal_loss(logits, target)
            + dice_weight * dice_loss(logits, target)
        )
    return torch.stack(losses).mean()
