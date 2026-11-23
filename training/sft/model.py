"""Qwen3-VL and the WeVisOne-VL pixel decoder as one trainable module."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from wevisone_vl.config import Qwen3VLSegConfig
from wevisone_vl.decoder import build_mask_decoder_from_qwen
from wevisone_vl.feature_bridge import (
    VisualFeatureCapture,
    per_sample_premerge_feature_maps,
    premerge_grid_hw,
)
from wevisone_vl.mask_decoder import WeVisOneMaskDecoder

from .losses import mask_loss

MASK_DECODER_FILE = "mask_decoder.pt"


def load_mask_decoder(
    qwen, path: str, obj_token_id: int
) -> tuple[WeVisOneMaskDecoder, Qwen3VLSegConfig]:
    """Build the pixel decoder from a ``mask_decoder.pt`` sidecar."""
    sidecar = torch.load(path, map_location="cpu", weights_only=True)
    saved_token_id = sidecar.get("seg_token_id")
    if saved_token_id is not None and int(saved_token_id) != obj_token_id:
        raise ValueError(f"{path} expects <OBJ> id {saved_token_id}, tokenizer has {obj_token_id}")
    cfg = Qwen3VLSegConfig.from_dict(sidecar.get("cfg"))
    decoder = build_mask_decoder_from_qwen(qwen, cfg)
    decoder.load_state_dict(sidecar["state_dict"], strict=True)
    return decoder.float(), cfg


def configure_trainable(qwen, *, tune_vision: bool, tune_merger: bool, tune_llm: bool) -> None:
    visual = qwen.model.visual
    qwen.requires_grad_(tune_llm)
    visual.requires_grad_(tune_vision)
    for merger in (visual.merger, *visual.deepstack_merger_list):
        merger.requires_grad_(tune_merger)


class FinalHiddenCapture:
    """Record the last decoder-layer output as the input of the final norm.

    Hooking the last decoder layer itself would record a tensor without
    autograd history when that layer runs under reentrant gradient checkpointing.
    """

    def __init__(self, qwen) -> None:
        self.norm = qwen.model.language_model.norm
        self.hidden: torch.Tensor | None = None
        self._handle = None

    def _hook(self, _module, inputs) -> None:
        self.hidden = inputs[0]

    def __enter__(self):
        self._handle = self.norm.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc) -> bool:
        self._handle.remove()
        return False


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


class WeVisOneForTraining(nn.Module):
    """Language-model loss on every sample plus a mask loss on samples with masks."""

    def __init__(
        self,
        qwen,
        mask_decoder: WeVisOneMaskDecoder,
        decoder_cfg: Qwen3VLSegConfig,
        obj_token_id: int,
        *,
        train_mask_decoder: bool,
        mask_loss_weight: float,
        mask_focal_weight: float,
        mask_dice_weight: float,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        self.mask_decoder = mask_decoder
        self.decoder_cfg = decoder_cfg
        self.obj_token_id = int(obj_token_id)
        self.train_mask_decoder = bool(train_mask_decoder)
        self.mask_loss_weight = float(mask_loss_weight)
        self.mask_focal_weight = float(mask_focal_weight)
        self.mask_dice_weight = float(mask_dice_weight)
        self.image_token_id = int(qwen.config.image_token_id)
        self.spatial_merge_size = int(qwen.config.vision_config.spatial_merge_size)
        mask_decoder.requires_grad_(self.train_mask_decoder)

    @property
    def config(self):
        return self.qwen.config

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.qwen.gradient_checkpointing_enable(**kwargs)
        self.qwen.enable_input_require_grads()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        decoder_images: list[torch.Tensor | None] | None = None,
        obj_indices: list[torch.Tensor | None] | None = None,
        obj_boxes: list[torch.Tensor | None] | None = None,
        obj_masks: list[torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "use_cache": False,
        }
        if not self.train_mask_decoder:
            return {"loss": self.qwen(**qwen_inputs).loss}

        if obj_boxes is None or all(boxes is None for boxes in obj_boxes):
            loss_text = self.qwen(**qwen_inputs).loss
            # Every rank must produce gradients for the same decoder parameters.
            unused = sum(p.sum() for p in self.mask_decoder.parameters()) * 0.0
            return {"loss": loss_text + unused, "loss_text": loss_text.detach()}

        with (
            VisualFeatureCapture(
                self.qwen, feature_layers=self.decoder_cfg.vision_feature_layers
            ) as visual,
            FinalHiddenCapture(self.qwen) as final,
        ):
            loss_text = self.qwen(**qwen_inputs).loss
        loss_mask = self._mask_loss(
            final.hidden,
            visual.premerge_features,
            input_ids,
            image_grid_thw,
            decoder_images,
            obj_indices,
            obj_boxes,
            obj_masks,
        )
        return {
            "loss": loss_text + self.mask_loss_weight * loss_mask,
            "loss_text": loss_text.detach(),
            "loss_mask": loss_mask.detach(),
        }

    def _mask_loss(
        self,
        hidden: torch.Tensor,
        premerge_features: dict[int, torch.Tensor],
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        decoder_images: list[torch.Tensor | None],
        obj_indices: list[torch.Tensor | None],
        obj_boxes: list[torch.Tensor | None],
        obj_masks: list[torch.Tensor | None],
    ) -> torch.Tensor:
        feature_maps = per_sample_premerge_feature_maps(
            premerge_features,
            self.decoder_cfg.vision_feature_layers,
            image_grid_thw,
            self.spatial_merge_size,
        )
        grids = premerge_grid_hw(image_grid_thw)
        image_slots = ((input_ids == self.image_token_id).any(dim=1).cumsum(0) - 1).tolist()
        dtype = next(self.mask_decoder.parameters()).dtype

        total, num_queries = hidden.new_zeros((), dtype=torch.float32), 0
        for b, boxes in enumerate(obj_boxes):
            if boxes is None:
                continue
            slot = image_slots[b]
            t_seg = hidden[b][input_ids[b] == self.obj_token_id][obj_indices[b]]
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                outputs = self.mask_decoder.decode(
                    decoder_images[b][None].to(dtype) / 255.0,
                    [feature.to(dtype) for feature in feature_maps[slot]],
                    t_seg.to(dtype),
                    boxes.to(dtype),
                    grids[slot],
                )
                loss = mask_loss(
                    outputs,
                    obj_masks[b].float() / 255.0,
                    focal_weight=self.mask_focal_weight,
                    dice_weight=self.mask_dice_weight,
                )
            total = total + loss * len(boxes)
            num_queries += len(boxes)
        return total / num_queries

    def save_pretrained(
        self, output_dir: str, state_dict: dict[str, torch.Tensor] | None = None
    ) -> None:
        """Write a Hugging Face model directory plus the ``mask_decoder.pt`` sidecar."""
        state_dict = self.state_dict() if state_dict is None else state_dict
        self.qwen.save_pretrained(output_dir, state_dict=_strip_prefix(state_dict, "qwen."))
        decoder_state = _strip_prefix(state_dict, "mask_decoder.")
        torch.save(
            {
                "cfg": self.decoder_cfg.to_dict(),
                "state_dict": {k: v.detach().float().cpu() for k, v in decoder_state.items()},
                "seg_token_id": self.obj_token_id,
                "spatial_merge_size": self.spatial_merge_size,
            },
            Path(output_dir) / MASK_DECODER_FILE,
        )

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load weights written by :meth:`save_pretrained`."""
        loaded = type(self.qwen).from_pretrained(checkpoint_dir, dtype=self.qwen.dtype)
        self.qwen.load_state_dict(loaded.state_dict())
        sidecar = torch.load(
            Path(checkpoint_dir) / MASK_DECODER_FILE, map_location="cpu", weights_only=True
        )
        self.mask_decoder.load_state_dict(sidecar["state_dict"], strict=True)
