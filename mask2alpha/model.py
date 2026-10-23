from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

import sam2
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

from .matting_head import RefineMattingHead

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_SHA256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
REPO = Path(__file__).resolve().parents[1]

if (REPO / "sam2").resolve() not in Path(sam2.__file__).resolve().parents:
    warnings.warn(
        f"using sam2 from {Path(sam2.__file__).parent}, not the pinned submodule {REPO / 'sam2'}; "
        "initialize the pinned SAM2 submodule to use the supported version"
    )


class SAM2AlphaRefiner(SAM2Base):
    def __init__(
        self,
        *args,
        head_feature_channels: Sequence[int] = (32, 64, 256),
        head_widths: Sequence[int] = (32, 64, 128, 256),
        head_stem_channels: Sequence[int] = (32, 64),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.matting_head = RefineMattingHead(head_feature_channels, head_widths, head_stem_channels)

    def image_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        backbone_out = self.forward_image(images)
        _, vision_feats, _, feat_sizes = self._prepare_backbone_features(backbone_out)
        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        batch_size = images.shape[0]
        return [feat.permute(1, 2, 0).reshape(batch_size, -1, *size) for feat, size in zip(vision_feats, feat_sizes)]

    def segment_from_prompts(self, features, point_inputs: Dict, multimask_output: bool) -> torch.Tensor:
        outputs = self._forward_sam_heads(
            backbone_features=features[-1],
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=features[:-1],
            multimask_output=multimask_output,
        )
        return outputs[3]

    def predict_alpha(self, images: torch.Tensor, prompt_masks: torch.Tensor, features) -> torch.Tensor:
        feat4, feat8, feat16 = features
        prompt = F.interpolate(prompt_masks.float(), size=feat4.shape[-2:], mode="bilinear", align_corners=False)
        return self.matting_head(images, prompt, feat4, feat8, feat16)[-1]


def file_sha256(path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_skeleton() -> SAM2AlphaRefiner:
    return build_sam2(
        SAM2_CONFIG,
        ckpt_path=None,
        device="cpu",
        mode="eval",
        hydra_overrides_extra=[f"++model._target_={SAM2AlphaRefiner.__module__}.{SAM2AlphaRefiner.__name__}"],
        apply_postprocessing=False,
    )


def build_mask2alpha(
    sam2_checkpoint=REPO / "checkpoints" / "sam2.1_hiera_base_plus.pt",
    head_checkpoint=REPO / "weights" / "mask2alpha_sam2_bplus.pt",
    device="cuda",
    verify_sam2_sha256: bool = True,
) -> SAM2AlphaRefiner:
    """Load the SAM2.1-B+ checkpoint and the head weights, and return the model."""
    head_state = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(head_state, Mapping) or not head_state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in head_state.items()
    ):
        raise ValueError(f"{head_checkpoint}: expected a pure tensor state_dict, without model/meta wrappers")
    if verify_sam2_sha256:
        actual = file_sha256(sam2_checkpoint)
        if actual != SAM2_SHA256:
            raise ValueError(f"SAM2.1 checkpoint sha256 {actual} != {SAM2_SHA256}; expected official SAM2.1-B+")
    model = build_skeleton()
    expected = {key: value for key, value in model.state_dict().items() if key.startswith("matting_head.")}
    if head_state.keys() != expected.keys():
        missing = sorted(expected.keys() - head_state.keys())
        unexpected = sorted(head_state.keys() - expected.keys())
        raise ValueError(f"{head_checkpoint}: head keys mismatch; missing={missing}, unexpected={unexpected}")
    for key, value in head_state.items():
        target = expected[key]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"{head_checkpoint}: {key} has shape/dtype {value.shape}/{value.dtype}, "
                             f"expected {target.shape}/{target.dtype}")
    sam2_state = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict({**sam2_state, **head_state}, strict=True)
    del sam2_state, head_state, expected
    return model.to(device).eval()
