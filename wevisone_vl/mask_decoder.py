"""Final WeVisOne-VL mask decoder used by the public checkpoint."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3VLSegConfig
from .msda import MultiScaleDeformableAttention


def _group_norm(channels: int, groups: int) -> nn.GroupNorm:
    groups = min(int(groups), int(channels))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _mask_feature_dim(cfg: Qwen3VLSegConfig) -> int:
    value = int(getattr(cfg, "mask_feature_dim", 0) or 0)
    value = value if value > 0 else int(cfg.hidden_dim)
    if value <= 0:
        raise ValueError(f"mask_feature_dim must resolve positive, got {value}")
    return value


class FourierPE1D(nn.Module):
    def __init__(self, num_freqs: int):
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_freqs, dtype=torch.float32),
            persistent=False,
        )

    @property
    def out_dim(self) -> int:
        return 2 * int(self.frequencies.numel())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * values[..., None].float() * self.frequencies
        return torch.cat([phase.sin(), phase.cos()], dim=-1).to(values.dtype)


class BoxFourierEncoder(nn.Module):
    """Fixed-frequency x1/y1/log(w)/log(h) box encoding."""

    def __init__(self, hidden_dim: int, num_freqs: int):
        super().__init__()
        self.pe = FourierPE1D(num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.pe.out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        width = (x2 - x1).clamp_min(1e-6)
        height = (y2 - y1).clamp_min(1e-6)
        encoded_w = 0.2 * width.log() + 0.5
        encoded_h = 0.2 * height.log() + 0.5
        encoded = torch.cat(
            [self.pe(x1), self.pe(y1), self.pe(encoded_w), self.pe(encoded_h)],
            dim=-1,
        )
        return self.mlp(encoded)


class MaskQueryBuilder(nn.Module):
    """Combine LLM ``<OBJ>`` content with fixed-Fourier box position."""

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int,
        num_freqs: int,
    ):
        super().__init__()
        self.content_proj = nn.Linear(llm_dim, hidden_dim)
        self.box_position = BoxFourierEncoder(hidden_dim, num_freqs)
        self.content_norm = nn.LayerNorm(hidden_dim)
        self.position_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        t_seg: torch.Tensor,
        boxes_xyxy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        content = self.content_norm(self.content_proj(t_seg))
        position = self.position_norm(self.box_position(boxes_xyxy))
        return content, position


class PositionEmbeddingSine2D(nn.Module):
    """DINO/DETR normalized 2-D sine/cosine position encoding."""

    def __init__(self, hidden_dim: int, temperature: float = 10000.0):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for 2-D sine position encoding")
        self.num_pos_feats = hidden_dim // 2
        self.temperature = float(temperature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feature.shape
        dtype = feature.dtype
        y = torch.arange(h, device=feature.device, dtype=torch.float32) + 0.5
        x = torch.arange(w, device=feature.device, dtype=torch.float32) + 0.5
        y = y / max(float(h), 1.0) * (2.0 * math.pi)
        x = x / max(float(w), 1.0) * (2.0 * math.pi)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dim_t = torch.arange(
            self.num_pos_feats, device=feature.device, dtype=torch.float32
        )
        dim_t = self.temperature ** (
            2.0 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = xx[..., None] / dim_t
        pos_y = yy[..., None] / dim_t
        pos_x = torch.stack(
            [pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            [pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1
        ).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1).to(dtype)


def _spatial_metadata(
    features: Sequence[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    shapes = torch.tensor(
        [[int(x.shape[-2]), int(x.shape[-1])] for x in features],
        device=device,
        dtype=torch.int32,
    )
    sizes = shapes[:, 0] * shapes[:, 1]
    starts = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
    return shapes, starts


def _boxes_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.clamp(0.0, 1.0)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            (x2 - x1).clamp_min(1e-4),
            (y2 - y1).clamp_min(1e-4),
        ],
        dim=-1,
    )


class SpatialConvEncoderLayer(nn.Module):
    """Linear-cost spatial token mixer used by the released encoder."""

    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        d = int(cfg.hidden_dim)
        self.depthwise = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
        self.pointwise = nn.Conv2d(d, d, 1, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, cfg.transformer_ffn_dim)
        self.linear2 = nn.Linear(cfg.transformer_ffn_dim, d)
        self.dropout = nn.Dropout(cfg.transformer_dropout)
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        sizes = [int(h) * int(w) for h, w in spatial_shapes.tolist()]
        mixed = []
        for tokens, (height, width) in zip(
            (src + pos).split(sizes, dim=1), spatial_shapes.tolist()
        ):
            feature = tokens.transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[-1], int(height), int(width)
            )
            feature = self.pointwise(F.gelu(self.depthwise(feature)))
            mixed.append(feature.flatten(2).transpose(1, 2))
        src = self.norm1(src + self.dropout(torch.cat(mixed, dim=1)))
        ffn = self.linear2(F.gelu(self.linear1(src)))
        return self.norm2(src + self.dropout(ffn))


class DINOEncoder(nn.Module):
    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SpatialConvEncoderLayer(cfg) for _ in range(cfg.msda_encoder_layers)]
        )

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, pos, spatial_shapes)
        return src


class DINODecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.self_attn = nn.MultiheadAttention(
            d,
            cfg.transformer_heads,
            dropout=cfg.transformer_dropout,
            batch_first=True,
        )
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=d,
            num_levels=cfg.msda_num_levels,
            num_heads=cfg.transformer_heads,
            num_points=cfg.msda_num_points,
        )
        self.dropout1 = nn.Dropout(cfg.transformer_dropout)
        self.dropout2 = nn.Dropout(cfg.transformer_dropout)
        self.dropout3 = nn.Dropout(cfg.transformer_dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, cfg.transformer_ffn_dim)
        self.linear2 = nn.Linear(cfg.transformer_ffn_dim, d)

    def forward(
        self,
        target: torch.Tensor,
        query_pos: torch.Tensor,
        references: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        qk = target + query_pos
        self_out = self.self_attn(qk, qk, target, need_weights=False)[0]
        target = self.norm1(target + self.dropout1(self_out))
        cross_out = self.cross_attn(
            target + query_pos,
            references,
            memory,
            spatial_shapes,
            level_start_index,
        )
        target = self.norm2(target + self.dropout2(cross_out))
        ffn = self.linear2(F.gelu(self.linear1(target)))
        return self.norm3(target + self.dropout3(ffn))


class DINOMaskDecoder(nn.Module):
    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        self.num_levels = cfg.msda_num_levels
        self.layers = nn.ModuleList(
            [DINODecoderLayer(cfg) for _ in range(cfg.transformer_layers)]
        )

    def forward(
        self,
        target: torch.Tensor,
        query_pos: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        references = _boxes_xyxy_to_cxcywh(boxes_xyxy)
        references = references[None, :, None].expand(
            target.shape[0], -1, self.num_levels, -1
        )
        for layer in self.layers:
            target = layer(
                target,
                query_pos,
                references,
                memory,
                spatial_shapes,
                level_start_index,
            )
        return target


class SameScaleMSDAEncoder(nn.Module):
    """Project raw ViT levels and apply the released convolutional encoder."""

    def __init__(self, visual_in_dims: Sequence[int], cfg: Qwen3VLSegConfig):
        super().__init__()
        if len(visual_in_dims) != cfg.msda_num_levels:
            raise ValueError(
                f"MSDA expects {cfg.msda_num_levels} levels, got {len(visual_in_dims)}"
            )
        d = cfg.hidden_dim
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(c), d, kernel_size=1, bias=False),
                    _group_norm(d, cfg.group_norm_groups),
                )
                for c in visual_in_dims
            ]
        )
        self.position = PositionEmbeddingSine2D(d)
        self.level_embed = nn.Parameter(torch.empty(len(visual_in_dims), d))
        nn.init.normal_(self.level_embed, std=0.02)
        self.encoder = DINOEncoder(cfg)

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != len(self.input_proj):
            raise ValueError(f"expected {len(self.input_proj)} ViT levels, got {len(features)}")
        real_hw = tuple(features[0].shape[-2:])
        if any(tuple(x.shape[-2:]) != real_hw for x in features):
            raise ValueError(
                "pre-merge ViT levels must retain their identical real token grid; "
                f"got {[tuple(x.shape[-2:]) for x in features]}"
            )
        projected = [proj(feature) for proj, feature in zip(self.input_proj, features)]
        spatial_shapes, starts = _spatial_metadata(projected, projected[0].device)
        src_parts = []
        for feature in projected:
            src_parts.append(feature.flatten(2).transpose(1, 2))
        src = torch.cat(src_parts, dim=1)
        pos_parts = []
        for level, feature in enumerate(projected):
            pos = self.position(feature)
            pos = pos + self.level_embed[level][None, :, None, None]
            pos_parts.append(pos.flatten(2).transpose(1, 2))
        pos = torch.cat(pos_parts, dim=1)
        memory = self.encoder(src, pos, spatial_shapes)
        split_sizes = [int(h) * int(w) for h, w in spatial_shapes.tolist()]
        chunks = memory.split(split_sizes, dim=1)
        maps = []
        for chunk, (height, width) in zip(chunks, spatial_shapes.tolist()):
            maps.append(
                chunk.transpose(1, 2).reshape(
                    chunk.shape[0], chunk.shape[-1], int(height), int(width)
                )
            )
        return maps, memory, spatial_shapes, starts


class SameScaleFPN(nn.Module):
    """Fuse semantic ViT levels while preserving their common token grid."""

    def __init__(self, hidden_dim: int, num_levels: int, gn_groups: int):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False) for _ in range(num_levels)]
        )
        self.output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                    _group_norm(hidden_dim, gn_groups),
                    nn.GELU(),
                )
                for _ in range(num_levels - 1)
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        fused = self.lateral[-1](features[-1])
        output_idx = 0
        for level in range(len(features) - 2, -1, -1):
            fused = fused + self.lateral[level](features[level])
            fused = self.output[output_idx](fused)
            output_idx += 1
        return fused


class ShallowImageStem(nn.Module):
    def __init__(
        self, hidden_dim: int, mid_dim: int, gn_groups: int, output_stride: int = 4
    ):
        super().__init__()
        output_stride = int(output_stride)
        if output_stride not in (2, 4):
            raise ValueError("decoder RGB stem output_stride must be 2 or 4")
        self.net = nn.Sequential(
            nn.Conv2d(3, mid_dim, 3, stride=2, padding=1, bias=False),
            _group_norm(mid_dim, gn_groups),
            nn.GELU(),
            nn.Conv2d(
                mid_dim, hidden_dim, 3,
                stride=2 if output_stride == 4 else 1,
                padding=1, bias=False,
            ),
            _group_norm(hidden_dim, gn_groups),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class PixelShuffleStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = int(in_dim if out_dim is None else out_dim)
        self.conv = nn.Conv2d(int(in_dim), out_dim * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.shuffle(self.conv(feature)))


class ResidualPixelShuffleStage(nn.Module):
    """Bilinear 2x base plus a zero-initialized learned subpixel residual."""

    def __init__(self, channels: int):
        super().__init__()
        channels = int(channels)
        self.conv = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            feature, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return base + self.shuffle(self.conv(feature))


class AdditivePixelFusion(nn.Module):
    """Final S/2 pixel-fusion path used by WeVisOne-VL v1."""

    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        hidden_dim = int(cfg.hidden_dim)
        mask_dim = _mask_feature_dim(cfg)
        groups = int(cfg.group_norm_groups)
        self.fpn = SameScaleFPN(hidden_dim, cfg.msda_num_levels, groups)
        self.upsample = nn.Sequential(
            PixelShuffleStage(hidden_dim, hidden_dim),
            PixelShuffleStage(hidden_dim, mask_dim),
        )
        self.s2_upsample = ResidualPixelShuffleStage(mask_dim)
        self.cnn = ShallowImageStem(
            mask_dim,
            cfg.stem_mid_dim,
            groups,
            output_stride=2,
        )
        self.fpn_proj = nn.Conv2d(mask_dim, mask_dim, 1, bias=False)
        self.cnn_proj = nn.Conv2d(mask_dim, mask_dim, 1, bias=False)
        self.smooth = nn.Sequential(
            _group_norm(mask_dim, groups),
            nn.GELU(),
            nn.Conv2d(mask_dim, mask_dim, 3, padding=1, bias=False),
        )

    def forward(
        self, image: torch.Tensor, encoded_levels: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        fpn = self.upsample(self.fpn(encoded_levels))
        cnn = self.cnn(image)
        target_hw = tuple(max(1, int(value) // 2) for value in cnn.shape[-2:])
        if tuple(fpn.shape[-2:]) != target_hw:
            src_h, src_w = (int(value) for value in fpn.shape[-2:])
            dst_h, dst_w = target_hw
            resize = fpn.float()
            pool_h, pool_w = min(src_h, dst_h), min(src_w, dst_w)
            if (pool_h, pool_w) != (src_h, src_w):
                resize = F.adaptive_avg_pool2d(resize, (pool_h, pool_w))
            if (pool_h, pool_w) != (dst_h, dst_w):
                resize = F.interpolate(
                    resize,
                    size=(dst_h, dst_w),
                    mode="bilinear",
                    align_corners=False,
                )
            fpn = resize.to(dtype=fpn.dtype)
        fused = self.smooth(self.fpn_proj(fpn))
        fused = self.s2_upsample(fused)
        return fused + self.cnn_proj(cnn)


class DynamicMaskPredictor(nn.Module):
    def __init__(self, hidden_dim: int, mask_dim: Optional[int] = None):
        super().__init__()
        mask_dim = int(hidden_dim if mask_dim is None else mask_dim)
        self.kernel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mask_dim + 1),
        )

    def forward(self, queries: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        params = self.kernel(queries)
        kernels, bias = params[:, :-1], params[:, -1]
        return torch.einsum("qc,chw->qhw", kernels, feature[0]) + bias[:, None, None]


class AlphaResidualBlock(nn.Module):
    def __init__(self, channels: int, gn_groups: int):
        super().__init__()
        self.net = nn.Sequential(
            _group_norm(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _group_norm(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.net(feature)


class FullCanvasAlphaCascadeHead(nn.Module):
    """Three-stage alpha cascade over the S/2 semantic prediction.

    RGB, semantic features, the coarse prediction and the preceding alpha
    estimate are fused successively at S/4, S/2 and S/1.  All outputs cover the
    complete image.
    """

    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        mask_dim = _mask_feature_dim(cfg)
        hidden = int(cfg.hidden_dim)
        channels = int(cfg.alpha_cascade_dim)
        if channels <= 0:
            raise ValueError("alpha_cascade_dim must be positive")
        groups = min(int(cfg.group_norm_groups), channels)
        blocks = int(cfg.alpha_cascade_blocks)
        if blocks < 0:
            raise ValueError("alpha_cascade_blocks must be non-negative")
        def project(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                _group_norm(channels, groups),
                nn.GELU(),
            )

        def residual_stack() -> nn.Sequential:
            return nn.Sequential(
                *[AlphaResidualBlock(channels, groups) for _ in range(blocks)]
            )

        self.semantic = project(mask_dim)
        self.rgb_s4 = project(3)
        self.rgb_s2 = project(3)
        self.coarse_s4 = project(1)
        self.coarse_s2 = project(1)
        self.previous_s2 = project(1)
        self.previous_s1 = project(1)

        self.fuse_s4 = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.blocks_s4 = residual_stack()
        self.up_s4_to_s2 = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.fuse_s2 = nn.Conv2d(5 * channels, channels, 1, bias=False)
        self.blocks_s2 = residual_stack()
        self.up_s2_to_s1 = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.rgb_s1 = project(3)
        self.fuse_s1 = nn.Conv2d(4 * channels, channels, 1, bias=False)
        self.blocks_s1 = residual_stack()

        self.query_s4 = nn.Linear(hidden, 2 * channels)
        self.query_s2 = nn.Linear(hidden, 2 * channels)
        self.query_s1 = nn.Linear(hidden, 2 * channels)
        self.output_s4 = nn.Conv2d(channels, 1, 1)
        self.output_s2 = nn.Conv2d(channels, 1, 1)
        self.output_s1 = nn.Conv2d(channels, 1, 1)
        for output in (self.output_s4, self.output_s2, self.output_s1):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _resize(value: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if tuple(value.shape[-2:]) == tuple(size):
            return value
        return F.interpolate(
            value.float(), size=size, mode="bilinear", align_corners=False
        ).to(dtype=value.dtype)

    @staticmethod
    def _condition(
        feature: torch.Tensor, queries: torch.Tensor, affine: nn.Linear
    ) -> torch.Tensor:
        scale, bias = affine(queries).chunk(2, dim=-1)
        return feature * (
            1.0 + 0.1 * scale.tanh()[:, :, None, None]
        ) + 0.1 * bias[:, :, None, None]

    def forward(
        self,
        image: torch.Tensor,
        pixel_feature: torch.Tensor,
        queries: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del boxes_xyxy  # Identity comes from the query and inherited coarse mask.
        num_queries = int(queries.shape[0])
        s2_hw = tuple(int(x) for x in coarse_logits.shape[-2:])
        s1_hw = tuple(int(x) for x in image.shape[-2:])
        if s1_hw != (2 * s2_hw[0], 2 * s2_hw[1]):
            raise ValueError(
                "alpha cascade requires an inherited S/2 canvas: "
                f"image={s1_hw}, coarse={s2_hw}"
            )
        s4_hw = (max(1, s2_hw[0] // 2), max(1, s2_hw[1] // 2))

        semantic_s2 = self.semantic(pixel_feature)
        semantic_s4 = self._resize(semantic_s2, s4_hw)
        rgb_s4 = self.rgb_s4(self._resize(image, s4_hw))
        coarse_s4_logits = self._resize(coarse_logits[:, None], s4_hw)[:, 0]
        feature_s4 = self.fuse_s4(
            torch.cat(
                [
                    semantic_s4.expand(num_queries, -1, -1, -1),
                    rgb_s4.expand(num_queries, -1, -1, -1),
                    self.coarse_s4(coarse_s4_logits[:, None].to(semantic_s4.dtype)),
                ],
                dim=1,
            )
        )
        feature_s4 = self._condition(feature_s4, queries, self.query_s4)
        feature_s4 = self.blocks_s4(feature_s4)
        delta_s4 = self.output_s4(feature_s4)[:, 0].float()
        logits_s4 = coarse_s4_logits.float() + delta_s4

        rgb_s2 = self.rgb_s2(self._resize(image, s2_hw))
        previous_s2 = self._resize(logits_s4[:, None], s2_hw)
        feature_s2 = self.fuse_s2(
            torch.cat(
                [
                    self.up_s4_to_s2(feature_s4),
                    semantic_s2.expand(num_queries, -1, -1, -1),
                    rgb_s2.expand(num_queries, -1, -1, -1),
                    self.coarse_s2(coarse_logits[:, None].to(semantic_s2.dtype)),
                    self.previous_s2(previous_s2.sigmoid().to(semantic_s2.dtype)),
                ],
                dim=1,
            )
        )
        feature_s2 = self._condition(feature_s2, queries, self.query_s2)
        feature_s2 = self.blocks_s2(feature_s2)
        delta_s2 = self.output_s2(feature_s2)[:, 0].float()
        logits_s2 = coarse_logits.float() + delta_s2

        semantic_s1 = self._resize(semantic_s2, s1_hw)
        rgb_s1 = self.rgb_s1(image)
        upsampled_s2 = self.up_s2_to_s1(feature_s2)
        previous_s1 = self._resize(logits_s2[:, None], s1_hw)
        feature_s1 = self.fuse_s1(
            torch.cat(
                [
                    upsampled_s2,
                    semantic_s1.expand(num_queries, -1, -1, -1),
                    rgb_s1.expand(num_queries, -1, -1, -1),
                    self.previous_s1(previous_s1.sigmoid().to(semantic_s1.dtype)),
                ],
                dim=1,
            )
        )
        feature_s1 = self._condition(feature_s1, queries, self.query_s1)
        feature_s1 = self.blocks_s1(feature_s1)
        delta_s1 = self.output_s1(feature_s1)[:, 0].float()
        logits_s1 = previous_s1[:, 0].float() + delta_s1
        return logits_s1, logits_s2


class WeVisOneMaskDecoder(nn.Module):
    """Final WeVisOne-VL v1 decoder with the S/1 alpha cascade."""

    def __init__(self, visual_in_dims: List[int], llm_dim: int, cfg: Qwen3VLSegConfig):
        super().__init__()
        self.visual_in_dims = [int(value) for value in visual_in_dims]
        self.llm_dim = int(llm_dim)
        self.visual_encoder = SameScaleMSDAEncoder(self.visual_in_dims, cfg)
        self.query_builder = MaskQueryBuilder(
            llm_dim,
            cfg.hidden_dim,
            cfg.num_fourier_freqs,
        )
        self.transformer = DINOMaskDecoder(cfg)
        self.pixel_fusion = AdditivePixelFusion(cfg)
        self.mask_predictor = DynamicMaskPredictor(
            cfg.hidden_dim, _mask_feature_dim(cfg)
        )
        # Keep the checkpoint module prefix ``highres_refine`` while exposing
        # only the one refinement implementation used by this release.
        self.highres_refine = FullCanvasAlphaCascadeHead(cfg)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        image: torch.Tensor,
        visual_features_2d: List[torch.Tensor],
        t_seg: torch.Tensor,
        boxes01: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        # The decoder weights expect FP32 execution regardless of the
        # backbone dtype.
        self.float()
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self.decode(
                image.float(),
                [feature.float() for feature in visual_features_2d],
                t_seg.float(),
                boxes01.float(),
                grid_hw,
            )

    def decode(
        self,
        image: torch.Tensor,
        visual_features_2d: List[torch.Tensor],
        t_seg: torch.Tensor,
        boxes01: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Run the decoder in the dtype of its inputs and parameters."""
        if t_seg.shape[0] != boxes01.shape[0]:
            raise ValueError(
                f"#OBJ={t_seg.shape[0]} does not match #boxes={boxes01.shape[0]}"
            )
        if tuple(visual_features_2d[0].shape[-2:]) != tuple(grid_hw):
            raise ValueError(
                f"raw ViT map {tuple(visual_features_2d[0].shape[-2:])} "
                f"does not match pre-merge grid {tuple(grid_hw)}"
            )

        encoded, memory, shapes, starts = self.visual_encoder(visual_features_2d)
        content, query_position = self.query_builder(t_seg, boxes01)
        queries = self.transformer(
            content.unsqueeze(0),
            query_position.unsqueeze(0),
            boxes01,
            memory,
            shapes,
            starts,
        )
        queries = queries.squeeze(0)
        pixel_feature = self.pixel_fusion(image, encoded)
        coarse_logits = self.mask_predictor(queries, pixel_feature)
        logits_s1, logits_s2 = self.highres_refine(
            image,
            pixel_feature,
            queries,
            boxes01,
            coarse_logits,
        )
        return {
            "mask_logits": logits_s1,
            "mask_logits_aux": logits_s2,
            "mask_logits_coarse": coarse_logits,
            "mask_logits_lowres": coarse_logits,
        }
