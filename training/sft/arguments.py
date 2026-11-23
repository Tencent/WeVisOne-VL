"""Command-line arguments for ``python -m training.sft.train``."""

from __future__ import annotations

from dataclasses import dataclass, field

import transformers

from wevisone_vl.engine import SYSTEM_PROMPT


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="tencent/WeVisOne-VL-4B",
        metadata={"help": "Hugging Face model id or local directory."},
    )
    mask_decoder_path: str | None = field(
        default=None,
        metadata={
            "help": "Pixel decoder weights. Defaults to <model_name_or_path>/mask_decoder.pt."
        },
    )
    tune_vision: bool = field(default=True, metadata={"help": "Train the ViT blocks."})
    tune_merger: bool = field(
        default=True, metadata={"help": "Train the vision-language mergers (projectors)."}
    )
    tune_llm: bool = field(
        default=True, metadata={"help": "Train the language model, embeddings and LM head."}
    )
    train_mask_decoder: bool = field(
        default=False,
        metadata={"help": "Train the pixel decoder on samples that provide masks."},
    )
    attn_implementation: str = field(
        default="sdpa", metadata={"help": "'sdpa', 'flash_attention_2' or 'eager'."}
    )


@dataclass
class DataArguments:
    data_config: str = field(
        metadata={
            "help": "Dataset mixture YAML (see training/configs/sft_data.yaml) or a "
            "single JSONL file."
        }
    )
    min_pixels: int = field(default=16 * 32 * 32)
    max_pixels: int = field(default=2025 * 32 * 32)
    system_message: str = field(default=SYSTEM_PROMPT)
    max_masks_per_sample: int = field(
        default=16,
        metadata={
            "help": "At most this many <OBJ> instances per sample are sent to the pixel "
            "decoder; the language model is still trained on every token."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Samples longer than this (text plus image tokens) are skipped."},
    )
    vision_lr: float | None = field(
        default=None, metadata={"help": "ViT learning rate. Defaults to learning_rate."}
    )
    merger_lr: float | None = field(
        default=None, metadata={"help": "Merger learning rate. Defaults to learning_rate."}
    )
    mask_decoder_lr: float | None = field(
        default=None,
        metadata={"help": "Pixel decoder learning rate. Defaults to learning_rate."},
    )
    mask_loss_weight: float = field(
        default=1.0, metadata={"help": "Weight of the mask loss relative to the LM loss."}
    )
    mask_focal_weight: float = field(default=200.0)
    mask_dice_weight: float = field(default=10.0)
    remove_unused_columns: bool = field(default=False)
