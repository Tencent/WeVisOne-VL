"""Supervised fine-tuning of WeVisOne-VL.

Run ``python -m training.sft.train --help`` for every option, or use
``training/scripts/sft.sh`` for a multi-GPU launch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

from .arguments import DataArguments, ModelArguments, TrainingArguments
from .data import OBJ_TOKEN, SFTCollator, SFTDataset, load_data_config
from .model import MASK_DECODER_FILE, WeVisOneForTraining, configure_trainable, load_mask_decoder
from .trainer import WeVisOneTrainer

logger = logging.getLogger(__name__)


def resolve_mask_decoder_path(model_args: ModelArguments) -> str:
    if model_args.mask_decoder_path:
        return model_args.mask_decoder_path
    local = Path(model_args.model_name_or_path).expanduser()
    if local.is_dir():
        return str(local / MASK_DECODER_FILE)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_args.model_name_or_path, MASK_DECODER_FILE)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=training_args.get_process_log_level(),
    )
    transformers.set_seed(training_args.seed)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if OBJ_TOKEN not in tokenizer.get_vocab():
        raise ValueError(f"{model_args.model_name_or_path} is not a WeVisOne-VL checkpoint")
    obj_token_id = tokenizer.convert_tokens_to_ids(OBJ_TOKEN)

    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    qwen = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        attn_implementation=model_args.attn_implementation,
    )
    configure_trainable(
        qwen,
        tune_vision=model_args.tune_vision,
        tune_merger=model_args.tune_merger,
        tune_llm=model_args.tune_llm,
    )

    mask_decoder, decoder_cfg = load_mask_decoder(
        qwen, resolve_mask_decoder_path(model_args), obj_token_id
    )
    model = WeVisOneForTraining(
        qwen,
        mask_decoder,
        decoder_cfg,
        obj_token_id,
        train_mask_decoder=model_args.train_mask_decoder,
        mask_loss_weight=training_args.mask_loss_weight,
        mask_focal_weight=training_args.mask_focal_weight,
        mask_dice_weight=training_args.mask_dice_weight,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable parameters: %s", f"{trainable:,}")

    dataset = SFTDataset(
        load_data_config(data_args.data_config),
        tokenizer,
        processor.image_processor,
        system_message=data_args.system_message,
        max_length=training_args.model_max_length,
        load_masks=model_args.train_mask_decoder,
        decoder_image_size=decoder_cfg.cnn_image_size,
        max_masks_per_sample=data_args.max_masks_per_sample,
        seed=training_args.seed,
    )
    trainer = WeVisOneTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
