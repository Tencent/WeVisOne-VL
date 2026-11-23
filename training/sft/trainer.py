"""Trainer with per-module learning rates and release-format checkpoints."""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from .data import KindGroupedSampler

MODULE_GROUPS = ("llm", "vision", "merger", "mask_decoder")
LOSS_COMPONENTS = ("loss_text", "loss_mask")


def module_group(parameter_name: str) -> str:
    if parameter_name.startswith("mask_decoder."):
        return "mask_decoder"
    if ".visual.merger." in parameter_name or ".visual.deepstack_merger_list." in parameter_name:
        return "merger"
    if ".visual." in parameter_name:
        return "vision"
    return "llm"


class WeVisOneTrainer(Trainer):
    """Trains :class:`~training.sft.model.WeVisOneForTraining`.

    Checkpoints contain the Hugging Face model, the processor and
    ``mask_decoder.pt``, i.e. they can be loaded directly by ``WeVisOneVL``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Sum and count of every loss component since the last log, on this rank.
        self._loss_stats = torch.zeros(len(LOSS_COMPONENTS), 2, dtype=torch.float64)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset or self.train_dataset
        group_size = self.args.per_device_train_batch_size * self.args.world_size
        return KindGroupedSampler(dataset.kinds, group_size, self.args.seed)

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer
        model = model or self.model
        args = self.args
        overrides = {
            "vision": args.vision_lr,
            "merger": args.merger_lr,
            "mask_decoder": args.mask_decoder_lr,
        }
        decay_names = set(self.get_decay_parameter_names(model))
        grouped = defaultdict(list)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                grouped[(module_group(name), name in decay_names)].append(parameter)
        # The Trainer logs the learning rate of the first group, so "llm" goes first.
        param_groups = [
            {
                "params": grouped[(group, decay)],
                "lr": args.learning_rate if overrides.get(group) is None else overrides[group],
                "weight_decay": args.weight_decay if decay else 0.0,
            }
            for group in MODULE_GROUPS
            for decay in (True, False)
            if grouped[(group, decay)]
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(args, model)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        for i, key in enumerate(LOSS_COMPONENTS):
            if key in outputs:
                self._loss_stats[i] += torch.tensor([float(outputs[key]), 1.0])
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            stats = self.accelerator.reduce(self._loss_stats.to(self.args.device), "sum").cpu()
            for key, (total, count) in zip(LOSS_COMPONENTS, stats.tolist()):
                if count:
                    logs[key] = round(total / count, 4)
            self._loss_stats.zero_()
        super().log(logs, start_time)

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        (model or self.model).load_checkpoint(resume_from_checkpoint)
