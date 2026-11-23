"""JSONL datasets for WeVisOne-VL supervised fine-tuning.

Every line of a dataset file is one conversation; see ``training/README.md``
for the full schema. Several files are mixed with per-file sampling ratios.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from wevisone_vl.config import SEG_SPECIAL_TOKENS

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
IMAGE_TAG = "<image>"
OBJ_TOKEN = SEG_SPECIAL_TOKENS[0]
_OBJ_OR_COORD_RE = re.compile(re.escape(OBJ_TOKEN) + r"|<(\d+)>")
_MAX_LOAD_ATTEMPTS = 10
KIND_TEXT, KIND_IMAGE, KIND_MASKS = 0, 1, 2


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    ratio: float = 1.0


def load_data_config(path: str) -> list[DatasetSpec]:
    """Read a mixture YAML, or wrap a single JSONL file as a one-entry mixture.

    Relative JSONL paths inside the YAML are resolved against the YAML's directory.
    """
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix == ".jsonl":
        return [DatasetSpec(config_path)]

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    entries = config.get("datasets")
    if not entries:
        raise ValueError(f"{config_path} must define a non-empty 'datasets' list")

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()

    specs = []
    for entry in entries:
        unknown = set(entry) - {"path", "ratio"}
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)} in dataset entry {entry}")
        jsonl = resolve(entry["path"])
        ratio = float(entry.get("ratio", 1.0))
        if ratio < 0:
            raise ValueError(f"ratio must be non-negative, got {ratio} for {jsonl}")
        specs.append(DatasetSpec(jsonl, ratio))
    return specs


def _index_lines(path: Path, load_masks: bool) -> tuple[np.ndarray, np.ndarray]:
    """Byte offset and kind (text, image, or image with masks) of every line."""
    offsets, kinds = [], []
    position = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                offsets.append(position)
                if not record.get("images"):
                    kinds.append(KIND_TEXT)
                elif load_masks and record.get("masks"):
                    kinds.append(KIND_MASKS)
                else:
                    kinds.append(KIND_IMAGE)
            position += len(line)
    return np.asarray(offsets, dtype=np.int64), np.asarray(kinds, dtype=np.int64)


def _resample(num_rows: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """``round(ratio * num_rows)`` row indices: whole repeats plus a random remainder."""
    if num_rows == 0:
        return np.zeros(0, dtype=np.int64)
    repeats, remainder = divmod(round(ratio * num_rows), num_rows)
    return np.concatenate(
        [np.tile(np.arange(num_rows), repeats), rng.permutation(num_rows)[:remainder]]
    )


def boxes_before_obj_tokens(text: str) -> list[list[float]]:
    """Normalized xyxy box written right before each ``<OBJ>`` token."""
    boxes, coords = [], []
    for match in _OBJ_OR_COORD_RE.finditer(text):
        if match.group(1) is not None:
            coords.append(int(match.group(1)))
            continue
        if len(coords) < 4:
            raise ValueError(f"{OBJ_TOKEN} must follow four coordinate tokens")
        boxes.append([value / 1000.0 for value in coords[-4:]])
        coords = []
    return boxes


def _load_mask(ref: Any, size: int) -> np.ndarray:
    """A full-image mask (PNG path or COCO RLE) resized to ``size x size`` uint8."""
    if isinstance(ref, dict):
        from pycocotools import mask as mask_utils

        rle = dict(ref)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        mask = Image.fromarray(mask_utils.decode(rle).astype(np.uint8) * 255)
    else:
        with Image.open(Path(ref).expanduser()) as source:
            mask = source.convert("L")
    return np.asarray(mask.resize((size, size), resample=Image.Resampling.BILINEAR))


class SFTDataset(torch.utils.data.Dataset):
    """Random-access mixture of JSONL conversations.

    Samples that cannot be loaded (missing files, too long, inconsistent masks)
    are logged and replaced by a random other sample of the same kind.
    """

    def __init__(
        self,
        specs: Sequence[DatasetSpec],
        tokenizer,
        image_processor,
        *,
        system_message: str,
        max_length: int,
        load_masks: bool,
        decoder_image_size: int,
        max_masks_per_sample: int,
        seed: int,
    ) -> None:
        self.specs = list(specs)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.system_message = system_message
        self.max_length = int(max_length)
        self.load_masks = bool(load_masks)
        self.decoder_image_size = int(decoder_image_size)
        self.max_masks_per_sample = int(max_masks_per_sample)
        self.obj_token_id = tokenizer.convert_tokens_to_ids(OBJ_TOKEN)
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.assistant_prefix_len = len(self._encode("<|im_start|>assistant\n"))

        rng = np.random.default_rng(seed)
        self.offsets, rows, kinds = [], [], []
        for source_id, spec in enumerate(self.specs):
            offsets, line_kinds = _index_lines(spec.path, self.load_masks)
            selected = _resample(len(offsets), spec.ratio, rng)
            self.offsets.append(offsets)
            rows.append(np.stack([np.full_like(selected, source_id), selected], axis=1))
            kinds.append(line_kinds[selected])
            logger.info(
                "%s: %d rows, ratio %.3g -> %d samples",
                spec.path,
                len(offsets),
                spec.ratio,
                len(selected),
            )
        self.rows = np.concatenate(rows)
        self.kinds = np.concatenate(kinds)
        self._indices_by_kind = {k: np.flatnonzero(self.kinds == k) for k in np.unique(self.kinds)}
        if len(self.rows) == 0:
            raise ValueError("the dataset mixture is empty")
        logger.info("training mixture: %d samples from %d files", len(self.rows), len(self.specs))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        for _ in range(_MAX_LOAD_ATTEMPTS):
            try:
                return self._load(index)
            except Exception as error:  # noqa: BLE001 - one bad row must not stop training
                logger.warning("skipping %s: %s", self._location(index), error)
                index = int(random.choice(self._indices_by_kind[self.kinds[index]]))
        raise RuntimeError(f"failed to load {_MAX_LOAD_ATTEMPTS} consecutive samples")

    def _location(self, index: int) -> str:
        source_id, row = (int(value) for value in self.rows[index])
        return f"{self.specs[source_id].path} (row {row})"

    def _read(self, index: int) -> dict[str, Any]:
        source_id, row = (int(value) for value in self.rows[index])
        with self.specs[source_id].path.open("rb") as handle:
            handle.seek(int(self.offsets[source_id][row]))
            return json.loads(handle.readline())

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _load(self, index: int) -> dict[str, Any]:
        record = self._read(index)
        images = record.get("images") or []
        if len(images) > 1:
            raise ValueError("at most one image per sample is supported")
        image = None
        sample: dict[str, Any] = {}
        num_image_tokens = 0
        if images:
            with Image.open(Path(images[0]).expanduser()) as source:
                image = source.convert("RGB")
            vision = self.image_processor(images=[image], return_tensors="pt")
            sample["pixel_values"] = vision["pixel_values"]
            sample["image_grid_thw"] = vision["image_grid_thw"]
            merge = int(self.image_processor.merge_size)
            num_image_tokens = int(vision["image_grid_thw"].prod()) // (merge * merge)

        input_ids, labels = self._tokenize(record["messages"], num_image_tokens)
        sample["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        sample["labels"] = torch.tensor(labels, dtype=torch.long)
        sample["mm_token_type_ids"] = (sample["input_ids"] == self.image_token_id).long()
        if self.load_masks and record.get("masks"):
            sample.update(self._load_objects(record, image, sample["input_ids"]))
        return sample

    def _tokenize(self, messages: list[dict[str, str]], num_image_tokens: int):
        system = self.system_message
        if messages and messages[0]["role"] == "system":
            system, messages = messages[0]["content"], messages[1:]
        input_ids = self._encode(f"<|im_start|>system\n{system}<|im_end|>\n")
        labels = [IGNORE_INDEX] * len(input_ids)

        image_tags = 0
        for message in messages:
            role, content = message["role"], message["content"]
            if role not in ("user", "assistant"):
                raise ValueError(f"unsupported role {role!r}")
            if role == "user":
                image_tags += content.count(IMAGE_TAG)
                vision = "<|vision_start|>" + "<|image_pad|>" * num_image_tokens + "<|vision_end|>"
                content = content.replace(IMAGE_TAG, vision)
            tokens = self._encode(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            input_ids += tokens
            if role == "assistant":
                prefix = self.assistant_prefix_len
                labels += [IGNORE_INDEX] * prefix + tokens[prefix:]
            else:
                labels += [IGNORE_INDEX] * len(tokens)

        if image_tags != (1 if num_image_tokens else 0):
            raise ValueError(f"expected one {IMAGE_TAG} per image, found {image_tags}")
        if len(input_ids) > self.max_length:
            raise ValueError(f"{len(input_ids)} tokens exceed model_max_length={self.max_length}")
        return input_ids, labels

    def _load_objects(self, record, image, input_ids) -> dict[str, torch.Tensor]:
        if image is None:
            raise ValueError("samples with masks need an image")
        masks = record["masks"]
        answer = "".join(m["content"] for m in record["messages"] if m["role"] == "assistant")
        boxes = boxes_before_obj_tokens(answer)
        num_obj = int((input_ids == self.obj_token_id).sum())
        if not num_obj == len(masks) == len(boxes):
            raise ValueError(
                f"{num_obj} {OBJ_TOKEN} tokens, {len(masks)} masks and {len(boxes)} boxes must match"
            )

        keep = sorted(random.sample(range(num_obj), min(num_obj, self.max_masks_per_sample)))
        size = self.decoder_image_size
        decoder_image = np.asarray(image.resize((size, size), resample=Image.Resampling.BICUBIC))
        return {
            "decoder_image": torch.from_numpy(decoder_image.copy()).permute(2, 0, 1),
            "obj_indices": torch.tensor(keep, dtype=torch.long),
            "obj_boxes": torch.tensor([boxes[i] for i in keep], dtype=torch.float32),
            "obj_masks": torch.from_numpy(np.stack([_load_mask(masks[i], size) for i in keep])),
        }


class KindGroupedSampler(torch.utils.data.Sampler[int]):
    """Shuffle samples so that each run of ``group_size`` indices has a single kind.

    With ``group_size = world_size * per_device_batch_size`` every rank sees the
    same kind in a training step, so all ranks run the vision tower and the
    pixel decoder together. Kinds whose size is not a multiple of
    ``group_size`` are topped up with random repeats.
    """

    def __init__(self, kinds: np.ndarray, group_size: int, seed: int) -> None:
        self.kinds = np.asarray(kinds)
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.epoch = 0
        counts = np.bincount(self.kinds)
        self.num_samples = int(sum(-(-n // self.group_size) * self.group_size for n in counts if n))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = []
        for kind in np.unique(self.kinds):
            indices = rng.permutation(np.flatnonzero(self.kinds == kind))
            missing = -len(indices) % self.group_size
            indices = np.concatenate([indices, rng.choice(indices, missing)])
            groups.extend(indices.reshape(-1, self.group_size))
        for group in rng.permutation(len(groups)):
            yield from groups[group].tolist()


@dataclass
class SFTCollator:
    """Right-pad text and keep per-sample decoder targets as lists."""

    pad_token_id: int

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        def pad(key: str, value: int) -> torch.Tensor:
            return torch.nn.utils.rnn.pad_sequence(
                [s[key] for s in samples], batch_first=True, padding_value=value
            )

        input_ids = pad("input_ids", self.pad_token_id)
        lengths = torch.tensor([len(s["input_ids"]) for s in samples])
        batch: dict[str, Any | None] = {
            "input_ids": input_ids,
            "attention_mask": (torch.arange(input_ids.shape[1])[None] < lengths[:, None]).long(),
            "mm_token_type_ids": pad("mm_token_type_ids", 0),
            "labels": pad("labels", IGNORE_INDEX),
            "pixel_values": None,
            "image_grid_thw": None,
        }
        with_images = [s for s in samples if "pixel_values" in s]
        if with_images:
            batch["pixel_values"] = torch.cat([s["pixel_values"] for s in with_images])
            batch["image_grid_thw"] = torch.cat([s["image_grid_thw"] for s in with_images])
        if any("obj_masks" in s for s in samples):
            batch["decoder_images"] = [s.get("decoder_image") for s in samples]
            batch["obj_indices"] = [s.get("obj_indices") for s in samples]
            batch["obj_boxes"] = [s.get("obj_boxes") for s in samples]
            batch["obj_masks"] = [s.get("obj_masks") for s in samples]
        return batch
