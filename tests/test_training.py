import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from training.rl import copy_mask_decoder
from training.rl.reward import compute_score, parse_boxes, parse_points
from training.sft.data import (
    IGNORE_INDEX,
    KindGroupedSampler,
    SFTCollator,
    SFTDataset,
    boxes_before_obj_tokens,
    load_data_config,
)
from training.sft.losses import mask_loss

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "WeVisOne-VL-4B"


def _score(task, prediction, reference):
    return compute_score("test", prediction, reference, {"task": task})


def test_box_reward():
    ref = "<ref>cat</ref><box><0><0><100><100>,<200><200><300><300></box>"
    assert _score("detection", ref, ref) == pytest.approx(1.0)
    half = "<ref>cat</ref><box><0><0><100><100></box>"
    assert _score("detection", half, ref) == pytest.approx(2 / 3)
    assert _score("detection", "<ref>dog</ref><box><0><0><100><100></box>", ref) == 0.0
    negative = "<ref>cat</ref><box>none</box>"
    assert _score("referring", "<ref>cat</ref><box>none</box>", negative) == 1.0
    assert _score("referring", half, negative) == 0.0


def test_segmentation_reward_requires_obj():
    ref = "<ref>cat</ref><box><0><0><100><100><OBJ></box>"
    assert _score("segmentation", ref, ref) == pytest.approx(1.0)
    assert _score("segmentation", "<ref>cat</ref><box><0><0><100><100></box>", ref) == 0.0
    assert parse_boxes(ref, require_obj=True) == {"cat": [[0.0, 0.0, 100.0, 100.0]]}


def test_ocr_reward_normalizes_text():
    ref = "<ref>STOP</ref><box><10><10><50><30></box>"
    pred = "<ref>stop!</ref><box><10><10><50><30></box>"
    assert _score("ocr", pred, ref) == pytest.approx(1.0)


def test_point_and_count_rewards():
    ref = "<ref>flag</ref><point><100><100>,<500><500></point>\nTotal number: 2."
    assert parse_points(ref) == {"flag": [[100.0, 100.0], [500.0, 500.0]]}
    assert _score("point", ref, ref) == pytest.approx(1.0)
    assert _score("count_point", ref, ref) == pytest.approx(1.5)
    one = "<ref>flag</ref><point><100><100></point>"
    assert _score("count_point", one, ref) < 0.5


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        _score("caption", "", "")


def test_copy_mask_decoder(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "mask_decoder.pt").write_bytes(b"decoder")
    exports = [tmp_path / "out" / f"global_step_{i}" / "actor" / "huggingface" for i in (5, 10)]
    for export in exports:
        export.mkdir(parents=True)
    monkeypatch.setattr("sys.argv", ["copy", str(model), str(tmp_path / "out")])
    copy_mask_decoder.main()
    assert all((export / "mask_decoder.pt").read_bytes() == b"decoder" for export in exports)


def test_boxes_before_obj_tokens():
    text = "<ref>a</ref><box><0><0><500><1000><OBJ>,<1><2><3><4><OBJ></box>"
    assert boxes_before_obj_tokens(text) == [[0, 0, 0.5, 1.0], [0.001, 0.002, 0.003, 0.004]]


def test_load_data_config(tmp_path):
    (tmp_path / "a.jsonl").write_text("{}\n")
    config = tmp_path / "mix.yaml"
    config.write_text("datasets:\n  - path: a.jsonl\n    ratio: 2\n")
    (spec,) = load_data_config(str(config))
    assert spec.path == tmp_path / "a.jsonl"
    assert spec.ratio == 2.0
    (single,) = load_data_config(str(tmp_path / "a.jsonl"))
    assert single.ratio == 1.0


def test_kind_grouped_sampler():
    kinds = np.array([0] * 5 + [1] * 3 + [2] * 8)
    sampler = KindGroupedSampler(kinds, group_size=4, seed=0)
    order = list(sampler)
    assert len(order) == len(sampler) == 8 + 4 + 8
    assert set(order) == set(range(len(kinds)))
    for start in range(0, len(order), 4):
        assert len(set(kinds[order[start : start + 4]])) == 1
    sampler.set_epoch(1)
    assert list(sampler) != order


def test_mask_loss_prefers_correct_logits():
    target = torch.zeros(1, 64, 64)
    target[:, 16:48, 16:48] = 1.0
    full = (target * 2 - 1) * 20
    half = torch.nn.functional.avg_pool2d(full[:, None], 2)[:, 0]
    good = {"mask_logits": full, "mask_logits_aux": half, "mask_logits_coarse": half}
    bad = {key: -value for key, value in good.items()}
    kwargs = {"focal_weight": 200.0, "dice_weight": 10.0}
    assert mask_loss(good, target, **kwargs) < 0.1 < mask_loss(bad, target, **kwargs)


@pytest.fixture(scope="module")
def processor():
    if not (CHECKPOINT / "tokenizer.json").exists():
        pytest.skip("released checkpoint not available")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        CHECKPOINT, min_pixels=64 * 32 * 32, max_pixels=256 * 32 * 32
    )


def _write_samples(root: Path) -> Path:
    Image.new("RGB", (320, 240), (120, 30, 200)).save(root / "img.jpg")
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[24:216, 32:160] = 255
    Image.fromarray(mask).save(root / "mask.png")
    rows = [
        {
            "messages": [
                {"role": "user", "content": "<image>\nLocate the instances: cat."},
                {
                    "role": "assistant",
                    "content": "<ref>cat</ref><box><100><100><500><900><OBJ></box>",
                },
            ],
            "images": [str(root / "img.jpg")],
            "masks": [str(root / "mask.png")],
        },
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi."},
            ]
        },
    ]
    path = root / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_dataset_and_collator(tmp_path, processor):
    tokenizer = processor.tokenizer
    dataset = SFTDataset(
        load_data_config(str(_write_samples(tmp_path))),
        tokenizer,
        processor.image_processor,
        system_message="You are a helpful assistant.",
        max_length=4096,
        load_masks=True,
        decoder_image_size=64,
        max_masks_per_sample=4,
        seed=0,
    )
    image_sample, text_sample = dataset[0], dataset[1]

    merge = processor.image_processor.merge_size
    num_pads = int(
        (image_sample["input_ids"] == tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()
    )
    assert num_pads == int(image_sample["image_grid_thw"].prod()) // merge**2
    supervised = image_sample["labels"][image_sample["labels"] != IGNORE_INDEX]
    assert (
        tokenizer.decode(supervised)
        == "<ref>cat</ref><box><100><100><500><900><OBJ></box><|im_end|>\n"
    )
    assert torch.allclose(image_sample["obj_boxes"], torch.tensor([[0.1, 0.1, 0.5, 0.9]]))
    assert image_sample["obj_masks"].shape == (1, 64, 64)
    assert image_sample["decoder_image"].shape == (3, 64, 64)
    assert tokenizer.decode(text_sample["input_ids"]).startswith("<|im_start|>system\nBe brief.")
    assert "obj_masks" not in text_sample

    batch = SFTCollator(tokenizer.pad_token_id)([image_sample, text_sample])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][1].sum() == len(text_sample["input_ids"])
    assert batch["obj_masks"][1] is None
    assert batch["pixel_values"].shape[0] == int(image_sample["image_grid_thw"].prod())
