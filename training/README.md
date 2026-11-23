# Training WeVisOne-VL

This directory contains the code to fine-tune WeVisOne-VL on your own data:

| Entry point | What it does |
| --- | --- |
| [`scripts/sft.sh`](scripts/sft.sh) | Supervised fine-tuning of the VLM and, optionally, the pixel decoder (Hugging Face `Trainer` + DeepSpeed). |
| [`scripts/rl.sh`](scripts/rl.sh) | GRPO with rule-based rewards on top of [verl](https://github.com/volcengine/verl). |
| [`rl/build_dataset.py`](rl/build_dataset.py) | Converts SFT-format JSONL into verl parquet files. |

All commands are run from the repository root.

## Installation

```bash
pip install -e ".[train]"   # SFT
pip install verl==0.9.1     # GRPO only; follow the verl docs for a matching vLLM
```

## Data format

Training data is JSONL, one conversation per line:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nLocate all the instances that match the following description: cat. dog."},
    {"role": "assistant", "content": "<ref>cat</ref><box><112><304><455><901>,<530><288><870><911></box>, <ref>dog</ref><box>none</box>"}
  ],
  "images": ["/data/images/0001.jpg"]
}
```

* `messages`: `user` / `assistant` turns, optionally preceded by one `system`
  message (default: `You are a helpful assistant.`). The loss is computed on
  assistant turns only.
* `images`: at most one image per sample, referenced by `<image>` in a user turn.
  Use absolute paths (relative paths are opened from the working directory).
  Text-only samples omit `images`.
* Prompts should follow the instruction templates in
  [`wevisone_vl/prompts.py`](../wevisone_vl/prompts.py) so that the fine-tuned
  model keeps working with `WeVisOneVL.predict`.

### Answer syntax

Coordinates are integers in `[0, 1000]`: pixel coordinates divided by the image
width or height and multiplied by 1000. Each coordinate is one token `<k>`.

| Output | Syntax |
| --- | --- |
| boxes | `<ref>label</ref><box><x1><y1><x2><y2>,<x1><y1><x2><y2></box>` |
| points | `<ref>label</ref><point><x><y>,<x><y></point>` |
| no match | `<ref>label</ref><box>none</box>` |
| masks | a box followed by `<OBJ>`: `<ref>cat</ref><box><112><304><455><901><OBJ></box>` |

Several labels are joined by `, `. Counting answers may append free text such
as `\nTotal number: 3.` after the geometry.

### Masks (segmentation and matting)

Samples that train the pixel decoder have one `<OBJ>` per instance in the
assistant answer and a `masks` list in the same order:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nLocate the instances that match the following description and provide the alpha mattes: person."},
    {"role": "assistant", "content": "<ref>person</ref><box><208><95><603><998><OBJ></box>"}
  ],
  "images": ["/data/images/0002.jpg"],
  "masks": ["/data/mattes/0002_person.png"]
}
```

Each mask covers the full image and is either a grayscale PNG path (binary masks
or alpha mattes in `[0, 255]`) or a COCO RLE dict `{"size": [h, w], "counts": ...}`.
The decoder is prompted with the box written before each `<OBJ>`. Samples
without `masks` only contribute the language-model loss.

### Dataset mixtures

`--data_config` accepts a single JSONL file or a YAML mixture such as
[`configs/sft_data.yaml`](configs/sft_data.yaml):

```yaml
datasets:
  - path: data/detection.jsonl      # relative to this YAML file
    ratio: 0.5                      # sample half of the file per epoch (default 1.0)
```

Text-only, image and image-with-mask samples can be mixed freely. Each training
step draws a single kind on all GPUs, so every rank runs the same modules.

## Supervised fine-tuning

```bash
DATA_CONFIG=training/configs/sft_data.yaml OUTPUT_DIR=outputs/sft bash training/scripts/sft.sh
```

`sft.sh` reads `MODEL_PATH`, `DATA_CONFIG`, `OUTPUT_DIR`, `DEEPSPEED_CONFIG` and
the usual `torchrun` variables (`NNODES`, `NODE_RANK`, `MASTER_ADDR`,
`MASTER_PORT`, `GPUS_PER_NODE`). Any argument after the script overrides its
defaults; `python -m training.sft.train --help` lists every option, including all
Hugging Face `TrainingArguments`.

Which parts are trained is controlled by four switches and per-module learning rates:

| Argument | Default | Trains |
| --- | --- | --- |
| `--tune_llm` | `True` | language model, embeddings and LM head (`--learning_rate`) |
| `--tune_vision` | `True` | ViT blocks (`--vision_lr`) |
| `--tune_merger` | `True` | vision-language mergers (`--merger_lr`) |
| `--train_mask_decoder` | `False` | pixel decoder on samples with `masks` (`--mask_decoder_lr`) |

Unset learning rates fall back to `--learning_rate`. Some typical setups:

```bash
# language model only
bash training/scripts/sft.sh --tune_vision False --tune_merger False
# language model and pixel decoder, e.g. for new segmentation / matting data
bash training/scripts/sft.sh --train_mask_decoder True --mask_decoder_lr 1e-5
# pixel decoder only
bash training/scripts/sft.sh --tune_llm False --tune_vision False --tune_merger False \
    --train_mask_decoder True --learning_rate 1e-5
```

The mask loss is `focal * 200 + dice * 10` (`--mask_focal_weight`,
`--mask_dice_weight`), averaged over the decoder outputs and added to the
language-model loss with `--mask_loss_weight`.

Every checkpoint (`checkpoint-*/` and the final `OUTPUT_DIR`) contains the
Hugging Face model, the processor and `mask_decoder.pt`, so it loads directly:

```python
from wevisone_vl import WeVisOneVL

model = WeVisOneVL("outputs/sft")
```

> [!NOTE]
> DeepSpeed is used by default. With plain DDP (`DEEPSPEED_CONFIG=`), batches
> without images leave the vision tower without gradients; pass
> `--ddp_find_unused_parameters True --gradient_checkpointing_kwargs '{"use_reentrant": false}'`
> when your data mixes samples with and without images.

## GRPO

1. Convert prompts with reference answers (the same JSONL format as above) into
   parquet. The reward is chosen with `--task`, or per line with a `"task"` field:

   ```bash
   python -m training.rl.build_dataset --input data/detection.jsonl --task detection \
       --output rl_data/detection.parquet
   ```

   | `task` | Reward ([`rl/reward.py`](rl/reward.py)) |
   | --- | --- |
   | `detection`, `referring`, `visual_prompt`, `ocr_grounding` | box F1: per-label Hungarian matching, `2 * sum(IoU) / (num_gt + num_pred)` |
   | `ocr` | box F1 over normalized text labels |
   | `segmentation`, `matting` | box F1 where a predicted box only counts when followed by `<OBJ>` |
   | `point` | soft point F1 with similarity `1 - distance / diagonal` |
   | `count_box`, `count_point` | `0.5 * F1 + (1 if the predicted count is correct else 0)` |

   A reference answer without objects (`<box>none</box>`) is a negative sample:
   the reward is 1 only when the prediction contains no objects either.

2. Launch training:

   ```bash
   TRAIN_FILES="[rl_data/detection.parquet,rl_data/ocr.parquet]" OUTPUT_DIR=outputs/rl \
       bash training/scripts/rl.sh
   ```

   Extra arguments are passed to verl as Hydra overrides (for example
   `trainer.total_epochs=2 actor_rollout_ref.rollout.n=16`). For multiple nodes,
   start a Ray cluster (`ray start --head` on the head node and
   `ray start --address=<head>:6379` on the others) and run the script on the head
   node with `NNODES` set.

   Hugging Face weights are written to `OUTPUT_DIR/global_step_*/actor/huggingface/`
   and load directly with `WeVisOneVL`. GRPO does not update the pixel decoder;
   `rl.sh` copies the starting model's `mask_decoder.pt` into each of these
   directories when training finishes.
