# Mask2Alpha

Mask2Alpha turns a coarse object prompt into a high-quality alpha matte. Given an image and **one** prompt (a coarse mask, a bounding box or a foreground click), it predicts a soft alpha matte at the original image resolution, recovering fine structures such as hair, fur and semi-transparent edges.

![Mask2Alpha predictions: input image, alpha matte and green-screen composite](assets/teaser.png)

Each example shows the input image, the predicted alpha matte, and the foreground composited on green.

Mask2Alpha is built on a frozen [SAM2.1-B+](https://github.com/facebookresearch/sam2), which provides the image features and, for box and point prompts, the initial object mask. A lightweight refinement head fuses these features with the image and the prompt mask to predict the alpha matte.

## Quick Start

```bash
python infer.py --image image.jpg --mask mask.png             # coarse mask
python infer.py --image image.jpg --bbox 120 40 860 1010      # bounding box
python infer.py --image image.jpg --point 512 400             # foreground click
```

## Installation

```bash
conda create -n mask2alpha python=3.10 -y
conda activate mask2alpha
pip install -r requirements.txt
git submodule update --init sam2
```

The last command fetches the official [SAM2](https://github.com/facebookresearch/sam2) code into `sam2/`; Mask2Alpha imports it from there.

### Download Weights

The repository contains only the Mask2Alpha head weights (`weights/mask2alpha_sam2_bplus.pt`). Download the official SAM2.1-B+ checkpoint into `checkpoints/`:

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

Both weight files are loaded together, and the SAM2 checkpoint is verified by its SHA256. If you already have the checkpoint elsewhere, pass `--sam2-checkpoint /path/to/sam2.1_hiera_base_plus.pt` instead of downloading it again.

### Test Installation

```bash
python infer.py --image sam2/notebooks/images/truck.jpg --bbox 75 275 1725 850
```

If the installation is successful, the alpha matte of the truck will be saved to `outputs/truck_alpha.png` and the transparent cutout to `outputs/truck_rgba.png`.

## Inference

`infer.py` processes one image per run. Give exactly one prompt; coordinates are pixel positions in the original image, with the origin at the top-left corner.

| Prompt | Argument | Notes |
| --- | --- | --- |
| Coarse mask | `--mask mask.png` | Pixels above `--mask-threshold` (default `0`) are foreground; masks of a different size are resized with nearest neighbour. |
| Bounding box | `--bbox X1 Y1 X2 Y2` | SAM2 first segments the object inside the box. |
| Foreground click | `--point X Y` | SAM2 first segments the clicked object, keeping the best of its three candidate masks. |

For an input `image.jpg`, two files are written to `outputs/` (set another folder with `--output-dir`):

- `image_alpha.png`: 8-bit alpha matte, where 0 is background and 255 is fully opaque foreground;
- `image_rgba.png`: the input image with the alpha matte as its alpha channel.

Run `python infer.py --help` to see all options.

## Python API

```python
import cv2
import numpy as np
from mask2alpha import SAM2AlphaPredictor, alpha_to_uint8, build_mask2alpha

model = build_mask2alpha(device="cuda")  # weights from checkpoints/ and weights/ by default
predictor = SAM2AlphaPredictor(model, device="cuda", amp_dtype="bf16")

bgr = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
mask = (cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)  # same size as the image

alphas, prompt_masks = predictor.predict([rgb], mode="mask", masks=[mask])
# alphas, _ = predictor.predict([rgb], mode="bbox", boxes=[np.array([120, 40, 860, 1010])])
# alphas, _ = predictor.predict([rgb], mode="point", points=[np.array([[512, 400]])])

alpha = alpha_to_uint8(alphas[0])  # alphas[0]: float32 in [0, 1], same size as the image
cv2.imwrite("alpha.png", alpha)
cv2.imwrite("rgba.png", np.dstack([bgr, alpha]))
```

- `predict` takes lists, so several images can be processed in one call. `prompt_masks` holds the binary mask given to the head; for box and point prompts this is the mask predicted by SAM2.
- Point prompts can have several clicks per image, e.g. `points=[np.array([[512, 400], [300, 900]])]`, with `point_labels=[np.array([1, 0])]` marking each click as positive (`1`) or negative (`0`). All clicks are positive by default.
- Custom weight paths can be passed as `build_mask2alpha(sam2_checkpoint=..., head_checkpoint=...)`. Load the model once and reuse the predictor for many images.
