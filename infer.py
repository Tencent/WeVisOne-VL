#!/usr/bin/env python3
"""Single-image matting from a coarse mask, a bounding box or a foreground point."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mask2alpha import SAM2AlphaPredictor, alpha_to_uint8, build_mask2alpha

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="input image file")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--mask", type=Path, help="coarse mask: foreground pixels > --mask-threshold")
    prompt.add_argument("--bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    prompt.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), help="one point on the foreground")
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--head-checkpoint", type=Path, default=ROOT / "weights/mask2alpha_sam2_bplus.pt")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def get_prompt(args, height, width):
    if args.mask is not None:
        if not 0 <= args.mask_threshold <= 255:
            raise ValueError("--mask-threshold must be in [0, 255]")
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"cannot read mask: {args.mask}")
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > args.mask_threshold).astype(np.uint8)
        if not mask.any():
            raise ValueError("the mask has no foreground pixels")
        return "mask", {"masks": [mask]}
    if args.bbox is not None:
        x1, y1, x2, y2 = args.bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bbox must be inside the image, with x1 < x2 and y1 < y2")
        return "bbox", {"boxes": [np.asarray(args.bbox, dtype=np.float32)]}
    x, y = args.point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("point must be inside the image")
    return "point", {"points": [np.asarray([[x, y]], dtype=np.float32)]}


def main():
    args = parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mode, prompt = get_prompt(args, *rgb.shape[:2])
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this GPU does not support bf16; use --amp-dtype fp32")
    model = build_mask2alpha(args.sam2_checkpoint, args.head_checkpoint, device=device)
    predictor = SAM2AlphaPredictor(model, device=device, amp_dtype=args.amp_dtype)
    alphas, _ = predictor.predict([rgb], mode, **prompt)
    alpha = alpha_to_uint8(alphas[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = args.output_dir / f"{args.image.stem}_alpha.png"
    rgba_path = args.output_dir / f"{args.image.stem}_rgba.png"
    if not cv2.imwrite(str(alpha_path), alpha):
        raise OSError(f"cannot write alpha: {alpha_path}")
    if not cv2.imwrite(str(rgba_path), np.dstack([bgr, alpha])):
        raise OSError(f"cannot write RGBA: {rgba_path}")
    print(f"alpha: {alpha_path.resolve()}")
    print(f"RGBA:  {rgba_path.resolve()}")


if __name__ == "__main__":
    main()
