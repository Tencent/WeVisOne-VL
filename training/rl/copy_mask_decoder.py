"""Copy the pixel decoder of the starting model into every verl export.

GRPO only updates the language model, so the exported Hugging Face weights in
``<output_dir>/global_step_*/actor/huggingface/`` need the unchanged
``mask_decoder.pt`` of the starting model to load with ``WeVisOneVL``::

    python -m training.rl.copy_mask_decoder tencent/WeVisOne-VL-4B outputs/rl
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MASK_DECODER_FILE = "mask_decoder.pt"


def mask_decoder_source(model: str) -> Path:
    local = Path(model).expanduser()
    if local.is_dir():
        return local / MASK_DECODER_FILE
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, MASK_DECODER_FILE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Hugging Face model id or directory GRPO started from.")
    parser.add_argument("output_dir", type=Path, help="verl trainer.default_local_dir.")
    args = parser.parse_args()

    exports = sorted(args.output_dir.glob("global_step_*/actor/huggingface"))
    if not exports:
        print(f"no Hugging Face exports under {args.output_dir}")
        return
    source = mask_decoder_source(args.model)
    for export in exports:
        shutil.copy2(source, export / MASK_DECODER_FILE)
        print(f"copied {MASK_DECODER_FILE} to {export}")


if __name__ == "__main__":
    main()
