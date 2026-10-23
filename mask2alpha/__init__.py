"""Mask2Alpha inference. The bundled ``sam2/`` directory is added to ``sys.path`` on import."""

import sys
from pathlib import Path

_SAM2_ROOT = Path(__file__).resolve().parents[1] / "sam2"
if (_SAM2_ROOT / "sam2" / "__init__.py").is_file() and str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))

from .model import SAM2AlphaRefiner, build_mask2alpha
from .predictor import MODES, SAM2AlphaPredictor, alpha_to_uint8, binarize_mask, tight_box_from_mask

__all__ = [
    "MODES",
    "SAM2AlphaRefiner",
    "SAM2AlphaPredictor",
    "alpha_to_uint8",
    "binarize_mask",
    "build_mask2alpha",
    "tight_box_from_mask",
]
