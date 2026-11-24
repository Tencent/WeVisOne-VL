"""Rule-based GRPO rewards for WeVisOne-VL.

verl loads this file by path (``reward.custom_reward_function.path``) and calls
:func:`compute_score` once per rollout. The ground truth is the reference
answer in the model's own output format, and both strings are parsed by the
same code. The task comes from ``extra_info["task"]``:

==============================  ===============================================
task                            reward
==============================  ===============================================
detection, referring,           box F1: per-label Hungarian matching,
visual_prompt, ocr_grounding    ``2 * sum(IoU**3) / (num_gt + num_pred)``
ocr                             box F1 over normalized text labels
segmentation, matting           box F1 where a predicted box only counts when
                                it is followed by ``<OBJ>``
point                           soft point F1 with ``1 - distance / diagonal``
count_box, count_point          ``0.5 * F1 + (1 if the counts match else 0)``
==============================  ===============================================

Coordinates stay in the 0-1000 token space. An empty reference answer is a
negative sample: the reward is 1 when the prediction contains no objects and
0 otherwise.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment

OBJ_TOKEN = "<OBJ>"
COUNT_F1_WEIGHT = 0.5
_DIAGONAL = math.hypot(1000.0, 1000.0)

_SPACE_RE = re.compile(r"\s*([<>/])\s*")
_OBJECT_RE = re.compile(r"<ref>(.*?)</ref><(box|point)>(.*?)(?:</\2>|(?=<ref>)|$)", re.DOTALL)
_TOKEN_RE = re.compile(r"<(\d+)>|" + re.escape(OBJ_TOKEN))

Geometry = dict[str, list[list[float]]]


def _objects(text: str) -> Iterator[tuple[str, str, str]]:
    text = _SPACE_RE.sub(r"\1", (text or "").split("<|im_end|>", 1)[0])
    for label, shape, body in _OBJECT_RE.findall(text):
        yield label.strip(), shape, body


def _tokens(segment: str) -> list[int | None]:
    """Coordinates of one comma-separated shape; ``None`` marks ``<OBJ>``."""
    return [int(m.group(1)) if m.group(1) else None for m in _TOKEN_RE.finditer(segment)]


def _valid_box(values) -> list[float] | None:
    x1, y1, x2, y2 = (float(v) for v in values)
    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _boxes_in(body: str, require_obj: bool) -> Iterator[list[float]]:
    for segment in body.split(","):
        tokens = _tokens(segment)
        i = 0
        while i + 4 <= len(tokens):
            quad = tokens[i : i + 4]
            if None in quad:
                i += 1
                continue
            has_obj = i + 4 < len(tokens) and tokens[i + 4] is None
            i += 5 if has_obj else 4
            box = _valid_box(quad)
            if box is not None and (has_obj or not require_obj):
                yield box


def parse_boxes(text: str, *, require_obj: bool = False) -> Geometry:
    """``{label: [[x1, y1, x2, y2], ...]}`` from ``<ref>..</ref><box>..</box>`` spans."""
    objects: Geometry = defaultdict(list)
    for label, shape, body in _objects(text):
        if shape == "box":
            objects[label].extend(_boxes_in(body, require_obj))
    return {label: boxes for label, boxes in objects.items() if boxes}


def parse_points(text: str) -> Geometry:
    """``{label: [[x, y], ...]}`` from ``<ref>..</ref><point>..</point>`` spans."""
    objects: Geometry = defaultdict(list)
    for label, shape, body in _objects(text):
        if shape != "point":
            continue
        for segment in body.split(","):
            coords = [t for t in _tokens(segment) if t is not None]
            objects[label].extend(
                [float(coords[i]), float(coords[i + 1])] for i in range(0, len(coords) - 1, 2)
            )
    return {label: points for label, points in objects.items() if points}


def normalize_ocr_text(text: str) -> str:
    """Case-folded letters only, or alphanumerics for labels without letters."""
    value = unicodedata.normalize("NFKC", text).casefold().strip()
    letters = "".join(ch for ch in value if ch.isalpha())
    return letters or "".join(ch for ch in value if ch.isalnum()) or value


def _merge_labels(objects: Geometry, normalize) -> Geometry:
    merged: Geometry = defaultdict(list)
    for label, items in objects.items():
        merged[normalize(label)].extend(items)
    return dict(merged)


def _count(objects: Geometry) -> int:
    return sum(len(items) for items in objects.values())


def _area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def _iou_matrix(gt: list[list[float]], pred: list[list[float]]) -> np.ndarray:
    a, b = np.asarray(gt)[:, None], np.asarray(pred)[None]
    top_left = np.maximum(a[..., :2], b[..., :2])
    bottom_right = np.minimum(a[..., 2:], b[..., 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=-1)
    return inter / (_area(a) + _area(b) - inter)


def box_f1(gt: Geometry, pred: Geometry) -> float:
    num_gt, num_pred = _count(gt), _count(pred)
    if num_gt == 0 and num_pred == 0:
        return 1.0
    matched = 0.0
    for label in gt.keys() & pred.keys():
        iou = _iou_matrix(gt[label], pred[label])
        rows, cols = linear_sum_assignment(iou, maximize=True)
        matched += float((iou[rows, cols] ** 3).sum())
    return 2.0 * matched / (num_gt + num_pred)


def point_f1(gt: Geometry, pred: Geometry) -> float:
    num_gt, num_pred = _count(gt), _count(pred)
    if num_gt == 0 or num_pred == 0:
        return float(num_gt == num_pred)

    def best_similarity(points: Geometry, others: Geometry) -> float:
        total = 0.0
        for label, items in points.items():
            if label not in others:
                continue
            distances = np.linalg.norm(
                np.asarray(items)[:, None] - np.asarray(others[label])[None], axis=-1
            )
            total += float(np.clip(1.0 - distances / _DIAGONAL, 0.0, 1.0).max(axis=1).sum())
        return total

    recall = best_similarity(gt, pred) / num_gt
    precision = best_similarity(pred, gt) / num_pred
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score(task: str, prediction: str, reference: str) -> float:
    if task in ("detection", "referring", "visual_prompt", "ocr_grounding"):
        return box_f1(parse_boxes(reference), parse_boxes(prediction))
    if task == "ocr":
        gt = _merge_labels(parse_boxes(reference), normalize_ocr_text)
        pred = _merge_labels(parse_boxes(prediction), normalize_ocr_text)
        return box_f1(gt, pred)
    if task in ("segmentation", "matting"):
        gt = parse_boxes(reference)
        if not gt:
            return float(not parse_boxes(prediction))
        return box_f1(gt, parse_boxes(prediction, require_obj=True))
    if task == "point":
        return point_f1(parse_points(reference), parse_points(prediction))
    if task == "count_box":
        gt, pred = parse_boxes(reference), parse_boxes(prediction)
        return COUNT_F1_WEIGHT * box_f1(gt, pred) + float(_count(gt) == _count(pred))
    if task == "count_point":
        gt, pred = parse_points(reference), parse_points(prediction)
        return COUNT_F1_WEIGHT * point_f1(gt, pred) + float(_count(gt) == _count(pred))
    raise ValueError(f"no reward is defined for task {task!r}")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """verl entry point (``reward.custom_reward_function.name=compute_score``)."""
    task = (extra_info or {}).get("task") or data_source
    return score(task, solution_str, ground_truth)
