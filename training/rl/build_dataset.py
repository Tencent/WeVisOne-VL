"""Convert SFT-format JSONL into a verl GRPO parquet file.

The last assistant message of each conversation becomes the reference answer
scored by ``training/rl/reward.py``; the preceding messages become the prompt.
The reward task is ``--task`` or a per-line ``"task"`` field.

Example::

    python -m training.rl.build_dataset --input detection.jsonl --task detection \\
        --output rl_data/detection.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wevisone_vl.engine import SYSTEM_PROMPT


def convert(record: dict, index: int, args: argparse.Namespace) -> dict:
    messages = list(record["messages"])
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the assistant answer")
    answer = messages.pop()["content"]
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("the prompt needs at least one user message")
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": args.system_message})

    task = record.get("task") or args.task
    if not task:
        raise ValueError("pass --task or set a 'task' field on every line")
    images = [
        {
            "image": str(Path(path).expanduser()),
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        }
        for path in record.get("images") or []
    ]
    return {
        "data_source": record.get("source") or args.input.stem,
        "prompt": [{"role": m["role"], "content": m["content"]} for m in messages],
        "images": images,
        "ability": task,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"task": task, "index": index},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="SFT-format JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file.")
    parser.add_argument("--task", help="Reward task, see training/rl/reward.py.")
    parser.add_argument("--system-message", default=SYSTEM_PROMPT)
    parser.add_argument("--min-pixels", type=int, default=16 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=2025 * 32 * 32)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    rows = [convert(record, index, args) for index, record in enumerate(records)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"wrote {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
