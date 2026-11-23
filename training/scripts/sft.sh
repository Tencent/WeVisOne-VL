#!/usr/bin/env bash
# Supervised fine-tuning of WeVisOne-VL.
#
#   DATA_CONFIG=my_data.yaml bash training/scripts/sft.sh [extra arguments]
#
# Extra arguments are passed to `python -m training.sft.train` and override the
# defaults below (run it with --help for every option). Set DEEPSPEED_CONFIG= to
# train without DeepSpeed. For multi-node runs, set NNODES, NODE_RANK,
# MASTER_ADDR and MASTER_PORT on every node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-tencent/WeVisOne-VL-4B}"
DATA_CONFIG="${DATA_CONFIG:?set DATA_CONFIG to a dataset YAML or JSONL file}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-training/configs/zero2.json}"
DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m training.sft.train \
    --model_name_or_path "${MODEL_PATH}" \
    --data_config "${DATA_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    "${DEEPSPEED_ARGS[@]}" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --vision_lr 5e-7 \
    --weight_decay 0.01 \
    --warmup_steps 0.03 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --model_max_length 8192 \
    --dataloader_num_workers 4 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --report_to none \
    "$@"
