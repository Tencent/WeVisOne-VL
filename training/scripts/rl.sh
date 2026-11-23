#!/usr/bin/env bash
# GRPO training of WeVisOne-VL with verl (https://github.com/volcengine/verl).
#
#   TRAIN_FILES=rl_data/detection.parquet bash training/scripts/rl.sh [overrides]
#
# TRAIN_FILES / VAL_FILES take one parquet file or a list such as
# "[a.parquet,b.parquet]" (see training/rl/build_dataset.py). Extra arguments are
# Hydra overrides for verl, e.g. trainer.total_epochs=2. For multi-node runs,
# start a Ray cluster first, set NNODES and launch this script on the head node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-tencent/WeVisOne-VL-4B}"
TRAIN_FILES="${TRAIN_FILES:?set TRAIN_FILES to the parquet file(s) to train on}"
VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wevisone_vl_grpo}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096
MAX_TOKENS_PER_GPU=24576

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.val_max_samples=256 \
    data.image_key=images \
    data.train_batch_size=128 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKENS_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="${REPO_ROOT}/training/rl/reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    trainer.logger="['console']" \
    trainer.project_name=wevisone_vl \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    "$@"

python3 -m training.rl.copy_mask_decoder "${MODEL_PATH}" "${OUTPUT_DIR}"
