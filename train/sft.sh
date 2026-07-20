#!/bin/bash
# PathoMLLM SFT — ms-swift LoRA fine-tune on Qwen3.5-9B (native vision).
# Images: jsonl stores s3:// paths; pathomllm_plugin patches load_file at train time.
set -euo pipefail

# =============================================================================
# Paths (relative to PathoMLLM = PROJECT_ROOT; overridable via env)
# =============================================================================
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${TRAIN_DIR}")"

MODEL_ID="${MODEL_ID:-${PROJECT_ROOT}/model/Qwen3.5-9B}"
SWIFT_JSONL="${SWIFT_JSONL:-${PROJECT_ROOT}/data/roi_cls_vqa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${train_url:-${PROJECT_ROOT}/outputs/sft}}"
PLUGIN="${TRAIN_DIR}/pathomllm_plugin.py"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/cache/modelscope}"
DATASET_MAP_DIR="${MODELSCOPE_CACHE}/datasets"

# =============================================================================
# Training hyperparameters
# =============================================================================
NUM_GPUS="${MA_NUM_GPUS:-8}"
PER_DEVICE_BATCH=4
GRAD_ACCUM=8
DEEPSPEED=zero2
SAVE_STEPS=500

LORA_RANK=64
LORA_ALPHA=256
LORA_DROPOUT=0.05

# =============================================================================
# Distributed (ModelArts multi-node or local single-node)
# =============================================================================
export NNODES="${MA_NUM_HOSTS:-${VC_WORKER_NUM:-1}}"
export NODE_RANK="${VC_TASK_INDEX:-0}"
export MASTER_PORT="${MASTER_PORT:-6060}"

if [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
    export MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
else
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

# global batch = PER_DEVICE_BATCH × GRAD_ACCUM × NUM_GPUS × NNODES
GLOBAL_BATCH=$((PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS * NNODES))

# =============================================================================
# Runtime environment
# =============================================================================
QWEN_ENV_BIN="${QWEN_ENV_BIN:-/home/ma-user/envs/qwen35/bin}"
[[ -d "${QWEN_ENV_BIN}" ]] && export PATH="${QWEN_ENV_BIN}:$PATH"

mkdir -p "${DATASET_MAP_DIR}"
export PYTHONPATH="${TRAIN_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12

# =============================================================================
# Preflight
# =============================================================================
if [[ ! -f "${SWIFT_JSONL}" ]]; then
    echo "ERROR: dataset not found: ${SWIFT_JSONL}"
    exit 1
fi

if [[ ! -f "${PLUGIN}" ]]; then
    echo "ERROR: plugin not found: ${PLUGIN}"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================
echo "=== PathoMLLM SFT ==="
echo "Model       : ${MODEL_ID}"
echo "Dataset     : ${SWIFT_JSONL}"
echo "Output      : ${OUTPUT_DIR}"
echo "Plugin      : ${PLUGIN}"
echo "Cache       : ${DATASET_MAP_DIR}"
echo "Distributed : nnodes=${NNODES} rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT} gpus/node=${NUM_GPUS}"
echo "Batch       : per_device=${PER_DEVICE_BATCH} grad_accum=${GRAD_ACCUM} global=${GLOBAL_BATCH}"
echo "LoRA        : rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "DeepSpeed   : ${DEEPSPEED}"
echo "Vision      : IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM}"
echo ""

# =============================================================================
# ms-swift SFT
# =============================================================================
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NUM_GPUS}" \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
swift sft \
    --model "${MODEL_ID}" \
    --tuner_type lora \
    --check_model false \
    --external_plugins "${PLUGIN}" \
    --dataset "${SWIFT_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed "${DEEPSPEED}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --max_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing true \
    --learning_rate 3e-4 \
    --aligner_lr 3e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0 \
    --group_by_length false \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --logging_steps 5 \
    --save_steps "${SAVE_STEPS}"
