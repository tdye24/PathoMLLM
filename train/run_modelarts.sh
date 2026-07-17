#!/bin/bash
# PathoMLLM on ModelArts —— 启动命令：
#   bash /home/ma-user/modelarts/user-job-dir/PathoMLLM/train/run_modelarts.sh
#
# OBS 代码目录（整包拷到 PROJECT_ROOT）：
#   s3://bucket-6038/00CV-stuff/yetiandi_y00959450/PathoMLLM/
#   ├── train/
#   ├── model/Qwen3.5-9B/
#   └── data/roi_cls_vqa.jsonl
#
# 网页「训练输出文件路径」填：
#   s3://bucket-6038/00CV-stuff/yetiandi_y00959450/PathoMLLM/outputs/
# 会挂载到 /home/ma-user/modelarts/outputs，平台自动回传 OBS，无需 mox 上传。
set -euo pipefail

# ==================== 写死路径 ====================
PROJECT_ROOT=/home/ma-user/modelarts/user-job-dir/PathoMLLM
TRAIN_DIR="${PROJECT_ROOT}/train"
MODEL_ID="${PROJECT_ROOT}/model/Qwen3.5-9B"
SWIFT_JSONL="${PROJECT_ROOT}/data/roi_cls_vqa.jsonl"
OUTPUT_DIR=/home/ma-user/modelarts/outputs
CONDA_ENV=/home/ma-user/work/yetiandi/envs/qwen35

# ==================== conda ====================
export PATH=/home/ma-user/anaconda3/bin:$PATH
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PATH="${CONDA_ENV}/bin:${PATH}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${TRAIN_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== PathoMLLM ModelArts boot ==="
echo "PROJECT_ROOT   : ${PROJECT_ROOT}"
echo "MODEL_ID       : ${MODEL_ID}"
echo "SWIFT_JSONL    : ${SWIFT_JSONL}"
echo "OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "VC_TASK_INDEX  : ${VC_TASK_INDEX:-0}"
echo "VC_WORKER_NUM  : ${VC_WORKER_NUM:-1}"
echo "MA_NUM_GPUS    : ${MA_NUM_GPUS:-<unset>}"
echo "python         : $(which python)"
echo "swift          : $(command -v swift || echo MISSING)"

[[ -d "${MODEL_ID}" ]] || { echo "ERROR: model not found: ${MODEL_ID}"; exit 1; }
[[ -f "${SWIFT_JSONL}" ]] || { echo "ERROR: jsonl not found: ${SWIFT_JSONL}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"

# ==================== 训练（写到平台挂载的 outputs，自动回传 OBS） ====================
export MODEL_ID SWIFT_JSONL OUTPUT_DIR
export SKIP_GPU_CHECK=1

bash "${TRAIN_DIR}/sft.sh"
