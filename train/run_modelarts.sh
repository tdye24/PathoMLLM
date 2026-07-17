#!/bin/bash
# PathoMLLM on 华为云 ModelArts —— 训练作业 boot 脚本。
#
# 借鉴同事 startup_mox.py 的模式：
#   - 模型 / jsonl 拷到本地 /cache；病理图像不拷，jsonl 里 s3:// 在线读（mox）
#   - 多机用 VC_WORKER_HOSTS / VC_TASK_INDEX / VC_WORKER_NUM 组网（sft.sh 内处理）
#   - 支持 ModelArts 常用路径变量：data_url / train_url / s3_output_dir
#
# 创建训练作业 → 启动命令：
#   bash /home/ma-user/modelarts/user-job-dir/PathoMLLM/train/run_modelarts.sh
#
# 运行参数（网页超参，key=value，会注入为环境变量）示例：
#   data_url=s3://bucket-xxx/PathoMLLM/data/train.jsonl
#   train_url=/cache/output
#   s3_output_dir=s3://bucket-xxx/PathoMLLM/outputs/run001/
#   model_obs=s3://bucket-xxx/models/Qwen3.5-9B
#   conda_env=qwen35
set -euo pipefail

# ==================== ModelArts 路径（优先读平台注入的 data_url / train_url / s3_output_dir） ====================
CONDA_ENV="${conda_env:-${CONDA_ENV:-qwen35}}"
MODEL_OBS="${model_obs:-${MODEL_OBS:-s3://bucket-xxx/models/Qwen3.5-9B}}"
DATA_OBS="${data_url:-${DATA_OBS:-s3://bucket-xxx/PathoMLLM/data/train.jsonl}}"
OUTPUT_OBS="${s3_output_dir:-${OUTPUT_OBS:-s3://bucket-xxx/PathoMLLM/outputs/}}"
LOCAL_OUTPUT="${train_url:-/cache/output}"

# ==================== 本地 SSD（对齐 startup_mox.py 的 /cache 用法） ====================
LOCAL_MODEL=/cache/model
LOCAL_DATA=/cache/data/train.jsonl

# ==================== conda ====================
export PATH=/home/ma-user/anaconda3/bin:$PATH
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# ==================== 定位代码（MA_JOB_DIR 下自动拷入） ====================
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${TRAIN_DIR}")"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${TRAIN_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== PathoMLLM ModelArts boot ==="
echo "MA_JOB_DIR        : ${MA_JOB_DIR:-<unset>}"
echo "MA_MOUNT_PATH     : ${MA_MOUNT_PATH:-<unset>}"
echo "VC_TASK_INDEX     : ${VC_TASK_INDEX:-0}"
echo "VC_WORKER_NUM     : ${VC_WORKER_NUM:-1}"
echo "MA_NUM_HOSTS      : ${MA_NUM_HOSTS:-<unset>}"
echo "MA_NUM_GPUS       : ${MA_NUM_GPUS:-<unset>}"
echo "MODEL_OBS         : ${MODEL_OBS}"
echo "DATA_OBS          : ${DATA_OBS}"
echo "LOCAL_OUTPUT      : ${LOCAL_OUTPUT}"
echo "OUTPUT_OBS        : ${OUTPUT_OBS}"

# ==================== mox：只拷模型 + jsonl；图像走 s3:// 在线读（同 pathology pass） ====================
mkdir -p "${LOCAL_MODEL}" "$(dirname "${LOCAL_DATA}")" "${LOCAL_OUTPUT}"
python -u <<PY
import moxing as mox

mox.file.copy_parallel("${MODEL_OBS}", "${LOCAL_MODEL}")
mox.file.copy("${DATA_OBS}", "${LOCAL_DATA}")
print("[run_modelarts] model + jsonl copied to /cache (images stay on s3://)")
PY

# ==================== 训练 ====================
export MODEL_ID="${LOCAL_MODEL}"
export SWIFT_JSONL="${LOCAL_DATA}"
export OUTPUT_DIR="${LOCAL_OUTPUT}"
export SKIP_GPU_CHECK=1

bash "${TRAIN_DIR}/sft.sh"

# ==================== 回传 OBS（仅 0 号节点，避免多机重复上传） ====================
if [[ "${VC_TASK_INDEX:-0}" == "0" ]]; then
    python -u <<PY
import moxing as mox
mox.file.copy_parallel("${LOCAL_OUTPUT}", "${OUTPUT_OBS}")
print("[run_modelarts] uploaded ${LOCAL_OUTPUT} -> ${OUTPUT_OBS}")
PY
fi
