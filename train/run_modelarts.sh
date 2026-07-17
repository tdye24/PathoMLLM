#!/bin/bash
# PathoMLLM on 华为云 ModelArts —— 训练作业 boot 脚本。
#
# 推荐 OBS 布局（代码目录选整个 PathoMLLM/，平台会整包拷到 MA_JOB_DIR）：
#   s3://bucket-6038/00CV-stuff/yetiandi_y00959450/PathoMLLM/
#   ├── train/   (本目录)
#   ├── model/Qwen3.5-9B/
#   └── data/roi_cls_vqa.jsonl
#
# 图像仍在 jsonl 里写 s3://，训练时 mox 在线读，不必放进代码目录。
#
# 运行参数可选:
#   conda_env=/home/ma-user/work/yetiandi/envs/qwen35
#   s3_output_dir=s3://bucket-6038/00CV-stuff/yetiandi_y00959450/PathoMLLM/outputs/run001/

# ==================== conda（你的环境是路径式激活，不是环境名） ====================
# 等价于: conda activate /home/ma-user/work/yetiandi/envs/qwen35
# 网页超参可覆盖: conda_env=/home/ma-user/work/yetiandi/envs/qwen35
CONDA_ENV="${conda_env:-${CONDA_ENV:-/home/ma-user/work/yetiandi/envs/qwen35}}"
export PATH=/home/ma-user/anaconda3/bin:$PATH
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PATH="${CONDA_ENV}/bin:${PATH}"
echo "[run_modelarts] conda: ${CONDA_ENV}  python=$(which python)  swift=$(command -v swift || echo MISSING)"

# ==================== 定位代码（ModelArts 已把 PathoMLLM/ 拷到 user-job-dir） ====================
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${TRAIN_DIR}")"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${TRAIN_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

# ==================== 路径：优先用代码目录里已随作业拷入的 model / data ====================
BUNDLED_MODEL="${PROJECT_ROOT}/model/Qwen3.5-9B"
BUNDLED_DATA="${PROJECT_ROOT}/data/roi_cls_vqa.jsonl"
LOCAL_OUTPUT="${train_url:-/cache/output}"
OUTPUT_OBS="${s3_output_dir:-${OUTPUT_OBS:-s3://bucket-6038/00CV-stuff/yetiandi_y00959450/PathoMLLM/outputs/}}"

# 可选覆盖（网页超参）：model_id=...  swift_jsonl=...
MODEL_ID="${model_id:-${MODEL_ID:-${BUNDLED_MODEL}}}"
SWIFT_JSONL="${swift_jsonl:-${SWIFT_JSONL:-${BUNDLED_DATA}}}"

# 若代码目录里没有，再回退到 OBS 单独路径（旧模式）
MODEL_OBS="${model_obs:-${MODEL_OBS:-s3://bucket-6038/00CV-stuff/yetiandi_y00959450/PathoMLLM/model/Qwen3.5-9B}}"
DATA_OBS="${data_url:-${DATA_OBS:-s3://bucket-6038/00CV-stuff/yetiandi_y00959450/PathoMLLM/data/roi_cls_vqa.jsonl}}"

echo "=== PathoMLLM ModelArts boot ==="
echo "MA_JOB_DIR        : ${MA_JOB_DIR:-<unset>}"
echo "PROJECT_ROOT      : ${PROJECT_ROOT}"
echo "VC_TASK_INDEX     : ${VC_TASK_INDEX:-0}"
echo "VC_WORKER_NUM     : ${VC_WORKER_NUM:-1}"
echo "MA_NUM_GPUS       : ${MA_NUM_GPUS:-<unset>}"
echo "MODEL_ID          : ${MODEL_ID}"
echo "SWIFT_JSONL       : ${SWIFT_JSONL}"
echo "OUTPUT_DIR        : ${LOCAL_OUTPUT}"
echo "OUTPUT_OBS        : ${OUTPUT_OBS}"

mkdir -p "${LOCAL_OUTPUT}"

# 模型 / jsonl：已在代码目录 → 直接用；否则 mox 从 OBS 拉到本地
if [[ ! -d "${MODEL_ID}" ]]; then
    echo "[run_modelarts] bundled model not found, copying from OBS: ${MODEL_OBS}"
    MODEL_ID="/cache/model"
    mkdir -p "${MODEL_ID}"
    python -u -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS}', '${MODEL_ID}')"
fi

if [[ ! -f "${SWIFT_JSONL}" ]]; then
    echo "[run_modelarts] bundled jsonl not found, copying from OBS: ${DATA_OBS}"
    SWIFT_JSONL="/cache/data/roi_cls_vqa.jsonl"
    mkdir -p "$(dirname "${SWIFT_JSONL}")"
    python -u -c "import moxing as mox; mox.file.copy('${DATA_OBS}', '${SWIFT_JSONL}')"
fi

if [[ ! -d "${MODEL_ID}" ]]; then
    echo "ERROR: model not found: ${MODEL_ID}"
    exit 1
fi
if [[ ! -f "${SWIFT_JSONL}" ]]; then
    echo "ERROR: jsonl not found: ${SWIFT_JSONL}"
    exit 1
fi

echo "[run_modelarts] using model=${MODEL_ID} data=${SWIFT_JSONL}"

# ==================== 训练 ====================
export MODEL_ID SWIFT_JSONL
export OUTPUT_DIR="${LOCAL_OUTPUT}"
export SKIP_GPU_CHECK=1

bash "${TRAIN_DIR}/sft.sh"

# ==================== 回传 OBS（仅 0 号节点） ====================
if [[ "${VC_TASK_INDEX:-0}" == "0" ]]; then
    python -u -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT}', '${OUTPUT_OBS}'); print('[run_modelarts] uploaded -> ${OUTPUT_OBS}')"
fi
