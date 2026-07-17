#!/bin/bash
# PathoMLLM on 华为云 ModelArts —— 训练作业 boot 脚本。
#
# 在「创建训练作业 → 启动命令」里填（代码目录选 PathoMLLM，会被拷到 user-job-dir 下）：
#   bash /home/ma-user/modelarts/user-job-dir/PathoMLLM/train/run_modelarts.sh
#
# 多机多卡：ModelArts 自动注入 MA_NUM_HOSTS / VC_TASK_INDEX / VC_WORKER_HOSTS / MA_NUM_GPUS，
# 由 sft.sh 读取并组网，本脚本无需手动传 rank/world_size。
set -euo pipefail

# ==================== 需按你的 OBS 改（也可在作业「超参/环境变量」里覆盖） ====================
CONDA_ENV="${CONDA_ENV:-qwen35}"                                     # 训练用的 conda 环境名
MODEL_OBS="${MODEL_OBS:-obs://bucket-xxx/models/Qwen3.5-9B}"         # 基座模型（OBS 目录）
DATA_OBS="${DATA_OBS:-obs://bucket-xxx/PathoMLLM/data/train.jsonl}"  # 训练 jsonl（OBS 文件；图像仍走 s3:// 在线读）
OUTPUT_OBS="${OUTPUT_OBS:-obs://bucket-xxx/PathoMLLM/outputs/}"      # 结果回传目录（OBS）

# ==================== 本地 SSD 路径（/cache，一般不用改） ====================
LOCAL_MODEL=/cache/model
LOCAL_DATA=/cache/data/train.jsonl
LOCAL_OUTPUT=/cache/output

# ==================== conda 环境 ====================
export PATH=/home/ma-user/anaconda3/bin:$PATH
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
# 若镜像未预装 ms-swift，取消下一行注释（更推荐用预装好的自定义镜像，启动更快）：
# pip install "ms-swift" deepspeed flash-attn --no-build-isolation

# ==================== 定位代码目录（自适应 user-job-dir 布局） ====================
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${TRAIN_DIR}")"
cd "${PROJECT_ROOT}"

# ==================== 从 OBS 拷模型 + 数据到本地 SSD ====================
# 模型反复读取，先拷到本地 SSD 更快；jsonl 小文件直接拷；图像不拷，训练时经 mox 在线读。
mkdir -p "${LOCAL_MODEL}" "$(dirname "${LOCAL_DATA}")" "${LOCAL_OUTPUT}"
python - <<PY
import moxing as mox
mox.file.copy_parallel("${MODEL_OBS}", "${LOCAL_MODEL}")
mox.file.copy("${DATA_OBS}", "${LOCAL_DATA}")
print("[run_modelarts] model + data copied to local SSD")
PY

# ==================== 交给 sft.sh（分布式变量由其内部读取 ModelArts env） ====================
export MODEL_ID="${LOCAL_MODEL}"
export SWIFT_JSONL="${LOCAL_DATA}"
export OUTPUT_DIR="${LOCAL_OUTPUT}"
export SKIP_GPU_CHECK=1   # ModelArts 独占整机，跳过僵尸进程占用检查

bash "${TRAIN_DIR}/sft.sh"

# ==================== 仅 0 号节点把结果回传 OBS ====================
if [[ "${VC_TASK_INDEX:-0}" == "0" ]]; then
    python - <<PY
import moxing as mox
mox.file.copy_parallel("${LOCAL_OUTPUT}", "${OUTPUT_OBS}")
print("[run_modelarts] output uploaded to ${OUTPUT_OBS}")
PY
fi
