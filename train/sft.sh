#!/bin/bash
# PathoMLLM SFT：用 ms-swift 微调 Qwen3.5-9B（LoRA + 原生 Vision）。
# 图像 resize 由 qwen_vl_utils 在线完成；jsonl images 为 s3://（mox 读入内存）。
set -euo pipefail

# ==================== 路径配置（按需修改） ====================
# PROJECT_ROOT 自动按本脚本位置推导（train/ 的上级目录），换机器 / 换 clone 路径都不用改
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${TRAIN_DIR}")"
MODEL_ID="/home/ma-user/work/yetiandi/Models/Qwen/Qwen3.5-9B"   # 基座模型
SWIFT_JSONL="${PROJECT_ROOT}/data/train.jsonl"                  # 训练 jsonl（messages + images + <image>）
OUTPUT_DIR="${PROJECT_ROOT}/outputs/sft"                        # checkpoint / 日志输出目录
# 预处理 map 缓存根目录（ms-swift 读 MODELSCOPE_CACHE → {根}/datasets/.../*.arrow）
# 运行前可覆盖：MODELSCOPE_CACHE=/other/path bash train/sft.sh
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/cache/modelscope}"
DATASET_MAP_DIR="${MODELSCOPE_CACHE}/datasets"                  # jsonl 加载 + map(lengths) 落盘位置

LIMIT_SAMPLES=0   # 冒烟：设 64 只训前 64 条；0 = 全量
NUM_GPUS=8        # 使用的 GPU 数量（对应 NPROC_PER_NODE）

# ==================== batch ====================
PER_DEVICE_BATCH=4
GRAD_ACCUM=8              # 等效 global batch = 4×8×8 = 256
DEEPSPEED=zero2
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"  # 1 = 跳过训练前 GPU 占用检查

# ==================== LoRA ====================
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
# all-linear + freeze_*：LLM + merger 挂 LoRA，ViT 冻结（方案 B）

# ==================== 环境 ====================
export PATH=/home/ma-user/work/yetiandi/envs/qwen35/bin:$PATH
mkdir -p "${DATASET_MAP_DIR}"
# train/ 放 sitecustomize.py，所有 python 子进程（含 dataset map）启动时：
#   - 放宽 PIL PNG/像素限制
#   - patch qwen_vl_utils.fetch_image：images 经 mox 读入内存（OBS）
export PYTHONPATH="${TRAIN_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

# ms-swift 导入 FSDPModule：torch>=2.6 在 torch.distributed.fsdp；
# torch 2.4/2.5 仅在 torch.distributed._composable.fsdp（pip 版 ms-swift 需启动时 patch）
python - <<'PY'
import pathlib
import site
import sys
import torch

FSDP_IMPORT_OLD = "from torch.distributed.fsdp import FSDPModule as FSDP2"
FSDP_IMPORT_NEW = (
    "try:\n"
    "    from torch.distributed.fsdp import FSDPModule as FSDP2\n"
    "except ImportError:\n"
    "    from torch.distributed._composable.fsdp import FSDPModule as FSDP2"
)


def fsdp_public() -> bool:
    try:
        from torch.distributed.fsdp import FSDPModule  # noqa: F401
        return True
    except ImportError:
        return False


def fsdp_composable() -> bool:
    try:
        from torch.distributed._composable.fsdp import FSDPModule  # noqa: F401
        return True
    except ImportError:
        return False


def find_activation_cpu_offload_py() -> pathlib.Path | None:
    rel = pathlib.Path("swift/callbacks/activation_cpu_offload.py")
    search_roots = []
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        if root:
            search_roots.append(pathlib.Path(root))
    for entry in sys.path:
        if entry:
            search_roots.append(pathlib.Path(entry))
    seen = set()
    for root in search_roots:
        candidate = root / rel
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def patch_ms_swift_fsdp_import() -> None:
    path = find_activation_cpu_offload_py()
    if path is None:
        print("ERROR: 找不到 swift/callbacks/activation_cpu_offload.py")
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    if FSDP_IMPORT_NEW in text:
        return
    if FSDP_IMPORT_OLD not in text:
        print(f"ERROR: 未识别的 ms-swift 文件格式，请手动检查: {path}")
        sys.exit(1)
    path.write_text(text.replace(FSDP_IMPORT_OLD, FSDP_IMPORT_NEW, 1), encoding="utf-8")
    print(f"[sft.sh] Patched ms-swift for torch {torch.__version__}: {path}")


if fsdp_public():
    sys.exit(0)

if fsdp_composable():
    patch_ms_swift_fsdp_import()
    sys.exit(0)

print(f"ERROR: 找不到 FSDPModule，当前 torch={torch.__version__}")
print("修复: pip install -U 'torch>=2.6.0'（推荐）")
sys.exit(1)
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"  # 可见 GPU 编号
export MKL_THREADING_LAYER=GNU    # MKL 与 libgomp 多进程兼容（避免 torchrun 启动失败）
export TOKENIZERS_PARALLELISM=false   # 避免 tokenizer 多进程死锁
export NCCL_IB_DISABLE=1              # 集群 InfiniBand 问题时关闭 IB
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"  # NCCL 通信网卡
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减轻 CUDA OOM 碎片
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VIDEO_MAX_TOKEN_NUM=128        # 本任务无视频，保持 ms-swift 默认即可
export FPS_MAX_FRAMES=12

SAVE_STEPS=500   # 每多少 step 存一次 checkpoint

if [[ "${LIMIT_SAMPLES}" -gt 0 ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/sft_smoke_${LIMIT_SAMPLES}"
    SWIFT_JSONL="${SWIFT_JSONL}#${LIMIT_SAMPLES}"
    SAVE_STEPS=10
    echo "=== Smoke test mode: LIMIT_SAMPLES=${LIMIT_SAMPLES} ==="
fi

if [[ ! -f "${SWIFT_JSONL%%#*}" ]]; then
    echo "ERROR: Dataset jsonl not found: ${SWIFT_JSONL%%#*}"
    exit 1
fi

# 训练前检查 GPU：OOM 日志里常见「一张卡上多个 python 进程」= 僵尸任务占显存
if [[ "${SKIP_GPU_CHECK}" != "1" ]]; then
    python - <<'PY'
import os
import subprocess
import sys

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
target_ids = {x.strip() for x in visible.split(",") if x.strip()}
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
except FileNotFoundError:
    sys.exit(0)

bad = []
for line in out.strip().splitlines():
    idx, used, total = [x.strip() for x in line.split(",")]
    if idx in target_ids and int(used) > 5000:
        bad.append(f"GPU {idx}: {used}MiB / {total}MiB already in use")

if bad:
    print("ERROR: GPU 上已有进程占用显存（可能是上次 OOM 残留的僵尸进程）:")
    for msg in bad:
        print(f"  {msg}")
    print("")
    print("先清理再训练，例如:")
    print("  nvidia-smi")
    print("  pkill -9 -f 'swift/cli/sft.py'")
    print("  pkill -9 -f 'torch.distributed.run'")
    print("  # 仍杀不掉时用 nvidia-smi 里的 PID: kill -9 <pid>")
    print("  # 或重启 notebook / 释放 GPU 实例")
    print("")
    print("确认 GPU 空闲后重跑；紧急跳过检查: SKIP_GPU_CHECK=1 bash train/sft.sh")
    sys.exit(1)
PY
fi

LOG_DIR="${PROJECT_ROOT}/logs/sft"
mkdir -p "${LOG_DIR}"
TIME="$(date +"%Y%m%d_%H%M%S")"

echo "=== SFT training ==="
echo "Model            : ${MODEL_ID}"
echo "Dataset          : ${SWIFT_JSONL}"
echo "Output           : ${OUTPUT_DIR}"
echo "Map cache root   : ${MODELSCOPE_CACHE}"
echo "Map cache dir    : ${DATASET_MAP_DIR}"
echo "Vision cap       : IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM}"
echo "Batch            : per_device=${PER_DEVICE_BATCH} grad_accum=${GRAD_ACCUM} gpus=${NUM_GPUS} (global=$((PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS)))"
echo "DeepSpeed        : ${DEEPSPEED}"
echo "LoRA             : rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "Trainable        : LLM LoRA + merger LoRA (all-linear); ViT frozen"

# ==================== ms-swift SFT ====================
# 有效 batch = PER_DEVICE_BATCH × GRAD_ACCUM × NUM_GPUS = 4×8×8 = 256
#
# --load_from_cache_file true    map 结果写入 DATASET_MAP_DIR（见上方 MODELSCOPE_CACHE），第二轮起加速
# --add_non_thinking_prefix true  数据 assistant 为纯答案时，训练前自动补空 <think> 块
# --loss_scale ignore_empty_think  上述空 think 块不参与 loss，只对答案算 loss
# --group_by_length true         相近长度样本组 batch（启动时会 map 算 lengths）
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE="${NUM_GPUS}" \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model "${MODEL_ID}" \
    --tuner_type lora \
    --dataset "${SWIFT_JSONL}" \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 1e-4 \
    --aligner_lr 1e-4 \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --weight_decay 0.1 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --group_by_length true \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 5 \
    --save_steps "${SAVE_STEPS}" \
    --max_length 8192 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed "${DEEPSPEED}" \
    2>&1 | tee "${LOG_DIR}/train_${TIME}.log"
