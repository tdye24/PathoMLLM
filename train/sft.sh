#!/bin/bash
# PathoMLLM SFT：用 ms-swift 微调 Qwen3.5-9B（LoRA + 原生 Vision）。
# 图像 resize 由 qwen_vl_utils 在线完成；jsonl images 为 s3://（mox 读入内存）。
set -euo pipefail

# ==================== 路径配置（全部相对 PathoMLLM = PROJECT_ROOT） ====================
# PROJECT_ROOT 自动按本脚本位置推导（train/ 的上级目录）
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${TRAIN_DIR}")"
# 均可被外部环境变量覆盖（run_modelarts.sh 会注入）
MODEL_ID="${MODEL_ID:-${PROJECT_ROOT}/model/Qwen3.5-9B}"
SWIFT_JSONL="${SWIFT_JSONL:-${PROJECT_ROOT}/data/roi_cls_vqa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/cache/modelscope}"
DATASET_MAP_DIR="${MODELSCOPE_CACHE}/datasets"
LOG_DIR="${PROJECT_ROOT}/logs/sft"

LIMIT_SAMPLES=0   # 冒烟：设 64 只训前 64 条；0 = 全量
NUM_GPUS="${MA_NUM_GPUS:-8}"   # 单节点 GPU 数（对应 NPROC_PER_NODE；ModelArts 注入 MA_NUM_GPUS）

# ==================== 分布式（单机 / 多机自动适配） ====================
# ModelArts 多机作业自动注入：MA_NUM_HOSTS(节点数)、VC_TASK_INDEX(节点序号)、
#   VC_WORKER_HOSTS(节点域名列表，逗号分隔，取第一个当 master)。
# 本地单机训练不设这些变量时，自动退化为单机（NNODES=1, NODE_RANK=0, localhost）。
export NNODES="${MA_NUM_HOSTS:-${VC_WORKER_NUM:-1}}"
export NODE_RANK="${VC_TASK_INDEX:-0}"
if [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
    export MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
else
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_PORT="${MASTER_PORT:-6060}"

# ==================== batch ====================
PER_DEVICE_BATCH=4
GRAD_ACCUM=8              # 等效 global batch = 4×8×8 = 256
DEEPSPEED=zero2
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"  # 单图视觉 token 上限（对齐论文 WSI=1024）
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"  # 1 = 跳过训练前 GPU 占用检查

# ==================== LoRA ====================
LORA_RANK=64
LORA_ALPHA=256
LORA_DROPOUT=0.05
# all-linear + freeze_*：LLM + merger 挂 LoRA，ViT 冻结；rank=64 多任务混训

# ==================== 环境 ====================
# 本地 conda 环境（存在才加入 PATH）；ModelArts 上由 run_modelarts.sh 提前 conda activate
QWEN_ENV_BIN="${QWEN_ENV_BIN:-/home/ma-user/envs/qwen35/bin}"
[[ -d "${QWEN_ENV_BIN}" ]] && export PATH="${QWEN_ENV_BIN}:$PATH"
mkdir -p "${DATASET_MAP_DIR}"
# train/ 放 sitecustomize.py + remote_image_io.py：s3:// 经 mox 读入内存
export PYTHONPATH="${TRAIN_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

# 写入 site-packages/*.pth，保证 torchrun / DataLoader 每个新进程都自动打 patch
# （仅靠 PYTHONPATH + sitecustomize 在部分环境不可靠）
python - <<PY
import pathlib
import site
import sys

train_dir = pathlib.Path(r"${TRAIN_DIR}").resolve()
candidates = []
try:
    candidates.extend(site.getsitepackages())
except Exception:
    pass
try:
    us = site.getusersitepackages()
    if us:
        candidates.append(us)
except Exception:
    pass
candidates.append(str(pathlib.Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"))

written = False
for sp in candidates:
    sp_path = pathlib.Path(sp)
    if not sp_path.is_dir():
        continue
    pth = sp_path / "pathomllm_remote_image.pth"
    try:
        pth.write_text(
            f"{train_dir}\n"
            "import remote_image_io; remote_image_io.apply_remote_image_patch()\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[sft] skip {pth}: {exc}", flush=True)
        continue
    print(f"[sft] wrote {pth}", flush=True)
    written = True
    break
if not written:
    print("[sft] WARN: could not write site .pth; relying on PYTHONPATH/sitecustomize", flush=True)

from remote_image_io import ensure_patched
ensure_patched()
print("[sft] remote_image_io preflight OK", flush=True)
PY

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

mkdir -p "${LOG_DIR}"
TIME="$(date +"%Y%m%d_%H%M%S")"

echo "=== SFT training ==="
echo "Model            : ${MODEL_ID}"
echo "Dataset          : ${SWIFT_JSONL}"
echo "Output           : ${OUTPUT_DIR}"
echo "Map cache root   : ${MODELSCOPE_CACHE}"
echo "Map cache dir    : ${DATASET_MAP_DIR}"
echo "Vision cap       : IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM}"
echo "Distributed      : nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT} gpus/node=${NUM_GPUS}"
echo "Batch            : per_device=${PER_DEVICE_BATCH} grad_accum=${GRAD_ACCUM} gpus=${NUM_GPUS} nodes=${NNODES} (global=$((PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS * NNODES)))"
echo "DeepSpeed        : ${DEEPSPEED}"
echo "LoRA             : rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "Trainable        : LLM LoRA + merger LoRA (all-linear); ViT frozen"

# ==================== ms-swift SFT ====================
# 有效 batch = PER_DEVICE_BATCH × GRAD_ACCUM × NUM_GPUS = 4×8×8 = 256
#
# --load_from_cache_file true    map 结果写入 DATASET_MAP_DIR（见上方 MODELSCOPE_CACHE），第二轮起加速
# --add_non_thinking_prefix true  数据 assistant 为纯答案时，训练前自动补空 <think> 块
# --loss_scale ignore_empty_think  上述空 think 块不参与 loss，只对答案算 loss
# --group_by_length false        跳过启动时的全量 length 预处理
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
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
    --learning_rate 3e-4 \
    --aligner_lr 3e-4 \
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
    --group_by_length false \
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
