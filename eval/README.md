# PathoMLLM Eval

Manifest 驱动的 benchmark 评测：内存采样 → LoRA 推理 → scorer 打分 → `summary.csv`。

## 运行

```bash
cd /path/to/PathoMLLM
export PATHOMLLM_DATA_ROOT=/path/to/tiles   # 可选，展开 JSON 里的 ${PATHOMLLM_DATA_ROOT}
pip install -r eval/requirements.txt

python -m eval.run_eval \
  --manifest eval/manifests/benchmarks.yaml \
  --run_config eval/manifests/run_sft.yaml
```

## 架构

```
manifest.yaml          run_config.yaml
     │                       │
     └─────────┬─────────────┘
               ▼
          run_eval.py  ──►  eval/batch_inference.py  ──►  *_pred.json
               │
               ▼
          score.py  ──►  eval/scorers/<name>.py  ──►  *_score.json
               │
               ▼
          aggregate.py  ──►  summary.csv
```

### 分类指标（MCQ benchmark 统一）

**acc**、**bacc**（balanced accuracy）、**f1**（macro F1）

### `eval/scorers/`

| scorer | 用于 | 额外 breakdown |
|--------|------|----------------|
| `mcq` | 通用 MCQ | — |
| `bcnb` | bcnb | `by_task` |
| `roi_cls` | ROI 分类 VQA | `by_task` |
| `pathmmu` | pathmmu | `by_subset` |

## 单独推理

```bash
python eval/batch_inference.py \
  --model_id model/Qwen3.5-9B \
  --adapter_dir outputs/sft/v0-xxx/checkpoint-500 \
  --input_json eval/data/bcnb.json \
  --output_json eval/results/bcnb_pred.json \
  --attn_implementation sdpa
```

`batch_inference.py` 会自动加载 `train/pathomllm_plugin.py`（s3:// 读图 + torch 2.4 FSDP shim）。

## 单独打分

```bash
python -m eval.score \
  --manifest eval/manifests/benchmarks.yaml \
  --dataset bcnb \
  --pred_json eval/results/bcnb_pred.json \
  --output_json eval/results/bcnb_score.json
```

## 单元测试（无 GPU）

```bash
cd /path/to/PathoMLLM
python -m eval.test_eval -v
```

## 画 checkpoint 折线图

评测完 `run_eval` 后，每个 checkpoint 目录下有 `<dataset>_score.json`，可用 `plot_curves` 画 acc/bacc/f1 随 step 变化的折线图。

**单个训练 run（多 checkpoint）：**

```bash
python -m eval.plot_curves \
  --run_dir eval/results/pathomllm_sft \
  --output_dir eval/plots
```

**从 summary.csv 画：**

```bash
python -m eval.plot_curves \
  --summary_csv eval/results/pathomllm_sft/summary.csv \
  --output_dir eval/plots
```

**对比多个实验（如不同 LoRA rank）：**

```bash
python -m eval.plot_curves \
  --runs rank32=eval/results/sft_r32 rank64=eval/results/sft_r64 \
  --datasets bcnb \
  --output_dir eval/plots/compare
```

**只画 acc，并输出一张多 dataset 总览图：**

```bash
python -m eval.plot_curves \
  --run_dir eval/results/pathomllm_sft \
  --metrics acc \
  --combined \
  --combined_metric acc \
  --output_dir eval/plots
```

输出：`eval/plots/<dataset>_curves.png`（每张图含 acc/bacc/f1 三条线）；加 `--combined` 另有 `all_datasets_acc.png`。

## 与 CPathOmni eval 的差异

| | CPathOmni | PathoMLLM |
|--|-----------|-----------|
| 图像字段 | `slide_images` / `roi_images` (h5) | `images` (`s3://` 或本地路径) |
| 占位符 | `<slide_image>` / `<roi_image>` | `<image>` |
| 推理脚本 | `v0/batch_inference.py` | `eval/batch_inference.py` |
| OBS 读图 | `remote_image_io` / sitecustomize | `train/pathomllm_plugin.py` |

若已有 CPathOmni 格式 JSON，需把 `slide_images`/`roi_images` 转为 `images`，并把 `<slide_image>` 换成 `<image>`。
