# Benchmark test JSON (PathoMLLM native format)

| File | scorer | 必填字段 |
|------|--------|----------|
| `bcnb.json` | `bcnb` | `id`, `messages`, `ground_truth`, `task`, `images` |
| `roi_cls_vqa_test.json` | `roi_cls` | `id`, `messages`, `ground_truth`, `task`, `images` |
| generic MCQ | `mcq` | `id`, `messages`, `ground_truth`, `images` |
| PathMMU-style | `pathmmu` | `id`, `messages`, `ground_truth`, `subset`, `images` |

## 公共字段

| Field | Notes |
|-------|-------|
| `id` | 唯一 |
| `messages` | 至少一条 `user`；`<image>` 数量 = `len(images)` |
| `images` | 路径或 `s3://` URL 列表（与训练 jsonl 一致） |
| `ground_truth` | MCQ 单字母 A–Z |
| `task` | BCNB / roi_cls：按 task 统计 |
| `subset` | PathMMU：按 subset 统计 |
| `chat_template_kwargs` | 可选，传给 ms-swift template |
| `max_tokens` | 可选，单图视觉 token 上限（会转成 max_pixels） |

## 示例

```json
[
  {
    "id": "sample_001",
    "task": "tumor_type",
    "messages": [
      {"role": "user", "content": "<image>\nWhat is shown?\n(A) Normal (B) Tumor"}
    ],
    "ground_truth": "B",
    "images": ["s3://bucket/path/tile.jpg"]
  }
]
```

环境变量 `${PATHOMLLM_DATA_ROOT}` 会在加载 manifest 时展开。
