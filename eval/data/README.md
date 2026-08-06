# Benchmark test JSON (PathoMLLM native format)

| File | scorer | 必填字段 |
|------|--------|----------|
| `bcnb.json` | `bcnb` | `id`, `messages`, `ground_truth`, `task`, `images` |
| `chaoyang.json` | `chaoyang` | `id`, `messages`, `ground_truth`, `images` |
| `roi_cls_vqa_test.json` | `roi_cls` | `id`, `messages`, `ground_truth`, `task`, `images` |
| generic MCQ | `mcq` | `id`, `messages`, `ground_truth`, `images` |
| PathMMU-style | `pathmmu` | `id`, `messages`, `ground_truth`, `subset`, `images` |
| detection / box segmentation | `bbox_seg` | `id`, `messages`, `ground_truth`, `images` |

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

## Bounding box 检测/分割

`bbox_seg` 将 bbox 当作矩形二值 mask，报告 macro **IoU、Dice、Precision、Recall**。
SmartPath-R1 风格的预测应为 `<bbox>[x1,y1,x2,y2]</bbox>`；也兼容裸数组和
`{"bbox": [...]}`。`ground_truth` 可使用相同格式。多目标时使用 bbox 数组，按 IoU
贪心一对一匹配；漏检、误检和无法解析的输出均按零分计入。GT 和预测必须采用同一
坐标系，例如均为 0--1000 归一化坐标。

```json
[
  {
    "id": "case_001",
    "messages": [
      {
        "role": "user",
        "content": "<image>\nLocate the lesion. Return <bbox>[x1,y1,x2,y2]</bbox> using 0-1000 coordinates."
      }
    ],
    "ground_truth": [125, 240, 680, 790],
    "images": ["/path/to/case_001.png"]
  }
]
```

Manifest 中增加：

```yaml
- name: my_detection_set
  path: data/my_detection_set.json
  scorer: bbox_seg
```

`images` 兼容本地路径与 `s3://`/`obs://` 路径。远端图片通过 ModelArts
`moxing.file.File` 在线读取，不会先下载到本地；两种图片路径可以混用：

```json
[
  {"id": "local", "images": ["/data/AGGC/local.png"], "ground_truth": "[[0,0,1024,1024]]"},
  {"id": "remote", "images": ["s3://bucket/AGGC/remote.png"], "ground_truth": "[[0,0,1024,681]]"}
]
```
