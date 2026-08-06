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

`bbox_seg` 当前用于 bounding-box 检测，报告 **AP50**。
SmartPath-R1 风格的预测应为 `<bbox>[x1,y1,x2,y2]</bbox>`；也兼容裸数组和
`{"bbox": [...]}`。`ground_truth` 可使用相同格式。多目标时使用 bbox 数组，按 IoU
在 IoU 阈值 0.5 下进行一对一匹配。GT 和预测必须采用同一
坐标系。Qwen3.5 的预测按 0--1000 相对坐标解析；scorer 会读取 `images` 中第一张
图片的真实宽高，将预测恢复为像素坐标，再与像素坐标 `ground_truth` 比较。
实现严格复现 SmartPath-R1：不使用置信度，按模型生成框的顺序累计 TP/FP，逐图
构建插值 PR 曲线并计算 AP50，最后对已评分图片的 AP50 求算术平均。GT 或预测框为空
时该图 AP50 为 0；完全缺失的预测记录不进入均值。像素级分割和 MedSAM Dice 暂未实现。

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
