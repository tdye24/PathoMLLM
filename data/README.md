# 结构化 CoT 润色脚本

`polish_structured_cot_to_swift.py` 用于将结构化的 MCTS 推理路径润色为 Swift 训练样本使用的 chat JSONL 格式。

脚本只把推理文本发送给 Qwen 或 OpenAI-compatible chat API 做语言润色。最终答案选项始终从输入数据的 `answer` 字段读取，并由脚本自动拼接，因此模型不会决定或修改标签。

## 输入格式

输入文件应为 JSONL，每条记录至少包含以下字段：

```json
{
  "id": "ccrcc_0005908",
  "question": "包含 A/B/C/D 选项的问题文本",
  "image_path": "s3://...",
  "reasoning_chain": [
    {"step_text": "..."},
    {"step_text": "..."},
    {"step_text": "...\n\nThe answer is D."}
  ],
  "answer": "D"
}
```

推荐输入：

```powershell
.\CoT\before_swift\CCRCC_pos.jsonl
```

脚本会把 `reasoning_chain` 拆成两部分：

- `Detailed observation nodes`：中间推理节点，视为细致的显微形态观察。
- `Key evidence from the finish node`：finish 节点中的关键证据，脚本会先去掉末尾的 `The answer is X.`

类似 `Let's inspect the pathology image systematically...` 的泛化根节点会被跳过。

## 输出格式

输出文件遵循 Swift CoT 样本格式：

```json
{
  "id": "ccrcc_0005908",
  "messages": [
    {
      "role": "user",
      "content": "<image>\n包含选项的问题文本"
    },
    {
      "role": "assistant",
      "content": "<think>润色后的推理文本。\n\nThe answer is D.</think><answer>D. Renal cancer</answer>"
    }
  ],
  "images": ["s3://..."]
}
```

`<answer>` 中的答案文本会从 `question` 的选项里自动解析。

## 当前 Prompt 行为

当前 prompt 要求模型：

- 保留已有视觉证据的含义；
- 不新增染色、免疫标志物、临床病史、标签、分子信息或其他新证据；
- 将中间节点视为细致观察；
- 将 finish 节点视为连接观察与最终诊断的关键证据；
- 自然合并重复表达，而不是重复同一形态描述；
- 让推理连贯、有逻辑，但不要新增证据；
- 只返回 `{"cot": "..."}` 形式的 JSON。

初始版写作要求已作为注释保存在脚本中，方便回退。其中被移除的长度限制如下：

```text
Keep the reasoning image-grounded and concise, usually 2 to 4 sentences.
```

## 使用 DashScope API 运行

PowerShell 多行命令需要使用反引号作为续行符：

```powershell
python .\tools\post_progress_cot\polish_structured_cot_to_swift.py `
  --input .\CoT\before_swift\CCRCC_pos.jsonl `
  --output .\tools\post_progress_cot\CCRCC_pos_qwen38_swift_cot_8samples.jsonl `
  --debug_output .\tools\post_progress_cot\CCRCC_pos_qwen38_debug_8samples.jsonl `
  --limit 8 `
  --api_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" `
  --api_model "qwen-plus" `
  --api_key "YOUR_API_KEY"
```

也可以先设置环境变量，然后运行时省略 `--api_key`：

```powershell
$env:DASHSCOPE_API_KEY="YOUR_API_KEY"

python .\tools\post_progress_cot\polish_structured_cot_to_swift.py `
  --input .\CoT\before_swift\CCRCC_pos.jsonl `
  --output .\tools\post_progress_cot\CCRCC_pos_qwen38_swift_cot_8samples.jsonl `
  --debug_output .\tools\post_progress_cot\CCRCC_pos_qwen38_debug_8samples.jsonl `
  --limit 8 `
  --api_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" `
  --api_model "qwen-plus"
```

## 使用本地 OpenAI-Compatible API 运行

如果使用本地 vLLM 或其他 OpenAI-compatible 服务：

```powershell
python .\tools\post_progress_cot\polish_structured_cot_to_swift.py `
  --input .\CoT\before_swift\CCRCC_pos.jsonl `
  --output .\tools\post_progress_cot\CCRCC_pos_qwen38_swift_cot_8samples.jsonl `
  --debug_output .\tools\post_progress_cot\CCRCC_pos_qwen38_debug_8samples.jsonl `
  --limit 8 `
  --api_base_url "http://127.0.0.1:2573/v1" `
  --api_model "Qwen/Qwen3.8-27B" `
  --api_key "dummy"
```

## 常用参数

- `--limit 8`：只生成 8 条记录，适合先抽样检查效果。
- `--debug_output PATH`：保存原始 API 响应、fallback 状态、最终 CoT、答案选项和答案文本。
- `--no_resume`：忽略输出文件中已有的 id，重新生成记录。
- `--num_chunks N --chunk_idx K`：按样本序号取模切分数据，适合多进程或多机器并行生成。
- `--dry_run`：打印第一条样本实际发送给模型的 prompt，然后退出，不调用 API。
- `--no_fallback_on_error`：API 调用或响应解析失败时直接报错，不使用 fallback 拼接。
- `--no_extra_body`：不向请求中发送 `enable_thinking=False` 相关的 `extra_body`。

## 注意事项

PowerShell 的续行符是反引号，不是 `^`。如果在 PowerShell 中使用 `^`，会出现类似 `MissingExpressionAfterOperator` 的解析错误。

如果 `debug_output` 中出现 `fallback:APIConnectionError:Connection error.`，说明没有连上 API，脚本使用的是 fallback 拼接结果，而不是模型润色结果。此时需要检查 `--api_base_url`、`--api_model`、API key 和网络连接。调试时可以加 `--no_fallback_on_error`，让错误直接暴露出来。

重新生成样本时建议使用新的输出路径，或者加 `--no_resume`。否则脚本会跳过输出文件中已经存在的 id。
