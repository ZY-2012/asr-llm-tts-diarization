# 实时会议链路：模块输出格式、时间线与延迟极限分析

> 分支 `diart-incremental`，2026-10-14。数据来自板端 2P 60s 截断 file-realtime 实测
> （`config-2pw-inc-preview3.yaml`：`max_segment_seconds=10`、`preview_seconds=3`、
> session chunk=step=10、增量聚类）。暂不涉及 LLM / TTS。

## 1. 架构与数据流

```
麦克风/文件 16kHz PCM
   │
   ├──► FastEnhance 降噪(48kHz) ──► 降采样16k ──► TEN-VAD(每16ms 有声/静音)
   │                                                   │ voiced hops
   │                                                   ▼
   │                                          Accumulator 分段器
   │                                     ┌──── 预览段(累计3s语音, 每段一次)
   │                                     └──── 正式段(1s静音结束 或 满10s)
   │                                                   │
   │                                                   ▼
   └──► diar_decimator ──► SessionDiarizer ──► 说话人时间线 ──► ResultWorker 编排器
        (原始音频, 不经降噪)   每满10s送一个窗                  │ 预览段: 直接整段ASR
                              给 diar-worker 子进程            │ 正式段: 等时间线覆盖段尾
                              (分割+embedding+增量聚类)        │        → 按说话人切片 → ASR
                                                               ▼
                                              SenseVoice ASR (NPU2)
                                                 │                 │
                                transcript_streaming.txt     transcript.txt
                                 pending@时间 文本            speaker-N@时间 文本
```

两条并行支路：①降噪→VAD→分段→ASR；②说话人分离。二者在 ResultWorker 汇合：
正式段必须等说话人时间线覆盖到段尾，才能切片送 ASR。

## 2. 时间线（会议开始 t=0，音频时间轴）

| 时刻 | 模块 | 事件 | 日志/输出 |
|---|---|---|---|
| 每16ms | TEN-VAD | 判有声/静音 | 内部 bool，不外吐 |
| ~3.0s | Accumulator | 预览段 | `[vad] preview submitted duration=3.008s` |
| ~3.3s | ASR | 预览转写（无说话人） | `pending@… 哎，小哥听得到不可以听到啊，是这样。` |
| ~10.0s | Accumulator | 首个正式段（满10s） | `[vad] segment submitted duration=10s` |
| ~10.2s | 说话人分离 | 首个窗提交 | `session window start=0s … parts=3 committed=10s wall=+10.2238s` |
| ~10.5s | ASR | 按说话人切片转写 | `speaker-1@… 哎，小崔听得到不？` / `speaker-2@… 可以听到。` |
| 每10s | 说话人分离 | 下一个窗提交 | `wall=+20.21s / +30.21s / …`，单窗推理 ~0.23s |
| 不定 | Accumulator | 1s静音提前结束或满10s | 每段一次预览 + 一次正式 |

## 3. 各模块输出格式

| 模块 | 输出 | 格式示例 |
|---|---|---|
| TEN-VAD | 每 16ms 一个 voiced bool（内部） | — |
| Accumulator | `Segment`：去静音拼接 PCM `samples`、`runs`（拼接偏移↔源偏移）、保留静音的 `raw_samples`、`is_preview`、`wall_start/end` | 段 WAV `20261014-10-44-45-380_20261014-10-44-53-252.wav` |
| 说话人分离（每窗） | 窗内说话人片段 + 聚类日志 | `[diar] wav=session-0.wav audio=10.00s clusters=2 parts=3 took=242ms`；`created new speaker-1` / `matched speaker-1 cosine=0.953` |
| 说话人分离（每段） | `<段>.wav.parts.txt`，单位 16k 采样点 | `3` / `0 23040 speaker-1` / `23040 34048 speaker-2` / `34048 160000 speaker-1` |
| 说话人档案 | `speakers.json` | `{"speaker_id":"speaker-1","sample_count":537198,"last_seen_time":…,"embedding_center":[256 维]}` |
| ASR 预览 | `transcript_streaming.txt` | `pending@261014_10-44-45-380 哎，小哥听得到不可以听到啊，是这样。` |
| ASR 正式 | `transcript.txt` | `speaker-1@261014_10-44-45-380 哎，小崔听得到不？` |
| 附属 | `runs.txt` / `joins.txt` | `# concat_offset source_offset` / `256 512` |

## 4. 为什么「说话人+文字」现在是 ~10.5s

说话人+文字 = 说话人时间线覆盖了这段话的段尾 + 这段的 ASR。下面三个因素叠加：

1. **分割模型窗长固定 10s。** community-1 的分割模型输入固定 `WINDOW_SAMPLES`=10s（输出 589 帧）。
   首个窗必须攒满 10s 真实音频，所以冷启动首标签 ≥10s。
   试过 `session_first_window_sec=5`（不足部分补零）：首标签提前到 5.2s，但后续窗无法匹配首窗的质心，
   60s 里多出 speaker-3/4 两个幻影说话人。分割前的均值/方差归一化覆盖整窗（含补零区），
   embedding 也算在补零的整窗 fbank 上，这两点是推测的原因，还没逐项验证。
2. **提交步长 = 10s，窗与窗不重叠。** 时间线只在 10s、20s、30s… 推进。这一点影响**稳态**，不只冷启动：
   一句在 t=11s 结束的话要等 t≈20.2s 那个窗提交后才有说话人，等待 0–10s，平均约 5s。
   之前说的「稳态标签延迟 ~0.2s」只是窗尾到提交的推理时间，不是每句话的延迟。
3. **ResultWorker 等整段被覆盖。** 正式段要等时间线覆盖到段尾（段最长 10s）才开始切片和 ASR，
   段内靠前的话也要跟着等。

所以 10.5s 是冷启动的值；稳态下每句话「说完 → 有说话人+文字」大约是 0–10s 的等待 + ~0.3s ASR，不是固定 0.2s。
单窗推理只有 ~0.23s（NPU，RTF 0.023），**算力不是瓶颈，窗长和步长才是**。

## 5. diart「~1s 出结果」：哪些是真的，哪些有前提

源码：`diart/src/diart/operators.py`、`blocks/aggregation.py`、`blocks/base.py`、`utils.py`。

**真的部分：** 5s 滚动窗、0.5s 步长、每步算量恒定；`DelayedAggregation` 每步只输出
`[buffer_end - latency, buffer_end - latency + step]` 这一小段，所以**稳态**下每 0.5s 的音频在约 latency（0.5–5s）后拿到说话人。
这正是我们缺的：步长短、窗重叠。

**有前提、宣传里不显眼的部分：**

| 点 | 源码依据 | 实际含义 |
|---|---|---|
| latency 是稳态算法延迟，不是首标签时间 | `rearrange_audio_stream` 里 `ops.filter(has_samples(chunk_samples))`：chunk 不满 5s 不输出 | 麦克风冷启动要等满 5s 才有第一个标签 |
| 文件评测把冷启动藏掉了 | `get_file_padding` → `get_padding_left`：流比 chunk 短时在左侧补零 | benchmark 的 DER/延迟不反映麦克风冷启动 |
| 首个 buffer 特殊处理 | `aggregation.py` `_prepend` 带 `FIXME`：首 buffer 直接把 `[0, end]` 整段输出 | 开头一段是一次性给出的，不是逐 0.5s |
| latency 不含计算时间 | README 的 11ms/26ms 是 Ryzen 9 / RTX 4060 上单独测的；`Benchmark` 预先批量算模型输出 | 慢设备上实际延迟 = latency + 推理 |
| 低延迟换精度 | README Figure 5 / 论文：延迟从 5s 降到 0.5s，DER 上升 | 0.5s 可用，但不是 5s 的精度 |
| 只有说话人，没有文字 | diart 不含 ASR | 说话人+文字还要等 ASR 有足够上下文；0.5s 片段 ASR 质量差 |
| 模型不同 | 默认 `pyannote/segmentation`（2021，5s 窗） | 我们用 community-1（10s 窗），不能直接套 5s |

结论：「~1s」指**稳态**下说话人标签的算法延迟，这部分不夸张，也是我们可以借鉴的。
但它不是「开口 1s 出说话人+文字」：麦克风冷启动 ≥5s，计算时间另算，文字也不在里面。

## 6. 还能怎么提速（按收益/风险排序）

| 方案 | 预期 | 风险/代价 |
|---|---|---|
| A. **滚动窗：窗 10s、步长 2s（已实测，2026-10-14）** | 稳态说话人延迟从平均 ~5s（最坏 10s）降到 ~2s + 推理 | 推理占空比 ~12%（0.24s/2s）；重叠窗下增量质心重复计数与 diart 同构（diart 每步也把整窗 embedding 累加进质心），2P 60s 实测恰好 2 人、无幻影；`session_step_sec` 基础设施 P1b 已备好，纯配置 |
| B. **ResultWorker 按片提交（已实现，见 §7）** | 段内靠前的说话人片一被覆盖就切片送 ASR，不等段尾 | 已与 A 叠加实测 |
| C. 预览段临时说话人 | 说话人档案建立后（>10s），预览音频只和已有质心比对、不新建说话人，~3s 出「临时说话人+文字」，正式结果再修正 | 3s 短音频 embedding 偏弱，需验证正确率 |
| D. 冷启动 <10s | 补零改为只用有效样本做归一化，或用反射填充 | 实验性；m5f5 已证明朴素补零不可行 |

目标已基本达成（A+B，2P 60s 板端实测）：冷启动 10.2s 不变（10s 模型窗 +
0.24s 推理），首条说话人+文字 11.1s；**之后每句「说完到有说话人+文字」
10.5–20.5s → 2.5–2.9s**（段尾 1s 静音判定 + 2s 步长 + 0.24s 推理 + ~0.3s
ASR）。文本、说话人归属、段边界与 step=10 逐字一致，无幻影说话人。

## 7. Plan B 已实现（2026-10-14）：段内逐片释放

**问题**：首条「说话人+文字」实测 20.3s，不是预期的 10.5s。根因是 VAD 段与
diar 提交窗错位：首个 VAD 段攒满 10s 有声后于 t≈10.9s 提交，其源时间轴终点
（静音被剔除）延伸到 10.94s；而首窗 diar 只覆盖到 10.0s，必须等第二窗 20.2s
提交。段内所有内容（包括 0–2s 的开头两句）都被段尾拖住。

**改动**（`framework/audio_pipeline.cpp`）：
- `SessionDiarizer` 新增 `Committed()`/`Finished()`；`ConcatCoveredBySource()`
  将 diar 覆盖前沿（源采样点）映射回拼接坐标。
- `ResultWorker::StreamSessionParts()`：不再等整段被覆盖，而是每当 diar 提交
  一次，就把「已覆盖前缀内、且距覆盖前沿 ≥0.3s（kMinPartSamples，防后续
  re-merge/absorb 改变切分）」的完整片段立刻切片送 ASR，写 transcript、发事件。
- 片段转写抽成 `TranscribePart()`，预览与正式路径共用。

**结果**（2P 60s，events.jsonl 实测）：

| 指标 | 改前 | 改后 |
|---|---|---|
| 首条说话人+文字 | 20.35s | **11.07s** |
| 说话人+文字平均延迟 | 9.4s | 7.0s |

文本与说话人归属逐字一致（无精度变化）。剩余延迟来自跨提交边界的段尾，
需 Plan A（滚动窗 step 1–2s）继续压。

**首字时间澄清**：预览首字 = preview_seconds 的**有声**样本攒满时刻 + ASR
（~0.2s）。2P 开头含 ~0.9s 静音，故 4.1s 而非 3.3s；连续说话时 ~3.3s。
冷启动「说话人+文字」下限 = 10s 模型窗 + 首片 ASR ≈ **11s**。

## 8. 实时演示页（tools/live-demo）

`tools/live-demo/server.py`（板端，仅 Python 标准库，端口 8090，无认证，内网用）
+ `index.html`：仿 diart 演示，按板端真实产出时刻回放 2P 音频，同步展示
说话人时间轴（含参考标注对照）、预览文字、说话人+文字、波形与 now 游标。
数据源是 pipeline 新增的 `events.jsonl`（start/diar/preview/final/end，含
板端时刻与源音频时间，写在 transcript.txt 旁）。

