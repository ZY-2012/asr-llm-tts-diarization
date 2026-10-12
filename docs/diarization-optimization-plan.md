# 说话人分离与 ASR 链路优化计划

> 更新时间：2026-10-12
> 配套分析：`docs/diarization-pipeline-analysis.md`（根因编号 L1~L7 / D1~D3 /
> M1~M8 均在分析文档中定义，本文直接引用）
> 目标：在 AX650 实时约束下，把"漏词、重复、说话人匹配错误"三类症状的
> 可修项全部落地。优先级按 **收益/成本比** 排序，不按模块顺序。
> 验收基线：AliMeeting R8001 + AMI EN2002d（`build/der-eval/`）与 2P/QA
> 听音集（`build/gt-compare/`、`build/3dspk/`）。

---

## 优先级总览

| 优先级 | 工作项 | 修复根因 | 预期收益 | 成本 |
| --- | --- | --- | --- | --- |
| P0-1 | shutdown 时序修复 | M3 | 消除退出期假超时+伪 ID | 极低（十几行） |
| P0-2 | 回退链改用降噪前音频 | M2 | 回退不再聚类塌缩 | 低 |
| P0-3 | part 切片 padding 0.3 s | L3 | solo 漏字 −1~3 pp（已验证） | 低 |
| P0-4 | 投影后碎片合并 + 最短 part 阈值 | L5 | 消灭亚秒 ASR 调用 | 低 |
| P1-1 | 跨块 ID 全局对齐（替代贪心） | M1 | 流式 conf 22%→目标 ≤5% | 中 |
| P1-2 | VAD 阈值 0.5→0.3 落配置 | L2 | Ali DER −15.8 pp（已验证） | 极低 + 噪声实测 |
| P1-3 | SpeakerTable 质量分级与延迟合并 | M1/M5/M6 | 消灭临界带新建 ID | 中 |
| P2-1 | 换人点投影修复（run 边界归属） | M4 | 归属错显著下降 | 中 |
| P2-2 | 30 s 强切静音切点 | L6 | 长轮尾部漏字 | 中 |
| P2-3 | 短句处理策略重做 | L4 | 附和不再丢/滞留 | 中 |
| P3-1 | 丢帧标记与时间轴补偿 | L7/M7 | 负载下时间戳不失真 | 中 |
| P3-2 | 接缝重复检测（只标不删） | D1/D2 | 重复可审计 | 低 |
| P4 | 重叠说话人可重叠标签 | L1/M8 | overlap delete 大项 | 高（模型/口径） |

---

## P0-1 修复 shutdown 时序（ResultWorker / SessionDiarizer）

**问题**：`ResultWorker::Close()` 先 `session_->Stop()`（置 `closed_=true`），
再 `ProcessRemaining()`。退出期所有话段的 `WaitCovered` 被立即唤醒且
`committed_ < end` → 打出误导性的 "timed out" 日志并走逐段 raw 回退，
而尾块随后才推理完成。T3 中退出期回退新建了 speaker-6（M3）。

**方案**：

1. `Close(abort=false)` 时把顺序反过来：**先 `ProcessRemaining()`，再
   `Stop()`**；`ProcessRemaining` 内部等待 chunk 推理完成是预期行为
   （退出延迟 +12 s，可接受）。
2. `WaitCovered` 区分三种返回：`covered` / `timeout` / `closed`，前两种
   才允许回退；`closed` 时应**等待尾块**而不是回退。
3. 日志文本改为 `session shutdown before coverage`，与真实超时分开统计。

**验收**：T3 复跑，退出期不再出现 per-segment fallback；表内 ID 数 ≤ 4；
退出总耗时增量 ≤ 尾块推理时间（~12 s）。

## P0-2 回退链改用降噪前音频（Accumulator / ResultWorker）

**问题**：逐段回退③raw 窗、④trimmed 用的是 FastEnhance 之后的 16k 音频，
隔离实验证明降噪流会让聚类塌缩成 1~2 簇（M2）。session 模式已经在用
`diar_decimator`（降噪前）音频，回退路径没有。

**方案**：

1. `Pipeline::Process` 在送 VAD 的同时，把 `diar_decimator` 输出的同一个
   16k hop 同步喂给 `Accumulator`（新增 `PushDiar(hop)`），在
   `Segment` 里维护第二组 `raw_samples_diar`（与现有 `raw_` 平行，
   光标用同一 `cursor_`，天然对齐，顺带解决 M7 的对时疑虑）。
2. ResultWorker 的③回退改用 `raw_samples_diar` 写 `*.raw.wav`。
3. ④trimmed 回退（对拼接音频做 diarization）建议直接删除——拼接音频
   本身就是 diarization 的病态输入（静音被删、相位断裂），保底链里
   有③和整段 ASR 已够。

**验收**：构造"session 超时"场景（把 `session_max_wait_sec` 设 0 复跑 T3），
逐段回退的 clusters 应恢复 3~4 而非 1~2；不再从回退路径新建 ID。

## P0-3 part 切片 padding 0.3 s（ResultWorker）

**问题**：`ResultWorker::Loop` 按 part 边界硬切、零上下文。label_transcript
已用 GT 证明 `--pad 0.3` 带来 equal +2.5 pp、solo delete 3.4→2.4，且
solo 漏字的 13.2% 是边界截断（L3/L5）。gt-compare README 明确标注该修复
需要同步到框架链路。

**方案**：

1. 切片时向两侧各外扩 0.3 s（配置项 `diarization.part_pad_sec`，默认 0.3），
   **朴素外扩、不跨相邻说话人钳制**（GT 实验证明钳制反而掉 2 pp：邻说话人
   方向的 0.3 s 更多时候是本句字头）。
2. 与 label_transcript 的差异：part 的两端本来就是同一段拼接音频，外扩后
   **相邻 part 之间存在 0.6 s 重叠区**，识别文本会双份——因此必须与 P0-4
   的"接缝归属"一起做：文本行仍按未外扩的 part 边界输出。
3. 注意 padding 会推高 insert（label_transcript 实测 +88），上线后重点
   监控紧接重复扫描指标（P3-2）。

**验收**：R8003 全片 token 对比：equal ≥ 基线 +2 pp，solo delete ≤ 2.5%，
insert 增幅与 label_transcript 同量级（可控）；2P "以。"案例修复。

## P0-4 投影输出碎片整理（MapTimelineToTrimmed / ResultWorker）

**问题**：`MapTimelineToTrimmed` 输出无最短 part 合并（引擎内部的
0.4 s 吸收发生在源坐标，投影后仍会切出 <0.4 s 甚至 <0.2 s 的碎片）；
亚秒级 WAV 直接进 SenseVoice 不可靠（"以。"案例，L5）；相邻同人 part
不合并导致一句被切多次。

**方案**（全部在投影输出后、写 WAV 前处理，不动 diar 引擎）：

1. **同人 part 合并**：相邻且同 `speaker_id` 的 part 直接合并。
2. **短 part 并入**：时长 < `diarization.min_part_sec`（默认 0.4 s，
   对齐引擎 `MIN_PART_SAMPLES`）的异标签碎片，并入**时间上更长的邻 part**
   （比引擎的"并入前一个"更稳）；段首/段尾碎片并入唯一邻居。
3. **保底**：合并后仍 < `min_part_sec` 的（整段都极短），不单独送 ASR，
   与相邻 part 拼成一个多标签 part 送"整段 ASR + 主说话人标签"，
   或者直接走整段无标签转录（宁可少标签不可漏字）。

**验收**：Ali session 模式转录行数下降（旧数据短碎片 51~65 行 → 目标 <20），
`<3 字符`行数显著减少；DER 不劣化（投影只影响切片不影响 RTTM 口径时用
parts 复评分确认）。

## P1-1 跨块 ID 全局对齐（diar_worker / SpeakerTable）

**问题**：每块独立聚类，簇→持久 ID 靠贪心表匹配（weight 序 + taken[]），
新建 ID 不可逆。T1 conf 22.6%、T3 6 ID；chunk2 中 44.3 s 的簇余弦 <0.55
被新建为 speaker-5（M1）。

**方案**（按成本递增，先做 1+2，3 为备选）：

1. **贪心改匈牙利**：`Assign()` 内对本块 K 个簇 × 表内 N 个说话人的余弦
   矩阵做全局最优分配（K≤16、N 常见 ≤10，匈牙利或贪心+交换改善都行，
   微秒级）；低于阈值的分配结果不落表，直接走"新建"判定。
2. **块间重叠上下文**：每块向前多带 `diarization.session_overlap_sec`
   （默认 15~30 s）音频参与聚类，但只提交非重叠区的 parts。这样
   本块聚类能看到上一块结尾的说话人，VBx 的初始标签更稳，跨块质心
   余弦整体抬高（同时天然给了一个"上一块 ID 直通"的先验：重叠区的
   标签可以直接继承上一块结果）。
3. **滚动联合重聚类（备选）**：每块结束后把"最近 N 块质心 + 表内活跃
   质心"合并做一次小规模 AHC，再把块内簇映射到联合簇。成本高、仅在
   1+2 不达标时做。

**验收**：T3/T1 全片：ID 数 = 实际人数（4），conf ≤ 5%，无等待超时堆积；
speaker-5/6 类"44 s 大簇被新建"不再出现（44 s 簇必须匹配到既有 ID）。

## P1-2 VAD 阈值落配置 + 场景化默认（config / apm-config / UI）

**问题**：实验已证 0.3 使 Ali DER 55.38→39.63、覆盖率 61%→90%，
但 `sdk-config.yaml`/`default-config.yaml` 仍是 0.5（L2）。

**方案**：

1. 默认值改 0.35~0.40（0.3 的 FA 已从 1.41→2.95 pp，噪声实测前不激进），
   并在 `docs` 与 UI 说明"安静远场 0.3 / 嘈杂近场 0.4~0.5"的调法。
2. UI 暴露 `vad.threshold` 与 `vad.min_speech_ms`（后者与 P2-3 联动）。
3. 增加一条板端噪声实测：会议室空调/风扇底噪下 0.35 的假话段率，
   用 transcript 行数 + MP3 时长监控。

**验收**：Ali/AMI session 模式 DER 复测达到 §4.1 的 0.3/0.4 行水平；
真实环境一周试运行无假话段刷屏。

## P1-3 SpeakerTable 质量分级与延迟合并（speaker_table.cpp）

**问题**：短段退化质心与正常 chunk 质心同台竞争（M5）；阈值单一（M6）；
新建 ID 不可逆导致同人永久分裂（M1 后半）。

**方案**：

1. **质心分级**：`Assign()` 增加来源标签（chunk / segment-fallback /
  short-<10s），质心带 `quality` 权重；回退短段质心只在没有任何高质量
   候选时参与匹配，且**默认不允许触发新建 ID**（新建仅允许来自
   session chunk 结果）。
2. **双阈值 + 滞回**：高置信（≥0.65）直接合并；中间带（0.45~0.65）
   先挂"临时 ID"，待第二个 chunk 复现再转正或与最近临时候选合并；
   <0.45 才允许 chunk 级新建。避免一次抖动永久分裂。
3. **表内合并**：提供 `Merge(a,b)`（把 b 的质心/计数并入 a），由 1.1 的
   匈牙利分配在"块内多簇映射到同一表内说话人"时自动触发；表文件保留
   `merged_into` 字段以便回滚审计。
4. **质心老化**：`samples` FIFO 改为按 `last_seen` 加权（超过 30 分钟的
   质心降权），`center` 滑动均值加遗忘因子，防长会议漂移。

**验收**：T3 复跑 ID 数 = 4；0.55~0.62 临界带匹配（speaker-1@0.613、
speaker-6@0.561/0.553 类）全部消除或转为可审计的临时 ID。

## P2-1 换人点投影归属修复（MapTimelineToTrimmed）

**问题**：最近标签继承会把换人点归错边（M4）；时间线端点有帧量化抖动。

**方案**：

1. **run 边界优先**：投影时不再先做标签填充；先在源坐标上把每个 run 的
   区间与时间线求交，**run 的标签由该 run 内时间线覆盖时长最长的说话人
   决定**（整 run 单标签）；跨 run 的填充只发生在时间线完全无覆盖的 run。
2. **换人点钳制**：两说话人边界落在 run 中间时，边界两侧各留 1 帧的
   "归属不确定区"，按时间线在不确定区内的覆盖比例分配（而不是简单
   向两侧渗透）。
3. 配置 `diarization.boundary_guard_sec`（默认 0.08，约 5 帧）内不产生
   新 part 边界，防止帧级抖动刷碎片（与 P0-4 叠加生效）。

**验收**：构造两轮紧挨着换人的短对话，转录中换人行不再互串 1~2 字；
R8003 "归属错"（solo 非 delete 的 1.6%）下降一半以上。

## P2-2 30 s 强切改静音切点（Accumulator）

**问题**：`max_` 一到立即 Emit，切口可落字中（L6）。

**方案**（与 label_transcript 的 `--chunk-search` 同思路，但注意 GT 实验
显示静音切在其工具里净收益为负——因为它当时与 pad 钳制问题纠缠；在
主管线里配合 P0-3 的 padding 一起上）：

1. 累积到 `max_ - search_win`（默认 4 s）后开始记录最近的 VAD 静音 hop；
   到 `max_` 时若有记录的静音点则提前在静音点切，否则仍在 `max_` 硬切。
2. `Segment` 增加字段标记"max-split 于静音点"，该信息写日志用于
   上线后统计两种切口的文本差异。

**验收**：2P `[3.052 34.844]` 长轮尾部案例在主管线复现并修复；
R8003 长跨切口句子的 delete 下降。

## P2-3 短句处理策略重做（Accumulator）

**问题**：<615 ms burst 被丢弃，或滞留到退出（L4）；不足 `min_segment`
的段不关闭导致时间戳漂移。

**方案**：

1. `min_speech_ms` 从 615 ms 降到 ~250 ms（TEN-VAD 单 hop 16 ms，
   250 ms ≈ 16 hop，附和词长度），配合 P1-2 的阈值下调一并回归 FA。
2. 静音达到 trailing 且 burst 不足 `min_segment` 时**也收段**（短段照发），
   消除"滞留到退出"；`min_segment_seconds` 仅用于控制"尾部静音计时器
   启动"，不再作为发送门槛。
3. 段内仍丢弃极短噪声毛刺（<100 ms）防 FA。

**验收**：短附和（"嗯/对/啊"）在实时 transcript 中按正确墙钟时间出现；
短行数量与 GT 轮数差距缩小；FA 行（噪声）无明显上升。

## P3-1 丢帧标记与时间轴补偿（Pipeline / BlockingQueue / Accumulator）

**问题**：实时链路三级队列（APM `recorded_queue_`、`vad_jobs_`、
monitor）满时静默丢最旧样本，`cursor_` 无感知，时间戳与 session 对齐
漂移（L7/M7）。

**方案**：

1. `TryDropOldest` 返回丢弃数量，`Pipeline::Process` 累计并周期性把
   "补偿静音 hop"注入 `Accumulator` 与 `SessionDiarizer::Append`
   （时长对齐 = 丢弃样本数），使 `cursor_` 与真实时间保持一致；
   日志输出 `compensated_samples`。
2. APM 侧 `dropped_recorded_samples` 已有计数，将其写进退出摘要并在
   >0 时打印醒目告警。
3. 【对时实验】用脉冲测试音频测 `diar_decimator` vs `decimator` 的
   样本级相位差，若非零，在 SessionDiarizer 坐标上做一次性校正。

**验收**：人工制造负载（开启 LLM/TTS 长回复）下，transcript 时间戳与
墙钟误差 < 100 ms；compensated 计数出现在退出摘要中。

## P3-2 接缝重复检测（ResultWorker，只标不删）

**问题**：接缝重复（D1/D2）与真实口吃（D3）无法区分，label_transcript
的 `--dedupe` 已证明盲删会误伤。

**方案**：

1. P0-3 的 padding 使相邻 part 文本存在已知重叠区；对相邻两行 transcript
   做"1~4 字单元紧接重复"扫描（QA-2P 已有同款脚本逻辑，可移植为
   C++ 轻量版或运行页脚本），在输出行尾附加 `[seam?]` 标记。
2. **不做自动删除**；运行状态页（runtime-ui）增加"重复标记"过滤视图，
   供人工抽检；积累一批后再评估是否对 seam 处的重复做定向删除。
3. 30 s 强切（P2-2）与 VAD join 点也纳入 seam 标记来源。

**验收**：QA-2P 的 35 处紧接重复中，系统能标出全部 7 处"疑似伪影"类
（与听音定案结果对齐），无误标真实口吃为伪影的记录。

## P4 重叠说话人可重叠标签（diar_engine / 口径）

**问题**：parts 单标签分区使重叠区第二说话人内容整体不送 ASR
（overlap delete 15~49%，文本漏词最大项 L1）；DER 口径也低估 miss。

**方案**（分三步，成本递增，此档只做 1）：

1. **口径先行**：`Run()` 已输出原生可重叠 rttm；把 DER 评分切到 native
   RTTM 口径（parts 只用于切片），先把重叠质量纳入回归指标（M8）。
2. parts 投影输出"次说话人"字段（重叠帧的第二长运行簇），ASR 切片仍用
   主说话人，但转录行标注 `[overlap]`，供下游与评测使用。
3. 重叠区双解码（对重叠 part 再跑一次以次说话人窗为中心的嵌入 +
   ASR）需要模型/算力再评估，不在本计划承诺。

**验收**：native 口径 DER 建立基线并进 `score_p1.py`；
`[overlap]` 行覆盖率与 GT 重叠占比（Ali 48%）同量级。

---

## 实施顺序与依赖

```text
第一批（纯管线，无模型改动，1~2 天）
  P0-1 → P0-2 → P0-3 + P0-4 + P3-2标记（同一改动集，一次回归）
  P1-2（改配置 + 噪声实测排期）
第二批（算法，3~5 天）
  P1-1(1) 匈牙利 → P1-3 质量分级/双阈值 → T3/T1 全片回归
第三批（体验，2~3 天）
  P2-1 投影修复 → P2-2 静音切 → P2-3 短句 → P3-1 丢帧补偿
第四批（口径与长线）
  P4-1 native DER → P4-2 overlap 标注 → （P1-1(3) 视 P1-1(1)+(2) 结果）
```

每批回归统一跑：

```bash
# DER（Ali/AMI，session + segment 两种模式）
python3 build/der-eval/score_p1.py
# 文本质量（R8003 + 2P 听音集）
python3 build/gt-compare/compare_txt.py --gt ... --sys ... --lang zh
# 流式专项
run-t3.sh（240 s 快迭代）→ run-t1.sh（Ali 26 min 全片）
```

**合并验收目标**（相对 2026-10-12 基线）：

| 指标 | 基线 | 目标 |
| --- | --- | --- |
| Ali session DER（VAD 0.4） | 46.70 | ≤ 42 |
| AMI session DER | 40.60 | ≤ 38 |
| 流式 T1/T3 conf | 21.8~22.6% | ≤ 5% |
| 流式 ID 数（4 人会） | 6 | 4 |
| R8003 equal / solo delete | 79.3% / 2.4% | ≥ 81% / ≤ 2% |
| 短碎片行（<3 字符） | ~50 行 | < 20 行 |
| 退出期假超时回退 | 每次必现 | 0 |

## 风险与回滚

- 所有行为改动走配置开关（`part_pad_sec`、`min_part_sec`、
  `session_overlap_sec`、`boundary_guard_sec`、`chunk_search_sec`、
  `min_speech_ms`），默认值可一键回退到当前行为。
- SpeakerTable schema 变更保持向后兼容（旧 JSON 可读；新字段
  `quality`/`merged_into` 缺省视为 legacy），部署前备份 `speakers.json`。
- P1-3 的双阈值涉及主观点，必须在 Ali/AMI 全片 A/B 后再定数值，
  不与代码一起上线。
