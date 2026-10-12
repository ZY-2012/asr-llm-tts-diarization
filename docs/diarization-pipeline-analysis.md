# ASR→Diarization 链路痛点分析：漏词、重复与说话人匹配错误

> 更新时间：2026-10-12
> 工程根目录：`asr-llm-tts-chain`
> 范围：ALSA/APM 前端 → FastEnhance → TEN-VAD → 话段累积 → pyannote 说话人分离 →
> 时间轴投影 → SenseVoice 切片识别（**不含 LLM 与 TTS**）。
> 方法：全部结论均来自源码走读（`framework/audio_pipeline.cpp`、`driver/pyannote/src/*`、
> `driver/alsa_briage/*`、`driver/ns/*`）+ 仓库内已有实测数据
> （`build/der-eval/`、`build/gt-compare/`）。标注【实验】的条目为板端实测结论，
> 标注【风险】的条目为源码推断、尚未用录音验证。

---

## 1. 链路总览

```text
ALSA / WAV 文件（48 kHz 或文件采样率，多声道）
        │ 10 ms 帧
        ▼
run-alsa-apm（WebRTC APM：NS(high)/AGC1/高通；AEC 当前关闭）
        │ stdout：单声道 48 kHz S16_LE PCM
        ▼
audio-pipeline 主循环（Pipeline::Process）
        ├─ diar_decimator（63 tap 低通 + 3:1 抽取，48k→16k）
        │     └─ 每逢 16 ms hop → SessionDiarizer::Append（仅 session 模式）
        │        ★ 会话 diarization 音频取自 FastEnhance **之前**
        ├─ FastEnhance（NPU1/LITTLE，48 kHz，512-sample hop，GRU 状态流式）
        │     └─ decimator（48k→16k）→ vad_jobs_ 队列（容量 ≈ input_buffer_ms/hop_ms）
        └─ VadLoop 线程：TEN-VAD（16 ms hop，threshold 0.5）→ Accumulator
              │
              ▼ 话段（Segment）
        ResultWorker 线程（FIFO）
        ├─ 写 WAV（VAD 去静音后的 voiced 拼接音频）+ .joins.txt/.runs.txt
        ├─ 说话人标注回退链：
        │    ① 流式 session 时间线（等 session_max_wait_sec，超时→②）
        │    ② 批量文件 session 预跑时间线（batch 文件输入）
        │    ③ 逐段 raw 窗（含静音，FastEnhance **之后**的 16k 音频）
        │    ④ 逐段 trimmed WAV（voiced 拼接）
        │    ⑤ 全部失败 → 整段 ASR（不带说话人）
        ├─ MapTimelineToTrimmed：说话人时间线 → 去静音拼接坐标
        └─ 按 part 切样本 → 逐 part 写 WAV → SenseVoice RunFile
           → transcript.txt（speaker-N@时间 文本）→ MP3
```

两条 16 kHz 时基并存：

| 时基 | 来源 | 用途 |
| --- | --- | --- |
| diar 时基 | FastEnhance 之前的 48k → `diar_decimator` | session 模式说话人分析 |
| VAD/ASR 时基 | FastEnhance 之后 → `decimator` | VAD、话段、raw 窗（③④回退）、ASR 切片 |

`Accumulator` 内部维护绝对游标 `cursor_`，`runs` 记录"拼接坐标 → 源坐标"映射，
`raw_` 记录从首个 voiced 采样到段尾（含静音）的原始窗口。所有说话人标签最终都要经过
`MapTimelineToTrimmed` 投影到"去静音拼接坐标"上才能用于 ASR 切片——**这条投影链是
匹配错误的主要聚集地**（见 §4.3）。

---

## 2. 模块逐个分析

### 2.1 ALSA 采集与 APM 前端（run-alsa-apm）

**实现要点**

- ALSA 按 `period_ms=10` 周期采集，WebRTC APM 以 10 ms 帧处理（NS level=high、
  AGC1 fixed_digital、高通滤波；AEC 当前 `enabled: false`）。
- 处理后只取左声道（`processed_left`），写入 `recorded_queue_`：
  - 实时模式用 `WriteDroppingOldest`，**队列满时丢最旧样本**并累计
    `dropped_recorded_samples_`（退出日志可见 "dropped N recorded samples"）；
  - batch 文件模式阻塞写满为止（不丢）。
- 文件输入采样率 ≠ 48 kHz 时用 WebRTC sinc 重采样器升/降采样（T3 实测
  16k→48k），realtime 模式还按音频时钟 `sleep_until` 对齐。

**痛点**

1. **静音抑制（NS high）+ AGC 与 VAD 阈值耦合**：NS 把低电平语音压得更低，
   TEN-VAD 0.5 阈值下低声/远场字头被判为静音（Ali 覆盖率仅 61.1%【实验】），
   这些样本在进入 ASR 之前就没了——是"源头漏词"的第一道闸门。
2. **实时队列丢帧无记录**：`WriteDroppingOldest` 丢的是"最旧"样本，下游
   `Accumulator::cursor_` 完全不感知，绝对时间轴出现空洞 → 墙钟时间戳、
   session 时间线对齐全部后移。板端负载重时（NPU2 Cell 被 LLM/TTS 长时间
   占用）最容易触发。
3. **多声道只取左声道**：`processed_left` 只复制 ch0。立体声设备若左右声道
   接不同麦克风，右路信息整链丢弃（当前配置 `capture.channels: 2`）。
4. **文件 16k→48k→APM→16k 往返**：sinc 重采样 + 63 tap 抽取的两级往返
   会在频带边缘引入幅度/相位畸变，且与"批量 session 直接读原始文件"路径
   （§2.8）存在微小时长差 → 时间轴投影偏移【风险】。

### 2.2 音频分发与两条时基（Pipeline::Process）

**实现要点**：每个 48k 输入块先过 `diar_decimator` 生成 16k hop 送
SessionDiarizer；再过 FastEnhance → `decimator` 生成 16k hop 送 VAD。
两个 Decimator 是同构的 63 tap 低通 + 3:1 抽取，代码注释要求"样本对齐"。

**痛点**

1. **两条时基没有公共对时基准**：对齐仅靠"两个滤波器实现相同"这一隐式
   约定，没有逐 hop 序号比对。任何一个 decimator 的 phase/历史状态被单独
   重置（例如 FastEnhance 失败重启路径），时基立刻错位数个样本；session
   时间线与 VAD 话段之间就会发生系统性偏移，说话人标签整体错位
   【风险，需构造实验验证】。
2. **session 音频含全部静音**：`AppendSessionSamples` 不区分 voiced/silence，
   120 s chunk 里有效语音可能只有几十秒，稀释聚类质量并拉高推理时间。

### 2.3 FastEnhance（NPU1 实时降噪）

**实现要点**：1024 点 STFT（hop 512 ≈ 10.67 ms @48k）、幂律压缩、NPU 掩码
推理、两路 GRU 状态跨 hop 流式传递，latency = 1 个 hop（宿主 overlap-add
已验证为恒等，延迟 512 样本 ≈ 10.7 ms；NPU 模型自身引入的延迟未单独测量）。
NPU 占用率实测仅 ~4.7%（T3），性能不构成瓶颈。

**痛点**

1. **说话人特征破坏已被隔离实验实锤**【实验】：120 s 切片 → 原始 4 簇；
   仅 APM 4 簇；APM+FastEnhance **1~2 簇**。FastEnhance 会抹平说话人间的
   韵律/频谱差异，导致嵌入聚类塌缩。session 模式已改用 FastEnhance 之前
   的音频规避；**但逐段回退链（③ raw 窗、④ trimmed）用的仍是降噪后
   音频**——凡走回退，聚类塌缩 + 表匹配新建 ID 就会发生（T3 中 speaker-6
   即由 9.58 s raw 回退创建，见 §4.3）。
2. 对 ASR 是正收益、对 diarization 是负收益，同一链路里两种诉求当前靠
   "回退链用错音频"来妥协，这是设计债。

### 2.4 TEN-VAD 与话段状态机（Accumulator）

**实现要点**：16 ms hop、threshold 0.5、`min_speech_ms=615`、
`min_segment_seconds=1.0`、`max_segment_seconds=30`、`trailing_silence=1.0`。
Accumulator 状态机：

- voiced hop 累积进 `burst_`；静音 gap < trailing 时语音续说，记 join point；
- 累积 ≥ `max_`（30 s）立即强制 Emit（**不找静音切点**）；
- 静音 ≥ trailing 且总量 ≥ `min_segment_` 时收段：burst ≥ 615 ms 才并入
  buffer，否则**整段丢弃**（`[vad] discarded short burst`）；
- `Flush()`（退出时）同样按 615 ms 阈值取舍。

**痛点（漏词的直接来源之一）**

1. **短句丢弃/延迟**：孤立短应答（"嗯"、"对"）不足 615 ms 时，
   - 若后续 1 s 内有人接话 → 并入下一段（时间戳归属错位）；
   - 若再无语音 → 静默滞留在 `burst_` 里**直到进程退出才被 Flush 丢弃**，
     会议中实时 transcript 永远看不到它。gt-compare 结论"系统轮数普遍少于
     GT，短附和常被漏掉"与此吻合【实验】。
2. **30 s 强切不找切点**：`max_` 一到立刻 Emit，切口可以落在字中间。
   下一段从切口后继续，两段 ASR 各自缺少上下文 → 字被切成两半 →
   漏字/错字，且切口两侧可能各自把半个字解码成相似字 → 重复观感。
   label_transcript 的对照实验证明静音切点可修复超长轮尾部
   （"更是。没有。"→"更实用一点？"）【实验】。
3. **min_segment 门槛的副作用**：总量不足 1 s 时段不关闭，短语音被
   无限期滞留（见 1），墙钟时间戳随之漂移。
4. **VAD 召回率是 DER miss 的主项**【实验】：threshold 0.5 时 TEN-VAD
   只保留参考语音 61.1%（Ali）/ 89.0%（AMI）；降到 0.3 后覆盖率 90%/101%，
   DER 从 55.38→39.63（Ali）、41.92→39.84（AMI）（2026-10-12 用
   `score_der.py` collar=0.25 重算）。**当前 `sdk-config.yaml` 仍是 0.5**，
   实验结论未落配置。
5. **去静音拼接本身就是有损变换**：`samples` 是 voiced run 的裸拼接，
   拼接点两侧语音相位不连续，SenseVoice 在接缝处易出伪重复/错字；
   `.joins.txt` 只是记录，识别并未使用。

### 2.5 VAD job 队列与线程调度

**实现要点**：`vad_jobs_` 容量 = `max(4, input_buffer_ms/hop_ms)`（默认
2000/16 = 125 hop）。实时模式下队满 `TryDropOldest`；batch 模式阻塞
（不丢）。VadLoop 串行 `TEN-VAD → Accumulator::Push`。

**痛点**

1. **实时丢 hop 无补偿无标记**：丢掉的不进 `cursor_`，VAD 概率状态机
   （ten_vad 内部）与话段绝对时间同时错位。轻则时间戳漂移，重则
   session 时间线与话段错位 → 说话人标签投影到错误时间。
2. **Accumulator 在 VAD 线程内同步执行**：`Push` 只做拷贝，尚可；
   但 `Emit` 里的 `submit_` → `jobs_.Push` 若 ResultWorker 阻塞在
   NPU2 Cell 锁上（LLM/TTS 长推理），FIFO 无界（`Segment` 直接入队）
   → 内存增长 + VAD 线程被 Push 阻塞 → vad_jobs 反压 → 丢 hop。
   LLM/TTS 关闭时无此问题，全链开启时是实际风险【风险】。

### 2.6 DiarWorker 协议与 diar_engine（pyannote community-1）

**实现要点**

- 常驻进程，stdin 收 "长度+WAV 路径"，fd3 回 `R/E` 帧；一次请求 = 一次完整
  10 s 窗/2 s 步进的分割 + 嵌入 + AHC(0.6) → PLDA → VBx(Fa .07/Fb .8) 聚类；
- 每窗每说话人一个嵌入（C×3×256），"干净帧"过滤（单说话人帧占比
  ≥20% 且 ≥2 帧才进训练集）；
- `Run()` 输出：非重叠 parts（**短于 0.4 s 的碎片并入前一个 part**，
  首尾强制扩展到 [0,N)）+ 每簇 256 维质心；
- 请求全程持有 NPU2 Cell flock（与 SenseVoice/Qwen3/Hojo 串行）。
  实测 RTF≈0.10（120 s chunk 推理 ~12 s）。

**痛点**

1. **请求即"局部聚类"**：每次请求（一个 120 s chunk 或一个话段）内部
   从零聚类，簇标签只在请求内有效；跨请求身份完全依赖 SpeakerTable 的
   余弦匹配——所有跨块一致性风险都汇聚到 §2.7。
2. **短窗退化**：窗数 C 小于 2 时（<12 s 音频），干净嵌入可能只有 0~1 个，
   AHC/VBx 退化为 K=1，质心取"全部嵌入均值"兜底——T3 中 9.58 s raw 回退
   就是 clusters=1，这种质心的代表性极差，却仍会进表匹配流程。
3. **`step_sec=2.0` 的边界量化**：帧步 16.875 ms、窗覆盖 [0,10s)+2s 步进，
   段尾不足一个完整窗时零填充；说话人边界精度受窗中心量化影响，
   单侧最多 ~0.5 窗的标签抖动。
4. **重叠语音不可表达**：parts 是单标签分区，重叠帧按"最长运行"归一个
   说话人（工程上合理，但 ASR 只转写主说话人声道内容）→ 重叠区第二
   说话人的字全部丢失（gt-compare：overlap delete 15~49%【实验】）。
5. **NPU2 串行放大回退概率**：chunk 推理 12 s 期间 Cell 锁被占；若
   SenseVoice/LLM 正在跑 Cell，chunk Label 排队 → `committed_` 推迟 →
   话段等待超时 → 走③④回退 → 降噪音频聚类塌缩 + 新建 ID，
   恶性循环（T3 shutdown 段实证，见 §4.3）。

### 2.7 SpeakerTable 跨段/跨块身份装配

**实现要点**（`speaker_table.cpp`）：

- 持久化 JSON；每个说话人保留最近 **8 个分段质心**，匹配取最大余弦；
- `Assign()`：请求内簇按 weight 降序，贪心遍历，`taken[]` 防止两簇映射
  同一人；无任何余弦 ≥ threshold（默认 0.55）则**新建 speaker-N**；
- 命中时用 weight 滑动平均更新 `center`，并把本次质心压入 `samples`。

**痛点（匹配错误的核心）**

1. **贪心 + taken 的次优分配**：weight 大的簇先挑，可能抢走次优簇的唯一
   最优匹配；后到的簇被迫新建 ID 或错配。没有全局（匈牙利）分配，
   也没有事后合并/分裂修复。
2. **新建 ID 不可逆**：一旦因为余弦略低于阈值新建 speaker-N，后续同
   说话人的块永远有两个候选质心在表里竞争，`taken` 会让同一请求内的
   不同簇分到这两个"同人 ID"上 → **同一人长期以两个 ID 交替出现**。
   T3 实证：chunk2 内一个 44.3 s（weight=708864）的簇与表内 speaker-1~4
   全部余弦 <0.55 → 新建 speaker-5；随后话段与 speaker-1 匹配仅 0.613
   （临界），1.6 s 段回退又新建 speaker-6，其后 5.2 s/1.8 s 段以
   0.561/0.553 悬崖值挂到 speaker-6。整场 4 人会议最终 6 个 ID【实验】。
3. **滑动均值质心漂移**：`center` 按累计 weight 加权更新，长会议后期
   说话人状态变化（疲劳、离麦）会让 center 偏离近期表现，老质心稀释
   新匹配；`samples` 只留 8 个又是无差别先进先出，没有按时间/质量加权。
4. **短段质心污染**：回退链把几秒的退化质心（K=1 兜底均值）也喂给表，
   与正常 chunk 质心同台竞争，进一步压低同人余弦。
5. **阈值单值全局生效**：chunk（120 s）与逐段回退（<10 s）的质心统计
   质量天差地别，却共用 0.55 阈值。

### 2.8 SessionDiarizer（流式会话）与批量预跑

**实现要点**：`Append` 缓冲全部 16k hop；满 `session_chunk_sec`（120 s）
写 WAV → `DiarWorker::Label`（阻塞至推理完成）→ parts 平移到绝对坐标
追加进 `timeline_`，`committed_ = chunk_start+len`；ResultWorker 对每个
话段等 `committed_ ≥ segment_end`，最多等 `session_max_wait_sec`（180 s）。

**痛点**

1. **标签延迟 ≈ chunk 尾部滞后 + 12 s 推理**：话段结束后最坏要等近
   2 分钟才有标签。等待期内 Cell 锁竞争（§2.6-5）会继续放大。
2. **shutdown 语义错误触发回退**【实验，T3 日志逐行核实】：文件播完
   （"Stopped after 240 s"）→ 主线程 `Close()` 先 `session_->Stop()`
   置 `closed_=true`，ResultWorker 里 `WaitCovered` 立刻被唤醒且
   `committed_<end` → 判定"超时"（日志文本有误导性）→ 走逐段 raw 回退；
   随后 `ProcessRemaining()` 才把尾块推理出来。**退出冲刷期所有未覆盖
   话段都被迫走质量最差的回退路径**，且这段时间新建的 ID（speaker-6）
   会永久留在表里。
3. **尾块不足 chunk 长度**：`ProcessRemaining` 要求 ≥1 s 才跑；不足 120 s
   的尾块窗数少、干净嵌入少 → 聚类质量差。
4. **批量预跑与实时路径的坐标换算**：`RunSessionDiarization` 直接把
   `input_wav` 原始文件（可能 16k）喂 worker，再按
   `scale = 16000/source_rate` 换算；而话段坐标经过"文件→48k 重采样→
   APM→抽取→16k"的实时链。两条链的样本数不保证严格相等（sinc 重采样
   长度误差、APM 帧对齐），投影存在几十 ms 级系统偏移【风险】。

### 2.9 MapTimelineToTrimmed 投影

**实现要点**：把说话人时间线（源坐标）经 `runs` 映射到拼接坐标；
重叠时"长区间优先"；未覆盖样本向前/向后**继承最近标签**；输出分区覆盖
整个 trimmed 音频。

**痛点**

1. **最近标签继承 = 跨静音标签渗透**：拼接音频里两段语音之间被删掉的
   静音不存在了，但时间线标签是按源坐标算的；一个 part 覆盖到 run 边界
   时，填充逻辑会把相邻 run 的头部归给上一个说话人——当说话人换人点
   恰在静音附近时，**换人点被系统性归错边**（1~2 个字挂错人）。
2. **无碎片吸收**：与 engine 内部 parts 的 0.4 s 吸收（`MIN_PART_SAMPLES`）
   不同，投影输出**没有**最短 part 合并。engine 的 0.4 s 吸收发生在
   raw 坐标上，经投影后仍可能产生 <0.4 s 甚至 <0.2 s 的 part
   （时间线端点量化 + run 截断）。ResultWorker 对任意短的 part 都
   单独写 WAV、单独调 SenseVoice。
3. **相邻同人 part 不合并**：时间线抖动（同说话人被 1~2 帧的异标签
   打断）会把一句连续的话切成 3 个 part，中间 0.4 s 以下的异标签
   碎片已并入，但**同标签两侧仍是两次独立 ASR**，接缝无上下文。

### 2.10 ResultWorker 切片与 SenseVoice 调用

**实现要点**：parts 逐个 `s.samples[begin:end)` 切样本 → 写 WAV →
`AX_ASR_RunFile`（非流式）→ `speaker-N@时间 文本`。无任何跨 part
上下文、无去重、无最短时长过滤。

**痛点（漏词/重复的直接来源）**

1. **切片零 padding**：`label_transcript` 已用 GT 实验证明 `--pad 0.3`
   是确定收益（R8003 equal 76.8→79.3，delete 14.0→12.4，solo delete
   3.4→2.4）【实验】；且其 solo 根因分类显示 **13.2% 的 solo 漏字是
   边界截断**。主管线的 part 切片是同一模式却**没有移植 padding**
   ——gt-compare README 明确标注"框架链路需要同步 --pad 级别的修复
   （归并行会话）"。这是当前最高性价比的未落地铁赢。
2. **碎片 part 直接进 ASR**：几百 ms 的 WAV 让 SenseVoice 输出不可靠
   （2P 实测："以。"实际是"可以听到。"）——半句话被 ASR 猜成短词。
3. **接缝重复无人处理**：label_transcript 的 `--dedupe` 实验证明硬切
   模式下接缝去重会误删真实口吃（默认已关）；但主管线连"候选重复
   检测"都没有。gt-compare insert 中"重复"占 16.1%（41/254）【实验】。
4. **每 part 一次 RunFile 的开销**：一个 30 s 话段切 12 个 part 就有
   12 次模型调用 + 12 次 WAV 落盘；NPU2 Cell 锁下串行，进一步拖慢
   session chunk 的 `committed_`（§2.8-1 闭环）。
5. **时间戳基准 bug 风险**：`part_start = s.wall_start + begin*1000/kRateOut`
   用的是"拼接坐标毫秒数"，但 `s.wall_start` 已对应 `s.start`（首个 voiced
   的源坐标）——数学上一致；可一旦 §2.5 的丢 hop 发生，`s.start` 本身
   已经偏了，输出的 `speaker-N@时间` 全部失真【风险，依赖丢帧复现】。

### 2.11 SenseVoice 调用方式

- 使用非流式 `sensevoice.axmodel`（`AX_ASR_RunFile`），输入 16k 单声道
  f32 WAV；`language: zh` 固定（配置为 auto 时内部自动判断）。
- 识别质量本身不是短板：gt-compare 11 场会议 solo delete 全部 ≤7.2%
  （最低 2.9%），equal 92.6%（R8003 solo）【实验】。
- 痛点集中在**喂给它的音频**：去静音拼接、无 padding 硬切、碎片 part、
  重叠区单声道混合。

---

## 3. 三大症状根因归纳

### 3.1 漏词

| # | 根因 | 机制 | 证据 | 量级估计 |
| --- | --- | --- | --- | --- |
| L1 | 重叠语音单分区 | parts 单标签，第二说话人内容不送 ASR | 11 场 GT：overlap delete 15~49%，且随重叠占比单调恶化（74% 重叠 → 31.2% delete） | **整场 delete 的大头** |
| L2 | VAD 召回不足（默认 0.5） | 字头/低声在进 ASR 前被删 | 覆盖率 Ali 61.1%；0.5→0.3 使 DER 55.38→39.63（miss 51.87→33.26） | miss 项 18 pp+ |
| L3 | 切片零 padding | 字头/字尾落在 part 边界外 | label_transcript：边界截断占 solo delete 13.2%；pad0.3 使 solo delete 3.4→2.4 | solo 漏字的主要可修项 |
| L4 | 短句丢弃/滞留 | <615 ms burst 被丢或并入下段 | "系统轮数普遍少于 GT，短附和漏掉"（11 场） | 附和/打断类 |
| L5 | 碎片 part 进 ASR | 亚秒级 WAV 识别不可靠 | 2P："以。" vs "可以听到。" | 与 L3 复合 |
| L6 | 30 s 强切 | 切口落字中，两侧 ASR 各丢半字 | "更是。没有。" vs "更实用一点？" | 长轮尾部 |
| L7 | 实时丢帧 | vad_jobs/APM 队列丢最旧，下游无感知 | 代码路径 + "dropped recorded samples" 计数（T3 为 0，重负载下才会出现） | 负载相关【风险】 |

### 3.2 重复

| # | 根因 | 机制 | 证据 |
| --- | --- | --- | --- |
| D1 | part/segment 接缝重复解码 | 切口在字中间，两侧各解码出相似的半字 | label_transcript 基线对照；gt-compare insert"重复"16.1% |
| D2 | 去静音拼接伪影 | join 点两侧相位不连续，SenseVoice 在接缝易复读 | `.joins.txt` 调试目的即此；2P"之后，那么。之后，那么"待听音定案 |
| D3 | 真实口吃与伪影无法区分 | 无听音/无重复检测 | 2P：35 处紧接重复中 28 处为真实口吃（应保留），≤7 处疑伪影 |

### 3.3 说话人匹配错误 / ID 碎片化

| # | 根因 | 机制 | 证据 |
| --- | --- | --- | --- |
| M1 | 逐块独立聚类 + 贪心表匹配 | chunk 簇标签无全局约束；贪心+taken 次优；新建 ID 不可逆 | T1 conf 22.6%；T3 4 人会议 6 个 ID；speaker-5（44.3 s 簇，余弦<0.55 被新建） |
| M2 | 回退链用降噪音频 | ③④回退用 FastEnhance 后音频 → 聚类塌缩 1~2 簇 | 隔离实验：原始/APM-only 4 簇，含 FastEnhance 1~2 簇 |
| M3 | shutdown 时序 bug | `Stop()` 置 closed_ 先于 `ProcessRemaining()`，退出期话段全部"假超时"走最差回退 | T3 日志："timed out"出现在 "Stopped" 之后、尾块完成之前 |
| M4 | 投影标签渗透 | MapTimelineToTrimmed 最近标签继承跨 run 渗透，换人点归错边 | 源码逻辑（§2.9-1）；与 frame 量化叠加 |
| M5 | 短段退化质心入表 | K=1 兜底均值质心参与匹配，压低同人余弦、催生新 ID | T3 speaker-6（9.58 s raw 回退）及其后 0.561/0.553 临界匹配 |
| M6 | 阈值/质心管理粗放 | 单阈值 0.55 通吃所有质心来源；滑动均值漂移；8 质心 FIFO 无加权 | §2.7；T3 匹配值密集分布在 0.55~0.62 临界带 |
| M7 | 两条 16k 时基无对时校验 | session 时间线 vs VAD 话段坐标错位 → 标签整体偏移 | 【风险】需构造实验 |
| M8 | 重叠区主说话人选择 | 重叠帧归"最长运行"，短促插话归错人 | gt-compare："归属错"占 solo 非 delete token 的 1.6%（R8003）；JER 口径未评 |

### 3.4 症状与模块的责任矩阵

```text
                    采集/APM  FastEnhance  VAD/累积  队列  diar引擎  Speaker表  Session流  投影  切片/ASR
漏词-重叠区 L1                                            ●(单分区)                                 ●
漏词-VAD召回 L2        ○(NS)              ●(0.5)
漏词-边界   L3/L5/L6                                                        ○(等待)    ○(碎片)  ●(无pad)
漏词-短句   L4                          ●(615ms)
重复 D1/D2                                                              ○(块边界)            ●(接缝)  ●
匹配错 M1/M5/M6                                  ●(局部)  ●(贪心/新建)
匹配错 M2/M3                  ●(回退源)                                      ●(时序)  ●(shutdown)
匹配错 M4/M7                                                            ○(对时)              ●(投影)
●=主责  ○=次责
```

---

## 4. 实测数据汇总（2026-10-12 复核）

### 4.1 DER（`score_der.py`，collar 0.25，10 ms 网格 + 匈牙利映射）

| 系统 | AliMeeting R8001 (4人, 1574s) | AMI EN2002d (4人, 2210s) |
| --- | --- | --- |
| 逐段 raw 窗（P1） | 76.26（miss 51.9 / conf 22.97） | 53.00（miss 38.2 / conf 13.7） |
| session 批量 + VAD 0.5 | 55.38（miss 51.9 / conf **2.10**） | 41.92（miss 38.2 / conf 2.65） |
| session 批量 + VAD 0.4 | 46.70 | 40.60 |
| session 批量 + VAD 0.3 | **39.63**（miss 33.3） | **39.84**（miss 32.9） |
| 板端整文件（上限参考） | 23.58 | 22.34 |

（注：与 `docs/diarization-progress.md` 旧表个别数字略有出入，以本次
`score_p1.py`/`score_der.py` 重跑输出为准；AMI 0.3 行旧文档记 43.09，
重算为 39.84。）

- session 全局聚类把 conf 从 ~23 pp 压到 2~3 pp：**标签混淆的根因是
  "逐段/逐块局部聚类"，不是模型本身**。
- 剩余 miss（VAD 0.5 时 ~52 pp/38 pp）由 VAD 覆盖 + 单分区不可表达重叠
  构成，其中 0.5→0.3 可收回 ~19 pp（Ali）。
- 实时流式（T1/T3）目前离批量 session 仍有明显差距：conf 21.8%~22.6%，
  ID 碎片化（4 人 → 6 ID）。

### 4.2 文本质量（`gt-compare`，label-transcript pad0.3 定版）

- solo delete 全部 11 场 ≤7.2% → **ASR 模型本身不是短板**；
- overlap delete 15%~49%，与重叠占比强相关 → **重叠处理是文本漏词最大项**；
- 说话人数 9/11 场正确；失败两场为高重叠（R8007-M8010，74% 重叠 → 3/4 人）
  与低话量（TS3003a → 2/4 人）；
- pad0.3 收益（R8003）：equal +2.5 pp、delete −1.6 pp、solo delete 3.4→2.4；
  代价 insert +88（其中"重复"占 insert 的 16.1%）→ **上 padding 必须同时
  监控接缝重复**。

### 4.3 T3 流式会话日志关键事实（`build/der-eval/ali-t3-run.log`）

1. chunk RTF≈0.10：120 s chunk 推理 11.9~12.2 s，NPU 空闲充足
   （FastEnhance util 4.7%，TEN-VAD 2.3%）。
2. chunk1（0~120 s）4 簇全部新建（表为空，正常）。
3. chunk2（120~240 s）3 簇匹配 speaker-1/2/4，但 weight=708864（44.3 s）
   的一簇与全表余弦 <0.55 → **新建 speaker-5**。随后话段与 speaker-1
   匹配余弦仅 0.613 —— 同人跨块余弦已贴着阈值带。
4. 文件播完后 "Stopped" 出现，随后的 30 s 话段报 "session timeline wait
   timed out → per-segment fallback"：这是 **shutdown 时序**造成的假超时
   （`closed_` 先于尾块推理被置位）。
5. 假超时话段走 44.3 s raw 回退：**clusters=1**（降噪音频短窗塌缩实锤），
   匹配了 3 个旧 ID；1.6 s 话段回退 **新建 speaker-6**，其后两个短话段以
   0.561/0.553 挂到 speaker-6。
6. 全场 4 人 → 表中 6 个 ID；MP3 编码因 `/tmp` 挂载问题全部失败（评测环境
   副作用，与链路无关，但说明评测时 output 路径要单独配）。

---

## 5. 症状发生链（把根因串成故事）

**漏词的典型链**：远场低声字头 → NS 压低 → TEN-VAD(0.5) 判静音 → 样本
在 Accumulator 之前消失（L2）；或 voiced 但落在 part 边界 → 无 padding 硬切
→ SenseVoice 只见半个字（L3）→ 输出短词或空（L5）；或两人同时说话 →
parts 单标签 → 副说话人整段不送 ASR（L1）。

**重复的典型链**：30 s 强切/说话人边界正好切在字中间 → 两侧 part 各自
把残缺音素解码成完整词 → 相邻两行出现相同/相似词（D1）；或 VAD 拼接点
相位不连续 → 解码器在接缝复读（D2）；真实口吃再贡献一部分（D3）。

**匹配错误的典型链**：话段结束 → 等 session chunk（最坏 2 分钟）→ 期间
NPU2 被 ASR/LLM 占用 → chunk 推理推迟 → 等待超时 → 回退到逐段 raw
（FastEnhance 后音频）→ 聚类塌缩成 1~2 簇 → 退化质心进表 → 余弦临界
→ 新建 speaker-N → 同人永久分裂为两个 ID（M1+M2+M3+M5 复合）。

---

## 6. 未决项（需要专门实验，不建议拍脑袋下结论）

1. **两条 16k 时基的实际对齐误差**：构造已知间隔的脉冲音频，比对
   `diar_decimator` 与 `decimator` 输出的样本级相位/长度差（M7）。
2. **FastEnhance 模型自身延迟**：宿主 overlap-add 已验证恒等，但 NPU
   掩码模型是否引入额外帧延迟未测（只影响延迟，不影响对齐，因为两条
   时基都从同一 48k 流出发）。
3. **VAD 阈值 0.3 的误唤醒率**：DER 的 FA 只从 1.41→2.95 pp，文本量
   稳定/略增；但真实噪声场景（空调、键盘、投影风扇）的 FA 与
   "假话段触发 ASR 幻觉"需要现场录音验证后再定默认值。
4. **重叠区 JER**：当前只有 DER 口径；`rttm` 原生输出支持重叠，但 parts
   投影会抹掉，重叠质量没有单独指标。
5. **`speaker_match_threshold` 的分布调参**：T3 匹配值密集在 0.55~0.62
   临界带，0.45~0.50 是否会把不同人合并需用 Ali/AMI 全片跑 A/B。
