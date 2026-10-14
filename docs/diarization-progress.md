# 说话人分离优化：进展与计划（campplus → pyannote）

> 更新时间：2026-10-10
> 项目：`asr-llm-tts-chain`（AX650 音频链路）
> 范围：旧模块（CAMPPlus + 逐段聚类）→ pyannote community-1 的替换、链路集成、
> 精度评测与实时接入。
> 说明：本文档合并自两个并行会话的进度记录（实现向 + 评测向）；
> 不同配置/版本的实测数字可能小幅不同，以各自 `config-*.yaml` 与 RTTM 产物为准。

---

## 1. 背景与目标

原链路用 CAMPPlus 在每个 VAD 话段结束后独立聚类，实测说话人混淆严重。
替换为 pyannote community-1 后，最初的"全链路分段"接入仍远差于整文件 diarization。
本次工作的目标：

1. 定量定位差距来源（模型 / 量化 / 去静音 / 逐段聚类 / 前处理）；
2. 在不牺牲实时性的前提下，把说话人正确率提升到接近整文件 diarization；
3. 建立可复现的板端 A/B 评测与文档。

## 2. 结论摘要

### 2.1 差距主因排序

| 因素 | 影响 | 说明 |
| --- | --- | --- |
| **逐 VAD 段独立 diarization** | conf +21~25pp（Ali）/ +10~13pp（AMI） | 段长多 <30s，community-1 在孤立短上下文下退化为 1~2 人；Ali 90 段中 88 段只返回 1 簇 |
| **VAD 去静音 + 非重叠输出** | miss 下限 +20.8pp（Ali）/ +8.5pp（AMI） | TEN-VAD 只保留参考语音的 61.1%（Ali）/ 89.0%（AMI），且分区输出无法表达重叠 |
| **FastEnhance 降噪（实时流式专属）** | 每块聚类 4→1~2 人 | 对 ASR 有益，但破坏说话人特征（2026-10-10 隔离实验）；会话 diarization 已改为从 FastEnhance **之前**取音频 |
| NPU 量化 | +2~6pp（Ali），≈0（AMI） | 次要；NPU2 与 NPU3 输出逐位一致 |

### 2.2 关键指标总表（离线，官方标注，`score_der.py` 同一评分器）

中文：AliMeeting `R8001_M8004_MS801`（1573.85 s，4 人远场）
英文：AMI `EN2002d.Mix-Headset`（2209.9 s，4 人）

| 系统 | 🇨🇳 ±0.125 / ±0.25 | 🇬🇧 no collar |
|---|---|---|
| 旧 campplus（去静音段） | 75.96 / 75.03 | 69.05 |
| 新 pyannote（去静音段，改造前） | 80.24 / 79.47 | 57.14 |
| P1 segment（原始窗口，逐段） | 76.26 / 74.83 | 55.67 |
| **P1 session（整场先分离）** | **55.38 / 52.07** | **45.07** |
| P1 session + VAD 阈值 0.4 | 46.70 | 43.85 |
| **P1 session + VAD 阈值 0.3** | **39.63** | **43.09** |
| pyannote 板端整文件（离线上限，native RTTM） | 23.58 / 20.14 | 25.94 |
| 旧族优化完全体（3D-Speaker +overlap，整文件） | 16.96 | 30.82 |

### 2.3 标签质量分解（仅链路有声帧，VAD 0.5）

| 系统 | 🇨🇳 miss / conf / DER | 🇬🇧 miss / conf / DER |
|---|---|---|
| 改造前 | 28.0 / 39.0 / 69.3 | 32.9 / 17.3 / 51.8 |
| P1 segment | 28.0 / 33.4 / 63.7 | 32.9 / 15.6 / 50.2 |
| **P1 session** | 28.0 / **3.8** / **34.1** | 32.9 / **3.7** / **38.2** |
| 整文件 | 18.2 / 4.5 / 25.5 | 17.5 / 5.7 / 24.8 |

- 整场/全局聚类几乎消除标签混淆；两场会议转录都正确得到 **4 个说话人**
  （逐段模式碎片化成 12 个 ID）。
- 剩余 miss 来自 VAD 覆盖与"单说话人分区无法表达重叠"，靠 diarization 本身继续下降空间有限。

### 2.4 VAD 阈值扫描（session 模式）

| VAD 阈值 | 🇨🇳 覆盖 / DER | 🇬🇧 覆盖 / DER | 转录（中文） |
|---|---|---|---|
| 0.5（原默认） | 63% / 55.38% | 89% / 45.07% | 273 行 / 2935 字 / 短碎片 65 |
| 0.4 | 77% / 46.70% | 94% / 43.85% | 267 行 / 3069 字 / 55 |
| **0.3** | **90% / 39.63%** | 101% / **43.09%** | 267 行 / 2993 字 / 51 |

中文收益最大（-15.8pp）；英文温和（-2.0pp）；FA 略升（1.4%→3.0%/4.2%）；
ASR 文本量稳定/略增，无退化信号。建议上板用现场录音复核噪声场景后再定默认值。

**整场复核（2026-10-14，session 模式，人工标注参考，与上表同口径）**：

| 场次 | vad0.5 | vad0.4 | vad0.3 |
|---|---|---|---|
| Ali 中文 DER / miss / FA | 55.38% / 51.87% / 1.41% | 46.70% / 41.86% / 2.17% | **39.63% / 33.26% / 2.95%** |
| AMI 英文 DER / miss / FA | 41.92% / 38.19% / 1.07% | 40.60% / 35.76% / 1.99% | **39.84% / 32.95% / 3.86%** |

中英两场都是单调改善，中文 -15.8pp、英文 -2.1pp——低阈值对**中文收益大、
英文温和**，且英文的 FA 涨得更多（1.07%→3.86%）。选 0.3 的前提是接受
FA 上升换 miss 下降。

**在线滑窗链路复核（T3 Ali 240s，`score_t3o.py` 同一评分器）**：

| 配置 | DER | miss | fa | conf | 覆盖语音 |
|---|---|---|---|---|---|
| win=120/step=10，vad 0.5 | 41.29% | 34.65% | 0.99% | 5.65% | 162.2s |
| win=120/step=10，vad 0.4 | 32.62% | 23.92% | 1.78% | 6.93% | 191.3s |
| win=120/step=10，**vad 0.3** | **26.83%** | **16.23%** | 2.78% | 7.82% | **213.8s** |

四个 60s 分区**全部改善**（vad0.5→vad0.3：DER 20.47→13.84 / 37.54→22.02 /
45.96→32.42 / 59.05→37.83），不是局部侥幸。人工标注在 240s 内共 222.5s 语音，
vad 0.3 覆盖 **213.8s = 96%**（vad 0.5 只有 73%）。

**结构：RTTM 覆盖量 = VAD 判定语音量**（2026-10-14，2P 双跑对照）。两次
run 的 diar 推理**逐窗完全一致**（185 个窗口的 clusters/centroids/parts
序列 md5 相同），因为 diar 走的是**原始音频**（`diar_decimator_` →
`AppendSessionSamples`），而 VAD/ASR 走 **FastEnhance 之后**的音频——两条
独立取音路径（`audio_pipeline.cpp` `Process()`）。阈值只改变"哪些样本进段"，
因此 RTTM 并集恰好等于 VAD 报告的语音量：

| | RTTM 并集 | 日志 VAD voiced |
|---|---|---|
| vad0.5 | 1092.4s | 1092.6s |
| vad0.3 | 1364.2s | 1364.3s |
| 差值 | **+271.8s** | **+271.7s** |

同一批样本上说话人标签映射一致率 **99.6%**。结论：在滑窗链路里
**miss 与 FA 基本是 VAD 指标，聚类只贡献 conf**——所以降 miss 只能从
VAD 阈值/参数下手，调聚类参数不会改善 miss。

**能量验证（判据）**：在原始波形上算各区间的 10ms 峰值。vad 0.3 从 vad 0.5
手里**抢回 45.8s，能量 p50=298**（人工标注语音本身 p50=373）——是真语音；
而它新多判的只有 **5.8s，p50=72**（近静音）。**恢复的是语音、新增的是噪声**，
所以 miss 下降不是靠把静音塞进时间轴换来的。

2P 侧同一判据（无人工标注，用能量分布比对参考）：

| 区间 | 时长 | 峰值 p50 | <300 占比 |
|---|---|---|---|
| v8-recheck（参考） | 1373.9s | 1759 | 26.5% |
| v8-pad03 | 1515.3s | 1330 | 32.8% |
| **我们 vad0.3** | 1364.2s | **1799** | **25.1%** |
| 我们 vad0.5 | 1092.4s | 2658 | 13.7% |
| pad03 比 recheck 多出的那段 | 142.8s | **35** | 93.6%（近纯静音） |

vad 0.3 的能量分布与参考几乎重合，vad 0.5 明显过度保守——**这同时说明
`v8-pad03` 不能当 2P 的评分基准**（它自己掺了 143s 静音，且 v8-recheck
拿它当参考评出 9.94% DER），2P 的 DER 应报 `v8-recheck` 口径。

**噪声复核（`run-noise-vad.sh` + `score_noise_vad.py`）**：工作区无现场录音，
用 T3 片段混入连续类语音噪声（实测 SNR 19.5 dB）作代理，参考仍为人工标注：

| 条件 | DER | miss | fa | conf |
|---|---|---|---|---|
| 干净 vad0.5 | 41.29% | 34.65% | 0.99% | 5.65% |
| 干净 vad0.3 | 26.83% | 16.23% | 2.78% | 7.82% |
| 噪声 vad0.5 | 66.63% | 50.57% | 0.34% | 15.72% |
| 噪声 vad0.3 | **52.28%** | **22.25%** | **3.06%** | 26.97% |

- 噪声下 vad0.3 的 **FA 只有 3.06%**（干净时 2.78%），低阈值没有把噪声大量判成语音；
  miss 50.57%→22.25%，DER 仍降 14.4pp。
- conf 的大幅上升来自**聚类**：diar 读原始音频，噪声直接污染嵌入。阈值不改变聚类
  （见上文"逐窗一致"），vad0.3 的 conf 更高只是因为有更多帧被打分。这是噪声鲁棒性
  问题，不属于 VAD 阈值。
- 结论：**推荐默认值 0.3**。混噪代理不等于真实现场，上线前仍建议用一段现场录音复核一次。

**注意**：这份扫描结论此前**从未落进默认值**——`apm-config.h` 与
`config/*.yaml` 仍为 `threshold: 0.5`，滑窗评测配置同样是 0.5。

### 2.5 实时流式现状（P1b+P1c 完成，2026-10-13）

- T1（Ali 全片 26 min 实时回放）：跑通，14 块、0 次等待超时，但 conf 22.6%；
  定位为每块聚类塌成 1~3 人。
- 隔离实验（板端量化引擎，120 s 切片）：
  原始文件切片 → 4 簇；仅 APM（不跑 FastEnhance）→ 4 簇；完整链路（APM + FastEnhance）→ 1~2 簇。
  **FastEnhance 是实时流式坍缩的根因**。
- 修复：会话 diarization 从 **FastEnhance 之前**的 16k 流取音频（独立 decimator）。
- P0 跨块 ID 装配（全局限性说话人表 + 在线匈牙利指派）完成后：
  T3 conf 21.8% → **4.82%**（滑窗 win=120/step=30）。
- 2P 全程实测（`report-2P-online-step30.txt`）：恰好 2 人、零幻影/漂移，
  与 v8 基线一致率 **91.8%**（2PO 时代 83%）；63 窗提交延迟实测
  avg 12.3s / max 17.2s（旧设计 ~73s/~135s）。
- P1c 子窗缓存后稳态推理 12.3s → **1.2s**（step=10，占空比 13%），
  T3 质量位级不变；2P step=10 全程实测（185 窗、恰好 2 人、一致率
  91.6%、稳态提交延迟 avg 1.3s/max 3.7s）见
  `build/3dspk/out/report-2P-online-step10.txt`。
- **冷启动首窗斜坡**（`session_first_window_sec`，默认 0=旧行为）：
  首次提交不再等满整块，只用该长度的声学上下文聚一次。T3（240s/4 人）
  实测首标签墙钟 **132.3s → 32.4s**（fw=30）、65.6s（fw=60）；代价集中在
  斜坡覆盖的 [0,120) 区间且**不扩散**（分区 DER：fw=30 时
  [0,60) conf 4.25%→27.62%、[60,120) 1.01%→19.47%，[120,240) 逐位不变）：
  fw=30 conf 5.65%→15.79%、fw=60 →12.89%。首窗越短越容易在开局把多人
  并成一个簇（fw=30 首窗只出 1 个说话人，fw=60 出 4 个）。**默认关闭**，
  仅在能接受开局区间质量下降、且更看重首标签及时性的场景启用。

### 2.6 diart 式增量聚类探索（2026-10-14，离线原型量化）

**目标**：参考 diart 的 `OnlineSpeakerClustering`（5s 滚窗 + 增量聚类，首标签 ~5s），
把冷启动首标签从 132.3s 降下来，同时保持 2P 精度（参考 `label-2P-v8-recheck.txt`）。

**方法**：逐 10s 子窗导出 segmentation mask + 3×256 embedding（`diar_worker --dump-features`，
纯追加调试路径、不动标签时间轴），host 上跑 diart 式增量聚类
（cosine 距离 + 质心累加 + `delta_new` 门控），从第 1 个子窗即出标签。

**结论（2P 全程 1955s，`score_der.py` collar 0.25，参考 v8-recheck）**：

| 指标 | online-vad30（批处理基线） | 增量聚类（raw cosine dn=0.5 rho=0.4） |
|---|---|---|
| 首标签 | 132.3s | **~11s**（10s 首窗 + ~1.2s 推理） |
| conf | 4.51% | **0.40%**（≈4.8s，5 个 <2s 边界片段） |
| miss | 3.79% | 19.60%（diar segmentation 比 VAD 保守） |
| fa | 31.33% | **0.04%** |
| DER | 39.63% | **20.04%** |

- 首标签提前 ~12 倍、conf 改善 ~11 倍；conf 4.8s 错分全是说话人切换边界的
  短暂混淆（684s/878s/1545s/1934s/1950s 处 <2s 片段），非冷启动问题。
- miss 19.60% 是 diar 在整段原始音频上自己判语音所致（覆盖 1120s vs VAD 1364s），
  **非聚类问题**；落地保持「VAD 判语音 + diar 只分说话人」结构即可降到 ~3.79%。
- 三场景验证（2P / EN2002c 3 人 / T3 4 人）：增量 conf 均不退化
  （0.40% / 4.61% / 4.28%，后者略优于批处理 5.65~7.82%）。
- **delta_new 按场景敏感**（说话人越多阈值越小）：raw cosine 无统一值
  （2P 需 ~1.0、T3 需 ~0.5）；PLDA 判别投影（256→128，`plda.bin` 的
  mean1/mean2/lda/plda_mu/plda_tr）后 delta_new=0.55 三场景统一（脆弱交集点，
  2P 单独用 raw cosine 更优）。
- step=2 重叠子窗 conf 0.32%（略优 0.40%）但 DER 略差、推理 5 倍、首标签不更早，
  工程上不值，保留 step=10。
- 产物：离线原型 `/tmp/opencode/proto_*.py`；最优 RTTM
  `/tmp/opencode/best-2p-incremental-step10.rttm`。

**C++ 落地（2026-10-14 完成，oneshot 模式）**：`diar_engine` 加
`ClusterIncremental()`（raw cosine / PLDA 投影可选，`IncrementalParams` 的
delta_new/rho_update/tau_active/max_speakers），`Run()` 加 `incremental` 参数分支；
`diar_worker` 加 `--incremental` / `--incremental-plda` / `--delta-new` / `--rho-update`。
板端 2P 全程（`--step 10 --incremental`）conf **0.39%** / DER **20.20%**，与离线原型
（0.40%/20.04%）逐位一致（差异为 RTTM 3 位小数舍入）。原型脚本固化到
`driver/pyannote/tools/incremental_proto/`（含 README + score_der.py）。

**流式落地（2026-10-14 完成，首标签 ~10s）**：增量质心状态（`inc_fc_/inc_rc_`）
提升为 engine 跨请求成员 + `ResetIncremental()`；`diar_worker` 流式循环支持逐子窗
提交（空请求 = session 边界，重置增量状态 + `speaker_table.Clear()`）。板端实测
逐 10s 子窗提交：**首标签推理延迟 0.24s**（端到端首标签 ≈ 10s 音频缓冲 + 0.24s ≈
**10.2s**，vs 132.3s 提前 ~13 倍）；第 1 子窗即正确发现 2 人并捕捉说话人切换，
增量状态跨子窗零幻影/漂移（全程 speaker-1×112 / speaker-2×173）。
注：流式响应为 `parts`（每子窗独立 partition、覆盖含静音），其 conf 3.42% 系
partition 边界断裂的口径假象；聚类质量本身与 oneshot 等价（rttm 口径 conf 0.39%）。

**audio_pipeline 全链路集成（2026-10-14 完成）**：`Diarization` 配置新增
`incremental` / `incremental_plda` / `incremental_delta_new` / `incremental_rho_update`
四字段（`apm-config.h/.cpp`，校验 delta_new∈(0,2]、rho_update∈(0,1)），
`AddAssignArgs()` 透传 `--incremental[--incremental-plda] --delta-new --rho-update`；
`session_chunk_sec` 校验下限从 20s 放宽到 10s（增量聚类语境下「窗」= 10s 分析子窗，
聚类上下文是跨请求的 `inc_fc_/inc_rc_` 状态，非重聚窗）。落地配置即
`session_chunk_sec == session_step_sec == 10`（`config-2pw-inc.yaml`），现有
`SessionDiarizer` 的滑窗提交在 chunk==step 时自然退化为逐 10s 非重叠子窗，无需改框架循环。
板端 2P 全程实测（`config-2pw-inc.yaml`，mode=session chunk=step=10，196 子窗）：
**首标签 wall=+10.22s**（10s 缓冲 + 0.24s 推理，vs 132.3s 提前 ~13 倍），第 1 子窗即
发现 2 人（speaker-1/speaker-2），稳态每 10s 窗推理 221~242ms（rtf≈0.023，标签延迟
~0.2s，推理占空比 44s/1956s=2.3%）。RTTM 口径 DER（vs v8-recheck，collar 0.25）：

| 运行 | miss | fa | conf | DER |
| --- | --- | --- | --- | --- |
| online-vad30（批处理 120s 基线） | 14.56% | 11.00% | 3.07% | 28.64% |
| online-step10（120s/step10 滑窗） | 24.01% | 2.70% | 2.54% | 29.25% |
| **inc（增量聚类 10s 子窗）** | **14.56%** | **11.00%** | **1.42%** | **26.99%** |

- conf 3.07%→**1.42%**（相对降 ~54%）、DER 28.64%→**26.99%**（三者最优）；miss/fa 与
  baseline 相同（VAD 0.3 覆盖不变，只换了聚类层）。
- 距 oneshot 0.39% 仍有 ~1pp 差：流式逐 10s 子窗每窗独立 segmentation/partition，
  边界断裂导致的短促说话人切换（summary 前文已定性为口径假象，聚类质量等价 oneshot）。
- 产物：`/tmp/der-2pw-inc/ali/transcript.txt`（220 行 speaker+文本）、
  `build/3dspk/out/label-2P-inc.txt`（源时间 transcript）、`rttm-2p-inc/`（RTTM）。

多说话人场景可开 `--incremental-plda`（delta_new 0.55 三场景统一），或探索自适应
delta_new；边界断裂可用子窗重叠 / 跨边界平滑进一步收窄（可选下一步）。

## 3. 已完成的工作

### 3.1 模型与引擎

- 8 个 axmodel 已从 **NPU3 重编为 NPU2**（`npu_mode: "NPU2"`，兼容链路 BIG_LITTLE），
  部署于 `models-for-asr-chain/pyannote/`（含 6 个主机参数 bin）；
- 实测 **NPU2 与 NPU3 输出逐位一致**（AMI/AliMeeting 对照 DER 完全相同），重编无精度损失；
- worker 实现 **NPU 模式自动回退**（先试 VNPU-DISABLE，失败再试 BIG_LITTLE），随链路/单机均可运行；
- 关键背景：AX650 的 VNPU 模式为设备级、跨进程互斥；NPU3 模型只能在 DISABLE 下加载，
  因此必须重编为 NPU2（错误码：Init 冲突 `0x80060087`、模型不匹配 `0x80060086`）。

### 3.2 代码集成（`driver/pyannote`）

- vendor MIT 源码：`diar_engine`（新增 parts 与聚类中心输出）、`EngineWrapper_axera`、
  `fbank/params/wav_reader`、kissfft；
- `diar_worker.cpp`：常驻 worker（协议同 qwen3-worker：stdin 请求 / fd3 响应帧）；
  支持单机、常驻、`--rttm-dir` 评测转储；
- `speaker_table.*`：跨段说话人表，JSON 兼容旧格式；每个说话人保留最近 8 个质心取最大余弦，
  默认阈值 0.55；
- 已修复：
  - speaker 表 `taken[]` 越界崩溃；
  - 短段无干净 embedding 时的空说话人 ID（引擎兜底质心 + worker 防空响应）；
  - 原始窗口→去静音切片映射的坐标偏移 bug（`MapTimelineToTrimmed(timeline, offset, runs, …)`）。

### 3.3 链路集成（`framework/audio_pipeline.cpp`）

| 变更 | 说明 |
| --- | --- |
| 移除 campplus | 编译、配置、部署全部下线（源码保留在 git 历史） |
| `Accumulator` 原始窗口捕获 | 从段首语音到段尾（含静音）保存 16k 原始样本 + 源偏移 |
| `MapTimelineToTrimmed` | 说话人时间线 → VAD 去静音拼接时间轴，供 ASR 切片 |
| `diarization.mode` | `segment`（默认，实时可用）/ `session`（文件输入整场先分离） |
| `RunSessionDiarization` | 文件输入：整场预跑一次（worker one-shot），标签按时间投影到各段 |
| `SessionDiarizer` | 实时流式：原始 16k 缓冲、`session_chunk_sec`（默认 120s）聚类窗、
`session_step_sec`（默认 0=整块提交；0<step<chunk 时滑窗提交：窗以步长边界结尾、
只提交新增核心区间）、绝对时间线、`session_max_wait_sec`（默认 180s）等待/超时回退、退出冲刷尾窗 |
| 音频路径 | 会话 diarization 取 **FastEnhance 之前**的 16k 流（独立 decimator，与 ASR 同 16k 时基） |
| ResultWorker 回退链 | 会话时间线覆盖 → 投影；未覆盖/失败 → 逐段 raw 窗 → 去静音段 → 整段 ASR |

### 3.4 配置 / UI / 构建 / 部署

- `apm-config.h/cpp`：`Diarization` 增加 `mode`、`session_chunk_sec`、`session_step_sec`、
  `session_max_wait_sec`，含合法值校验与 YAML 解析；
- `sdk-config-ui.py`：分离模式下拉、分块长度、最长等待，加载/保存顺序一致；
- `sdk-config.yaml` / `default-config.yaml`：`diarization` 段替换 `campplus`；
- `scripts/build.sh` 编译 `build/diar-worker`；deploy/check 白名单加 diar-worker 与 pyannote 14 个模型文件；
- README / `driver/pyannote/README.md` 已更新。

## 4. 模式与行为定义

- `mode: segment`（默认）：逐话段 raw 窗 diarization（低延迟；实时可用）；
- `mode: session`：
  - `input: file` + `batch`：启动时整场预跑一次，输出非重叠说话人时间轴，映射到各话段
    （当前质量最佳）；
  - 实时（ALSA / file realtime）：原始 16k 缓冲、120 s 聚类窗送常驻 worker，
    每到 `session_step_sec`（默认 0=整块；设 30 时窗以步长边界结尾）提交一次
    新增核心区间，话段最多等待 `session_max_wait_sec` 获取覆盖，超时回退逐段；
    退出前用保留上下文冲刷尾窗；
- 所有模式共用持久化 SpeakerTable（`speaker-N` 跨段稳定）；
- 延迟：实时流式标签延迟 ≈ `session_step_sec`（0 时为分块长度）内的等待位置 +
  窗推理 + part ASR，上限再叠加 `session_max_wait_sec`；
  置 0 可退回逐段低延迟模式。滑窗（0<step<chunk）时延迟 ≈ step + 推理，
  聚类上下文仍为整窗（见 P1b）。

## 5. 差距定位与对策

| 损失项 | 量级 | 原因 | 对策 |
|---|---|---|---|
| VAD 召回 | 中文 miss 主导（覆盖 63%） | TEN-VAD 阈值偏保守 | 阈值 0.3/0.4（已验证，见 2.4） |
| 重叠语音 | miss 28–33%（有声帧内） | 输出为单说话人分区 | 可重叠标签（DER 口径用 native RTTM；ASR 仍按主说话人切片） |
| 局部聚类碎片 | segment 模式 conf 15–33% | 逐窗独立聚类 | session / 流式 session + 身份表（批量已解决） |
| 跨块 ID 装配 | 流式 conf 22% | 分块独立聚类 + 表匹配 | 阈值调参（0.45~0.5）/ 块间全局对齐 / 块间重叠上下文 |
| 切片偏碎 | 中文短碎片 ~50 行 | 帧级边界直接使用 | 最短切片合并（0.3–0.6 s） |

## 6. 计划与待办

### P0 实时流式跨块 ID 一致性（已完成，2026-10-13）

结论：幻影 ID 的根因不是块间表匹配，而是**关闭竞态**——文件结束时
`SessionDiarizer::Stop()` 的 notify 把等待最后一块时间线的话段全部误判为
"等待超时"，回退逐段 raw 推理，混合质心匹配不到任何人而新建 speaker-5/6；
最后一块的 12s NPU 结果反而作废。修复（`framework/audio_pipeline.cpp`）：

- `SessionDiarizer` 增加 `finished_` 终态；`WaitCovered` 只认
  `committed_ >= end || finished_`（Stop 不再骗醒等待者）；
- `ResultWorker::Close` 顺序改为 Stop → ProcessRemaining → Finish，
  `session_` 生命周期延长到结果线程 join 之后（消除释放竞态）。

决策层升级（`speaker_table.cpp` `AssignOnline`，diart 式，`mode: online` 启用）：
匈牙利整体最优分配 + 新建门控 `online_min_new_speech_sec` + 更新门控
`online_update_min_sec` + 封顶 `online_max_speakers`；逐段回退请求
（`.raw.wav`）identify-only，混合质心无出生权。

实测：

| 场景 | 指标 | 旧二进制 session | 新二进制 session | 新二进制 online | 滑窗 online（win=120/step=30） |
|---|---|---|---|---|---|
| T3（Ali 240s，4 人，流式） | IDs / conf / DER | 6 / 21.83% / 57.47% | 4 / 9.54% / 45.18% | **4 / 9.54% / 45.18%** | **4 / 4.82% / 40.46%** |
| 2P（20200327，32.6 min，2 人，流式） | IDs | — | — | **2（全程 17 块零幻影）** | 2（见 P1a 滑窗实测） |

2P 全程（`build/3dspk/out/report-2P-online.txt`）：块1 精确建 2 人
（47.8s + 72.2s），后续 16 块全部静默匹配；说话占比 55.9%，与 v8 离线
基线说话人一致率 90%（真实错分 ~53s ≈ 4.9%，其余为 VAD 覆盖差异）；
转录按 part 输出 278 行（`label-2P-online.txt`），逐轮抽查与 v8 标签一致。
T3 的簇余弦普遍 0.89–1.00，贪心与匈牙利决策无差异；online 决策层的价值
在质心漂移 / 短附和 / 开放麦封顶等难场景。

### P1 完整实时验证

- Ali R8001 + AMI EN2002d 实时（`file_input_mode: realtime`）各跑一遍；
- 与 session 批量对比 DER/conf/miss，确认流式损失可接受。

### P1a 实时性评估与 chunk 扫描（2026-10-13 完成，结论：标签延迟由块长决定）

标签（连同 transcript 行）只随块提交产生，端到端延迟 ≈ 块内等待位置 +
块推理 + part ASR。实测推理 rtf ≈ 0.10（120s 块 12.2s、30s 块 3.1s）。

| chunk | 标签延迟（平均/最坏） | T3 IDs | T3 conf | 结论 |
|---|---|---|---|---|
| 30s | ~19s / ~34s | 2 | 34.49% | 块内欠分割，不可用 |
| 60s | ~37s / ~67s | 3 | 20.73% | 仍欠分割 |
| 120s | ~73s / ~135s | 4 | 9.54% | 精度可用，延迟非交互 |

**根因**：社区模型块内 AHC 需要每簇足够语音才能分人，短块内说话人语音
不足 → 欠分割；online 决策层无法修复不存在的簇。2P（2 人）全程 32.6 min
用 120s 块零幻影、一致率 83%（对 v8 离线基线轴）。

缩延迟候选架构（详见 `build/3dspk/out/report-2P-online-vs-v8-and-latency.txt`）：
按优先级：① 块间重叠窗（块长不变、提交只出核心区间，延迟降为
chunk/2+overlap+12s）；② 双速度层（30s 粗标实时预览 + 120s 精标覆盖修正，
即 diart DelayedAggregation 思想）；③ 首块小后续块大（冷启动友好）。

#### P1b 滑窗提交（方案 ① 已实现，2026-10-13）

diart 精度随延迟缓降的关键在于**分析上下文与提交粒度解耦**（5s 窗每
0.5s 提交、聚类记忆全局累积、latency 只是输出聚合参数）。照此把
`session_chunk_sec`（聚类窗，质量轴）与新增的 `session_step_sec`（提交
步长，延迟轴）拆开：`SessionDiarizer` 每到步长边界就送一个以该边界
**结尾**的整窗给 worker，但只把新增核心区间的 parts 提交进全局时间线；
首个窗口（[0,120s)）整体提交保冷启动聚类质量；尾部用保留的上下文冲刷
最后一次部分提交。`step=0` 或 `step=chunk` 时与旧行为完全一致（回归
已验证）。

实测（T3W，`config-t3w-ali.yaml`，win=120/step=30）：

| 设计 | 标签延迟（平均/最坏） | T3 IDs | T3 conf | T3 DER | 推理占空比 |
|---|---|---|---|---|---|
| 120s 整块提交 | ~73s / ~135s | 4 | 9.54% | 45.18% | 10% |
| **滑窗 win=120/step=30** | **~27s / ~42s** | **4** | **4.82%** | **40.46%** | ~40% |

每窗仍是完整 120s 聚类上下文且核心区恒在窗尾（说话人已充分展开），
conf 反而比整块提交低一半；代价是推理占空比 10%→40%（4 倍冗余聚类，
diart 自身为 10 倍），仍远低于实时极限。2P 全程实测见
`build/3dspk/out/report-2P-online-step30.txt`。

**(win, step) 网格结论**（T3 240s，同一评分器）：

| win / step | conf | IDs | 判定 |
|---|---|---|---|
| 120 / 30 | 4.82% | 4 | 质量基准 |
| 120 / 15 | 5.32% | 4 | 可用，step↑质量微降 |
| 120 / 10 | 5.65% | 4 | 可用（缓存后） |
| 60 / 15 | 25.11% | 3 | **拒绝**：窗 60s 时块内 AHC 欠分割 |

**win=120 是质量地板**（每窗 AHC 需要足够上下文），延迟只能沿 step 轴压。

#### P1c 子窗缓存：推理项与窗长解耦（2026-10-13 完成）

滑窗相邻窗重叠 90s，但引擎每次对整窗重算 56 个 10s 子窗的分割+嵌入
——其中 ~73% 是纯重复。10s 子窗的（分割掩码 + 3×256 嵌入）是其音频
内容的纯函数，因此引擎内建**内容哈希键的子窗缓存**（仅缓存未补零的
完整子窗；FIFO 上限 1024 条 ≈ 5MB）：相邻滑窗直接复用重叠区的子窗
结果，每步只算新增音频的 5 个（step=10）子窗。命中结果与重算**位级
一致**（T3W conf 4.82% 复现无误），质量零代价。

实测（T3，worker 日志 `cache=H/56`、`took=`）：

| step | 稳态每窗推理 | cache 命中 | 占空比 | 提交延迟（末） |
|---|---|---|---|---|
| 30（无缓存） | 12343ms | 0/56 | 39% | 3.4~17s |
| 30（缓存） | ~3350ms | 41/56 | 11% | ~3.4s |
| 10（缓存） | **~1250ms** | **51/56** | **13%** | **~1.2s** |

**新延迟模型**：延迟 = step/2（等待）+ 推理。win=120/step=10 下
平均 ≈ 5+1.2 ≈ **6.2s**、最坏 ≈ 11.3s——比旧设计（73s/135s）低一个
量级，平均进入 diart 量级（<10s 目标达成）；占空比 13% 为 ASR/LLM
留足余量。引擎改动见 `driver/pyannote/src/diar_engine.{hpp,cpp}`。

2P 全程实测（32.6 min 实时回放，`report-2P-online-step10.txt`）：
- 185 窗全部恰好 2 人、零幻影/漂移；与 v8 基线一致率 **91.6%**
  （step=30 为 91.8%，质量不随 step 细化下降）
- 提交延迟（核心区末）：稳态 184 窗 **min 1.2s / avg 1.3s / max 3.7s**
  （冷启动首窗 12.3s，同旧行为）
- 推理占空比 **12.5%**（245s/1956s）
- 转录行数 418（step=30 为 323）：碎片化随 step 增长，见 P2
  （P2 落地后重跑：236 行、空文本行 0、一致率 91.5%）



### P2 转录碎片整理（A+B 已实现，2026-10-13；C 待定）

滑窗提交后每个核心区边界会把连续语音切成两段、每次说话人切换也切一刀，
每段 part 都是一次独立 ASR 请求（产生 "。"/空串垃圾行）。在
`MapTimelineToTrimmed()` 输出的最终循环里两级清理：

- **(A) 相邻同说话人合并**（标签中性，先离线重放验证再上线）：
  提交边界切开的同人相邻 part 重新拼回一段——样本与说话人完全不变，
  只消除人为切分。离线重放实测 2P 418→277 段，T3 conf/DER 逐位不变；
- **(B) 亚秒碎片吸收**（阈值 **0.3s**，吸收进较长邻段）：短碎片的说话人
  归属本就不可靠、ASR 也只产出垃圾行。阈值扫描（离线同库重放）：
  T3 conf 代价在 T≤0.3s 时为零（4.82%），0.35s +0.15pp、0.4s +0.31pp、
  0.5s +0.52pp；线上 T3W 回归确认 0.3s 下 conf/DER 完全复现 4.82%/40.46%。
  2P 离线：418→236 段、<0.5s 碎片 99→17、一致率 91.6%→91.5%；
  **线上 2P 全程重跑实测：418→236 行、空文本行 46→0、亚秒片段
  32→2、覆盖 1092.6s 不变、一致率 91.5%、延迟与占空比不变**
  （`label/report-2P-online-step10.txt`，185 窗）；
- **(C) part 级 ASR 头部 pad（已实现 0.2s，`kAsrPadSamples`）**：说话人切换
  边界可能落在词头中间，ASR 只拿到词的尾半截（2P 第 2 句 "可以听到。" 的
  "可" 起音在源 2.06s，part 边界 2.080s 切在爆破前 → ASR 只输出 "以。"）。
  只加长送 ASR 的切片、不动 parts.txt/RTTM（标签与 DER 口径不变）。用管线
  自身 ASR 在整轮 part 上 A/B 重放（`run-pad-replay.sh`）：pad 0→0.2s 改变
  47/236 条 part 文本（含 "以。"→"可以听。"），全场净增 ~11 字符——修的是
  可见的"无头词"病例，不是漏字量的主体；0.5s 起会串入前一说话人，故取 0.2s；
- 可选：整段 ASR + 词级时间戳分配说话人，替代按 part 切片 ASR。

验证资产：`build/der-eval/proto_p2.py`（策略重放/计数）、
`proto_p2_score.py`（官方评分器隔离评估）、`run-pad-replay.sh` +
`run-head-loss.sh`（part 切片 / pad 的受控 ASR 实验）、
`cmp_2p_v8.py` 与 `/tmp/opencode/analyze_2p_drops.py`（对 v8 基线的漏字计量）。

### P3 覆盖率（miss）优化（量化完成，2026-10-13）

miss 34.65% 是 DER 的最大项，拆开看到底是"没送到 ASR"还是"ASR 听错"：

- **受控切片实验（`run-head-loss.sh`，173 段 / 1117 次 ASR）**：同一段音频用
  "整段一次 ASR" vs "按 part 切片多次 ASR"（相同模型/前端/VAD/标签）。
  **切片只损失 3.2% 字符（净 +9）**——所以漏字主体不是切片；
- **part 头部 pad（pad-replay）**：净 +11 字符，只修"无头词"；
- **段起点前导（`run-vad-onset.sh`，0.3s 原始前导）**：净 +33 字符
  （39 段变好 / 25 段变差）——VAD 段起点的词头损失也**不是主体**；
- **2P 全程实测（vad0.5 → vad0.3，`compare_2p_runs.py`）**：

  | 指标 | vad0.5 | vad0.3 |
  |---|---|---|
  | RTTM 语音覆盖 | 1092.6s（55.9%） | **1364.3s（69.8%）** |
  | v8 参考语音未被覆盖 | 32.2s（2.3%） | **7.7s（0.6%）** |
  | 转录行数 | 236 | 219 |
  | 归一化字符 | 5628 | **5952** |
  | vs v8 相似度 | 86.7% | **89.4%** |
  | **纯漏字**（delete） | 6.0% | **3.1%** |
  | 听错（replace） | 10.8% | 8.8% |
  | 多字（insert） | 1.3% | 1.8% |

  三档对照（同一 2P 全程，参考 `v8-recheck`）：

  | 阈值 | 覆盖 | 字符 | 相似度 | 纯漏字 | 听错 | 多字 | DER | miss | fa |
  |---|---|---|---|---|---|---|---|---|---|
  | 0.5 | 1092.6s | 5628 | 86.7% | 6.0% | 10.8% | 1.3% | 29.25% | 24.01% | 2.70% |
  | 0.4 | 1213.6s | 5822 | 88.4% | 4.3% | 9.6% | 1.7% | **27.91%** | 19.32% | 5.79% |
  | 0.3 | 1364.3s | **5952** | **89.4%** | **3.1%** | **8.8%** | 1.8% | 28.64% | **14.56%** | 11.00% |

  文本侧随阈值单调改善；2P 的 DER 在 0.4 最低，原因是 v8-recheck 的边界较紧，
  0.3 的边界抖动被它计成 FA（见上文 FA 分布：70% 在 part 头部、中位 0.04s）。
  三档的延迟一致（185 窗，稳态 avg 1.27s）。

  **纯漏字减半（6.0%→3.1%），字符 +5.8%，行数反而减少**（低阈值让整句不再
  被切碎）。延迟零回退：首窗 132.29s / 稳态 avg 1.27s / max 3.57s，与
  vad0.5 逐项相同（185 窗，cache 命中一致）。近静音段没有幻觉放大
  （<100 峰值行 28→41 行、+78 字符，抽查为真实低音量话语）。
- **真正的缺口是语音覆盖**（`analyze_2p_coverage.py`）：

  | 指标 | 值 |
  |---|---|
  | 本链路判定语音 | 1092.6s（占 1955.2s 文件的 **55.9%**） |
  | v8 参考判定语音 | 1373.9s（**70.3%**） |
  | v8 有语音、我们完全未覆盖 | **338.0s**（= v8 语音的 24.6%、文件的 17.3%），1796 段 |
  | 其中 >1s 的整段漏掉 | 72 段 / 131s，含 48 条 v8 句子、**547 字**（v8 文本的 7.9%） |

  分布：>3s 22.3s、1–3s **108.7s**、0.5–1s 44.0s、0.2–0.5s 81.4s、
  0.05–0.2s 59.4s、<0.05s 22.2s——**三分之二集中在 0.2–3s 的中片段**，
  属于 VAD 把整句/半句判成静音（`min_speech_ms` / `threshold` 口径），
  不是边界抖动。
- **VAD 阈值是最大杠杆**：§2.4 的扫描结论（Ali 中文 0.5→0.3：
  覆盖 63%→90%，DER 55.38%→39.63%，-15.8pp）**从未落进默认值**——
  `apm-config.h` 仍是 `threshold = 0.5f`，`config/*.yaml` 也是 0.5，
  T3W/2P 的滑窗配置同样 0.5。2P 实测（`score_t3o` 同一评分器口径）
  正在固化：`config-2pwvad30.yaml` + `config-t3wvad30-ali.yaml`。
- 待办：把 0.5→0.3 落到 `apm-config.h` 默认值与配置页（同时复核 FA 略升
  在噪声现场录音下是否可接受）；可选：DER 口径使用引擎原生（允许重叠）
  RTTM 输出，parts 仅用于 ASR 切片。

### P4 收尾

- README / 部署清单 / 配置页面最终核对（`mode`、阈值等字段）；
- Git 提交与推送（滑窗/缓存/P2 均已提交至 `diarization` 远端），
  保留实验脚本与结论数据。

## 7. 复现与产物

### 7.1 评测资产（工作区 `asr-llm-tts-chain/build/der-eval/`，非 Git）

- `score_p1.py`：P1/P1S/历史基线与整文件的统一评分；
- 配置：`config-p1-*.yaml`、`config-p1s-*.yaml`、`config-s1-*.yaml`、`config-t1-*.yaml`、
  `config-t3-ali.yaml`、`config-vad{30,40}-*.yaml`；
- 实验脚本：`run-s1.sh`、`run-t1.sh`、`run-t3.sh`、`board_emit.py`、`board_emit_raw.py`、
  `make_windows.py`、`run_worker_batch.py`；
- RTTM 输出：`rttm/`（改造前）、`rttm-p1/`、`rttm-p1s/`、`rttm-vad30/`、`rttm-vad40/`、
  `rttm-t1/`、`rttm-t3/`；
- 参考与对照：`eval-alimeeting/`、`eval-ami/`、`eval-ref/`（3D-Speaker、GPU 参考）；
- 旧模块批处理工具（评估用）：`build/campplus-batch` + `build/old-campplus/`。

### 7.2 数据与评分

- 中文：`.../vad_seg_sr_cluster/alimeeting/{audio_mono,ref}`；
- 英文：`.../vad_seg_sr_cluster/ami_eval/{audio,ref_nxt}`；
- 评分器：`pyannote-speaker-diarization.axera/benchmark/score_der.py`（10 ms 帧级 + 匈牙利映射）。

```bash
# 统一对比
python3 build/der-eval/score_p1.py
# 或单场评分
python3 pyannote-speaker-diarization.axera/benchmark/score_der.py \
  --ref build/eval-alimeeting/ref --hyp build/der-eval/rttm-p1s/new --collar 0.25
# 运行链路（板端）
export LD_LIBRARY_PATH="$PWD/build:$PWD/driver/apm/lib:$PWD/driver/vad/lib:$PWD/driver/asr/sensevoice/lib:/soc/lib:/opt/lib"
./build/audio-pipeline --config build/der-eval/config-p1s-ali.yaml
```

## 8. 风险与注意事项

1. **NPU 模式互斥**：VNPU 模式为设备级，禁止在链路运行时用 DISABLE 模式进程
   （会撞 `0x80060087`）；worker 已自动回退，但批量实验要注意串行。
2. **模型版本**：必须使用 NPU2 版 axmodel；NPU3 版只能单机 DISABLE 运行。
3. **FastEnhance 差异**：会话 diarization 使用 FastEnhance 之前的音频；ASR 仍使用增强后音频；
   两条路径共用同一 16k 时基（独立 decimator，样本对齐），不要改回增强后音频。
4. **session 模式输入**：`input_wav` 需为 16 kHz（worker 不做重采样）；实时 ALSA 依赖流式 session。
5. **VAD 低阈值**：FA 会上升，噪声环境需实测（不同语言/场景可在 0.3–0.4 微调）。
6. **NPU2 争用**：diarization 与 SenseVoice/Qwen3/Hojo 共享 NPU2，经 Cell 锁串行；
   分块推理约 0.14 RTF（120 s 音频约 17 s），26 分钟会议约 14% 占空比。
7. **并行开发**：源码曾有多个会话并行推进；改动前先同步，避免冲突。
8. **许可**：模型来源 `pyannote/speaker-diarization-community-1`（cc-by-4.0）；
   部署模型为 Pulsar2 `npu_mode: NPU2` 版本；工程代码 MIT（`driver/pyannote/LICENSE`）。

## 9. 关键数据速查

- 板端整文件 NPU2：Ali 23.6% / AMI 25.9%（native RTTM，标准 DER）；
- 批量会话（P1S，VAD 0.5）：Ali 55.4% / AMI 45.1%；conf 2.1% / 3.7%；
- **批量会话 + VAD 0.3：Ali 39.6% / AMI 43.1%**；
- 逐段原始窗口（P1）：Ali 76.3% / AMI 55.7%；
- 旧 campplus 同段：Ali 76.0% / AMI 69.1%；
- 实时流式现状（T3）：块聚类 4/3 簇，conf 21.8%，ID 不稳（6 个），待 P0 修复；
- 隔离结论：120 s 原始切片 4 簇；APM-only 4 簇；含 FastEnhance 1~2 簇；
- VAD 覆盖：TEN-VAD voiced = 参考语音量的 61.1%（Ali, 阈值 0.5）/ 89.0%（AMI, 阈值 0.5）。
