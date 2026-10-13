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

### 2.5 实时流式现状（进行中）

- T1（Ali 全片 26 min 实时回放）：跑通，14 块、0 次等待超时，但 conf 22.6%；
  定位为每块聚类塌成 1~3 人。
- 隔离实验（板端量化引擎，120 s 切片）：
  原始文件切片 → 4 簇；仅 APM（不跑 FastEnhance）→ 4 簇；完整链路（APM + FastEnhance）→ 1~2 簇。
  **FastEnhance 是实时流式坍缩的根因**。
- 修复：会话 diarization 从 **FastEnhance 之前**的 16k 流取音频（独立 decimator）。
- T3 短测（前 240 s）：每块聚类恢复正常（4/3 簇），但整体 conf 仍 21.8%、
  转录出现 6 个 ID → **跨块说话人 ID 装配（表匹配）仍不稳定**，进行中。

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

### P2 转录碎片整理

- 映射后合并过短 part（建议 <0.4~0.6 s 并入邻段），减少短文本行
  （当前 Ali 42 / AMI 106 行 <3 字符；2P online 278 行中空文本 36 / ≤2 字 23）；
- part 级首尾 pad（各 0.3 s，不跨相邻 part）：2P 已知病例——"可以听到。"被
  1 s part 切成"以。"（词头在 part 边界外）；
- 可选：整段 ASR + 词级时间戳分配说话人，替代按 part 切片 ASR。

### P3 覆盖率（miss）优化

- **采纳 VAD 阈值扫描结论**（`der-vad`：0.3/0.4，见 2.4），更新默认值与 UI；
- 可选：DER 口径使用引擎原生（允许重叠）RTTM 输出；parts 仅用于 ASR 切片。

### P4 收尾

- README / 部署清单 / 配置页面最终核对（`mode`、阈值等字段）；
- Git 提交（当前改动未提交），保留实验脚本与结论数据。

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
