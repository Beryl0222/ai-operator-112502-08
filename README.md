# 采样率变化下的流式重排与重叠相加

纯 Python 标准库实现的流式音频算子包 `operator_audio_chunks`。接收
**带会话 ID、采样率和会话内序号 seq 的不等长音频块**，输出：

* 固定 `hop`、具有稳定时间基准（帧序号 + 精确 PTS）的模型帧；
* 每帧可向解码器确认的**来源范围**（哪些 `seq`、块内哪些样本、何种采样率、
  哪段会话时间）；
* 序号重排、重复/冲突/缺口的显式处理，绝不静默拼接；
* 乐观交付 + **撤回 / 确认**协议，告诉解码器哪些已交付帧可保留、哪些必须丢弃；
* 按 seq 区分的恢复计划：**可重放状态** vs **必须重新取样（补传）的区间**；
* 来源可追踪、修订可传播的**重叠相加（OLA）**算子；
* `high`（Kaiser 加窗 sinc，抗混叠）与 `low_latency`（线性插值）两档滤波，
  输出保持相同的时间/来源契约。

## 环境与验证

项目运行只依赖 Python ≥ 3.11 标准库。

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt   # 仅打包工具
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
```

仓库自带 60+ 个 `unittest` 用例（无需 pytest / numpy），包括：

* data/input.json 的精确帧值、来源与撤回断言；
* 120 种“缺口 + 乱序 + 迟到补齐”排列下的最终收敛；
* 随机排列下“**stable 帧永不被撤回**”不变量；
* 采样率段隔离的“毒性样本”与差分测试；
* Hann/矩形窗 COLA 重建、OLA 修订/删除传播；
* 检查点序列化/恢复/重放、CLI 与 wheel 打包。

## 60 秒上手

```python
from operator_audio_chunks import StreamingAudioOperator

op = StreamingAudioOperator(
    "call-3",
    hop=2,                 # 模型样本为单位的帧移
    window_size=4,         # 缺省 = 2 * hop（50% 重叠）
    model_rate=16000,      # 稳定时间基准的目标采样率
    quality="low_latency", # 或 "high"
    gap_policy="provisional",
)

for chunk in chunks:       # {"seq": 0, "sample_rate": 16000, "samples": [...]}
    result = op.ingest(chunk)
    for frame in result.frames:        # 新帧 / 修订帧
        ...
    for retract in result.retractions: # 必须撤回的旧修订
        decoder.drop(retract.frame_id, retract.revision)
    for confirm in result.confirmations:
        ...                            # provisional -> stable，可保留
```

多会话入口 `AudioChunkOperator`（别名 `StreamingReorderOLAOperator` /
`Operator`）按 `session_id` 自动路由：

```python
from operator_audio_chunks import AudioChunkOperator
mgr = AudioChunkOperator(hop=160, model_rate=16000)
result = mgr.process({"session_id": "A", "seq": 0,
                      "sample_rate": 16000, "samples": [...]})
plan = mgr.recovery_plan("A")
```

也可以直接跑资料文件：

```bash
python -m operator_audio_chunks --input data/input.json --summary-only
python -m operator_audio_chunks --input data/input.json --inject 2   # 演示迟到 seq2
```

函数式入口：

```python
from operator_audio_chunks import process_payload, process_chunks
report = process_payload("data/input.json", quality="high")
```

## 数据契约

### AudioChunk（输入）

| 字段 | 说明 |
| --- | --- |
| `session_id` | 会话标识（字典也接受 `session` / `sid`） |
| `seq` | 会话内块序号，从 `first_seq`（默认 0）开始，唯一标识一个块 |
| `sample_rate` | 本块输入采样率（也接受 `sr` / `rate` / `fs`） |
| `samples` | 样本序列（也接受 `audio` / `data` / `pcm` / `values`） |

块在媒体时间线上按 seq 顺序首尾相接，块时长精确为
`len(samples) / sample_rate` 秒。seq 是**块**序号；缺失 seq 的真实时长未知，
算子不会编造其样本或时长。

### Frame（模型帧）

稳定身份是 `(session_id, index)` / `frame_id == "<session>#F<index>"`：

* `index`：从 0 开始的帧序号；
* `pts_samples == index * hop`，`pts_seconds == pts_samples / model_rate`；
* `window_size`、`hop`、`model_rate`、`samples`（长度恰为 window_size）；
* `revision`：内容更正时递增，撤回通知引用被作废的旧修订；
* `status`：`stable`（不会再变）/ `provisional`（可能被修订或撤回）；
* `sources` / `source_range` / `provenance`：来源范围；
* `assumed_gap`、`filter_padding`、`reasons`：临时/填充原因。

对象同时支持属性与 `frame["index"]` 字典式访问，`to_dict()` 是 JSON 安全结构。

### SourceRange（帧来源范围）

```
session_id, seq, sample_rate
start_sample, end_sample     # 该 seq 块内的半开样本区间 [start, end)
start_seconds, end_seconds   # 会话级精确秒区间（内部为 Fraction）
```

一帧可包含多个有序 SourceRange；跨越采样率边界的帧会同时列出旧速率 seq 与
新速率 seq。解码器可用 `frame.source_range` 一次性确认整帧来源包络，或用
`frame.sources` 逐段核对，处理完成后可调 `op.acknowledge(frame_id)`。

## 关键语义

### 1. 稳定时间基准与无漂移切换

全局模型 tick `g` 的时间是 `g / model_rate` 秒，内部全部用
`fractions.Fraction` 做有理数换算。每个输入块/速率段的起点由
`Σ len / sample_rate` 精确推出，不做“每段四舍五入再累加”，因此 44100→16000
这类非整除比率与多次速率切换都不会产生累积漂移。

### 2. 采样率边界的滤波状态隔离

连续 seq 且同采样率的块组成一个重采样 run，共享滤波器历史（块边界不丢状态）。
一旦采样率改变或 seq 之间存在缺口，就开启**全新 run / 全新滤波器**：

* 旧 run 的边缘抽头只在旧速率自身样本上做边缘填充并 flush；
* 新 run 不接收任何旧速率状态；
* 全局 tick 到源坐标的映射、抽头支撑区间都被限制在对应 run 内。

因此“旧滤波状态只能影响旧速率覆盖的样本”是数据结构保证，而非约定。
独立流式重采样器见 `StreamingResampler` / `BoundaryAwareResampler`
（别名 `Resampler` / `BoundaryResampler`）。

### 3. 迟到、重复、缺失与冲突

`ReorderBuffer`（函数式 `reorder_chunks`）负责序号：

* **乱序/迟到**：任意到达顺序都按 seq 重排；已推进高水位后到达的低 seq 标记
  `kind="late"`；
* **重复**：同 seq 且 `sample_rate/samples` 相同 => `duplicate_ignored`，
  幂等无副作用；`strict_duplicates=True` 时抛 `DuplicateChunkError`；
* **内容冲突**：同 seq 但速率/样本不同 => 默认 `conflict` 并保留旧内容，
  **绝不静默覆盖或拼接**；可配 `on_conflict="replace"` 显式替换（随后触发
  重算与撤回），或 `"error"` 抛 `ChunkConflictError`；
* **缺失**：`missing_seqs()` / `gap_ranges()` 显式给出缺口。

缺口策略 `gap_policy`：

* `"provisional"`（默认，低延迟）：缺口先按**零时长**乐观拼接，缺口之后的帧
  标记 `provisional` 且 `assumed_gap=True`；缺口块迟到后用“确定性重算 +
  签名 diff”发出撤回与修订；
* `"wait"`：缺口后的块只缓存，不进入时间线；补齐后首次成帧，因此没有撤回。

### 4. 先交付 + 保留/撤回协议

每来一个块，算子基于块日志确定性重算并 diff：

| 情况 | 事件 |
| --- | --- |
| 新帧索引 | `result.frames` 中 `revision=1` |
| 内容变化 | 先 `result.retractions`（旧 `revision`），再发 `revision+1` 的替换帧 |
| 仅 provisional→stable | `result.confirmations`（内容不变，解码器可标记保留） |
| 帧变得不可物化 | 撤回，reason=`no_longer_available`，并联动撤回 OLA 块 |

`result.retained_frame_ids` 列出本次更新中**未被撤回、可继续保留**的已交付帧。
稳定性判定是保守的：只要帧的摆放依赖假设零时长缺口，或滤波抽头需要尚未到达的
前瞻/历史，就仍是 provisional。测试验证了在数百种乱序/缺口排列下，**一旦某
修订被标记 stable，就绝不会出现在后续 retractions 中**。

### 5. 恢复计划：可重放 vs 必须重新取样

`op.recovery_plan()` 返回 `RecoveryPlan`（别名 `GapReport`），例如对
data/input.json 的 seq `[0,1,3]`：

```text
contiguous_through_seq = 1     # 连续前缀
checkpoint_seq         = 1     # 可建立检查点的位置
replay_from_seq        = 2
missing_seqs           = [2]
resample_required_seqs = [2]   # 这些区间必须由对端重新取样/补传
replayable_seqs        = [0, 1, 3]  # 已接收块均可确定性重放
buffered_after_gap_seqs = [3]  # 缺口后已缓存、补齐即可重放
retain_frame_indices    = [0, 1, 2, 3]
provisional_frame_indices = [4, 5]
```

`op.checkpoint()` 给出 JSON 可序列化快照（配置 + 块日志 + 各帧修订号 +
撤回记录），`StreamingAudioOperator.from_checkpoint(cp)` 可恢复；恢复后再喂
迟到块产生的事件与未中断运行完全一致。`op.replay(from_seq=...)` 可从块日志
确定性重放帧序列。

### 6. 重叠相加 OverlapAdd

`OverlapAdd(session_id, window_size, hop, window="hann", normalize=True)`：

* `submit(frame)` / `apply_batch(frames, removals=...)` 输入帧；
* 输出按 hop 切分的 `OlaBlock`（同样带 PTS、合并来源、贡献帧及修订号）；
* Hann + `window_size=2*hop` 满足 COLA，内部区间可精确重建；矩形窗 +
  权重归一化可对任意 hop/window 比例稳定重建；
* 帧修订/删除会原子地传播为重叠块的 `OlaRetraction` 与更高修订的替换块，
  同一次 ingest 中每个块最多撤回一次；迟到的旧修订号会被拒绝，防止状态回退；
* 支持乱序提交（依赖帧到齐才交付块）与 `acknowledge("s#B3")` 确认。

引擎默认挂载一个 OLA，其事件在 `OperatorResult.ola_blocks` /
`ola_retractions`；也可设 `enable_ola=False` 后单独使用。函数式 API：
`overlap_add(frames, ...)`。

### 7. 两档滤波精度

| `quality` | 核 | 特征 |
| --- | --- | --- |
| `"low_latency"`（别名 `low`/`fast`/`realtime`） | 两点线性插值 `LinearKernel` | 支撑半径 1，最快稳定 |
| `"high"`（别名 `hq`/`best`） | Kaiser(β=8.6) 加窗 sinc，默认半径 8，下采样带抗混叠截止 | 精度高、前瞻大，尾部更多 provisional，待块到达或 `flush()` 后确认 |

两档产生相同的帧索引、PTS 网格以及**与精度无关的契约级 SourceRange**
（由帧时间窗口映射）；数值与滤波器抽头支撑可以不同。

## data/input.json 的实际行为

输入 seq 为 `[0,1,3]`、采样率 `16000/8000`、hop=2。以 model_rate=16000、
window_size=4、低延迟档为例，算子立即交付 6 帧：

```text
F0 [0,1,2,3]        stable     src seq0
F1 [2,3,4,5]        stable     src seq0 + seq1   (跨越 16k->8k)
F2 [4,5,5.5,6]      stable     src seq0 + seq1
F3 [5.5,6,6.5,7]    stable     src seq1
F4 [6.5,7,7,9]      provisional (缺口 seq2 的零时长假设 + 滤波器前瞻)
F5 [7,9,9.5,10]     provisional
```

迟到补入 `seq2=[8]` 后：撤回 `F4@r1`、`F5@r1`，交付 `F4@r2`、`F5@r2` 与
新的 `F6`；`F0..F3` 原样保留，对应 OLA 块 `B4/B5` 也各撤回一次并给出修订块。
最终模型样本流正是 `0..10` 在 8k 段线性二倍上采样后的连续序列。

## 模块导览

| 模块 | 主要名称 |
| --- | --- |
| `types.py` | `AudioChunk` `Frame` `SourceRange` `RecoveryPlan` `OperatorConfig` 枚举 |
| `buffer.py` / `reorder.py` | `ReorderBuffer` `Checkpoint` `reorder_chunks` 异常 |
| `kernels.py` | `LinearKernel` `WindowedSincKernel` `make_kernel` |
| `resampler.py` | `StreamingResampler` `BoundaryAwareResampler` |
| `timeline.py` | 有理数时间线、run 隔离、`materialize` |
| `ola.py` | `OverlapAdd` `OlaBlock` `OlaRetraction` `overlap_add` |
| `engine.py` / `operator.py` | `StreamingAudioOperator` `AudioChunkOperator` `OperatorResult` |
| `io.py` / `pipeline.py` | `process_payload` `run_payload` `process_chunks` |

根包 `operator_audio_chunks` 重导出全部公开名称，并提供 `ModelFrame`、
`AudioFrame`、`FrameSource`、`GapReport`、`OLA`、`Operator` 等常见别名。
