"""会话时间线与模型帧物化（算子的数学核心）。

设计要点
--------
1. **稳定时间基准**：以模型采样率的整数 tick/帧序号为骨干。所有边界用
   :class:`fractions.Fraction` 精确换算，逐块四舍五入不会造成漂移。
2. **采样率段隔离**：连续 seq 且同采样率的块组成一个 resampling run，
   共享滤波历史；采样率切换或 seq 缺口都会开启新 run，旧滤波器状态在
   结构上不可能触及新速率段的样本。
3. **来源契约**：每帧给出覆盖的 ``(seq, 块内样本区间, 采样率, 秒区间)``。
4. **最终性**：只有当帧不依赖“假设零时长的缺口摆放”，且所有滤波抽头
   要么自然可用、要么位于已闭合的速率段边缘时，帧才标记为 stable。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .buffer import ReorderBuffer
from .kernels import frac_ceil, frac_floor, make_kernel
from .types import (
    Frame,
    FrameStatus,
    GapPolicy,
    OperatorConfig,
    SourceRange,
    merge_source_ranges,
)


@dataclass
class Run:
    """连续 seq、同采样率的重采样段（共享一份滤波历史）。"""

    rate: int
    start_time: Fraction
    tick_start: Fraction
    seqs: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    samples: list[float] = field(default_factory=list)
    start_open: bool = False
    end_open: bool = True
    placement_assumed: bool = False
    model_rate: int = 16000

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def tick_end(self) -> Fraction:
        # 必须动态计算：run 会在连续同速率块上继续 append 样本
        return self.tick_start + Fraction(
            self.n_samples * self.model_rate, self.rate
        )

    @property
    def end_time(self) -> Fraction:
        return self.start_time + Fraction(self.n_samples, self.rate)


@dataclass
class Materialization:
    frames: list[Frame]
    total_ticks: Fraction
    tick_count: int
    runs: list[Run]


def _build_runs(
    config: OperatorConfig, buffer: ReorderBuffer, final: bool
) -> tuple[list[Run], Fraction]:
    session_id = buffer.session_id
    first_seq = config.first_seq
    max_seq = buffer.max_seq()
    if max_seq is None:
        return [], Fraction(0)

    if config.gap_policy == GapPolicy.WAIT:
        seq_iter = range(first_seq, (buffer.contiguous_through() or first_seq - 1) + 1)
        known_missing: set[int] = set()
    else:
        seq_iter = range(first_seq, max_seq + 1)
        known_missing = set(buffer.missing_seqs())

    runs: list[Run] = []
    t = Fraction(0)
    gap_seen = False
    current: Run | None = None

    for s in seq_iter:
        chunk = buffer.get(s)
        if chunk is None:
            # 缺失块：provisional 策略下按零时长处理，但污染后续摆放可信度
            gap_seen = True
            current = None
            continue

        can_append = (
            current is not None
            and chunk.seq == current.seqs[-1] + 1
            and chunk.sample_rate == current.rate
        )
        if can_append:
            assert current is not None
            current.offsets.append(current.n_samples)
            current.lengths.append(chunk.num_samples)
            current.seqs.append(chunk.seq)
            current.samples.extend(chunk.samples)
        else:
            current = Run(
                rate=chunk.sample_rate,
                start_time=t,
                tick_start=t * config.model_rate,
                seqs=[chunk.seq],
                lengths=[chunk.num_samples],
                offsets=[0],
                samples=list(chunk.samples),
                placement_assumed=gap_seen,
                model_rate=config.model_rate,
            )
            runs.append(current)

        t += chunk.duration

    # 判定每个 run 两端是否“开放”（未来可能有同速率块延伸滤波器支撑）
    for run in runs:
        first_s = run.seqs[0]
        last_s = run.seqs[-1]
        if final:
            run.start_open = False
            run.end_open = False
        else:
            if first_s == first_seq:
                run.start_open = False  # 会话起点之前约定为静音/边缘填充
            else:
                run.start_open = buffer.get(first_s - 1) is None
            nxt = buffer.get(last_s + 1)
            # 下一块存在时，最大连续同速率 run 必然因速率不同而闭合
            run.end_open = nxt is None
    return runs, t


def _map_run_interval_to_sources(
    run: Run, a_tick: Fraction, b_tick: Fraction, session_id: str
) -> tuple[SourceRange, ...]:
    """把 run 的 tick 坐标区间 ``[a_tick, b_tick)`` 映射为块级来源段。"""

    rate = run.rate
    model_rate = run.model_rate
    sa = (a_tick - run.tick_start) * Fraction(rate, model_rate)
    sb = (b_tick - run.tick_start) * Fraction(rate, model_rate)
    lo = frac_floor(sa)
    hi_excl = frac_ceil(sb)
    if hi_excl <= lo:
        hi_excl = lo + 1
    lo = max(0, lo)
    hi_excl = min(run.n_samples, hi_excl)
    if hi_excl <= lo:
        return ()

    ranges: list[SourceRange] = []
    pos = lo
    j = max(0, bisect_right(run.offsets, pos) - 1)
    while pos < hi_excl and j < len(run.seqs):
        off = run.offsets[j]
        length = run.lengths[j]
        chunk_end = off + length
        end = min(hi_excl, chunk_end)
        start_in_chunk = pos - off
        end_in_chunk = end - off
        start_time = run.start_time + Fraction(off + start_in_chunk, rate)
        end_time = run.start_time + Fraction(off + end_in_chunk, rate)
        ranges.append(
            SourceRange(
                session_id=session_id,
                seq=run.seqs[j],
                sample_rate=rate,
                start_sample=start_in_chunk,
                end_sample=end_in_chunk,
                start_time=start_time,
                end_time=end_time,
            )
        )
        pos = end
        j += 1
    return merge_source_ranges(ranges)


def materialize(
    config: OperatorConfig,
    buffer: ReorderBuffer,
    *,
    final: bool = False,
) -> Materialization:
    """根据当前缓冲内容重算全部候选模型帧。

    重算是确定性的：相同的已接纳块集合 + 配置必然得到逐值相同的候选帧，
    因此引擎可以安全地用“重算 + 签名 diff”产生撤回/修订。
    """

    session_id = buffer.session_id
    runs, total_seconds = _build_runs(config, buffer, final)
    total_ticks = total_seconds * config.model_rate
    tick_count = frac_ceil(total_ticks) if total_ticks > 0 else 0

    if not runs:
        return Materialization([], total_ticks, tick_count, [])

    kernels: dict[int, Any] = {}

    def kernel_for(rate: int) -> Any:
        if rate not in kernels:
            kernels[rate] = make_kernel(
                config.quality.value if hasattr(config.quality, "value") else config.quality,
                in_rate=rate,
                out_rate=config.model_rate,
                filter_radius=config.filter_radius,
                kaiser_beta=config.kaiser_beta,
            )
        return kernels[rate]

    hop = config.hop
    window = config.window_size
    assert window is not None

    frames: list[Frame] = []
    k = 0
    # 帧需要完整窗口的整数 tick；最后一个 tick 下标为 k*hop+window-1，
    # 它必须 < total_ticks，等价于 k*hop+window <= ceil(total_ticks)。
    while k * hop + window <= tick_count:
        g0 = k * hop
        g1 = g0 + window
        values: list[float] = []
        provisional = False
        used_padding = False
        reasons: set[str] = set()
        sources: list[SourceRange] = []

        # run 按 tick 有序，用单调指针查找
        run_idx = 0
        for g in range(g0, g1):
            gt = Fraction(g, 1)
            while run_idx < len(runs) and not (
                runs[run_idx].tick_start <= gt < runs[run_idx].tick_end
            ):
                # 可能恰好落在零时长缺口（不会发生，缺口时长为 0），
                # 或越过当前 run；推进到下一个
                if gt >= runs[run_idx].tick_end:
                    run_idx += 1
                else:  # pragma: no cover - 防御性分支
                    break
            if run_idx >= len(runs):
                # 超出已知媒体范围（理论上被 tick_count 条件排除）
                reasons.add("out_of_range")
                provisional = True
                values.append(0.0)
                continue

            run = runs[run_idx]
            p = (gt - run.tick_start) * Fraction(run.rate, config.model_rate)
            kernel = kernel_for(run.rate)
            values.append(kernel.interpolate(run.samples, p))

            lo, hi = kernel.required_range(p)
            if lo < 0 or hi >= run.n_samples:
                used_padding = True
                if (lo < 0 and run.start_open) or (hi >= run.n_samples and run.end_open):
                    provisional = True
                    if hi >= run.n_samples and run.end_open:
                        reasons.add("filter_lookahead_pending")
                    if lo < 0 and run.start_open:
                        reasons.add("filter_history_pending")
            if run.placement_assumed:
                provisional = True
                reasons.add("assumed_gap_placement")

        # 契约级来源：帧窗口的模型时间区间映射（与滤波精度无关）
        for run in runs:
            a = max(Fraction(g0), run.tick_start)
            b = min(Fraction(g1), run.tick_end)
            if a < b:
                sources.extend(
                    _map_run_interval_to_sources(run, a, b, session_id)
                )
        sources = list(merge_source_ranges(sources))

        status = FrameStatus.PROVISIONAL if provisional else FrameStatus.STABLE
        frame = Frame(
            session_id=session_id,
            index=k,
            hop=hop,
            window_size=window,
            model_rate=config.model_rate,
            samples=tuple(values),
            sources=tuple(sources),
            status=status,
            revision=1,
            quality=config.quality.value,
            assumed_gap="assumed_gap_placement" in reasons,
            filter_padding=used_padding,
            reasons=tuple(sorted(reasons)),
        )
        frames.append(frame)
        k += 1

    return Materialization(frames, total_ticks, tick_count, runs)
