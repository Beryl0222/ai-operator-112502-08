"""边界感知的流式重采样器（独立可复用算子）。

与 ``timeline`` 中“全局模型 tick 网格 + 确定性重算”不同，本模块给出经典的
相位递增流式重采样器，适合直接串在重排缓冲之后：

* 同采样率的连续块共享滤波器/样本历史，块边界不丢状态；
* 采样率切换时先把旧段在**旧速率样本覆盖范围**内 flush 完，再以全新状态
  开启新段 —— 旧滤波器状态在数据结构上不可能参与新速率输出；
* 每个输出样本携带精确的输入支撑区间，并可映射到 ``(seq, 块内偏移)``。

两档精度由 :func:`operator_audio_chunks.kernels.make_kernel` 决定，
输出的来源契约字段保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from .kernels import make_kernel
from .types import Quality, SourceRange


@dataclass(frozen=True)
class ResampledSample:
    """单个重采样输出。``position`` 是当前速率段内的输入样本坐标。"""

    index: int
    value: float
    in_rate: int
    out_rate: int
    position: Fraction
    source_lo: int
    source_hi: int
    padded: bool = False
    sources: tuple[SourceRange, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "value": self.value,
            "in_rate": self.in_rate,
            "out_rate": self.out_rate,
            "position": (
                self.position.numerator
                if self.position.denominator == 1
                else f"{self.position.numerator}/{self.position.denominator}"
            ),
            "source_lo": self.source_lo,
            "source_hi": self.source_hi,
            "padded": self.padded,
            "sources": [s.to_dict() for s in self.sources],
        }


class StreamingResampler:
    """固定输入/输出速率、跨块保留历史的相位递增重采样器。"""

    def __init__(
        self,
        in_rate: int,
        out_rate: int = 16000,
        quality: Quality | str = Quality.LOW_LATENCY,
        filter_radius: int | None = None,
        kaiser_beta: float = 8.6,
        session_id: str = "session",
    ) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError("采样率必须为正")
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self.session_id = session_id
        self.kernel = make_kernel(
            quality, in_rate, out_rate, filter_radius, kaiser_beta
        )
        self._samples: list[float] = []
        self._next_index: int = 0
        self.finished: bool = False

    # -- 相位 -------------------------------------------------------------
    def position_of(self, output_index: int) -> Fraction:
        return Fraction(output_index * self.in_rate, self.out_rate)

    def push(self, samples: Sequence[float]) -> list[ResampledSample]:
        if self.finished:
            raise RuntimeError("resampler 已 finish；速率切换请新建实例")
        self._samples.extend(float(x) for x in samples)
        return self._emit(natural_only=True)

    def finish(self) -> list[ResampledSample]:
        """段结束（速率切换/会话结束）：用边缘填充交付剩余输出。"""

        tail = self._emit(natural_only=False)
        self.finished = True
        return tail

    flush = finish

    # -- 核心 -------------------------------------------------------------
    def _emit(self, *, natural_only: bool) -> list[ResampledSample]:
        out: list[ResampledSample] = []
        n = len(self._samples)
        while True:
            j = self._next_index
            p = self.position_of(j)
            if p >= n:
                # 位置到达/越过输入跨度末端：段内输出已穷尽
                break
            lo, hi = self.kernel.required_range(p)
            if natural_only and hi >= n:
                # 前瞻不足，保留 phase，等下一块带来滤波器历史后重试 j
                break
            value = self.kernel.interpolate(self._samples, p)
            padded = lo < 0 or hi >= n
            out.append(
                ResampledSample(
                    index=j,
                    value=value,
                    in_rate=self.in_rate,
                    out_rate=self.out_rate,
                    position=p,
                    source_lo=max(0, lo),
                    source_hi=min(n - 1, hi),
                    padded=padded,
                )
            )
            self._next_index += 1
        return out


@dataclass
class _ChunkLoc:
    seq: int
    start: int
    length: int


class BoundaryAwareResampler:
    """按块输入、自动在采样率边界隔离状态并映射 seq 来源。"""

    def __init__(
        self,
        out_rate: int = 16000,
        quality: Quality | str = Quality.LOW_LATENCY,
        filter_radius: int | None = None,
        session_id: str = "session",
    ) -> None:
        self.out_rate = out_rate
        self.quality = quality
        self.filter_radius = filter_radius
        self.session_id = session_id
        self.in_rate: int | None = None
        self._r: StreamingResampler | None = None
        self._locs: list[_ChunkLoc] = []
        self._segment_sample_base: int = 0
        self.outputs: list[ResampledSample] = []

    def process(
        self,
        chunk: Any = None,
        *,
        seq: int | None = None,
        sample_rate: int | None = None,
        samples: Sequence[float] | None = None,
    ) -> list[ResampledSample]:
        if chunk is not None:
            seq = chunk.seq if seq is None else seq
            sample_rate = (
                chunk.sample_rate if sample_rate is None else sample_rate
            )
            samples = chunk.samples if samples is None else samples
        if seq is None or sample_rate is None or samples is None:
            raise TypeError("process 需要 seq/sample_rate/samples 或 chunk")

        emitted: list[ResampledSample] = []
        if self._r is not None and sample_rate != self.in_rate:
            # 速率切换：旧状态只 flush 旧速率段，随后彻底丢弃
            assert self._r is not None
            emitted.extend(self._attach_sources(self._r.finish()))
            self._r = None
            self._locs = []
            self._segment_sample_base = 0

        if self._r is None:
            self.in_rate = int(sample_rate)
            self._r = StreamingResampler(
                self.in_rate,
                self.out_rate,
                quality=self.quality,
                filter_radius=self.filter_radius,
                session_id=self.session_id,
            )

        start = len(self._r._samples)  # 当前段坐标
        self._locs.append(_ChunkLoc(int(seq), start, len(samples)))
        emitted.extend(self._attach_sources(self._r.push(samples)))
        self.outputs.extend(emitted)
        return emitted

    process_chunk = process
    push = process

    def finish(self) -> list[ResampledSample]:
        if self._r is None:
            return []
        tail = self._attach_sources(self._r.finish())
        self.outputs.extend(tail)
        return tail

    flush = finish

    # -- 来源映射 ---------------------------------------------------------
    def _attach_sources(
        self, items: Sequence[ResampledSample]
    ) -> list[ResampledSample]:
        result: list[ResampledSample] = []
        for item in items:
            ranges = self._map_support(item.source_lo, item.source_hi)
            result.append(
                ResampledSample(
                    index=item.index,
                    value=item.value,
                    in_rate=item.in_rate,
                    out_rate=item.out_rate,
                    position=item.position,
                    source_lo=item.source_lo,
                    source_hi=item.source_hi,
                    padded=item.padded,
                    sources=tuple(ranges),
                )
            )
        return result

    def _map_support(self, lo: int, hi: int) -> list[SourceRange]:
        """把段坐标闭区间 [lo, hi] 映射到块级 SourceRange。"""

        if not self._locs:
            return []
        out: list[SourceRange] = []
        for loc in self._locs:
            c_lo = loc.start
            c_hi = loc.start + loc.length - 1
            if hi < c_lo or lo > c_hi:
                continue
            use_lo = max(lo, c_lo) - loc.start
            use_hi = min(hi, c_hi) - loc.start + 1  # 转半开
            rate = self.in_rate or 0
            out.append(
                SourceRange(
                    session_id=self.session_id,
                    seq=loc.seq,
                    sample_rate=rate,
                    start_sample=use_lo,
                    end_sample=use_hi,
                    start_time=Fraction(loc.start + use_lo, rate) if rate else Fraction(0),
                    end_time=Fraction(loc.start + use_hi, rate) if rate else Fraction(0),
                )
            )
        return out


# 直观别名
Resampler = StreamingResampler
BoundaryResampler = BoundaryAwareResampler
BoundaryAwareStreamingResampler = BoundaryAwareResampler
ResamplingOperator = BoundaryAwareResampler
