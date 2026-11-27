"""带来源追踪与撤回传播的重叠相加（OLA）算子。

输入若干固定 hop、长度 window_size 的模型帧（允许同一 index 以更高
revision 提交修订帧），输出按 hop 切分的重建块（默认 Hann 综合窗 +
权重归一化，在 window_size = 2*hop 时内部区间满足 COLA，可精确重建）。

当某帧被修订或撤回时，所有与该帧重叠、且已经交付的 OLA 块都会收到
:block:`BlockRetraction`，随后给出重算的新修订块；未受影响的块不会
重复发送。每个输出块同样携带合并后的来源范围，时间/来源契约与 Frame
保持一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

from .types import ApiObject, SourceRange, merge_source_ranges


def _clip_source_range(
    sr: SourceRange, t0: Fraction, t1: Fraction
) -> SourceRange | None:
    """把来源段按 OLA 块的秒区间裁剪，并同步换算块内样本偏移。"""

    start_time = sr.start_time
    end_time = sr.end_time
    if end_time <= start_time and sr.end_sample > sr.start_sample:
        # 调用方只给了样本偏移、未填精确时间时，按块内 0 点推导
        start_time = Fraction(sr.start_sample, sr.sample_rate)
        end_time = Fraction(sr.end_sample, sr.sample_rate)

    a = max(start_time, t0)
    b = min(end_time, t1)
    if a >= b:
        return None
    rate = sr.sample_rate
    start_sample = sr.start_sample + int(
        (a - start_time) * rate
    )  # floor：向左保守覆盖
    end_exact = sr.start_sample + (b - start_time) * rate
    # ceil：向右保守覆盖，保证区间内样本不丢来源
    end_sample = -((-end_exact.numerator) // end_exact.denominator)
    start_sample = max(sr.start_sample, min(start_sample, sr.end_sample))
    end_sample = max(start_sample, min(sr.end_sample, end_sample))
    if end_sample <= start_sample:
        # 时间相交但未覆盖完整样本中心：至少保留一个样本的来源
        mid_exact = sr.start_sample + ((a + b) / 2 - start_time) * rate
        idx = mid_exact.numerator // mid_exact.denominator
        idx = max(sr.start_sample, min(idx, sr.end_sample - 1))
        start_sample, end_sample = idx, idx + 1
    return SourceRange(
        session_id=sr.session_id,
        seq=sr.seq,
        sample_rate=rate,
        start_sample=start_sample,
        end_sample=end_sample,
        start_time=a,
        end_time=b,
    )


@dataclass(frozen=True)
class OlaBlock(ApiObject):
    session_id: str
    index: int
    hop: int
    model_rate: int
    samples: tuple[float, ...]
    revision: int = 1
    frame_indices: tuple[int, ...] = ()
    frame_revisions: tuple[int, ...] = ()
    sources: tuple[SourceRange, ...] = ()
    window: str = "rect"
    normalized: bool = True

    @property
    def block_id(self) -> str:
        return f"{self.session_id}#B{self.index}"

    @property
    def id(self) -> str:
        return self.block_id

    @property
    def pts_samples(self) -> int:
        return self.index * self.hop

    @property
    def pts_seconds(self) -> float:
        return self.index * self.hop / self.model_rate

    @property
    def values(self) -> tuple[float, ...]:
        return self.samples

    @property
    def source_ranges(self) -> tuple[SourceRange, ...]:
        return self.sources

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ola_block",
            "session_id": self.session_id,
            "index": self.index,
            "block_index": self.index,
            "block_id": self.block_id,
            "id": self.block_id,
            "revision": self.revision,
            "hop": self.hop,
            "model_rate": self.model_rate,
            "pts_samples": self.pts_samples,
            "pts_seconds": self.pts_seconds,
            "samples": list(self.samples),
            "values": list(self.samples),
            "frame_indices": list(self.frame_indices),
            "frame_revisions": list(self.frame_revisions),
            "sources": [r.to_dict() for r in self.sources],
            "source_ranges": [r.to_dict() for r in self.sources],
            "window": self.window,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class OlaRetraction(ApiObject):
    session_id: str
    index: int
    revision: int
    reason: str = "frame_revised"
    block_id: str | None = None
    replacement_revision: int | None = None
    kind: str = "ola_block"

    def __post_init__(self) -> None:
        if self.block_id is None:
            object.__setattr__(self, "block_id", f"{self.session_id}#B{self.index}")

    @property
    def target_revision(self) -> int:
        return self.revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ola_retraction",
            "target_kind": "ola_block",
            "session_id": self.session_id,
            "index": self.index,
            "block_index": self.index,
            "block_id": self.block_id,
            "id": self.block_id,
            "revision": self.revision,
            "target_revision": self.revision,
            "replacement_revision": self.replacement_revision,
            "reason": self.reason,
        }


BlockRetraction = OlaRetraction  # 直观别名


def synthesis_window(name: str, size: int) -> list[float]:
    name = (name or "rect").lower()
    if name in ("rect", "rectangular", "boxcar", "none"):
        return [1.0] * size
    if name == "hann":
        # 周期 Hann：与 hop=size/2 满足 COLA
        return [0.5 - 0.5 * math.cos(2.0 * math.pi * n / size) for n in range(size)]
    if name == "hamming":
        return [0.54 - 0.46 * math.cos(2.0 * math.pi * n / size) for n in range(size)]
    if name in ("oneshot", "trapezoid"):
        # 极简梯形窗，首尾各 1/4 斜坡
        w = [1.0] * size
        ramp = max(1, size // 4)
        for i in range(ramp):
            v = (i + 1) / ramp
            w[i] = v
            w[size - 1 - i] = v
        return w
    raise ValueError(f"未知综合窗: {name!r}")


@dataclass
class _StoredFrame:
    values: tuple[float, ...]
    revision: int
    sources: tuple[SourceRange, ...]


@dataclass
class _BlockRecord:
    block: OlaBlock
    signature: tuple[Any, ...]


@dataclass
class OlaResult:
    blocks: list[OlaBlock] = field(default_factory=list)
    retractions: list[OlaRetraction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "retractions": [r.to_dict() for r in self.retractions],
        }


class OverlapAdd:
    """流式重叠相加器 / 修订传播器。

    参数
    ----
    session_id:
        会话标识（决定 block_id 命名空间）。
    window_size, hop, model_rate:
        与 Frame 契约一致。
    window:
        ``hann``（默认）/ ``hamming`` / ``rect`` / ``trapezoid``。
    normalize:
        按累积窗权重归一化，保证任意 hop/window 比例下幅度稳定。
    """

    def __init__(
        self,
        session_id: str,
        window_size: int,
        hop: int,
        model_rate: int = 16000,
        window: str = "hann",
        normalize: bool = True,
    ) -> None:
        if hop <= 0 or window_size <= 0:
            raise ValueError("hop/window_size 必须为正")
        self.session_id = session_id
        self.window_size = window_size
        self.hop = hop
        self.model_rate = model_rate
        self.window_name = window
        self.normalize = normalize
        self.window_weights = synthesis_window(window, window_size)
        self._frames: dict[int, _StoredFrame] = {}
        self._blocks: dict[int, _BlockRecord] = {}
        self.acked_blocks: set[int] = set()

    # -- 对外操作 ---------------------------------------------------------
    def submit(
        self,
        frame: Any = None,
        *,
        index: int | None = None,
        values: Sequence[float] | None = None,
        revision: int = 1,
        sources: Sequence[SourceRange] = (),
    ) -> OlaResult:
        """提交一帧（可以是 :class:`~operator_audio_chunks.Frame`）。

        相同 index 且 revision 更新 => 修订；相同 revision 相同内容 => no-op。
        """

        if frame is not None:
            index = frame.index
            values = frame.samples
            revision = getattr(frame, "revision", 1)
            sources = getattr(frame, "sources", ())
        if index is None or values is None:
            raise TypeError("submit 需要 frame 或 index+values")
        if len(values) != self.window_size:
            raise ValueError(
                f"OLA 期望帧长 {self.window_size}，收到 {len(values)} (index={index})"
            )

        view = SimpleNamespace(
            index=int(index),
            samples=tuple(float(v) for v in values),
            revision=int(revision),
            sources=tuple(sources),
        )
        return self.apply_batch([view])

    # 常见方法名别名
    add_frame = submit
    process_frame = submit
    push_frame = submit

    def retract_frame(self, index: int, reason: str = "frame_retracted") -> OlaResult:
        """删除一帧贡献；受影响且已交付的块会被撤回。"""

        if index not in self._frames:
            return OlaResult()
        del self._frames[index]
        return self.apply_batch(removals=[index])

    def apply_batch(
        self,
        frames: Iterable[Any] | None = None,
        *,
        upserts: Iterable[Any] | None = None,
        removals: Iterable[int] = (),
    ) -> OlaResult:
        """原子应用一批帧修订/删除，每个受影响块最多产生一次撤回。

        位置参数 ``frames`` 是 :class:`~operator_audio_chunks.Frame` 的便捷
        传入方式；也可用 ``upserts`` 传同样的对象。
        """

        upsert_list = list(frames if frames is not None else (upserts or []))
        removal_set = set(int(i) for i in removals)

        affected: set[int] = set()
        for frame in upsert_list:
            idx = frame.index
            old = self._frames.get(idx)
            new_rev = int(getattr(frame, "revision", 1))
            if old is not None and new_rev < old.revision:
                continue  # 迟到的旧修订：拒绝状态回退
            self._frames[idx] = _StoredFrame(
                tuple(float(v) for v in frame.samples),
                new_rev,
                tuple(getattr(frame, "sources", ())),
            )
            affected.update(range(idx, idx + self._m_frames()))
        for idx in removal_set:
            self._frames.pop(idx, None)
            affected.update(range(idx, idx + self._m_frames()))

        result = OlaResult()
        for b in sorted(affected):
            if b < 0:
                continue
            old_rec = self._blocks.get(b)
            if not self._block_complete(b):
                if old_rec is not None:
                    self._blocks.pop(b, None)
                    result.retractions.append(
                        OlaRetraction(
                            session_id=self.session_id,
                            index=b,
                            revision=old_rec.block.revision,
                            reason="dependency_removed",
                        )
                    )
                continue

            block, signature = self._build_block(b)
            if old_rec is None:
                self._blocks[b] = _BlockRecord(block, signature)
                result.blocks.append(block)
            elif old_rec.signature != signature:
                new_rev = old_rec.block.revision + 1
                revised = OlaBlock(
                    session_id=block.session_id,
                    index=block.index,
                    hop=block.hop,
                    model_rate=block.model_rate,
                    samples=block.samples,
                    revision=new_rev,
                    frame_indices=block.frame_indices,
                    frame_revisions=block.frame_revisions,
                    sources=block.sources,
                    window=block.window,
                    normalized=block.normalized,
                )
                result.retractions.append(
                    OlaRetraction(
                        session_id=self.session_id,
                        index=b,
                        revision=old_rec.block.revision,
                        reason="frames_revised",
                        replacement_revision=new_rev,
                    )
                )
                self._blocks[b] = _BlockRecord(revised, signature)
                result.blocks.append(revised)
        return result

    def acknowledge(self, block_id_or_index: str | int) -> None:
        idx = self._coerce_index(block_id_or_index)
        if idx is not None:
            self.acked_blocks.add(idx)

    ack = acknowledge

    def delivered_blocks(self) -> list[OlaBlock]:
        return [rec.block for _, rec in sorted(self._blocks.items())]

    def get_block(self, index: int) -> OlaBlock | None:
        rec = self._blocks.get(index)
        return rec.block if rec else None

    # -- 内部计算 ---------------------------------------------------------
    def _coerce_index(self, token: str | int) -> int | None:
        if isinstance(token, int):
            return token
        if isinstance(token, str) and "#B" in token:
            try:
                return int(token.rsplit("#B", 1)[1])
            except ValueError:
                return None
        return None

    def _m_frames(self) -> int:
        """一个窗口跨越的 hop 块数（向上取整）。"""
        return (self.window_size + self.hop - 1) // self.hop

    def _required_frame_indices(self, block_index: int) -> list[int]:
        m = self._m_frames()
        lo = max(0, block_index - m + 1)
        return list(range(lo, block_index + 1))

    def _block_complete(self, block_index: int) -> bool:
        return all(k in self._frames for k in self._required_frame_indices(block_index))

    def _build_block(self, b: int) -> tuple[OlaBlock, tuple[Any, ...]]:
        h = self.hop
        w = self.window_size
        start = b * h
        ys = [0.0] * h
        ws = [0.0] * h
        contributing: list[int] = []
        revisions: list[int] = []
        sources: list[SourceRange] = []

        block_t0 = Fraction(start, self.model_rate)
        block_t1 = Fraction(start + h, self.model_rate)

        # 精确枚举与块区间 [start, start+h) 相交的帧
        k_lo = max(0, b - self._m_frames())
        for k in range(k_lo, b + 1):
            stored = self._frames.get(k)
            if stored is None:
                continue
            frame_start = k * h
            frame_end = frame_start + w
            if not (frame_start < start + h and frame_end > start):
                continue  # 数学上不相交，不能计入来源/依赖

            for j, weight in enumerate(self.window_weights):
                global_sample = frame_start + j
                local = global_sample - start
                if 0 <= local < h:
                    ys[local] += weight * stored.values[j]
                    ws[local] += weight
            contributing.append(k)
            revisions.append(stored.revision)
            for sr in stored.sources:
                clipped = _clip_source_range(sr, block_t0, block_t1)
                if clipped is not None:
                    sources.append(clipped)

        if self.normalize:
            out = [
                (ys[i] / ws[i]) if ws[i] > 1e-12 else 0.0
                for i in range(h)
            ]
        else:
            out = ys

        merged_sources = merge_source_ranges(sources)
        block = OlaBlock(
            session_id=self.session_id,
            index=b,
            hop=h,
            model_rate=self.model_rate,
            samples=tuple(out),
            revision=1,
            frame_indices=tuple(contributing),
            frame_revisions=tuple(revisions),
            sources=merged_sources,
            window=self.window_name,
            normalized=self.normalize,
        )
        signature = (
            tuple(round(v, 12) for v in out),
            tuple(contributing),
            tuple(revisions),
            tuple((r.seq, r.start_sample, r.end_sample) for r in merged_sources),
        )
        return block, signature

    def flush_tail(self) -> OlaResult:
        """会话结束：当前实现按 hop 块交付，尾部不足一窗的帧不会产生新块。

        保留该方法以表达显式结束边界；最终化由引擎以 ``final=True``
        重算帧后再次提交完成。
        """

        return OlaResult()


# 函数式便捷 API ------------------------------------------------------------


def overlap_add(
    frames: Iterable[Any],
    *,
    window_size: int | None = None,
    hop: int | None = None,
    model_rate: int = 16000,
    session_id: str = "session",
    window: str = "hann",
    normalize: bool = True,
) -> list[OlaBlock]:
    """一次性对可迭代 Frame 做重叠相加，返回按 index 排序的块。"""

    frames = list(frames)
    if not frames:
        return []
    first = frames[0]
    hop = hop or getattr(first, "hop", None)
    window_size = window_size or getattr(first, "window_size", None)
    if hop is None or window_size is None:
        raise ValueError("需要显式 hop/window_size")
    ola = OverlapAdd(
        session_id=getattr(first, "session_id", session_id),
        window_size=window_size,
        hop=hop,
        model_rate=getattr(first, "model_rate", model_rate),
        window=window,
        normalize=normalize,
    )
    blocks: list[OlaBlock] = []
    for frame in sorted(frames, key=lambda f: f.index):
        blocks.extend(ola.submit(frame).blocks)
    return sorted(blocks, key=lambda b: b.index)
