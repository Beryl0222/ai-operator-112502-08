"""公开数据契约：会话音频块、来源范围、模型帧、撤回与恢复计划。

本模块只依赖 Python 标准库。所有时间在内部使用 :class:`fractions.Fraction`
精确表示，对外同时给出 ``float`` 秒值，避免采样率切换处的累积漂移。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# 枚举（同时提供字符串值，方便 JSON 断言）
# ---------------------------------------------------------------------------


class Quality(str, enum.Enum):
    """滤波精度档位。

    * ``HIGH``：加窗 sinc（Kaiser）高精度重采样，滤波器支撑更大、前瞻更多；
    * ``LOW_LATENCY``：线性插值，支撑小、可尽快判定为稳定帧。

    ``LOW`` / ``FAST`` 是 ``LOW_LATENCY`` 的枚举别名（同值成员）。
    """

    HIGH = "high"
    LOW_LATENCY = "low_latency"
    LOW = "low_latency"
    FAST = "low_latency"


_QUALITY_ALIASES = {
    "high": Quality.HIGH,
    "hq": Quality.HIGH,
    "best": Quality.HIGH,
    "quality": Quality.HIGH,
    "low_latency": Quality.LOW_LATENCY,
    "low-latency": Quality.LOW_LATENCY,
    "lowlatency": Quality.LOW_LATENCY,
    "linear": Quality.LOW_LATENCY,
    "low": Quality.LOW_LATENCY,
    "fast": Quality.LOW_LATENCY,
    "realtime": Quality.LOW_LATENCY,
    "rt": Quality.LOW_LATENCY,
}


def coerce_quality(value: "Quality | str") -> Quality:
    if isinstance(value, Quality):
        return value
    key = str(value).strip().lower()
    if key in _QUALITY_ALIASES:
        return _QUALITY_ALIASES[key]
    # 允许直接传枚举值
    return Quality(key)


_CONFIG_KWARG_ALIASES = {
    "frame_size": "window_size",
    "frame_length": "window_size",
    "win_size": "window_size",
    "window_len": "window_size",
    "samples_per_frame": "window_size",
    "hop_size": "hop",
    "frame_hop": "hop",
    "hop_length": "hop",
    "target_rate": "model_rate",
    "out_rate": "model_rate",
    "output_rate": "model_rate",
    "target_sample_rate": "model_rate",
    "output_sample_rate": "model_rate",
    "target_sr": "model_rate",
}


def normalize_config_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """把常见参数别名归一到 :class:`OperatorConfig` 的正式字段名。"""

    normalized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        canonical = _CONFIG_KWARG_ALIASES.get(key, key)
        # 正式名与别名同时出现时，以正式名为准
        if canonical in normalized and key in _CONFIG_KWARG_ALIASES:
            continue
        normalized[canonical] = value
    return normalized


class GapPolicy(str, enum.Enum):
    """序号缺口（缺失块）的处理策略。

    * ``PROVISIONAL``：缺口按零时长乐观拼接，先交付临时帧；缺口块迟到后
      通过撤回/修订纠正时间基准；
    * ``WAIT``：缺口之后的块只缓存，不参与时间线，直到缺口补齐。
    """

    PROVISIONAL = "provisional"
    WAIT = "wait"


class FrameStatus(str, enum.Enum):
    STABLE = "stable"          # 不会再被修订/撤回，解码器可保留
    PROVISIONAL = "provisional"  # 已交付但可能被后续块修订/撤回
    RETRACTED = "retracted"    # 已被撤回


class NoticeKind(str, enum.Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    DUPLICATE_IGNORED = "duplicate_ignored"
    CONFLICT = "conflict"
    LATE = "late"
    GAP = "gap"
    RATE_CHANGE = "rate_change"
    INVARIANT = "invariant_violation"


class ConflictPolicy(str, enum.Enum):
    REJECT = "reject"
    REPLACE = "replace"
    ERROR = "error"


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    """把契约对象递归转换成 ``json.dumps`` 可接受的纯数据。"""

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Fraction):
        # 精确有理数：整数直接给 int，否则给 "n/d" 字符串
        return obj.numerator if obj.denominator == 1 else f"{obj.numerator}/{obj.denominator}"
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    return str(obj)


class ApiObject:
    """让数据对象同时支持属性、字典下标与 ``get`` 访问。

    这样调用方无论习惯 ``frame.index`` 还是 ``frame["index"]`` 都能工作。
    """

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - 被子类覆盖
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key in data:
            return data[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_dict()


# ---------------------------------------------------------------------------
# 输入音频块
# ---------------------------------------------------------------------------

_CHUNK_KEY_ALIASES = {
    "session_id": ("session_id", "session", "sid", "conversation_id", "call_id"),
    "seq": ("seq", "sequence", "sequence_number", "seq_num", "index", "sn"),
    "sample_rate": ("sample_rate", "sr", "rate", "sampling_rate", "fs"),
    "samples": ("samples", "audio", "data", "pcm", "payload", "values", "chunk"),
    "pts": ("pts", "start_pts", "timestamp", "start_time", "offset_seconds"),
}


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


@dataclass(frozen=True)
class AudioChunk(ApiObject):
    """带会话、采样率和序号的音频块。

    序号是**会话内块序号**：同一会话中 ``seq`` 唯一标识一个块。块在媒体
    时间线上按 seq 顺序首尾相接；缺失 seq 的真实时长未知，不能静默填补。
    """

    session_id: str
    seq: int
    sample_rate: int
    samples: tuple[float, ...] = field(default=())
    pts: Fraction | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id 必须是非空字符串")
        if not isinstance(self.seq, int) or self.seq < 0:
            raise ValueError("seq 必须是非负整数")
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate 必须是正整数")
        coerced = tuple(float(x) for x in self.samples)
        object.__setattr__(self, "samples", coerced)
        if self.pts is not None and not isinstance(self.pts, Fraction):
            object.__setattr__(self, "pts", Fraction(float(self.pts)))

    # -- 常用派生量 --------------------------------------------------------
    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> Fraction:
        """块时长（秒，精确有理数）。"""
        return Fraction(len(self.samples), self.sample_rate)

    @property
    def duration_seconds(self) -> float:
        return float(self.duration)

    @property
    def key(self) -> tuple[str, int]:
        return (self.session_id, self.seq)

    def fingerprint(self) -> tuple[Any, ...]:
        """重复/冲突判定所用的内容指纹。"""
        return (self.sample_rate, tuple(self.samples))

    # -- 构造 / 序列化 -----------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: "AudioChunk | Mapping[str, Any]",
        session_id: str | None = None,
    ) -> "AudioChunk":
        if isinstance(data, AudioChunk):
            if session_id is not None and data.session_id != session_id:
                raise ValueError(
                    f"块会话 {data.session_id!r} 与指定会话 {session_id!r} 不一致"
                )
            return data
        if not isinstance(data, Mapping):
            raise TypeError(f"无法把 {type(data)!r} 解析为 AudioChunk")

        sid = _first_present(data, _CHUNK_KEY_ALIASES["session_id"]) or session_id
        if sid is None:
            raise ValueError("音频块缺少 session_id")
        seq = _first_present(data, _CHUNK_KEY_ALIASES["seq"])
        if seq is None:
            raise ValueError("音频块缺少 seq")
        rate = _first_present(data, _CHUNK_KEY_ALIASES["sample_rate"])
        if rate is None:
            raise ValueError("音频块缺少 sample_rate")
        samples = _first_present(data, _CHUNK_KEY_ALIASES["samples"])
        if samples is None:
            raise ValueError("音频块缺少 samples")
        pts = _first_present(data, _CHUNK_KEY_ALIASES["pts"])
        return cls(
            session_id=str(sid),
            seq=int(seq),
            sample_rate=int(rate),
            samples=tuple(samples),
            pts=Fraction(float(pts)) if pts is not None else None,
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "session_id", "session", "sid", "seq", "sequence",
                    "sample_rate", "sr", "rate", "samples", "audio", "data",
                    "pcm", "values", "pts", "start_pts", "timestamp",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "seq": self.seq,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "samples": list(self.samples),
            "duration_seconds": self.duration_seconds,
            "pts_seconds": float(self.pts) if self.pts is not None else None,
        }


# ---------------------------------------------------------------------------
# 来源范围
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRange(ApiObject):
    """一个模型帧区间所覆盖的、可向解码器确认的原始音频来源段。

    区间为半开 ``[start_sample, end_sample)``，偏移在 ``seq`` 块内。
    ``start_time`` / ``end_time`` 是会话级秒时间轴。
    """

    session_id: str
    seq: int
    sample_rate: int
    start_sample: int
    end_sample: int
    start_time: Fraction = Fraction(0)
    end_time: Fraction = Fraction(0)

    @property
    def start_seconds(self) -> float:
        return float(self.start_time)

    @property
    def end_seconds(self) -> float:
        return float(self.end_time)

    @property
    def duration_seconds(self) -> float:
        return float(self.end_time - self.start_time)

    @property
    def seq_range(self) -> tuple[int, int]:
        return (self.seq, self.seq)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "seq": self.seq,
            "sample_rate": self.sample_rate,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_seconds": float(self.start_time),
            "end_seconds": float(self.end_time),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class SourceSpan(ApiObject):
    """一帧的完整来源包络：若干有序 :class:`SourceRange`。"""

    session_id: str
    ranges: tuple[SourceRange, ...]

    @property
    def start_seq(self) -> int | None:
        return self.ranges[0].seq if self.ranges else None

    @property
    def end_seq(self) -> int | None:
        return self.ranges[-1].seq if self.ranges else None

    @property
    def seqs(self) -> tuple[int, ...]:
        return tuple(r.seq for r in self.ranges)

    @property
    def sample_rates(self) -> tuple[int, ...]:
        return tuple(sorted({r.sample_rate for r in self.ranges}))

    @property
    def start_seconds(self) -> float | None:
        return float(self.ranges[0].start_time) if self.ranges else None

    @property
    def end_seconds(self) -> float | None:
        return float(self.ranges[-1].end_time) if self.ranges else None

    # 让 source_range 可直接迭代/下标访问其 SourceRange 列表
    def __iter__(self):
        return iter(self.ranges)

    def __len__(self) -> int:
        return len(self.ranges)

    def __getitem__(self, item):
        return self.ranges[item]

    def __bool__(self) -> bool:
        return bool(self.ranges)

    def to_dict(self) -> dict[str, Any]:
        first = self.ranges[0] if self.ranges else None
        last = self.ranges[-1] if self.ranges else None
        return {
            "session_id": self.session_id,
            "start_seq": first.seq if first else None,
            "end_seq": last.seq if last else None,
            "seqs": list(self.seqs),
            "sample_rates": list(self.sample_rates),
            "start_seconds": float(first.start_time) if first else None,
            "end_seconds": float(last.end_time) if last else None,
            "ranges": [r.to_dict() for r in self.ranges],
        }


def merge_source_ranges(ranges: Iterable[SourceRange]) -> tuple[SourceRange, ...]:
    """排序并合并同 seq 的相邻/重叠来源段。"""

    ordered = sorted(ranges, key=lambda r: (r.seq, r.start_sample, r.end_sample))
    merged: list[SourceRange] = []
    for r in ordered:
        if not merged:
            merged.append(r)
            continue
        prev = merged[-1]
        if (
            prev.seq == r.seq
            and prev.sample_rate == r.sample_rate
            and r.start_sample <= prev.end_sample
        ):
            merged[-1] = SourceRange(
                session_id=prev.session_id,
                seq=prev.seq,
                sample_rate=prev.sample_rate,
                start_sample=min(prev.start_sample, r.start_sample),
                end_sample=max(prev.end_sample, r.end_sample),
                start_time=min(prev.start_time, r.start_time),
                end_time=max(prev.end_time, r.end_time),
            )
        else:
            merged.append(r)
    return tuple(merged)


# ---------------------------------------------------------------------------
# 模型帧与撤回
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame(ApiObject):
    """固定 hop 的模型帧。

    稳定身份是 ``(session_id, index)`` / :attr:`frame_id`，修订号
    :attr:`revision` 随内容更正递增；撤回通知引用旧修订号。
    """

    session_id: str
    index: int
    hop: int
    window_size: int
    model_rate: int
    samples: tuple[float, ...]
    sources: tuple[SourceRange, ...] = ()
    status: FrameStatus = FrameStatus.PROVISIONAL
    revision: int = 1
    quality: str = Quality.LOW_LATENCY.value
    assumed_gap: bool = False
    filter_padding: bool = False
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.samples) != self.window_size:
            raise ValueError(
                f"帧 {self.index} 样本数 {len(self.samples)} != window_size {self.window_size}"
            )

    # -- 身份 / 时间基准 ---------------------------------------------------
    @property
    def frame_id(self) -> str:
        return f"{self.session_id}#F{self.index}"

    @property
    def id(self) -> str:
        return self.frame_id

    @property
    def pts_samples(self) -> int:
        return self.index * self.hop

    @property
    def pts(self) -> float:
        return self.pts_seconds

    @property
    def pts_seconds(self) -> float:
        return self.pts_samples / self.model_rate

    @property
    def duration_seconds(self) -> float:
        return self.window_size / self.model_rate

    @property
    def end_seconds(self) -> float:
        return (self.pts_samples + self.window_size) / self.model_rate

    @property
    def stable(self) -> bool:
        return self.status == FrameStatus.STABLE

    @property
    def provisional(self) -> bool:
        return self.status == FrameStatus.PROVISIONAL

    # -- 样本字段的常见别名 ------------------------------------------------
    @property
    def values(self) -> tuple[float, ...]:
        return self.samples

    @property
    def data(self) -> tuple[float, ...]:
        return self.samples

    @property
    def audio(self) -> tuple[float, ...]:
        return self.samples

    @property
    def features(self) -> tuple[float, ...]:
        return self.samples

    @property
    def source_ranges(self) -> tuple[SourceRange, ...]:
        return self.sources

    @property
    def provenance(self) -> tuple[SourceRange, ...]:
        return self.sources

    @property
    def source_span(self) -> SourceSpan:
        return SourceSpan(self.session_id, self.sources)

    @property
    def source_range(self) -> SourceSpan:
        """单数形式返回完整来源包络，便于解码器一次性确认。"""
        return self.source_span

    @property
    def source_seqs(self) -> tuple[int, ...]:
        return tuple(sorted({r.seq for r in self.sources}))

    def signature(self) -> tuple[Any, ...]:
        """内容指纹：相同输入重算必须得到相同签名。"""
        return (
            tuple(round(float(v), 12) for v in self.samples),
            tuple(
                (r.seq, r.sample_rate, r.start_sample, r.end_sample)
                for r in self.sources
            ),
            bool(self.assumed_gap),
            bool(self.filter_padding),
            tuple(self.reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        span = self.source_span.to_dict()
        return {
            "session_id": self.session_id,
            "index": self.index,
            "frame_id": self.frame_id,
            "id": self.frame_id,
            "revision": self.revision,
            "status": self.status.value if isinstance(self.status, FrameStatus) else self.status,
            "stable": self.stable,
            "provisional": self.provisional,
            "hop": self.hop,
            "window_size": self.window_size,
            "model_rate": self.model_rate,
            "pts_samples": self.pts_samples,
            "pts_seconds": self.pts_seconds,
            "pts": self.pts_seconds,
            "duration_seconds": self.duration_seconds,
            "end_seconds": self.end_seconds,
            "quality": self.quality,
            "samples": list(self.samples),
            "values": list(self.samples),
            "assumed_gap": self.assumed_gap,
            "filter_padding": self.filter_padding,
            "reasons": list(self.reasons),
            "sources": [r.to_dict() for r in self.sources],
            "source_ranges": [r.to_dict() for r in self.sources],
            "source_range": span,
            "provenance": [r.to_dict() for r in self.sources],
            "source_seqs": list(self.source_seqs),
        }


@dataclass(frozen=True)
class Retraction(ApiObject):
    """撤回通知：已交付的 ``(frame_id, revision)`` 必须被解码器丢弃。"""

    session_id: str
    index: int
    revision: int
    reason: str = "content_changed"
    frame_id: str | None = None
    replacement_revision: int | None = None
    kind: str = "frame"

    def __post_init__(self) -> None:
        if self.frame_id is None:
            object.__setattr__(
                self, "frame_id", f"{self.session_id}#F{self.index}"
            )

    @property
    def target_revision(self) -> int:
        return self.revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "index": self.index,
            "frame_index": self.index,
            "frame_id": self.frame_id,
            "id": self.frame_id,
            "revision": self.revision,
            "target_revision": self.revision,
            "replacement_revision": self.replacement_revision,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 块通知
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkNotice(ApiObject):
    kind: NoticeKind
    session_id: str
    seq: int | None = None
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value if isinstance(self.kind, NoticeKind) else self.kind,
            "type": self.kind.value if isinstance(self.kind, NoticeKind) else self.kind,
            "session_id": self.session_id,
            "seq": self.seq,
            "message": self.message,
            "details": jsonable(dict(self.details)),
        }


# ---------------------------------------------------------------------------
# 恢复计划
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryPlan(ApiObject):
    """按会话序号区分可重放状态与必须重新取样（补传）的区间。"""

    session_id: str
    first_seq: int
    max_seq_seen: int | None
    contiguous_through_seq: int | None
    checkpoint_seq: int | None
    seen_seqs: tuple[int, ...] = ()
    missing_seqs: tuple[int, ...] = ()
    resample_required_seqs: tuple[int, ...] = ()
    replayable_seqs: tuple[int, ...] = ()
    buffered_after_gap_seqs: tuple[int, ...] = ()
    replay_from_seq: int | None = None
    stable_frame_indices: tuple[int, ...] = ()
    provisional_frame_indices: tuple[int, ...] = ()
    retracted_frame_indices: tuple[int, ...] = ()
    retain_frame_indices: tuple[int, ...] = ()

    @property
    def gaps(self) -> tuple[dict[str, int], ...]:
        if not self.missing_seqs:
            return ()
        ranges: list[dict[str, int]] = []
        start = prev = self.missing_seqs[0]
        for s in self.missing_seqs[1:]:
            if s == prev + 1:
                prev = s
                continue
            ranges.append({"start_seq": start, "end_seq": prev})
            start = prev = s
        ranges.append({"start_seq": start, "end_seq": prev})
        return tuple(ranges)

    @property
    def resample_required_ranges(self) -> tuple[dict[str, int], ...]:
        return self.gaps

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "first_seq": self.first_seq,
            "max_seq_seen": self.max_seq_seen,
            "contiguous_through_seq": self.contiguous_through_seq,
            "checkpoint_seq": self.checkpoint_seq,
            "replay_from_seq": self.replay_from_seq,
            "seen_seqs": list(self.seen_seqs),
            "missing_seqs": list(self.missing_seqs),
            "gaps": list(self.gaps),
            "resample_required_seqs": list(self.resample_required_seqs),
            "required_seqs": list(self.resample_required_seqs),
            "resample_required_ranges": list(self.resample_required_ranges),
            "replayable_seqs": list(self.replayable_seqs),
            "buffered_after_gap_seqs": list(self.buffered_after_gap_seqs),
            "stable_frame_indices": list(self.stable_frame_indices),
            "provisional_frame_indices": list(self.provisional_frame_indices),
            "retracted_frame_indices": list(self.retracted_frame_indices),
            "retain_frame_indices": list(self.retain_frame_indices),
        }


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorConfig(ApiObject):
    hop: int = 160
    window_size: int | None = None  # None -> 2 * hop（50% 重叠）
    model_rate: int = 16000
    quality: Quality = Quality.LOW_LATENCY
    gap_policy: GapPolicy = GapPolicy.PROVISIONAL
    first_seq: int = 0
    filter_radius: int | None = None
    kaiser_beta: float = 8.6
    ola_window: str = "hann"
    normalize_ola: bool = True
    enable_ola: bool = True
    on_conflict: ConflictPolicy = ConflictPolicy.REJECT
    strict_duplicates: bool = False
    default_sample_rate: int | None = None

    def __post_init__(self) -> None:
        if self.hop <= 0:
            raise ValueError("hop 必须为正整数")
        if self.window_size is None:
            object.__setattr__(self, "window_size", 2 * self.hop)
        elif self.window_size <= 0:
            raise ValueError("window_size 必须为正整数")
        if self.model_rate <= 0:
            raise ValueError("model_rate 必须为正整数")
        object.__setattr__(self, "quality", coerce_quality(self.quality))
        gp = self.gap_policy
        if isinstance(gp, str) and not isinstance(gp, GapPolicy):
            object.__setattr__(self, "gap_policy", GapPolicy(gp))
        elif isinstance(gp, GapPolicy):
            object.__setattr__(self, "gap_policy", GapPolicy(gp.value))
        cp = self.on_conflict
        if isinstance(cp, str) and not isinstance(cp, ConflictPolicy):
            object.__setattr__(self, "on_conflict", ConflictPolicy(cp))

    @property
    def overlap(self) -> int:
        return max(0, (self.window_size or 0) - self.hop)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop": self.hop,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "model_rate": self.model_rate,
            "quality": self.quality.value,
            "gap_policy": self.gap_policy.value,
            "first_seq": self.first_seq,
            "filter_radius": self.filter_radius,
            "kaiser_beta": self.kaiser_beta,
            "ola_window": self.ola_window,
            "normalize_ola": self.normalize_ola,
            "enable_ola": self.enable_ola,
            "on_conflict": self.on_conflict.value,
            "default_sample_rate": self.default_sample_rate,
        }
