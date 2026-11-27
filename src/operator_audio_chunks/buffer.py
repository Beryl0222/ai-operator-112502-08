"""会话内序号重排缓冲。

职责：

* 按 ``(session_id, seq)`` 唯一接纳音频块；
* 区分新块、重复块（幂等忽略）、内容冲突块（绝不静默覆盖/拼接）；
* 计算连续前缀、序号缺口、迟到块；
* 提供可序列化检查点，供故障恢复时区分“可重放状态”和“必须补传区间”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

from .types import AudioChunk, NoticeKind


class DuplicateChunkError(ValueError):
    """严格模式下收到重复 seq。"""


class ChunkConflictError(ValueError):
    """同 seq 但采样率或样本内容不一致，且策略为 error。"""


class SequenceGapError(RuntimeError):
    """严格模式下尝试在缺口未补齐时越过缺口推进。"""


@dataclass(frozen=True)
class AddOutcome:
    status: str  # accepted | duplicate_ignored | duplicate_identical | conflict | replaced
    chunk: AudioChunk
    notice_kind: NoticeKind
    message: str = ""
    existing: AudioChunk | None = None

    @property
    def accepted(self) -> bool:
        return self.status in ("accepted", "replaced")

    @property
    def duplicate(self) -> bool:
        return self.status.startswith("duplicate")

    @property
    def conflict(self) -> bool:
        return self.status == "conflict"


@dataclass
class Checkpoint:
    """重排缓冲的可序列化快照（可重放状态）。"""

    session_id: str
    first_seq: int
    chunks: list[dict[str, Any]]
    delivered_through_seq: int | None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "first_seq": self.first_seq,
            "chunks": list(self.chunks),
            "delivered_through_seq": self.delivered_through_seq,
        }


class ReorderBuffer:
    """单会话序号缓冲。"""

    def __init__(self, session_id: str, first_seq: int = 0) -> None:
        self.session_id = session_id
        self.first_seq = first_seq
        self._chunks: dict[int, AudioChunk] = {}
        # 已向调用方交付到的 seq 高水位（用于判定 late）
        self.delivered_through_seq: int | None = None
        self.arrival_order: list[int] = []

    # -- 基本容器 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, seq: object) -> bool:
        return seq in self._chunks

    def __iter__(self) -> Iterator[AudioChunk]:
        return iter(self.ordered_chunks())

    def get(self, seq: int) -> AudioChunk | None:
        return self._chunks.get(seq)

    def seqs(self) -> list[int]:
        return sorted(self._chunks)

    def items(self) -> list[tuple[int, AudioChunk]]:
        return [(s, self._chunks[s]) for s in self.seqs()]

    # -- 写入 / 去重 / 冲突 ----------------------------------------------
    def add(
        self,
        chunk: AudioChunk | Mapping[str, Any],
        *,
        on_conflict: str = "reject",
        strict_duplicates: bool = False,
    ) -> AddOutcome:
        if not isinstance(chunk, AudioChunk):
            chunk = AudioChunk.from_dict(chunk, session_id=self.session_id)
        if chunk.session_id != self.session_id:
            raise ValueError(
                f"chunk session {chunk.session_id!r} != buffer session {self.session_id!r}"
            )

        seq = chunk.seq
        if seq < self.first_seq:
            outcome = AddOutcome(
                status="rejected_before_start",
                chunk=chunk,
                notice_kind=NoticeKind.CONFLICT,
                message=f"seq {seq} 小于会话起始序号 {self.first_seq}",
            )
            if on_conflict == "error":
                raise ChunkConflictError(outcome.message)
            return outcome

        existing = self._chunks.get(seq)
        if existing is not None:
            if existing.fingerprint() == chunk.fingerprint():
                if strict_duplicates:
                    raise DuplicateChunkError(f"重复块 seq={seq}")
                return AddOutcome(
                    status="duplicate_ignored",
                    chunk=chunk,
                    notice_kind=NoticeKind.DUPLICATE_IGNORED,
                    message=f"重复块 seq={seq} 内容一致，已幂等忽略",
                    existing=existing,
                )
            # 同 seq 内容冲突：不能静默拼接或覆盖
            if on_conflict == "replace":
                self._chunks[seq] = chunk
                return AddOutcome(
                    status="replaced",
                    chunk=chunk,
                    notice_kind=NoticeKind.CONFLICT,
                    message=f"seq={seq} 内容冲突，按策略替换并触发重算/撤回",
                    existing=existing,
                )
            message = (
                f"seq={seq} 已存在但 sample_rate/samples 不一致；"
                "拒绝静默覆盖，请显式 on_conflict='replace' 或要求对端重传"
            )
            if on_conflict == "error":
                raise ChunkConflictError(message)
            return AddOutcome(
                status="conflict",
                chunk=chunk,
                notice_kind=NoticeKind.CONFLICT,
                message=message,
                existing=existing,
            )

        self._chunks[seq] = chunk
        self.arrival_order.append(seq)
        late = (
            self.delivered_through_seq is not None
            and seq <= self.delivered_through_seq
        )
        return AddOutcome(
            status="accepted",
            chunk=chunk,
            notice_kind=NoticeKind.LATE if late else NoticeKind.ACCEPTED,
            message=f"迟到块 seq={seq}" if late else f"已接收 seq={seq}",
        )

    # 常见方法名别名（签名与 add 相同）
    push = add
    receive = add
    insert = add
    add_chunk = add

    def mark_delivered_through(self, seq: int) -> None:
        if self.delivered_through_seq is None or seq > self.delivered_through_seq:
            self.delivered_through_seq = seq

    # -- 顺序 / 缺口 ------------------------------------------------------
    def max_seq(self) -> int | None:
        return max(self._chunks) if self._chunks else None

    def min_seq(self) -> int | None:
        return min(self._chunks) if self._chunks else None

    def contiguous_through(self) -> int | None:
        """从 ``first_seq`` 开始全部到齐的最大 seq。

        一个块都没有时返回 None；first_seq 缺失时也返回 ``first_seq - 1``
        还是 None？这里返回 None 表示“连起点都未确认”。
        """
        if self.first_seq not in self._chunks:
            return None
        cur = self.first_seq
        while cur + 1 in self._chunks:
            cur += 1
        return cur

    def missing_seqs(self, *, up_to: int | None = None) -> list[int]:
        """已知范围内（first_seq..max_seq 或 up_to）缺失的序号。"""
        upper = self.max_seq() if up_to is None else up_to
        if upper is None or upper < self.first_seq:
            return []
        return [s for s in range(self.first_seq, upper + 1) if s not in self._chunks]

    def gap_ranges(self) -> list[tuple[int, int]]:
        missing = self.missing_seqs()
        if not missing:
            return []
        ranges: list[tuple[int, int]] = []
        start = prev = missing[0]
        for s in missing[1:]:
            if s == prev + 1:
                prev = s
            else:
                ranges.append((start, prev))
                start = prev = s
        ranges.append((start, prev))
        return ranges

    def buffered_beyond_gap(self) -> list[int]:
        """连续前缀之后、因缺口而暂时不能进入稳定时间线的已缓存 seq。"""
        c = self.contiguous_through()
        if c is None:
            return self.seqs()
        return [s for s in self.seqs() if s > c + 1 or (s > c and (c + 1) not in self)]

    def ordered_chunks(self, *, only_contiguous_prefix: bool = False) -> list[AudioChunk]:
        if only_contiguous_prefix:
            c = self.contiguous_through()
            if c is None:
                return []
            return [self._chunks[s] for s in range(self.first_seq, c + 1)]
        return [self._chunks[s] for s in self.seqs()]

    def consecutive_runs(self, *, only_contiguous_prefix: bool = False) -> list[list[AudioChunk]]:
        """按 seq 连续性切分运行段（不考虑采样率）。"""
        chunks = self.ordered_chunks(only_contiguous_prefix=only_contiguous_prefix)
        runs: list[list[AudioChunk]] = []
        for chunk in chunks:
            if runs and chunk.seq == runs[-1][-1].seq + 1:
                runs[-1].append(chunk)
            else:
                runs.append([chunk])
        return runs

    # -- 检查点 / 重放 ----------------------------------------------------
    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            session_id=self.session_id,
            first_seq=self.first_seq,
            chunks=[self._chunks[s].to_dict() for s in self.seqs()],
            delivered_through_seq=self.delivered_through_seq,
        )

    def restore(self, checkpoint: Checkpoint | Mapping[str, Any]) -> None:
        if not isinstance(checkpoint, Checkpoint):
            data = dict(checkpoint)
            checkpoint = Checkpoint(
                session_id=data["session_id"],
                first_seq=data.get("first_seq", 0),
                chunks=list(data.get("chunks", [])),
                delivered_through_seq=data.get("delivered_through_seq"),
            )
        if checkpoint.session_id != self.session_id:
            raise ValueError("checkpoint 属于其他会话")
        self.first_seq = checkpoint.first_seq
        self._chunks = {}
        self.arrival_order = []
        for d in checkpoint.chunks:
            chunk = AudioChunk.from_dict(d, session_id=self.session_id)
            # 恢复时跳过冲突策略，直接重建
            self._chunks[chunk.seq] = chunk
            self.arrival_order.append(chunk.seq)
        self.delivered_through_seq = checkpoint.delivered_through_seq

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint | Mapping[str, Any]) -> "ReorderBuffer":
        data = checkpoint.to_dict() if isinstance(checkpoint, Checkpoint) else dict(checkpoint)
        buf = cls(data["session_id"], first_seq=data.get("first_seq", 0))
        buf.restore(checkpoint)
        return buf


def reorder_chunks(
    chunks: Iterable[AudioChunk | Mapping[str, Any]],
    *,
    session_id: str | None = None,
    first_seq: int = 0,
) -> list[AudioChunk]:
    """便捷函数：把块（字典或 AudioChunk）按 seq 排序返回。

    重复 seq 以首次出现为准（幂等）；冲突内容抛出 :class:`ChunkConflictError`。
    缺口不会被填补，调用方可据返回 seq 列表自行检查。
    """

    raw_list = list(chunks)
    if session_id is None:
        for raw in raw_list:
            if isinstance(raw, AudioChunk):
                session_id = raw.session_id
                break
            sid = raw.get("session_id") or raw.get("session") or raw.get("sid")
            if sid is not None:
                session_id = str(sid)
                break
    buf = ReorderBuffer(session_id or "session", first_seq=first_seq)
    for raw in raw_list:
        chunk = AudioChunk.from_dict(raw, session_id=buf.session_id)
        buf.add(chunk, on_conflict="error")
    return buf.ordered_chunks()
