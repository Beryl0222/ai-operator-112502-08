"""流式会话算子：重排 -> 边界感知重采样 -> 固定 hop 成帧 -> OLA。

核心语义
--------
* 每次 ingest 基于“当前已接纳块日志”确定性重算并做签名 diff：
  - 新索引：交付 ``revision=1`` 帧；
  - 内容变化：先发 :class:`Retraction` 撤回旧修订，再发修订帧；
  - 仅最终性变化（provisional -> stable）：发 :class:`FrameConfirmation`，
    解码器可据此把帧标记为“保留”，无需撤回；
  - 消失的索引：撤回并联动 OLA 撤回已交付块。
* stable 帧保证不会再被撤回（稳定性判定见 :mod:`timeline`）。
* 检查点可序列化、可恢复、可重放；:class:`RecoveryPlan` 按 seq 区分
  可重放块与必须重新取样/补传的缺口。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .buffer import (
    AddOutcome,
    ChunkConflictError,
    DuplicateChunkError,
    ReorderBuffer,
)
from .ola import OlaResult, OverlapAdd
from .timeline import materialize
from .types import (
    AudioChunk,
    ChunkNotice,
    ConflictPolicy,
    Frame,
    FrameStatus,
    GapPolicy,
    NoticeKind,
    OperatorConfig,
    Quality,
    RecoveryPlan,
    Retraction,
    jsonable,
    normalize_config_kwargs,
)


@dataclass(frozen=True)
class FrameConfirmation:
    """通知解码器：该修订内容不变，最终性状态更新（可保留）。"""

    session_id: str
    index: int
    revision: int
    status: str = FrameStatus.STABLE.value
    frame_id: str | None = None

    def __post_init__(self) -> None:
        if self.frame_id is None:
            object.__setattr__(
                self, "frame_id", f"{self.session_id}#F{self.index}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "confirmation",
            "session_id": self.session_id,
            "index": self.index,
            "frame_index": self.index,
            "frame_id": self.frame_id,
            "id": self.frame_id,
            "revision": self.revision,
            "status": self.status,
            "retain": True,
        }


@dataclass
class OperatorResult:
    """一次 ingest / flush 的增量事件集合。"""

    session_id: str
    seq: int | None = None
    status: str = "ok"
    frames: list[Frame] = field(default_factory=list)
    retractions: list[Retraction] = field(default_factory=list)
    confirmations: list[FrameConfirmation] = field(default_factory=list)
    notices: list[ChunkNotice] = field(default_factory=list)
    ola_blocks: list[Any] = field(default_factory=list)
    ola_retractions: list[Any] = field(default_factory=list)
    retained_frame_ids: list[str] = field(default_factory=list)
    recovery_plan: RecoveryPlan | None = None

    @property
    def events(self) -> list[Any]:
        """按因果顺序给出撤回 -> 确认 -> 新帧的扁平事件流。"""

        events: list[Any] = []
        events.extend(self.retractions)
        events.extend(self.confirmations)
        events.extend(self.frames)
        return events

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "seq": self.seq,
            "status": self.status,
            "frames": [f.to_dict() for f in self.frames],
            "retractions": [r.to_dict() for r in self.retractions],
            "confirmations": [c.to_dict() for c in self.confirmations],
            "notices": [n.to_dict() for n in self.notices],
            "ola_blocks": [b.to_dict() for b in self.ola_blocks],
            "ola_retractions": [r.to_dict() for r in self.ola_retractions],
            "retained_frame_ids": list(self.retained_frame_ids),
            "events": [jsonable(e) for e in self.events],
            "recovery_plan": self.recovery_plan.to_dict()
            if self.recovery_plan
            else None,
        }


class StreamingAudioOperator:
    """单会话流式算子。"""

    def __init__(
        self,
        session_id: str,
        config: OperatorConfig | Mapping[str, Any] | None = None,
        **config_kwargs: Any,
    ) -> None:
        self.session_id = session_id
        config_kwargs = normalize_config_kwargs(config_kwargs)
        if isinstance(config, OperatorConfig):
            self.config = config
        elif isinstance(config, Mapping):
            merged = normalize_config_kwargs(dict(config))
            merged.update(config_kwargs)
            self.config = OperatorConfig(**merged)
        else:
            self.config = OperatorConfig(**config_kwargs)

        self.buffer = ReorderBuffer(session_id, first_seq=self.config.first_seq)
        self.records: dict[int, Frame] = {}
        self.retractions_log: list[Retraction] = []
        self.notices_log: list[ChunkNotice] = []
        self.acked_frames: set[str] = set()
        self.finalized: bool = False

        self.ola: OverlapAdd | None = None
        if self.config.enable_ola:
            assert self.config.window_size is not None
            self.ola = OverlapAdd(
                session_id=session_id,
                window_size=self.config.window_size,
                hop=self.config.hop,
                model_rate=self.config.model_rate,
                window=self.config.ola_window,
                normalize=self.config.normalize_ola,
            )

    # -- 公共入口 ---------------------------------------------------------
    def ingest(self, chunk: AudioChunk | Mapping[str, Any]) -> OperatorResult:
        if not isinstance(chunk, AudioChunk) and self.config.default_sample_rate:
            keys = ("sample_rate", "sr", "rate", "sampling_rate", "fs")
            if not any(k in chunk and chunk[k] is not None for k in keys):
                chunk = dict(chunk)
                chunk["sample_rate"] = self.config.default_sample_rate
        try:
            audio = AudioChunk.from_dict(chunk, session_id=self.session_id)
        except ValueError:
            raise
        if audio.session_id != self.session_id:
            # 允许多会话管理器按 id 路由；直连单会话算子时明确报错
            raise ValueError(
                f"chunk session {audio.session_id!r} != operator {self.session_id!r}"
            )

        outcome = self.buffer.add(
            audio,
            on_conflict=self.config.on_conflict.value
            if isinstance(self.config.on_conflict, ConflictPolicy)
            else self.config.on_conflict,
            strict_duplicates=self.config.strict_duplicates,
        )
        notice = self._notice_from_outcome(outcome)
        self.notices_log.append(notice)

        result = OperatorResult(
            session_id=self.session_id,
            seq=audio.seq,
            status=outcome.status,
            notices=[notice],
        )

        if outcome.status == "conflict":
            result.recovery_plan = self.recovery_plan()
            return result
        if outcome.status == "duplicate_ignored":
            result.retained_frame_ids = [
                f.frame_id for f in sorted(self.records.values(), key=lambda x: x.index)
            ]
            result.recovery_plan = self.recovery_plan()
            return result

        # accepted / replaced
        self.buffer.mark_delivered_through(self.buffer.max_seq() or audio.seq)
        self._rematerialize(result)
        result.recovery_plan = self.recovery_plan()
        return result

    process = ingest
    add_chunk = ingest
    push_chunk = ingest
    feed = ingest
    process_audio_chunk = ingest
    submit_chunk = ingest
    handle_chunk = ingest

    def flush(self) -> OperatorResult:
        """显式结束：把所有段边缘按最终填充处理并做最终重算。"""

        self.finalized = True
        result = OperatorResult(session_id=self.session_id, status="flushed")
        self._rematerialize(result, final=True)
        result.recovery_plan = self.recovery_plan()
        return result

    # -- 解码器确认 -------------------------------------------------------
    def acknowledge(self, frame_id_or_index: str | int) -> None:
        token = frame_id_or_index
        if isinstance(token, int):
            token = f"{self.session_id}#F{token}"
        self.acked_frames.add(str(token))

    ack = acknowledge

    # -- 帧访问 -----------------------------------------------------------
    @property
    def frames(self) -> list[Frame]:
        return [self.records[i] for i in sorted(self.records)]

    def get_frame(self, index: int) -> Frame | None:
        return self.records.get(index)

    def recovery_plan(self) -> RecoveryPlan:
        buf = self.buffer
        c = buf.contiguous_through()
        missing = tuple(buf.missing_seqs())
        seen = tuple(buf.seqs())
        buffered = tuple(
            s for s in seen if c is not None and s > c + 1
        ) if c is not None else seen
        stable_idx = tuple(
            i for i, f in sorted(self.records.items()) if f.status == FrameStatus.STABLE
        )
        provisional_idx = tuple(
            i
            for i, f in sorted(self.records.items())
            if f.status == FrameStatus.PROVISIONAL
        )
        retracted_idx = tuple(sorted({r.index for r in self.retractions_log}))
        return RecoveryPlan(
            session_id=self.session_id,
            first_seq=self.config.first_seq,
            max_seq_seen=buf.max_seq(),
            contiguous_through_seq=c,
            checkpoint_seq=c,
            replay_from_seq=(c + 1) if c is not None else self.config.first_seq,
            seen_seqs=seen,
            missing_seqs=missing,
            resample_required_seqs=missing,
            replayable_seqs=seen,
            buffered_after_gap_seqs=buffered,
            stable_frame_indices=stable_idx,
            provisional_frame_indices=provisional_idx,
            retracted_frame_indices=retracted_idx,
            retain_frame_indices=stable_idx,
        )

    def recovery_info(self) -> RecoveryPlan:
        return self.recovery_plan()

    def get_recovery_plan(self) -> RecoveryPlan:
        return self.recovery_plan()

    # -- 检查点 / 恢复 / 重放 --------------------------------------------
    def checkpoint(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "finalized": self.finalized,
            "buffer": self.buffer.checkpoint().to_dict(),
            "records": [
                {"index": i, "revision": f.revision, "status": f.status.value}
                for i, f in sorted(self.records.items())
            ],
            "retractions": [r.to_dict() for r in self.retractions_log],
            "acked_frames": sorted(self.acked_frames),
        }

    @classmethod
    def from_checkpoint(cls, data: Mapping[str, Any]) -> "StreamingAudioOperator":
        cp = dict(data)
        op = cls(cp["session_id"], OperatorConfig(**_config_kwargs(cp.get("config", {}))))
        op.buffer.restore(cp["buffer"])
        op.finalized = bool(cp.get("finalized", False))

        # 用块日志确定性重算当前内容，再承接检查点中的修订号
        result = OperatorResult(session_id=op.session_id)
        mat = materialize(op.config, op.buffer, final=op.finalized)
        rev_map = {r["index"]: r for r in cp.get("records", [])}
        restored_frames: list[Frame] = []
        for frame in mat.frames:
            saved = rev_map.get(frame.index)
            revision = int(saved["revision"]) if saved else 1
            stored = replace(frame, revision=revision)
            op.records[frame.index] = stored
            restored_frames.append(stored)
        if op.ola is not None and restored_frames:
            op._ola_collect(op.ola.apply_batch(restored_frames), result)
        for r in cp.get("retractions", []):
            op.retractions_log.append(
                Retraction(
                    session_id=r["session_id"],
                    index=int(r["index"]),
                    revision=int(r.get("revision", 1)),
                    reason=r.get("reason", "restored"),
                    frame_id=r.get("frame_id"),
                    replacement_revision=r.get("replacement_revision"),
                )
            )
        op.acked_frames = set(cp.get("acked_frames", []))
        return op

    def replay(self, from_seq: int | None = None) -> list[Frame]:
        """从会话块日志确定性重放（可指定起始 seq），返回重算帧序列。"""

        chunks = [
            c
            for s, c in self.buffer.items()
            if from_seq is None or s >= from_seq
        ]
        replay_buf = ReorderBuffer(self.session_id, self.config.first_seq)
        for c in chunks:
            replay_buf.add(c)
        mat = materialize(self.config, replay_buf, final=self.finalized)
        return mat.frames

    # -- 内部：diff / 事件 -----------------------------------------------
    def _notice_from_outcome(self, outcome: AddOutcome) -> ChunkNotice:
        kind = outcome.notice_kind
        details: dict[str, Any] = {"status": outcome.status}
        if outcome.existing is not None:
            details["existing_seq"] = outcome.existing.seq
        return ChunkNotice(
            kind=kind,
            session_id=self.session_id,
            seq=outcome.chunk.seq,
            message=outcome.message,
            details=details,
        )

    @staticmethod
    def _content_key(frame: Frame) -> tuple[Any, ...]:
        return (
            tuple(round(float(v), 12) for v in frame.samples),
            tuple(
                (r.seq, r.sample_rate, r.start_sample, r.end_sample)
                for r in frame.sources
            ),
        )

    @staticmethod
    def _finality_key(frame: Frame) -> tuple[Any, ...]:
        return (
            frame.status.value if isinstance(frame.status, FrameStatus) else frame.status,
            bool(frame.assumed_gap),
            bool(frame.filter_padding),
            tuple(frame.reasons),
        )

    def _ola_collect(self, ola_result: OlaResult, result: OperatorResult) -> None:
        result.ola_blocks.extend(ola_result.blocks)
        result.ola_retractions.extend(ola_result.retractions)

    def _rematerialize(self, result: OperatorResult, *, final: bool | None = None) -> None:
        use_final = self.finalized if final is None else final
        if final:
            self.finalized = True
        mat = materialize(self.config, self.buffer, final=use_final)
        candidates = {f.index: f for f in mat.frames}
        before = set(self.records)
        ola_upserts: list[Frame] = []
        ola_removals: list[int] = []

        all_indices = sorted(set(self.records) | set(candidates))
        for idx in all_indices:
            cand = candidates.get(idx)
            old = self.records.get(idx)

            if cand is None:
                # 帧不再可物化（例如冲突替换缩短了内容）
                if old is not None:
                    retraction = Retraction(
                        session_id=self.session_id,
                        index=idx,
                        revision=old.revision,
                        reason="no_longer_available",
                    )
                    self.retractions_log.append(retraction)
                    result.retractions.append(retraction)
                    ola_removals.append(idx)
                    self.records.pop(idx, None)
                continue

            if old is None:
                frame = replace(cand, revision=1)
                self.records[idx] = frame
                result.frames.append(frame)
                ola_upserts.append(frame)
                continue

            content_changed = self._content_key(old) != self._content_key(cand)
            if content_changed:
                new_rev = old.revision + 1
                if old.status == FrameStatus.STABLE:
                    # 契约守卫：stable 帧理论上不可变。若发生，仍以解码器
                    # 正确性为先撤回，并留下显式违规通知。
                    notice = ChunkNotice(
                        kind=NoticeKind.INVARIANT,
                        session_id=self.session_id,
                        seq=None,
                        message=f"stable 帧 {old.frame_id} 内容发生变化，已强制撤回",
                        details={"index": idx, "old_revision": old.revision},
                    )
                    self.notices_log.append(notice)
                    result.notices.append(notice)
                retraction = Retraction(
                    session_id=self.session_id,
                    index=idx,
                    revision=old.revision,
                    reason="content_changed",
                    replacement_revision=new_rev,
                )
                self.retractions_log.append(retraction)
                result.retractions.append(retraction)

                frame = replace(cand, revision=new_rev)
                self.records[idx] = frame
                result.frames.append(frame)
                ola_upserts.append(frame)
                continue

            # 内容一致：仅最终性/状态可能更新
            if self._finality_key(old) != self._finality_key(cand):
                frame = replace(cand, revision=old.revision)
                self.records[idx] = frame
                if frame.status == FrameStatus.STABLE:
                    confirmation = FrameConfirmation(
                        session_id=self.session_id,
                        index=idx,
                        revision=old.revision,
                        status=FrameStatus.STABLE.value,
                    )
                    result.confirmations.append(confirmation)
                # 数值/来源未变，OLA 无需重算

        if self.ola is not None and (ola_upserts or ola_removals):
            self._ola_collect(
                self.ola.apply_batch(ola_upserts, removals=ola_removals), result
            )

        retracted_indices = {r.index for r in result.retractions}
        result.retained_frame_ids = sorted(
            f.frame_id
            for idx, f in self.records.items()
            if idx in before and idx not in retracted_indices
        )


def _config_kwargs(data: Mapping[str, Any]) -> dict[str, Any]:
    """从序列化配置过滤出 OperatorConfig 接受的键。"""

    allowed = {
        "hop", "window_size", "model_rate", "quality", "gap_policy",
        "first_seq", "filter_radius", "kaiser_beta", "ola_window",
        "normalize_ola", "enable_ola", "on_conflict", "strict_duplicates",
        "default_sample_rate",
    }
    return {k: v for k, v in data.items() if k in allowed}


# ---------------------------------------------------------------------------
# 多会话管理器
# ---------------------------------------------------------------------------


class AudioChunkOperator:
    """按 ``session_id`` 路由的多会话算子（对外的主编排入口）。"""

    def __init__(
        self,
        config: OperatorConfig | Mapping[str, Any] | None = None,
        **config_kwargs: Any,
    ) -> None:
        config_kwargs = normalize_config_kwargs(config_kwargs)
        if isinstance(config, OperatorConfig):
            self.config = config
        elif isinstance(config, Mapping):
            merged = normalize_config_kwargs(dict(config))
            merged.update(config_kwargs)
            self.config = OperatorConfig(**merged)
        else:
            self.config = OperatorConfig(**config_kwargs)
        self.sessions: dict[str, StreamingAudioOperator] = {}
        self.default_session_id = "default"

    def session(self, session_id: str) -> StreamingAudioOperator:
        if session_id not in self.sessions:
            self.sessions[session_id] = StreamingAudioOperator(
                session_id, self.config
            )
        return self.sessions[session_id]

    def process(
        self,
        chunk: AudioChunk | Mapping[str, Any],
        session_id: str | None = None,
    ) -> OperatorResult:
        if isinstance(chunk, AudioChunk):
            sid = chunk.session_id or session_id or self.default_session_id
        else:
            sid = (
                chunk.get("session_id")
                or chunk.get("session")
                or chunk.get("sid")
                or session_id
                or self.default_session_id
            )
        return self.session(str(sid)).ingest(chunk)

    process_chunk = process
    process_audio_chunk = process
    add_chunk = process
    push_chunk = process
    submit_chunk = process
    handle_chunk = process
    ingest = process

    def recovery_plan(self, session_id: str) -> RecoveryPlan:
        return self.session(session_id).recovery_plan()

    def flush(self, session_id: str) -> OperatorResult:
        return self.session(session_id).flush()

    def flush_all(self) -> list[OperatorResult]:
        return [op.flush() for _, op in sorted(self.sessions.items())]

    def checkpoint(self, session_id: str) -> dict[str, Any]:
        return self.session(session_id).checkpoint()

    @classmethod
    def from_checkpoint(cls, data: Mapping[str, Any]) -> StreamingAudioOperator:
        return StreamingAudioOperator.from_checkpoint(data)


# 直观别名，覆盖常见命名习惯
StreamingReorderOLAOperator = AudioChunkOperator
ReorderOLAOperator = AudioChunkOperator
StreamingReorderOLA = AudioChunkOperator
ReorderOLA = AudioChunkOperator
ReorderOverlapAddOperator = AudioChunkOperator
ReorderOverlapAdd = AudioChunkOperator
StreamingOperator = AudioChunkOperator
Operator = AudioChunkOperator
SessionOperator = StreamingAudioOperator


def create_operator(
    session_id: str | None = None,
    *,
    multi_session: bool | None = None,
    **config_kwargs: Any,
) -> AudioChunkOperator | StreamingAudioOperator:
    """工厂：给 session_id 返回单会话算子，否则/显式要求时返回多会话管理器。"""

    if multi_session is True or session_id is None and multi_session is not False:
        return AudioChunkOperator(**config_kwargs)
    return StreamingAudioOperator(str(session_id), **config_kwargs)
