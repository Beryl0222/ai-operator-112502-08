"""JSON 资料读取与函数式批处理入口。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .engine import AudioChunkOperator, OperatorResult, StreamingAudioOperator
from .types import OperatorConfig, jsonable, normalize_config_kwargs


@dataclass
class SessionRunReport:
    """按输入顺序喂完所有块后的完整报告。"""

    session_id: str
    config: OperatorConfig
    results: list[OperatorResult] = field(default_factory=list)
    operator: StreamingAudioOperator | None = None
    flushed: bool = False

    @property
    def num_ingests(self) -> int:
        return len(self.results) - (1 if self.flushed else 0)

    @property
    def frames(self):
        return self.operator.frames if self.operator else []

    @property
    def retractions(self):
        return self.operator.retractions_log if self.operator else []

    @property
    def recovery_plan(self):
        return self.operator.recovery_plan() if self.operator else None

    @property
    def ola_blocks(self):
        return self.operator.ola.delivered_blocks() if self.operator and self.operator.ola else []

    def to_dict(self) -> dict[str, Any]:
        op = self.operator
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "num_ingests": self.num_ingests,
            "flushed": self.flushed,
            "results": [r.to_dict() for r in self.results],
            "frames": [f.to_dict() for f in (op.frames if op else [])],
            "retractions": [r.to_dict() for r in (op.retractions_log if op else [])],
            "recovery_plan": op.recovery_plan().to_dict() if op else None,
            "ola_blocks": [b.to_dict() for b in self.ola_blocks],
            "checkpoint": op.checkpoint() if op else None,
        }


def load_payload(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8")
        payload = json.loads(text)
    else:
        payload = dict(source)
    if "chunks" not in payload:
        raise ValueError("payload 缺少 chunks 字段")
    return payload


def _config_from_payload(
    payload: Mapping[str, Any], overrides: Mapping[str, Any]
) -> OperatorConfig:
    kwargs: dict[str, Any] = {}
    for key in (
        "hop", "window_size", "model_rate", "quality", "gap_policy",
        "first_seq", "filter_radius", "kaiser_beta", "ola_window",
        "normalize_ola", "enable_ola", "on_conflict", "strict_duplicates",
        "default_sample_rate",
    ):
        if key in payload and payload[key] is not None:
            kwargs[key] = payload[key]
    if "hop_size" in payload and "hop" not in kwargs:
        kwargs["hop"] = payload["hop_size"]
    if "sample_rate" in payload and "model_rate" not in kwargs:
        # 顶层 sample_rate 视为目标模型速率
        kwargs["model_rate"] = payload["sample_rate"]
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    if "hop" not in kwargs:
        raise ValueError("必须在 payload 或参数中提供 hop")
    kwargs = normalize_config_kwargs(kwargs)
    return OperatorConfig(**kwargs)


def run_payload(
    source: str | Path | Mapping[str, Any],
    *,
    flush: bool = False,
    **overrides: Any,
) -> SessionRunReport:
    """读取 payload（路径或字典），按数组顺序喂入全部块。"""

    payload = load_payload(source)
    session_id = str(payload.get("session_id") or payload.get("session") or "session")
    config = _config_from_payload(payload, overrides)
    op = StreamingAudioOperator(session_id, config)

    results: list[OperatorResult] = []
    for raw in payload["chunks"]:
        if not isinstance(raw, Mapping):
            raise TypeError("chunks[] 必须是对象")
        chunk = dict(raw)
        chunk.setdefault("session_id", session_id)
        results.append(op.ingest(chunk))
    flushed = False
    if flush:
        results.append(op.flush())
        flushed = True
    return SessionRunReport(
        session_id=session_id,
        config=config,
        results=results,
        operator=op,
        flushed=flushed,
    )


def process_payload(
    source: str | Path | Mapping[str, Any],
    *,
    flush: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """:func:`run_payload` 的 JSON 安全字典版本。"""

    return jsonable(run_payload(source, flush=flush, **overrides).to_dict())


def process_chunks(
    chunks: Sequence[Mapping[str, Any] | Any],
    *,
    session_id: str = "session",
    hop: int = 160,
    window_size: int | None = None,
    model_rate: int = 16000,
    quality: str = "low_latency",
    gap_policy: str = "provisional",
    flush: bool = False,
    **kwargs: Any,
) -> SessionRunReport:
    """直接处理块序列的函数式入口。"""

    config = OperatorConfig(
        hop=hop,
        window_size=window_size,
        model_rate=model_rate,
        quality=quality,  # type: ignore[arg-type]
        gap_policy=gap_policy,  # type: ignore[arg-type]
        **kwargs,
    )
    op = StreamingAudioOperator(session_id, config)
    results = [op.ingest(c) for c in chunks]
    flushed = False
    if flush:
        results.append(op.flush())
        flushed = True
    return SessionRunReport(session_id, config, results, op, flushed)
