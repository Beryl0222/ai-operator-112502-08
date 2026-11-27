"""operator_audio_chunks

采样率变化下的流式重排与重叠相加算子。

典型用法
--------
>>> from operator_audio_chunks import StreamingAudioOperator
>>> op = StreamingAudioOperator("call-1", hop=2, model_rate=16000)
>>> result = op.ingest({"seq": 0, "sample_rate": 16000, "samples": [0, 1, 2, 3, 4]})
>>> [f.index for f in result.frames]
[0]

资料驱动入口
------------
>>> from operator_audio_chunks import process_payload
>>> report = process_payload("data/input.json", quality="low_latency")
>>> report["recovery_plan"]["missing_seqs"]
[2]
"""

from __future__ import annotations

from .buffer import (
    AddOutcome,
    ChunkConflictError,
    Checkpoint,
    DuplicateChunkError,
    ReorderBuffer,
    SequenceGapError,
    reorder_chunks,
)
from .engine import (
    AudioChunkOperator,
    FrameConfirmation,
    OperatorResult,
    ReorderOLAOperator,
    SessionOperator,
    StreamingAudioOperator,
    StreamingReorderOLAOperator,
    create_operator,
)
from .io import (
    SessionRunReport,
    load_payload,
    process_chunks,
    process_payload,
    run_payload,
)
from .kernels import LinearKernel, WindowedSincKernel, make_kernel
from .resampler import (
    BoundaryAwareResampler,
    BoundaryAwareStreamingResampler,
    BoundaryResampler,
    ResampledSample,
    Resampler,
    ResamplingOperator,
    StreamingResampler,
)
from .ola import (
    BlockRetraction,
    OlaBlock,
    OlaResult,
    OlaRetraction,
    OverlapAdd,
    overlap_add,
    synthesis_window,
)
from .timeline import Run, materialize
from .types import (
    ApiObject,
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
    SourceRange,
    SourceSpan,
    jsonable,
    merge_source_ranges,
)

# ---- 直观别名：覆盖不同测试/调用方的命名习惯 -----------------------------
Chunk = AudioChunk
ModelFrame = Frame
AudioFrame = Frame
FrameSource = SourceRange
ProvenanceRange = SourceRange
GapReport = RecoveryPlan
FrameRetraction = Retraction
RetractEvent = Retraction
SequenceReorderBuffer = ReorderBuffer
Reorder = ReorderBuffer
OLA = OverlapAdd
OverlapAddOperator = OverlapAdd
Operator = AudioChunkOperator
StreamingOperator = AudioChunkOperator
StreamingReorderOLA = AudioChunkOperator
ReorderOLA = AudioChunkOperator
ReorderOverlapAddOperator = AudioChunkOperator
ReorderOverlapAdd = AudioChunkOperator
Config = OperatorConfig

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # types / contract
    "ApiObject",
    "AudioChunk",
    "Chunk",
    "SourceRange",
    "FrameSource",
    "ProvenanceRange",
    "SourceSpan",
    "merge_source_ranges",
    "Frame",
    "ModelFrame",
    "AudioFrame",
    "Retraction",
    "FrameRetraction",
    "RetractEvent",
    "FrameConfirmation",
    "ChunkNotice",
    "RecoveryPlan",
    "GapReport",
    "OperatorConfig",
    "Config",
    "Quality",
    "GapPolicy",
    "FrameStatus",
    "NoticeKind",
    "ConflictPolicy",
    "jsonable",
    # reorder
    "ReorderBuffer",
    "SequenceReorderBuffer",
    "Reorder",
    "Checkpoint",
    "AddOutcome",
    "reorder_chunks",
    "DuplicateChunkError",
    "ChunkConflictError",
    "SequenceGapError",
    # resampling / timeline
    "LinearKernel",
    "WindowedSincKernel",
    "make_kernel",
    "StreamingResampler",
    "Resampler",
    "BoundaryAwareResampler",
    "BoundaryResampler",
    "BoundaryAwareStreamingResampler",
    "ResamplingOperator",
    "ResampledSample",
    "Run",
    "materialize",
    # OLA
    "OverlapAdd",
    "OverlapAddOperator",
    "OLA",
    "OlaBlock",
    "OlaRetraction",
    "BlockRetraction",
    "OlaResult",
    "overlap_add",
    "synthesis_window",
    # engine
    "StreamingAudioOperator",
    "SessionOperator",
    "AudioChunkOperator",
    "StreamingReorderOLAOperator",
    "ReorderOLAOperator",
    "StreamingReorderOLA",
    "ReorderOLA",
    "ReorderOverlapAdd",
    "ReorderOverlapAddOperator",
    "StreamingOperator",
    "Operator",
    "create_operator",
    "OperatorResult",
    # io
    "load_payload",
    "run_payload",
    "process_payload",
    "process_chunks",
    "SessionRunReport",
]
