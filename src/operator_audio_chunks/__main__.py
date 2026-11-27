"""命令行入口：在 data/input.json 或任意同结构 payload 上运行算子。

示例::

    python -m operator_audio_chunks --input data/input.json --quality low_latency
    python -m operator_audio_chunks --input data/input.json --inject 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import process_payload, run_payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="operator_audio_chunks")
    p.add_argument("--input", "-i", default="data/input.json", help="payload JSON 路径")
    p.add_argument("--quality", "-q", default="low_latency",
                   choices=["high", "low_latency", "low", "fast"])
    p.add_argument("--model-rate", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--gap-policy", default="provisional",
                   choices=["provisional", "wait"])
    p.add_argument("--inject", type=int, nargs="*", default=None,
                   help="在原 payload 之后追加注入指定 seq 的合成块（演示迟到恢复）")
    p.add_argument("--flush", action="store_true", help="喂完后显式 flush")
    p.add_argument("--summary-only", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.input)
    payload = json.loads(path.read_text(encoding="utf-8"))

    overrides = {"quality": args.quality, "gap_policy": args.gap_policy}
    if args.model_rate:
        overrides["model_rate"] = args.model_rate
    if args.window_size:
        overrides["window_size"] = args.window_size

    if args.inject:
        # 演示用：为指定缺失 seq 注入一个单位样本块（真实场景由对端补传）
        existing = {c["seq"] for c in payload["chunks"]}
        rate = payload["chunks"][-1]["sample_rate"]
        for seq in args.inject:
            if seq not in existing:
                payload["chunks"].append(
                    {"seq": seq, "sample_rate": rate, "samples": [seq + 6]}
                )

    if args.summary_only:
        report = run_payload(payload, flush=args.flush or bool(args.inject), **overrides)
        plan = report.recovery_plan
        out = {
            "session_id": report.session_id,
            "quality": report.config.quality.value,
            "frame_count": len(report.frames),
            "frame_indices": [f.index for f in report.frames],
            "stable_indices": [f.index for f in report.frames if f.stable],
            "provisional_indices": [f.index for f in report.frames if f.provisional],
            "retractions": [r.to_dict() for r in report.retractions],
            "missing_seqs": list(plan.missing_seqs) if plan else [],
            "replayable_seqs": list(plan.replayable_seqs) if plan else [],
            "ola_block_count": len(report.ola_blocks),
        }
    else:
        out = process_payload(payload, flush=args.flush or bool(args.inject), **overrides)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
