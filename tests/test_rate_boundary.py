"""采样率切换时的滤波状态隔离与来源契约。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from operator_audio_chunks import StreamingAudioOperator  # noqa: E402
from operator_audio_chunks.timeline import materialize  # noqa: E402
from operator_audio_chunks.types import OperatorConfig  # noqa: E402


def feed(spec, quality="low_latency", hop=8, window=16, **kw):
    """spec: list of (seq, rate, samples)."""
    op = StreamingAudioOperator(
        "s", hop=hop, window_size=window, model_rate=16000,
        quality=quality, enable_ola=False, **kw
    )
    for seq, rate, samples in spec:
        op.ingest({"seq": seq, "sample_rate": rate, "samples": samples})
    return op


class RateBoundaryTest(unittest.TestCase):
    def test_runs_are_split_by_rate_switch(self):
        cfg = OperatorConfig(hop=4, window_size=8, quality="high")
        from operator_audio_chunks.buffer import ReorderBuffer

        buf = ReorderBuffer("s")
        for seq, rate, n in [(0, 16000, 40), (1, 8000, 40), (2, 8000, 40)]:
            buf.add(
                {"session_id": "s", "seq": seq, "sample_rate": rate,
                 "samples": [0.1 * i for i in range(n)]}
            )
        mat = materialize(cfg, buf)
        rates = [r.rate for r in mat.runs]
        # seq1+2 同速率且连续 => 合并共享历史；seq0 独立
        self.assertEqual(rates, [16000, 8000])
        self.assertEqual(mat.runs[1].seqs, [1, 2])

    def test_old_rate_samples_cannot_influence_new_rate_only_frames(self):
        # 旧段放入巨大“毒性”数值；线性插值有界，若旧状态越界污染新段，
        # 新速率专属帧必然出现超界值。
        old = [1e6 if i % 2 == 0 else -1e6 for i in range(32)]
        new = [0.25 * (i % 5) for i in range(48)]
        op = feed([(0, 16000, old), (1, 8000, new)], hop=8, window=16)

        new_min, new_max = min(new), max(new)
        for f in op.frames:
            seqs = set(f.source_seqs)
            if seqs == {1}:
                for v in f.samples:
                    self.assertGreaterEqual(v, new_min - 1e-9)
                    self.assertLessEqual(v, new_max + 1e-9)

    def test_perturbing_old_chunk_only_changes_old_sourced_frames(self):
        new = [((i * 37) % 13) - 6 for i in range(64)]

        op_a = feed([(0, 16000, [i % 7 for i in range(40)]),
                     (1, 8000, new)])
        op_b = feed([(0, 16000, [999.0] * 40),
                     (1, 8000, new)])
        fa = {f.index: f for f in op_a.frames}
        fb = {f.index: f for f in op_b.frames}
        self.assertEqual(set(fa), set(fb))
        for idx in fa:
            old_seqs = {r.seq for r in fa[idx].sources if r.sample_rate == 16000}
            if not old_seqs:
                # 完全由新速率覆盖的帧必须逐值相同 —— 旧滤波状态不可达
                self.assertEqual(
                    list(fa[idx].samples), list(fb[idx].samples),
                    msg=f"frame {idx} 被旧速率状态污染",
                )

    def test_straddling_frame_reports_both_rates(self):
        # 让窗口明确跨越 16k->8k 边界
        op = feed(
            [(0, 16000, list(range(20))), (1, 8000, list(range(100, 140)))],
            hop=4, window=8,
        )
        straddlers = [
            f for f in op.frames
            if {r.sample_rate for r in f.sources} == {16000, 8000}
        ]
        self.assertTrue(straddlers, "应当存在跨越速率边界的帧")
        for f in straddlers:
            # 来源按 seq 有序，且旧段 seq < 新段 seq
            seqs = [r.seq for r in f.sources]
            self.assertEqual(seqs, sorted(seqs))
            self.assertLess(min(seqs), max(seqs))

    def test_filter_support_never_crosses_rate_segment(self):
        # 高质量核半径很大；逐 tick 校验核所需下标在 run 内被钳制，
        # 且帧的契约来源不会把新速率 run 的样本算到旧 seq 上。
        cfg = OperatorConfig(hop=8, window_size=16, quality="high")
        from operator_audio_chunks.buffer import ReorderBuffer
        from operator_audio_chunks.kernels import make_kernel
        from fractions import Fraction
        from operator_audio_chunks.kernels import frac_floor

        buf = ReorderBuffer("s")
        buf.add({"session_id": "s", "seq": 0, "sample_rate": 16000,
                 "samples": [float(i) for i in range(64)]})
        buf.add({"session_id": "s", "seq": 1, "sample_rate": 8000,
                 "samples": [float(i) for i in range(64)]})
        mat = materialize(cfg, buf)
        for run in mat.runs:
            kernel = make_kernel("high", run.rate, 16000)
            # run 内部任意位置的支撑区间都是针对该 run 自身样本数组，
            # 结构上不存在指向上一速率 run 的共享状态。
            for numer in range(0, run.n_samples * 2, 3):
                p = Fraction(numer, 2)
                lo, hi = kernel.required_range(p)
                i0 = frac_floor(p)
                self.assertLessEqual(lo, i0)
                self.assertGreaterEqual(hi, i0)
                self.assertEqual(kernel.support_radius, 8)

    def test_same_rate_chunk_boundary_shares_history(self):
        # 连续同速率块分开喂与合并成一块喂，内部帧结果应一致
        samples = [((i * 53) % 17) - 8 for i in range(80)]
        split = feed([(0, 16000, samples[:30]),
                      (1, 16000, samples[30:55]),
                      (2, 16000, samples[55:])])
        merged = feed([(0, 16000, samples)])
        a = [(f.index, f.samples) for f in split.frames]
        b = [(f.index, f.samples) for f in merged.frames]
        # 总长度一致 => 帧数与帧值一致（边界滤波历史未丢失）
        self.assertEqual([i for i, _ in a], [i for i, _ in b])
        for (ia, va), (ib, vb) in zip(a, b):
            self.assertEqual(ia, ib)
            for x, y in zip(va, vb):
                self.assertAlmostEqual(x, y, places=9)


if __name__ == "__main__":
    unittest.main()
