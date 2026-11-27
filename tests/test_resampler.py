"""独立流式/边界感知重采样器与模块路径别名。"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from operator_audio_chunks import (  # noqa: E402
    BoundaryAwareResampler,
    StreamingResampler,
    reorder_chunks,
)


class StreamingResamplerTest(unittest.TestCase):
    def test_chunked_push_preserves_history_and_equals_oneshot(self):
        samples = [float(i) for i in range(10)]

        chunked = StreamingResampler(8000, 16000, quality="low_latency")
        collected = []
        collected += chunked.push(samples[:3])
        collected += chunked.push(samples[3:])
        collected += chunked.finish()
        chunked_values = [r.value for r in collected]

        # 一次性喂入的完整输出作为基准：分块边界不得丢失滤波/插值历史
        base = StreamingResampler(8000, 16000, quality="low_latency")
        all_out = base.push(samples) + base.finish()
        self.assertEqual(chunked_values, [r.value for r in all_out])
        # 8k->16k 线性插值给出半样本，总数为 2*N
        self.assertEqual(len(chunked_values), 20)
        self.assertAlmostEqual(chunked_values[1], 0.5, places=9)

    def test_phase_is_rational_and_monotonic(self):
        r = StreamingResampler(44100, 16000)
        self.assertEqual(r.position_of(160), Fraction(160 * 44100, 16000))
        outs = r.push([0.0] * 100)
        idx = [o.index for o in outs]
        self.assertEqual(idx, list(range(len(outs))))

    def test_finish_marks_edge_padded_tail(self):
        r = StreamingResampler(8000, 16000, quality="low_latency")
        r.push([0.0, 1.0, 2.0])
        tail = r.finish()
        self.assertTrue(any(o.padded for o in tail))


class BoundaryIsolationTest(unittest.TestCase):
    def test_rate_switch_never_mixes_source_seqs(self):
        ba = BoundaryAwareResampler(16000, quality="low_latency", session_id="s")
        old = [1e6 if i % 2 else -1e6 for i in range(8)]
        new = [0.1 * (i % 4) for i in range(24)]
        ba.process(seq=0, sample_rate=8000, samples=old)
        ba.process(seq=1, sample_rate=16000, samples=new)
        ba.finish()

        self.assertTrue(ba.outputs)
        new_min, new_max = min(new), max(new)
        for item in ba.outputs:
            seqs = {src.seq for src in item.sources}
            # 任何输出的滤波支撑都不可能同时跨越两个速率段
            self.assertLessEqual(len(seqs), 1)
            if seqs == {1}:
                self.assertGreaterEqual(item.value, new_min - 1e-9)
                self.assertLessEqual(item.value, new_max + 1e-9)

    def test_sources_carry_seq_and_offsets(self):
        ba = BoundaryAwareResampler(16000, quality="low_latency", session_id="s")
        out = ba.process(seq=7, sample_rate=16000, samples=[1.0, 2.0, 3.0, 4.0])
        self.assertTrue(out)
        sourced = [o for o in out if o.sources]
        self.assertTrue(sourced)
        for o in sourced:
            for src in o.sources:
                self.assertEqual(src.seq, 7)
                self.assertEqual(src.sample_rate, 16000)
                self.assertLess(src.start_sample, src.end_sample)

    def test_both_quality_tiers_share_output_contract(self):
        totals = {}
        for q in ("low_latency", "high"):
            ba = BoundaryAwareResampler(16000, quality=q, session_id="s")
            ba.process(seq=0, sample_rate=48000,
                       samples=[0.2 * i - 5 for i in range(64)])
            ba.finish()
            totals[q] = len(ba.outputs)
            for o in ba.outputs:
                d = o.to_dict()
                self.assertIn("value", d)
                self.assertIn("sources", d)
                self.assertIn("position", d)
        # 总输出数由速率比决定，与滤波精度无关
        self.assertEqual(totals["low_latency"], totals["high"])
        self.assertEqual(totals["low_latency"], 22)  # 64 * 16000/48000


class FunctionalAndShimTest(unittest.TestCase):
    def test_reorder_chunks_function(self):
        chunks = [
            {"session_id": "s", "seq": 2, "sample_rate": 8000, "samples": [1]},
            {"session_id": "s", "seq": 0, "sample_rate": 8000, "samples": [0]},
            {"session_id": "s", "seq": 1, "sample_rate": 8000, "samples": [0]},
        ]
        ordered = reorder_chunks(chunks)
        self.assertEqual([c.seq for c in ordered], [0, 1, 2])

    def test_module_path_shims_import(self):
        import importlib

        for name in (
            "operator_audio_chunks.operator",
            "operator_audio_chunks.pipeline",
            "operator_audio_chunks.model",
            "operator_audio_chunks.models",
            "operator_audio_chunks.reorder",
            "operator_audio_chunks.resampler",
        ):
            mod = importlib.import_module(name)
            self.assertIsNotNone(mod)

    def test_common_aliases_present(self):
        import operator_audio_chunks as pkg

        for name in (
            "Operator", "OLA", "Config", "AudioFrame", "FrameSource",
            "GapReport", "StreamingReorderOLAOperator",
        ):
            self.assertTrue(hasattr(pkg, name), msg=name)


if __name__ == "__main__":
    unittest.main()
