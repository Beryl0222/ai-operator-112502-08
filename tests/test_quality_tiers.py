"""高/低延迟两档滤波：精度不同，但时间/来源契约一致。"""

import json
import math
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from operator_audio_chunks import (  # noqa: E402
    LinearKernel,
    StreamingAudioOperator,
    WindowedSincKernel,
)
from operator_audio_chunks.kernels import make_kernel  # noqa: E402


class KernelPrecisionTest(unittest.TestCase):
    def test_tiers_have_different_support(self):
        low = make_kernel("low_latency", 8000, 16000)
        high = make_kernel("high", 8000, 16000)
        self.assertIsInstance(low, LinearKernel)
        self.assertIsInstance(high, WindowedSincKernel)
        self.assertGreater(high.support_radius, low.support_radius)

    def test_high_quality_differs_from_linear_on_fractional_position(self):
        y = [0.0, 1.0, 1.0, 0.0]
        p = __import__("fractions").Fraction(1, 2)
        linear = LinearKernel().interpolate(y, p)
        hq = WindowedSincKernel(16000, 16000, radius=4).interpolate(y, p)
        self.assertAlmostEqual(linear, 0.5, places=9)
        self.assertFalse(math.isclose(hq, linear, abs_tol=1e-6))

    def test_downsampling_kernel_uses_antialias_cutoff(self):
        hq = WindowedSincKernel(16000, 8000, radius=8)
        self.assertLess(hq.cutoff, 1.0)
        up = WindowedSincKernel(8000, 16000, radius=8)
        self.assertEqual(up.cutoff, 1.0)


class TierContractParityTest(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((BASE / "data" / "input.json").read_text("utf-8"))

    def _run(self, quality):
        op = StreamingAudioOperator(
            "call-3", hop=2, model_rate=16000, quality=quality
        )
        for c in self.payload["chunks"]:
            op.ingest(c)
        return op

    def test_frame_identity_time_and_source_contract_match(self):
        low = self._run("low_latency")
        high = self._run("high")
        self.assertEqual(
            [f.index for f in low.frames], [f.index for f in high.frames]
        )
        for fl, fh in zip(low.frames, high.frames):
            # 相同的稳定时间基准
            self.assertEqual(fl.pts_samples, fh.pts_samples)
            self.assertEqual(fl.hop, fh.hop)
            self.assertEqual(fl.window_size, fh.window_size)
            # 相同的契约级来源（seq/速率/块内区间），与滤波精度解耦
            self.assertEqual(
                [(r.seq, r.sample_rate, r.start_sample, r.end_sample)
                 for r in fl.sources],
                [(r.seq, r.sample_rate, r.start_sample, r.end_sample)
                 for r in fh.sources],
            )
            # 序列化契约键集合一致
            self.assertEqual(set(fl.to_dict()), set(fh.to_dict()))
            self.assertTrue(fl.to_dict()["source_range"])

    def test_values_can_differ_between_tiers_on_rich_signal(self):
        # 高频衰减信号下两档结果不应逐值相同
        samples = [
            math.sin(i * 0.9) + 0.5 * math.sin(i * 2.3)
            for i in range(120)
        ]
        chunk = {"seq": 0, "sample_rate": 16000, "samples": samples}

        def run(q):
            op = StreamingAudioOperator(
                "s", hop=16, window_size=32, model_rate=8000, quality=q
            )
            op.ingest(chunk)
            return op.frames

        low, high = run("low_latency"), run("high")
        differences = sum(
            1 for a, b in zip(low, high)
            for x, y in zip(a.samples, b.samples)
            if not math.isclose(x, y, abs_tol=1e-6)
        )
        self.assertGreater(differences, 0)

    def test_string_aliases_normalize(self):
        from operator_audio_chunks.types import coerce_quality, Quality

        for alias in ("low", "fast", "realtime", "LOW"):
            self.assertIs(coerce_quality(alias), Quality.LOW_LATENCY)
        self.assertIs(coerce_quality("hq"), Quality.HIGH)


if __name__ == "__main__":
    unittest.main()
