"""稳定时间基准、有理数无漂移、任意到达顺序最终收敛。"""

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from operator_audio_chunks import StreamingAudioOperator  # noqa: E402


def deterministic_samples(seed, n):
    out = []
    state = seed & 0x7FFFFFFF
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(((state >> 8) % 2001) - 1000)
    return out


def make_spec():
    # 含非整除比率 44100/16000、24000/16000，以及多次速率切换
    return [
        {"seq": 0, "sample_rate": 44100, "samples": deterministic_samples(1, 97)},
        {"seq": 1, "sample_rate": 44100, "samples": deterministic_samples(2, 61)},
        {"seq": 2, "sample_rate": 8000, "samples": deterministic_samples(3, 53)},
        {"seq": 3, "sample_rate": 24000, "samples": deterministic_samples(4, 71)},
        {"seq": 4, "sample_rate": 16000, "samples": deterministic_samples(5, 40)},
    ]


def run_order(spec, order, quality="high", with_gap_fill=None):
    op = StreamingAudioOperator(
        "s", hop=32, window_size=64, model_rate=16000,
        quality=quality, enable_ola=False,
    )
    stable_ids = set()
    for seq in order:
        r = op.ingest(next(c for c in spec if c["seq"] == seq))
        for f in r.frames:
            if f.stable:
                stable_ids.add((f.index, f.revision))
        for c in r.confirmations:
            stable_ids.add((c.index, c.revision))
        for rt in r.retractions:
            if (rt.index, rt.revision) in stable_ids:
                raise AssertionError(
                    f"stable frame {rt.index} r{rt.revision} was retracted"
                )
    return op


class TimeBaseTest(unittest.TestCase):
    def test_pts_grid_is_exact_and_monotonic(self):
        op = run_order(make_spec(), [0, 1, 2, 3, 4], quality="low_latency")
        indices = [f.index for f in op.frames]
        self.assertEqual(indices, list(range(len(indices))))
        for f in op.frames:
            self.assertEqual(f.pts_samples, f.index * 32)
            self.assertAlmostEqual(
                f.pts_seconds * 16000, f.index * 32, places=9
            )
            self.assertGreater(f.end_seconds, f.pts_seconds)

    def test_source_totals_match_raw_durations_without_drift(self):
        spec = make_spec()
        op = run_order(spec, [0, 1, 2, 3, 4])
        # 每个来源段时间区间与 (样本数/速率) 自洽
        for f in op.frames:
            for r in f.sources:
                self.assertAlmostEqual(
                    r.end_seconds - r.start_seconds,
                    (r.end_sample - r.start_sample) / r.sample_rate,
                    places=9,
                )
        # 时间线总时长 == 各块原始时长之和（有理数精确累加，无重采样取整漂移）
        total_raw_seconds = sum(
            len(c["samples"]) / c["sample_rate"] for c in spec
        )
        last_end = max(
            r.end_seconds for f in op.frames for r in f.sources
        )
        self.assertLessEqual(last_end, total_raw_seconds + 1e-9)
        self.assertGreater(last_end, total_raw_seconds - 64 / 16000 - 1e-6)

    def test_all_arrival_orders_converge_same_frames(self):
        spec = make_spec()
        for quality in ("low_latency", "high"):
            reference = run_order(
                spec, [0, 1, 2, 3, 4], quality=quality
            ).frames
            ref_key = [
                (
                    f.index,
                    tuple(round(v, 9) for v in f.samples),
                    tuple(
                        (r.seq, r.sample_rate, r.start_sample, r.end_sample)
                        for r in f.sources
                    ),
                    f.status.value,
                )
                for f in reference
            ]
            for order in itertools.permutations([0, 1, 2, 3, 4]):
                op = run_order(spec, list(order), quality=quality)
                got = [
                    (
                        f.index,
                        tuple(round(v, 9) for v in f.samples),
                        tuple(
                            (r.seq, r.sample_rate, r.start_sample, r.end_sample)
                            for r in f.sources
                        ),
                        f.status.value,
                    )
                    for f in op.frames
                ]
                self.assertEqual(got, ref_key, msg=f"order={order} q={quality}")

    def test_gap_and_late_fill_all_permutations_converge(self):
        spec = make_spec()
        # 所有把 seq2 当作“最后补的缺口”的排列
        others = [0, 1, 3, 4]
        for quality in ("low_latency", "high"):
            ref = run_order(spec, [0, 1, 2, 3, 4], quality=quality)
            ref_key = [(f.index, f.samples, f.sources) for f in ref.frames]
            for perm in itertools.permutations(others):
                order = list(perm) + [2]
                op = run_order(spec, order, quality=quality)
                got_key = [(f.index, f.samples, f.sources) for f in op.frames]
                self.assertEqual(
                    len(got_key), len(ref_key), msg=f"order={order}"
                )
                for (gi, gv, gs), (ri, rv, rs) in zip(got_key, ref_key):
                    self.assertEqual(gi, ri)
                    for a, b in zip(gv, rv):
                        self.assertAlmostEqual(a, b, places=8)
                    self.assertEqual(
                        [(x.seq, x.start_sample, x.end_sample) for x in gs],
                        [(x.seq, x.start_sample, x.end_sample) for x in rs],
                    )


if __name__ == "__main__":
    unittest.main()
