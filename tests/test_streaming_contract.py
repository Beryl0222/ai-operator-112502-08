"""基于 data/input.json 的端到端契约：时间基准、撤回、恢复计划、检查点。"""

import json
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from operator_audio_chunks import (  # noqa: E402
    Retraction,
    StreamingAudioOperator,
    process_payload,
)


def load_fixture():
    return json.loads((BASE / "data" / "input.json").read_text(encoding="utf-8"))


def make_operator(**kw):
    payload = load_fixture()
    kw.setdefault("hop", payload["hop"])
    kw.setdefault("model_rate", 16000)
    kw.setdefault("quality", "low_latency")
    return StreamingAudioOperator(payload["session_id"], **kw)


class FixtureStreamingTest(unittest.TestCase):
    def setUp(self):
        self.payload = load_fixture()
        self.op = make_operator()
        self.results = [self.op.ingest(c) for c in self.payload["chunks"]]

    def test_fixture_shape(self):
        self.assertEqual([c["seq"] for c in self.payload["chunks"]], [0, 1, 3])

    def test_exact_frame_values_and_stable_boundary(self):
        frames = self.op.frames
        self.assertEqual([f.index for f in frames], [0, 1, 2, 3, 4, 5])
        expected = {
            0: [0, 1, 2, 3],
            1: [2, 3, 4, 5],
            2: [4, 5, 5.5, 6],
            3: [5.5, 6, 6.5, 7],
            4: [6.5, 7, 7, 9],
            5: [7, 9, 9.5, 10],
        }
        for f in frames:
            self.assertEqual(
                list(f.samples), [float(x) for x in expected[f.index]]
            )
        self.assertEqual(
            [f.index for f in frames if f.stable], [0, 1, 2, 3]
        )
        self.assertEqual(
            [f.index for f in frames if f.provisional], [4, 5]
        )

    def test_stable_time_base_identity_and_pts(self):
        for f in self.op.frames:
            self.assertEqual(f.frame_id, f"call-3#F{f.index}")
            self.assertEqual(f.pts_samples, f.index * 2)
            self.assertAlmostEqual(
                f.pts_seconds, f.index * 2 / 16000, places=12
            )
            self.assertEqual(len(f.samples), f.window_size)

    def test_source_ranges_identify_seq_rate_and_offset(self):
        by_index = {f.index: f for f in self.op.frames}
        # 跨采样率边界的帧同时携带旧速率段与新速率段来源
        f1 = by_index[1]
        self.assertEqual(
            [(r.seq, r.sample_rate) for r in f1.sources],
            [(0, 16000), (1, 8000)],
        )
        self.assertEqual(
            (f1.sources[0].start_sample, f1.sources[0].end_sample), (2, 5)
        )
        # 每个来源段的秒时长与样本数/速率自洽
        for f in self.op.frames:
            for r in f.sources:
                self.assertAlmostEqual(
                    r.duration_seconds,
                    (r.end_sample - r.start_sample) / r.sample_rate,
                    places=12,
                )

    def test_recovery_plan_distinguishes_replay_and_resample(self):
        plan = self.op.recovery_plan()
        self.assertEqual(plan.missing_seqs, (2,))
        self.assertEqual(plan.resample_required_seqs, (2,))
        self.assertEqual(plan.resample_required_ranges, ({"start_seq": 2, "end_seq": 2},))
        # 已收到的块都在日志中，可确定性重放
        self.assertEqual(plan.replayable_seqs, (0, 1, 3))
        # 连续前缀到 seq1；seq3 是缺口后已缓存、补齐后可重放的块
        self.assertEqual(plan.contiguous_through_seq, 1)
        self.assertEqual(plan.checkpoint_seq, 1)
        self.assertEqual(plan.replay_from_seq, 2)
        self.assertEqual(plan.buffered_after_gap_seqs, (3,))
        self.assertEqual(plan.stable_frame_indices, (0, 1, 2, 3))

    def test_late_seq2_retracts_only_boundary_frames(self):
        result = self.op.ingest(
            {"seq": 2, "sample_rate": 8000, "samples": [8]}
        )
        retracted = [(r.index, r.revision) for r in result.retractions]
        self.assertEqual(retracted, [(4, 1), (5, 1)])
        self.assertEqual(
            [(f.index, f.revision) for f in result.frames],
            [(4, 2), (5, 2), (6, 1)],
        )
        # 0..3 保留：结果明确给出 retained 列表，且不含撤回
        retained_ids = set(result.retained_frame_ids)
        for i in range(4):
            self.assertIn(f"call-3#F{i}", retained_ids)
        self.assertNotIn("call-3#F4", retained_ids)
        self.assertNotIn("call-3#F5", retained_ids)

        by_index = {f.index: f for f in self.op.frames}
        self.assertEqual(list(by_index[4].samples), [6.5, 7, 7.5, 8])
        self.assertEqual(list(by_index[5].samples), [7.5, 8, 8.5, 9])
        self.assertEqual(list(by_index[6].samples), [8.5, 9, 9.5, 10])
        # 缺口消除后恢复计划不再要求补传
        plan = self.op.recovery_plan()
        self.assertEqual(plan.missing_seqs, ())
        self.assertEqual(plan.resample_required_seqs, ())

    def test_duplicate_delivery_is_idempotent_at_operator_level(self):
        before = len(self.op.frames)
        result = self.op.ingest(
            {"seq": 1, "sample_rate": 8000, "samples": [5, 6, 7]}
        )
        self.assertEqual(result.status, "duplicate_ignored")
        self.assertEqual(result.frames, [])
        self.assertEqual(result.retractions, [])
        self.assertEqual(len(self.op.frames), before)

    def test_checkpoint_restore_reproduces_future_events(self):
        # 在 seq 0,1 之后打检查点
        op = make_operator()
        op.ingest(self.payload["chunks"][0])
        op.ingest(self.payload["chunks"][1])
        cp = op.checkpoint()

        restored = StreamingAudioOperator.from_checkpoint(cp)
        # 重放 seq3：恢复出的算子应产生与原始算子一致的新帧
        original_event = op.ingest(self.payload["chunks"][2])
        restored_event = restored.ingest(self.payload["chunks"][2])
        self.assertEqual(
            [f.to_dict() for f in restored_event.frames],
            [f.to_dict() for f in original_event.frames],
        )
        self.assertEqual(
            [r.to_dict() for r in restored_event.retractions],
            [r.to_dict() for r in original_event.retractions],
        )

    def test_full_checkpoint_roundtrip_deep_equality(self):
        self.op.ingest({"seq": 2, "sample_rate": 8000, "samples": [8]})
        cp = self.op.checkpoint()
        restored = StreamingAudioOperator.from_checkpoint(cp)
        self.assertEqual(
            [f.to_dict() for f in restored.frames],
            [f.to_dict() for f in self.op.frames],
        )
        self.assertEqual(
            [r.to_dict() for r in restored.retractions_log],
            [r.to_dict() for r in self.op.retractions_log],
        )

    def test_replay_from_seq_is_deterministic(self):
        self.op.ingest({"seq": 2, "sample_rate": 8000, "samples": [8]})
        replayed = self.op.replay(from_seq=2)
        # 从 seq2 重放得到的帧是非空且确定性的
        self.assertTrue(replayed)
        again = self.op.replay(from_seq=2)
        self.assertEqual(
            [f.samples for f in replayed], [f.samples for f in again]
        )

    def test_decoder_ack_accepted(self):
        self.op.acknowledge("call-3#F0")
        self.op.acknowledge(1)
        self.assertIn("call-3#F0", self.op.acked_frames)
        self.assertIn("call-3#F1", self.op.acked_frames)


class WaitGapPolicyTest(unittest.TestCase):
    def test_wait_policy_holds_post_gap_frames_until_filled(self):
        payload = load_fixture()
        op = StreamingAudioOperator(
            payload["session_id"],
            hop=2,
            model_rate=16000,
            quality="low_latency",
            gap_policy="wait",
        )
        for c in payload["chunks"]:  # 0,1,3
            op.ingest(c)
        # seq3 在缺口后被缓存，但不进入时间线：只有 seq0,1 产生的 4 帧
        self.assertEqual([f.index for f in op.frames], [0, 1, 2, 3])
        result = op.ingest({"seq": 2, "sample_rate": 8000, "samples": [8]})
        # 从未交付过 4,5，因此补齐时没有撤回，只有新帧
        self.assertEqual(result.retractions, [])
        self.assertEqual([f.index for f in result.frames], [4, 5, 6])


class ProcessPayloadTest(unittest.TestCase):
    def test_process_payload_is_json_safe(self):
        report = process_payload(
            BASE / "data" / "input.json", quality="low_latency"
        )
        text = json.dumps(report, ensure_ascii=False)
        self.assertIn("call-3#F4", text)
        self.assertEqual(report["recovery_plan"]["missing_seqs"], [2])
        # 每个帧字典都带有解码器确认所需的来源范围
        for f in report["frames"]:
            self.assertIn("source_range", f)
            self.assertTrue(f["sources"])


if __name__ == "__main__":
    unittest.main()
