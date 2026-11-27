"""OverlapAdd：重建精度、来源合并、乱序提交、修订/删除传播。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from operator_audio_chunks import OLA, Frame, OverlapAdd, overlap_add  # noqa: E402,F401
from operator_audio_chunks.types import FrameStatus  # noqa: E402


def make_frame(index, values, hop=2, window=4, session="s", revision=1, sources=()):
    return Frame(
        session_id=session,
        index=index,
        hop=hop,
        window_size=window,
        model_rate=16000,
        samples=tuple(values),
        sources=tuple(sources),
        status=FrameStatus.STABLE,
        revision=revision,
    )


class ColaReconstructionTest(unittest.TestCase):
    def test_rect_window_normalized_reconstructs_signal(self):
        signal = [float(i % 11) - 5 for i in range(24)]
        hop, w = 2, 4
        ola = OverlapAdd("s", w, hop, window="rect", normalize=True)
        n_frames = (len(signal) - w) // hop + 1
        for k in range(n_frames):
            ola.submit(make_frame(k, signal[k * hop:k * hop + w]))
        blocks = ola.delivered_blocks()
        out = [v for b in blocks for v in b.samples]
        # 归一化矩形窗在重叠区是相同样本的平均，整体精确重建
        self.assertEqual(out, signal[:len(out)])

    def test_hann_50pct_overlap_interior_is_perfect(self):
        signal = [float((i * 29) % 17) - 8 for i in range(40)]
        hop, w = 2, 4
        ola = OverlapAdd("s", w, hop, window="hann", normalize=True)
        n_frames = (len(signal) - w) // hop + 1
        for k in range(n_frames):
            ola.submit(make_frame(k, signal[k * hop:k * hop + w]))
        for b in ola.delivered_blocks():
            if b.index >= 1 and (b.index + 1) * hop <= len(signal):
                for j, v in enumerate(b.samples):
                    src_idx = b.index * hop + j
                    self.assertAlmostEqual(
                        v, signal[src_idx], places=9,
                        msg=f"block {b.index} sample {j}",
                    )

    def test_functional_overlap_add(self):
        frames = [make_frame(k, [float(k), k + 1, k + 2, k + 3]) for k in range(4)]
        blocks = overlap_add(frames, window="rect")
        self.assertEqual([b.index for b in blocks], [0, 1, 2, 3])


class OlaRevisionTest(unittest.TestCase):
    def _filled_ola(self):
        ola = OverlapAdd("s", 4, 2, window="rect")
        for k in range(4):
            ola.submit(make_frame(k, [k + 0.1 * j for j in range(4)]))
        self.assertEqual([b.index for b in ola.delivered_blocks()], [0, 1, 2, 3])
        return ola

    def test_revised_frame_retracts_only_overlapping_blocks(self):
        ola = self._filled_ola()
        result = ola.submit(
            make_frame(1, [9.0, 9.0, 9.0, 9.0], revision=2)
        )
        retracted = sorted(r.index for r in result.retractions)
        # 帧1覆盖样本2..5 => 块1[2,4) 与块2[4,6)
        self.assertEqual(retracted, [1, 2])
        revised = {b.index: b for b in result.blocks}
        self.assertEqual(revised[1].revision, 2)
        self.assertEqual(revised[2].revision, 2)
        # 块0、块3不受影响
        self.assertEqual(ola.get_block(0).revision, 1)
        self.assertEqual(ola.get_block(3).revision, 1)

    def test_batch_emits_single_retraction_per_block(self):
        ola = self._filled_ola()
        result = ola.apply_batch([
            make_frame(1, [9.0] * 4, revision=2),
            make_frame(2, [8.0] * 4, revision=2),
        ])
        # 同一批次内每个块最多撤回一次（不产生中间修订噪声）
        counts = {}
        for r in result.retractions:
            counts[r.index] = counts.get(r.index, 0) + 1
        self.assertTrue(all(c == 1 for c in counts.values()))
        self.assertEqual(sorted(counts), [1, 2, 3])

    def test_remove_frame_retracts_dependent_blocks(self):
        ola = self._filled_ola()
        result = ola.retract_frame(2)
        retracted = sorted(r.index for r in result.retractions)
        # 删除帧2后，块2、块3的完整依赖被破坏
        self.assertEqual(retracted, [2, 3])
        self.assertIsNone(ola.get_block(2))
        self.assertIsNone(ola.get_block(3))
        self.assertIsNotNone(ola.get_block(1))

    def test_out_of_order_submission_emits_when_complete(self):
        ola = OverlapAdd("s", 4, 2, window="rect")
        r1 = ola.submit(make_frame(1, [4, 5, 6, 7]))
        self.assertEqual(r1.blocks, [])  # 块1还缺帧0
        r0 = ola.submit(make_frame(0, [0, 1, 2, 3]))
        self.assertEqual(sorted(b.index for b in r0.blocks), [0, 1])

    def test_stale_revision_is_rejected(self):
        ola = self._filled_ola()
        # 先把帧1升级到 revision=2，块1随之重算
        ola.submit(make_frame(1, [9.0] * 4, revision=2))
        self.assertEqual(ola.get_block(1).revision, 2)
        before = tuple(ola.get_block(1).samples)
        # 迟到的 revision=1 重放必须被拒绝，不能让状态/数值回退
        stale_result = ola.submit(make_frame(1, [0.0] * 4, revision=1))
        self.assertEqual(stale_result.blocks, [])
        self.assertEqual(stale_result.retractions, [])
        self.assertEqual(ola.get_block(1).revision, 2)
        self.assertEqual(tuple(ola.get_block(1).samples), before)

    def test_block_carries_source_provenance(self):
        from operator_audio_chunks.types import SourceRange
        sr0 = SourceRange("s", 0, 16000, 0, 4)
        sr1 = SourceRange("s", 1, 8000, 0, 3)
        ola = OverlapAdd("s", 4, 2)
        ola.submit(make_frame(0, [1, 2, 3, 4], sources=[sr0]))
        ola.submit(make_frame(1, [3, 4, 5, 6], sources=[sr1]))
        block1 = ola.get_block(1)
        seqs = [r.seq for r in block1.sources]
        self.assertIn(0, seqs)
        self.assertIn(1, seqs)

    def test_ack_tracking(self):
        ola = self._filled_ola()
        ola.acknowledge("s#B1")
        ola.acknowledge(2)
        self.assertIn(1, ola.acked_blocks)
        self.assertIn(2, ola.acked_blocks)


if __name__ == "__main__":
    unittest.main()
