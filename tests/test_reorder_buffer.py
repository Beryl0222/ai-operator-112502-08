"""序号重排、重复、冲突、缺口与检查点语义。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from operator_audio_chunks import (  # noqa: E402
    ChunkConflictError,
    DuplicateChunkError,
    ReorderBuffer,
)
from operator_audio_chunks.types import AudioChunk  # noqa: E402


def chunk(seq, rate=16000, samples=None, session="s1"):
    return AudioChunk(
        session_id=session,
        seq=seq,
        sample_rate=rate,
        samples=tuple(samples if samples is not None else [seq]),
    )


class ReorderBufferTest(unittest.TestCase):
    def test_out_of_order_insertion_is_sorted_by_seq(self):
        buf = ReorderBuffer("s1")
        for s in (3, 0, 2, 1):
            buf.add(chunk(s))
        self.assertEqual(buf.seqs(), [0, 1, 2, 3])
        self.assertEqual([c.seq for c in buf.ordered_chunks()], [0, 1, 2, 3])

    def test_contiguous_frontier_and_gap_ranges(self):
        buf = ReorderBuffer("s1")
        for s in (0, 1, 3, 5):
            buf.add(chunk(s))
        self.assertEqual(buf.contiguous_through(), 1)
        self.assertEqual(buf.missing_seqs(), [2, 4])
        self.assertEqual(buf.gap_ranges(), [(2, 2), (4, 4)])

    def test_multi_seq_gap_range(self):
        buf = ReorderBuffer("s1")
        buf.add(chunk(0))
        buf.add(chunk(4))
        self.assertEqual(buf.missing_seqs(), [1, 2, 3])
        self.assertEqual(buf.gap_ranges(), [(1, 3)])

    def test_missing_first_seq_has_no_frontier(self):
        buf = ReorderBuffer("s1")
        buf.add(chunk(2))
        self.assertIsNone(buf.contiguous_through())
        self.assertEqual(buf.missing_seqs(), [0, 1])

    def test_identical_duplicate_is_idempotent(self):
        buf = ReorderBuffer("s1")
        buf.add(chunk(1, samples=[7, 8]))
        outcome = buf.add(chunk(1, samples=[7, 8]))
        self.assertEqual(outcome.status, "duplicate_ignored")
        self.assertTrue(outcome.duplicate)
        self.assertEqual(buf.seqs(), [1])
        self.assertEqual(len(buf.arrival_order), 1)

    def test_strict_duplicate_raises(self):
        buf = ReorderBuffer("s1")
        buf.add(chunk(0))
        with self.assertRaises(DuplicateChunkError):
            buf.add(chunk(0), strict_duplicates=True)

    def test_conflicting_duplicate_is_not_silently_accepted(self):
        buf = ReorderBuffer("s1")
        buf.add(chunk(2, rate=8000, samples=[1, 2]))
        outcome = buf.add(chunk(2, rate=8000, samples=[9, 9]))
        self.assertEqual(outcome.status, "conflict")
        self.assertTrue(outcome.conflict)
        # 旧内容保持不变 —— 没有静默覆盖或拼接
        self.assertEqual(list(buf.get(2).samples), [1.0, 2.0])

    def test_conflict_error_policy_raises(self):
        buf = ReorderBuffer("s1")
        buf.add(chunk(2, samples=[1]))
        with self.assertRaises(ChunkConflictError):
            buf.add(chunk(2, samples=[2]), on_conflict="error")

    def test_conflict_replace_policy_overwrites_explicitly(self):
        buf = ReorderBuffer("s1")
        buf.add(chunk(2, samples=[1]))
        outcome = buf.add(chunk(2, samples=[2]), on_conflict="replace")
        self.assertEqual(outcome.status, "replaced")
        self.assertEqual(list(buf.get(2).samples), [2.0])

    def test_late_chunk_is_classified(self):
        buf = ReorderBuffer("s1")
        for s in (0, 1, 3):
            buf.add(chunk(s))
        buf.mark_delivered_through(buf.max_seq())
        outcome = buf.add(chunk(2))
        self.assertEqual(outcome.notice_kind.value, "late")

    def test_checkpoint_roundtrip_restores_state(self):
        buf = ReorderBuffer("s1")
        for s in (0, 1, 3):
            buf.add(chunk(s, samples=[s, s + 10]))
        cp = buf.checkpoint().to_dict()

        restored = ReorderBuffer.from_checkpoint(cp)
        self.assertEqual(restored.seqs(), [0, 1, 3])
        self.assertEqual(restored.missing_seqs(), [2])
        self.assertEqual(list(restored.get(1).samples), [1.0, 11.0])
        self.assertEqual(restored.session_id, "s1")

    def test_cross_session_chunk_rejected(self):
        buf = ReorderBuffer("s1")
        with self.assertRaises(ValueError):
            buf.add(chunk(0, session="s2"))


if __name__ == "__main__":
    unittest.main()
