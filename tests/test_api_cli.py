"""多会话路由、JSON/CLI 入口与公共 API 表面。"""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from operator_audio_chunks import (  # noqa: E402
    AudioChunkOperator,
    OverlapAdd,
    ReorderBuffer,
    process_payload,
)
from operator_audio_chunks.__main__ import main as cli_main  # noqa: E402


class MultiSessionTest(unittest.TestCase):
    def test_chunks_are_routed_by_session(self):
        mgr = AudioChunkOperator(hop=2, model_rate=16000, enable_ola=False)
        mgr.process({"session_id": "A", "seq": 0, "sample_rate": 16000,
                     "samples": [0, 1, 2, 3, 4]})
        mgr.process({"session_id": "B", "seq": 0, "sample_rate": 16000,
                     "samples": [9, 8, 7, 6, 5]})
        self.assertEqual(set(mgr.sessions), {"A", "B"})
        fa = mgr.session("A").frames[0].samples
        fb = mgr.session("B").frames[0].samples
        self.assertEqual(list(fa), [0, 1, 2, 3])
        self.assertEqual(list(fb), [9, 8, 7, 6])

    def test_dict_style_access_on_objects(self):
        mgr = AudioChunkOperator(hop=2, model_rate=16000, enable_ola=False)
        result = mgr.process(
            {"session_id": "A", "seq": 0, "sample_rate": 16000,
             "samples": [0, 1, 2, 3, 4]}
        )
        frame = result.frames[0]
        self.assertEqual(frame["index"], 0)
        self.assertEqual(frame["session_id"], "A")
        self.assertIn("frame_id", frame)
        self.assertEqual(frame.get("nope", 42), 42)

    def test_recovery_plan_routing(self):
        mgr = AudioChunkOperator(hop=2, model_rate=16000)
        mgr.process({"session_id": "A", "seq": 0, "sample_rate": 16000,
                     "samples": [0, 1, 2, 3]})
        plan = mgr.recovery_plan("A")
        self.assertEqual(plan.session_id, "A")


class CliAndPayloadTest(unittest.TestCase):
    def test_cli_summary_prints_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main([
                "--input", str(BASE / "data" / "input.json"),
                "--quality", "low_latency", "--summary-only",
            ])
        self.assertEqual(code, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["session_id"], "call-3")
        self.assertEqual(out["missing_seqs"], [2])
        self.assertIn(0, out["stable_indices"])

    def test_cli_inject_late_seq_shows_retraction(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main([
                "--input", str(BASE / "data" / "input.json"),
                "--quality", "low_latency", "--summary-only",
                "--inject", "2",
            ])
        out = json.loads(buf.getvalue())
        self.assertTrue(out["retractions"])
        self.assertEqual(out["missing_seqs"], [])

    def test_process_payload_accepts_mapping_and_path(self):
        payload = json.loads((BASE / "data" / "input.json").read_text("utf-8"))
        r1 = process_payload(payload, quality="low_latency")
        r2 = process_payload(
            BASE / "data" / "input.json", quality="low_latency"
        )
        self.assertEqual(
            [f["index"] for f in r1["frames"]],
            [f["index"] for f in r2["frames"]],
        )

    def test_public_surface_exposes_core_operators(self):
        # 这些名称对应题目明确要求的两类算子
        self.assertTrue(callable(ReorderBuffer))
        self.assertTrue(callable(OverlapAdd))
        self.assertTrue(callable(AudioChunkOperator))


if __name__ == "__main__":
    unittest.main()
