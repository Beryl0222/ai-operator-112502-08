import json
from pathlib import Path
import unittest


class FixtureContractTest(unittest.TestCase):
    def test_audio_fixture_exposes_rate_change_and_sequence_gap(self):
        base = Path(__file__).parents[1]
        payload = json.loads((base / "data/input.json").read_text(encoding="utf-8"))
        chunks = payload["chunks"]
        self.assertEqual([chunk["seq"] for chunk in chunks], [0, 1, 3])
        self.assertEqual(len({chunk["sample_rate"] for chunk in chunks}), 2)
        self.assertTrue(all(chunk["samples"] for chunk in chunks))

if __name__ == "__main__":
    unittest.main()
