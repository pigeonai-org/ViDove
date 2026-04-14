import __init_path__
import sys
import types
import unittest

openai_stub = types.ModuleType("openai")


class _AzureOpenAI:
    pass


openai_stub.AzureOpenAI = _AzureOpenAI
sys.modules.setdefault("openai", openai_stub)

from src.SRT.srt import SrtScript, split_script


class ChunkAlignmentTests(unittest.TestCase):
    def test_split_script_preserves_last_chunk_range(self):
        script = "s0\n\ns1\n\ns2\n\ns3\n\ns4"

        chunks, ranges = split_script(script, chunk_size=10)

        self.assertEqual(chunks, ["s0\n\ns1\n\ns2", "s3\n\ns4"])
        self.assertEqual(ranges, [(1, 3), (4, 5)])

    def test_set_translation_writes_full_tail_range(self):
        srt = SrtScript("EN", "ZH", segments=["s0", "s1", "s2", "s3", "s4"], task_id="test")

        srt.set_translation("t0\n\nt1\n\nt2", (1, 3), "test-model", "video")
        srt.set_translation("t3\n\nt4", (4, 5), "test-model", "video")

        self.assertEqual(
            [seg.translation for seg in srt.segments],
            ["t0", "t1", "t2", "t3", "t4"],
        )

    def test_get_source_only_uses_consistent_segment_separator(self):
        srt = SrtScript("EN", "ZH", segments=["alpha", "beta", "gamma"], task_id="test")

        self.assertEqual(srt.get_source_only(), "alpha\n\nbeta\n\ngamma")


if __name__ == "__main__":
    unittest.main()
